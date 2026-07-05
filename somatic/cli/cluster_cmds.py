"""`somatic run` / `somatic down` / `somatic ps` — the split-model launcher CLI.

Thin argument-parsing skin over `somatic.cluster` (Cluster + teardown_run). All
the real work lives in the package so the CLI, the SDK, and the UI share one path.
"""

from __future__ import annotations

import signal
import time

import typer

from somatic.cluster import Cluster, teardown_run
from somatic.cluster.errors import ClusterError
from somatic.cluster.hosts import parse_hosts
from somatic.cluster.runstate import RunRecord


def cluster_run(
    model: str = typer.Argument(..., help="Model id, e.g. Qwen/Qwen3-1.7B (any Llama-family HF model)."),
    host: list[str] = typer.Option(..., "--host", "-h", help="A machine: 'localhost' or 'user@ip'. Repeat; first is the driver."),
    precision: str = typer.Option("bf16", help="Weight precision the cluster holds (bf16|fp16|fp32)."),
    mode: str = typer.Option("relay", help="Boundary wire: 'relay' (fp16, ~2x fewer bytes, near-exact), 'exact' (identity, provably-identical full precision), or 'compact' (int8, ~4x fewer bytes, more lossy). Run 'somatic verify' to see the tradeoff for your model."),
    port: int = typer.Option(8000, help="Port for the OpenAI API + chat UI on the driver."),
    expose: bool = typer.Option(False, "--expose", help="Bind to 0.0.0.0 so a UI on another device (or in Docker) can reach the API over the LAN."),
    headroom: float = typer.Option(0.80, help="Fraction of each host's free RAM the fit may use."),
    plan_only: bool = typer.Option(False, "--plan-only", help="Print the split and exit; launch nothing."),
    skip_preflight: bool = typer.Option(False, "--skip-preflight", help="Skip host preflight checks."),
) -> None:
    """Auto-split MODEL across the given hosts and serve it (OpenAI-compatible API + chat UI)."""

    hosts = parse_hosts(host)
    try:
        if plan_only:
            plan = Cluster.plan(model, hosts, precision=precision, headroom=headroom, quiet=False)
            typer.echo(f"\nsplit: {plan.layout}")
            raise typer.Exit(0)

        cluster = Cluster.launch(
            model, hosts, precision=precision, mode=mode,
            serve_host="0.0.0.0" if expose else "127.0.0.1",
            serve_port=port, headroom=headroom, skip_preflight=skip_preflight,
        )
    except ClusterError as exc:
        typer.secho(f"\nsomatic ✗ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    api = cluster.advertised_base_url
    typer.secho(f"\n  chat UI        {cluster.advertised_web_url}", fg=typer.colors.GREEN)
    typer.secho(f"  OpenAI API     {api}", fg=typer.colors.GREEN)
    typer.echo(f"  model name     {cluster.model}")
    typer.echo("")
    typer.echo("  point any OpenAI-compatible app (Open WebUI, LibreChat, Chatbox, the openai SDK)")
    typer.echo(f"  at the API above — no key needed. {'' if expose else 'Add --expose to reach it from another device.'}")
    typer.echo("  stop  somatic down   (Ctrl-C here also stops it)\n")

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))
    try:
        while not stop["now"]:
            time.sleep(0.4)
    finally:
        typer.echo("\nsomatic ▸ stopping …")
        cluster.down()
        typer.echo("somatic ▸ stopped.")


def cluster_down(
    run_id: str = typer.Argument(None, help="Run id to stop (default: latest)."),
    sweep: bool = typer.Option(False, "--sweep", help="Kill ALL somatic workers on the hosts of every recorded run."),
) -> None:
    """Stop a running cluster and free its machines."""

    if sweep:
        hosts = set()
        for rid in RunRecord.list_runs():
            for w in RunRecord.load(rid).workers:
                hosts.add(w.ssh)
        teardown_run(sweep_hosts=list(hosts) or ["localhost"])
        typer.secho("somatic ▸ swept all somatic workers.", fg=typer.colors.GREEN)
        return

    stopped = teardown_run(run_id)
    if stopped is None:
        typer.secho("somatic ▸ no running cluster found.", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    typer.secho(f"somatic ▸ stopped {stopped}.", fg=typer.colors.GREEN)


def cluster_verify(
    model: str = typer.Argument(..., help="Model id (any Llama-family HF model)."),
    host: list[str] = typer.Option(..., "--host", "-h", help="A machine: 'localhost' or 'user@ip'. Repeat; first is the driver."),
    precision: str = typer.Option("bf16", help="Weight precision the cluster holds."),
    max_new_tokens: int = typer.Option(48, help="Tokens to generate per prompt."),
    prompts: int = typer.Option(5, help="How many built-in prompts to test (max 5)."),
    headroom: float = typer.Option(0.80, help="Fraction of each host's free RAM the fit may use."),
    skip_preflight: bool = typer.Option(False, "--skip-preflight"),
) -> None:
    """Measure the exactness/bytes tradeoff of each boundary mode for MODEL.

    Runs prompts through the split cluster under exact (identity), relay (fp16),
    and compact (int8) boundaries and reports how faithfully each reproduces the
    exact reference, plus the wire bytes each costs.
    """

    from somatic.cluster.errors import ClusterError
    from somatic.cluster.verify import DEFAULT_PROMPTS, verify

    hosts = parse_hosts(host)
    try:
        report = verify(
            model, hosts,
            prompts=DEFAULT_PROMPTS[: max(1, min(prompts, len(DEFAULT_PROMPTS)))],
            precision=precision, max_new_tokens=max_new_tokens,
            headroom=headroom, skip_preflight=skip_preflight,
        )
    except ClusterError as exc:
        typer.secho(f"\nsomatic ✗ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo("")
    typer.secho(f"  model   {report.model}", fg=typer.colors.CYAN)
    typer.echo(f"  split   {report.layout}")
    typer.echo(f"  tested  {report.prompts} prompts × {report.max_new_tokens} tokens")
    typer.echo("")
    det = "yes ✓" if report.exact_deterministic else "NO ✗ (nondeterminism!)"
    exact_b = report.exact_wire_bytes_per_boundary_token
    typer.secho(f"  exact  (identity)  deterministic: {det}", fg=typer.colors.GREEN)
    typer.echo(f"                     wire {exact_b} B / boundary·token  (this is the reference)")
    typer.echo("")
    typer.echo(f"  {'mode':<9}{'strategy':<16}{'prompts exact':<15}{'token match':<14}{'wire vs exact'}")
    typer.echo(f"  {'-'*9}{'-'*16}{'-'*15}{'-'*14}{'-'*14}")
    for m in report.modes:
        ratio = m.wire_bytes_per_boundary_token / exact_b if exact_b else 1.0
        typer.echo(
            f"  {m.mode:<9}{m.strategy:<16}"
            f"{f'{m.prompts_token_exact}/{report.prompts}':<15}"
            f"{f'{m.token_match_rate*100:.0f}%':<14}"
            f"{m.wire_bytes_per_boundary_token} B  ({ratio:.2f}×)"
        )
    typer.echo("")
    typer.secho(
        "  exact is byte-identical to full precision; relay/compact trade measured "
        "fidelity for bytes.", fg=typer.colors.BRIGHT_BLACK,
    )


def cluster_provision(
    host: list[str] = typer.Option(..., "--host", "-h", help="A machine to set up: 'user@ip'. Repeat."),
    model: str = typer.Option(None, "--model", "-m", help="Also warm this model's cache on each host."),
    push_model: bool = typer.Option(False, "--push-model", help="Copy the model from THIS machine instead of downloading on the host (for hosts with slow/no internet)."),
    no_env: bool = typer.Option(False, "--no-env", help="Skip the dependency sync (code + model only)."),
) -> None:
    """Set up hosts so `somatic run` just works: push code, ensure deps, warm the model cache."""

    from somatic.cluster.provision import provision

    results = provision(
        parse_hosts_for_provision(host),
        model=model, push_model=push_model, sync_env=not no_env,
    )
    typer.echo("")
    all_ready = True
    for r in results:
        color = typer.colors.GREEN if r.ready else typer.colors.RED
        typer.secho(f"  {r.host}  {'ready ✓' if r.ready else 'NOT ready ✗'}", fg=color)
        for name, ok, detail in r.steps:
            mark = "✓" if ok else "✗"
            typer.echo(f"      {mark} {name}: {detail}")
        all_ready = all_ready and r.ready
    typer.echo("")
    if all_ready:
        typer.secho("  all hosts ready — run:  somatic run <model> " +
                    " ".join(f"--host {r.host}" for r in results), fg=typer.colors.GREEN)
    else:
        raise typer.Exit(1)


def parse_hosts_for_provision(tokens: list[str]) -> list[str]:
    # provision takes raw host strings (no layer pins); pass through.
    return tokens


def cluster_bench(
    model: str = typer.Argument(..., help="Model id (any Llama-family HF model)."),
    host: list[str] = typer.Option(..., "--host", "-h", help="A machine: 'localhost' or 'user@ip'. Repeat; first is the driver."),
    mode: str = typer.Option("relay", help="Boundary wire mode to benchmark."),
    steps: int = typer.Option(40, help="Timed decode steps (after warmup)."),
    no_frontier: bool = typer.Option(False, "--no-frontier", help="Skip the single-machine floor measurement."),
    headroom: float = typer.Option(0.80, help="Fraction of each host's free RAM the fit may use."),
    skip_preflight: bool = typer.Option(False, "--skip-preflight"),
) -> None:
    """Report reproducible, honest tok/s for MODEL split across the hosts.

    Times real decoding (warmed-up, synced) and, when the model fits the driver,
    also the memory-bandwidth frontier so you see how close the split runs to the
    physical floor. This is the number the rest of the category refuses to publish.
    """

    from somatic.cluster.bench import benchmark
    from somatic.cluster.errors import ClusterError

    try:
        r = benchmark(model, parse_hosts(host), mode=mode, steps=steps,
                      measure_frontier=not no_frontier, headroom=headroom,
                      skip_preflight=skip_preflight)
    except ClusterError as exc:
        typer.secho(f"\nsomatic ✗ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo("")
    typer.secho(f"  model    {r.model}   (mode: {r.mode})", fg=typer.colors.CYAN)
    typer.echo(f"  split    {r.layout}")
    typer.echo(f"  method   {r.steps} decode steps, warmed-up, wall-clock, synced")
    typer.echo("")
    typer.secho(f"  decode   {r.decode_tok_s:.1f} tok/s   ({r.decode_ms_per_token:.1f} ms/token)", fg=typer.colors.GREEN)
    typer.echo(f"  prefill  {r.prefill_tok_s:.0f} tok/s")
    if r.frontier_tok_s is not None:
        typer.echo("")
        typer.secho(f"  frontier {r.frontier_tok_s:.1f} tok/s   (single-machine memory-bandwidth floor)", fg=typer.colors.BRIGHT_BLACK)
        bar = "#" * int(round((r.pct_of_frontier or 0) / 5))
        typer.secho(f"  ON FRONTIER  {r.pct_of_frontier:.0f}%   [{bar:<20}]", fg=typer.colors.YELLOW)
        typer.secho("  (splitting gives capacity, not speed — the frontier is one machine's bandwidth)",
                    fg=typer.colors.BRIGHT_BLACK)
    typer.echo("")


def cluster_ps() -> None:
    """List recorded cluster runs and their workers."""

    runs = RunRecord.list_runs()
    if not runs:
        typer.echo("no cluster runs recorded.")
        return
    for rid in runs:
        rec = RunRecord.load(rid)
        typer.secho(f"{rid}  {rec.model_id}  ({rec.precision}, {rec.mode})  api :{rec.serve_port}", fg=typer.colors.CYAN)
        for w in rec.workers:
            where = "localhost" if w.is_local else w.ssh
            typer.echo(f"    [{w.layer_start:>2},{w.layer_end:>2})  {where}:{w.port}  pid={w.pid}")
