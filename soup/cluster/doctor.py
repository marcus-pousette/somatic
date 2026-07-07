"""Preflight each host before spawning anything — turn silent hangs into loud errors.

The failures this catches were all real, painful ones: an SSH key that isn't
authorized, a venv missing torch, a stale worker script that rejects
`--shard-loading`, or a model that isn't in the host's HF cache (which otherwise
manifests as a worker that starts and then tries to hit the network and hangs).
Each check returns an actionable message; `launch` refuses to proceed on any red.
"""

from __future__ import annotations

from dataclasses import dataclass

from soup.cluster.errors import PreflightError
from soup.cluster.hosts import Host, validate_model_id
from soup.cluster import ssh


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    detail: str


def preflight(host: Host, model_id: str) -> list[tuple[str, CheckResult]]:
    """Run the ordered checks for one host. Returns (name, result) pairs."""

    validate_model_id(model_id)  # never let an unsafe id reach the shell below
    checks: list[tuple[str, CheckResult]] = []

    reach = ssh.run(host, "echo soup-ok", timeout=12.0)
    checks.append((
        "ssh",
        CheckResult(reach.ok and "soup-ok" in reach.stdout,
                    "reachable" if reach.ok else f"unreachable: {reach.stderr or reach.stdout}"),
    ))
    if not checks[-1][1].ok:
        return checks  # nothing else will work

    deps = ssh.run(
        host,
        f"{host.remote_python()} -c 'import torch, transformers, safetensors; print(\"deps-ok\")'",
        timeout=40.0,
    )
    checks.append((
        "deps",
        CheckResult("deps-ok" in deps.stdout,
                    "torch+transformers+safetensors present" if "deps-ok" in deps.stdout
                    else f"missing deps in {host.remote_python()}: {deps.stderr or deps.stdout}"),
    ))

    script = ssh.run(host, "grep -c -- --shard-loading scripts/live_split_worker.py", timeout=12.0)
    has_shard = script.ok and script.stdout.strip() not in ("", "0")
    checks.append((
        "worker-script",
        CheckResult(has_shard,
                    "live_split_worker.py supports --shard-loading" if has_shard
                    else "scripts/live_split_worker.py is stale/missing --shard-loading — rsync the repo"),
    ))

    cached = ssh.run(
        host,
        f"{host.remote_python()} -c \"from huggingface_hub import try_to_load_from_cache as t; "
        f"import sys; sys.exit(0 if t('{model_id}','config.json') else 3)\"",
        timeout=25.0,
    )
    checks.append((
        "model-cache",
        CheckResult(cached.returncode == 0,
                    f"{model_id} is cached" if cached.returncode == 0
                    else f"{model_id} not in HF cache on {host.display} — download it there first "
                         f"(and if you rsynced the cache, repair refs/main)"),
    ))
    return checks


def assert_ready(host: Host, model_id: str) -> list[tuple[str, CheckResult]]:
    results = preflight(host, model_id)
    failed = [(name, r) for name, r in results if not r.ok]
    if failed:
        lines = "\n".join(f"    ✗ {name}: {r.detail}" for name, r in failed)
        raise PreflightError(f"preflight failed on {host.display}:\n{lines}")
    return results
