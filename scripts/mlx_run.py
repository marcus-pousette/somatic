"""One command to run an MLX split cluster: deploy → launch workers → drive.

The MLX backend counterpart to `soup run`. It deploys the two mlx-only files
to each remote host, launches a worker there over SSH, TCP-health-gates it, then
runs `MLXClusterEngine` (driver = embed/head + its own layer range; remote workers
= the rest) and generates. Tears the workers down on exit.

Run it with the MLX env + soup on PYTHONPATH:

    PYTHONPATH=. ~/mlxenv/bin/python scripts/mlx_run.py \
        --model Qwen/Qwen3-14B --local 0:20 \
        --worker you@other-machine=192.168.1.42:5601:20:40 \
        --identity ~/.ssh/<key> --remote-python '~/mlxenv/bin/python'

A `--worker` is ``ssh_target=ip:port:start:end`` — ``ssh_target`` is what SSH/scp
use, ``ip`` is what the driver's socket dials. Full launcher/provisioner/CLI
integration (auto-fit, mlx-env provisioning, OpenAI serving) is the next slice.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

from soup.serving.mlx_engine import MLXClusterEngine, RemoteRange

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_FILES = [
    os.path.join(REPO, "soup", "serving", "mlx_shard.py"),
    os.path.join(REPO, "scripts", "mlx_split_worker.py"),
]


@dataclass
class Worker:
    ssh_target: str  # e.g. user@192.168.1.42 — for ssh/scp
    ip: str          # e.g. 192.168.1.42     — for the driver socket
    port: int
    start: int
    end: int


def parse_worker(spec: str) -> Worker:
    ssh_target, rest = spec.split("=", 1)
    ip, port, start, end = rest.rsplit(":", 3)
    return Worker(ssh_target, ip, int(port), int(start), int(end))


def ssh_args(identity: str | None) -> list[str]:
    args = ["-o", "ConnectTimeout=25", "-o", "StrictHostKeyChecking=accept-new"]
    if identity:
        args += ["-i", os.path.expanduser(identity)]
    return args


def probe_free_bytes(ssh_target: str, identity: str | None, is_local: bool) -> int:
    """Free RAM on a host via Computer Soup's dependency-free probe snippet (vm_stat on
    macOS). Runs locally for the driver, over SSH (system python3) for a remote."""
    from soup.cluster.capacity import _PROBE_SNIPPET

    b64 = base64.b64encode(_PROBE_SNIPPET.encode()).decode()
    expr = f"exec(__import__('base64').b64decode('{b64}').decode())"  # one token, no spaces
    if is_local:
        out = subprocess.run([sys.executable, "-c", expr], capture_output=True, text=True).stdout
    else:
        # SSH runs the command through the remote shell, so quote it as one token.
        remote = f"python3 -c {shlex.quote(expr)}"
        out = subprocess.run(
            ["ssh", *ssh_args(identity), ssh_target, remote],
            capture_output=True, text=True, timeout=45,
        ).stdout
    for line in out.splitlines():
        if line.startswith("FREE="):
            return int(line[5:])
    raise RuntimeError(f"free-RAM probe failed on {ssh_target}: {out!r}")


def auto_fit(model: str, host_specs: list[str], identity: str | None, headroom: float):
    """Probe each machine's free RAM and split the layers *proportional to
    capacity* (not greedy-fill) so the pipeline stays balanced — a machine with
    2× the room holds 2× the layers, rather than the first machine hogging them
    and hitting its own memory cliff. Driver = localhost (holds head + its range).
    Reuses Computer Soup's tested `model_footprint`. Returns (workers, driver_range)."""
    from soup.cluster.footprint import model_footprint
    from soup.cluster.hosts import Host

    overhead = int(1.5 * 1024 ** 3)  # torch/MLX + activations + KV headroom per host
    fp = model_footprint(model, precision="bf16", local_files_only=True)
    total_layer_bytes = sum(fp.per_layer_bytes)

    hosts = [Host(ssh="localhost", driver=True)]
    ip_by_ssh = {"localhost": "127.0.0.1"}
    for spec in host_specs:
        ssh_target, ip = spec.split("=", 1)
        hosts.append(Host(ssh=ssh_target, identity=identity))
        ip_by_ssh[ssh_target] = ip

    usable = []
    for h in hosts:
        free = probe_free_bytes(h.ssh, identity, h.is_local)
        room = free * headroom - overhead - (fp.head_bytes if h.driver else 0)
        usable.append(max(room, 0.0))
        print(f"  free RAM {h.display}: {free / 1e9:.1f} GB (usable ~{usable[-1] / 1e9:.1f})", flush=True)

    total_usable = sum(usable)
    if total_usable <= 0 or total_layer_bytes > total_usable:
        raise RuntimeError(
            f"{model}: {total_layer_bytes / 1e9:.1f} GB of layers don't fit the pool's "
            f"~{total_usable / 1e9:.1f} GB usable — add a host or free RAM"
        )

    n = fp.total_layers
    counts = [int(n * u / total_usable) for u in usable]
    while sum(counts) < n:  # hand out the remainder to the roomiest-per-next-layer host
        j = max(range(len(counts)), key=lambda i: usable[i] / (counts[i] + 1))
        counts[j] += 1

    ranges, cursor = [], 0
    for c in counts:
        ranges.append((cursor, cursor + c))
        cursor += c

    workers, port = [], 5601
    for h, (start, end) in list(zip(hosts, ranges))[1:]:
        if end > start:
            workers.append(Worker(h.ssh, ip_by_ssh[h.ssh], port, start, end))
            port += 1
    return workers, ranges[0]


def deploy_and_launch(w: Worker, model: str, identity: str | None, remote_python: str) -> None:
    print(f"  deploy → {w.ssh_target}", flush=True)
    subprocess.run(
        ["scp", *ssh_args(identity), *DEPLOY_FILES, f"{w.ssh_target}:~/"],
        check=True, capture_output=True, text=True,
    )
    # kill any stale worker, then launch ours detached, logging to ~/mlx_worker.log
    # (the model id flows into a remote shell command — quote it; layer/port are ints)
    launch = (
        "pkill -f mlx_split_worker 2>/dev/null; sleep 1; "
        f"nohup {remote_python} ~/mlx_split_worker.py "
        f"--model {shlex.quote(model)} --start {int(w.start)} --end {int(w.end)} --port {int(w.port)} "
        "> ~/mlx_worker.log 2>&1 & echo launched"
    )
    print(f"  launch → {w.ssh_target}  layers [{w.start},{w.end}) port {w.port}", flush=True)
    subprocess.run(
        ["ssh", *ssh_args(identity), w.ssh_target, launch],
        check=True, capture_output=True, text=True,
    )


def wait_tcp(ip: str, port: int, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((ip, port), timeout=3):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"worker {ip}:{port} never accepted a connection within {int(timeout)}s")


def teardown(workers: list[Worker], identity: str | None) -> None:
    for w in workers:
        try:
            subprocess.run(
                ["ssh", *ssh_args(identity), w.ssh_target, "pkill -f mlx_split_worker 2>/dev/null; true"],
                check=False, capture_output=True, text=True, timeout=30,
            )
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--local", default="0:0", help="[manual] driver in-process layer range start:end")
    ap.add_argument("--worker", action="append", default=[], help="[manual] ssh_target=ip:port:start:end")
    ap.add_argument("--host", action="append", default=[], help="[auto-fit] remote host ssh_target=ip (layers picked by free RAM)")
    ap.add_argument("--headroom", type=float, default=0.8, help="[auto-fit] fraction of free RAM to use")
    ap.add_argument("--identity", default=None)
    ap.add_argument("--remote-python", default="~/mlxenv/bin/python")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--draft", default=None,
                    help="draft model for speculative decoding (same family as --model, "
                         "runs whole on the driver) — e.g. Qwen/Qwen3-1.7B for a 14B target")
    ap.add_argument("--num-draft", type=int, default=6,
                    help="draft tokens per verify pass (measured optimum ~6)")
    ap.add_argument("--prompt", default="Explain how a computer network routes packets across the internet.")
    ap.add_argument("--serve", action="store_true", help="serve an OpenAI API + chat UI instead of a one-shot generate")
    ap.add_argument("--serve-host", default="127.0.0.1")
    ap.add_argument("--serve-port", type=int, default=8000)
    args = ap.parse_args()

    from soup.cluster.hosts import validate_model_id

    validate_model_id(args.model)  # it flows into remote shell commands
    if args.draft:
        validate_model_id(args.draft)
        if args.num_draft < 1:
            ap.error("--num-draft must be >= 1 (drop --draft to disable speculation)")

    if args.host:  # auto-fit by free RAM
        print(f"soup-mlx ▸ {args.model}  auto-fit across driver + {len(args.host)} host(s)")
        workers, (ls, le) = auto_fit(args.model, args.host, args.identity, args.headroom)
    else:  # manual ranges
        workers = [parse_worker(s) for s in args.worker]
        ls, le = (int(x) for x in args.local.split(":"))

    print(f"soup-mlx ▸ {args.model}  driver[{ls},{le}) + {len(workers)} worker(s)")
    for w in workers:
        deploy_and_launch(w, args.model, args.identity, args.remote_python)
    # Guarantee the remote workers are killed however we exit — a returning
    # finally isn't reached under uvicorn's SIGTERM shutdown, but atexit is.
    _torn_down = {"done": False}

    def _teardown_once() -> None:
        if _torn_down["done"]:
            return
        _torn_down["done"] = True
        teardown(workers, args.identity)
        print("soup-mlx ▸ workers stopped.", flush=True)

    atexit.register(_teardown_once)
    for w in workers:
        print(f"  health   waiting on {w.ip}:{w.port} …", flush=True)
        wait_tcp(w.ip, w.port)
        print(f"  healthy  {w.ip}:{w.port}", flush=True)

    if args.draft:
        print(f"soup-mlx ▸ speculative decoding: draft {args.draft}, k={args.num_draft}")
    engine = MLXClusterEngine(
        model_id=args.model,
        remote_workers=[RemoteRange(w.ip, w.port, w.start, w.end) for w in workers],
        local_range=(ls, le),
        draft_model_id=args.draft,
        num_draft=args.num_draft,
    )
    try:
        if args.serve:
            _serve(engine, args)
        else:
            engine.start()
            print("soup-mlx ▸ layout:", engine.layout, flush=True)
            messages = [{"role": "user", "content": args.prompt}]
            engine.generate_ids(messages, max_new_tokens=6)  # warm
            t0 = time.perf_counter()
            ids = engine.generate_ids(messages, max_new_tokens=args.max_tokens)
            dt = time.perf_counter() - t0
            print(f"soup-mlx ▸ {len(ids)} tokens in {dt:.2f}s = {len(ids)/dt:.2f} tok/s")
            print("soup-mlx ▸", repr(engine.shard.tokenizer.decode(ids[:40])))
            engine.close()
    finally:
        _teardown_once()


def _serve(engine: MLXClusterEngine, args) -> None:
    import uvicorn

    from soup.cluster.server import build_cluster_app
    from soup.serving.mlx_engine import AsyncMLXEngine

    async_engine = AsyncMLXEngine(engine)  # app startup calls .start() (connects sockets)
    status = {
        "model": args.model,
        "backend": "mlx",
        "workers": engine.layout,
        "endpoint": f"http://{args.serve_host}:{args.serve_port}/v1",
    }
    if args.draft:
        status["draft"] = f"{args.draft} (k={args.num_draft})"
    app = build_cluster_app(async_engine, model_name=args.model, status=status)
    base = f"http://{args.serve_host}:{args.serve_port}"
    print(f"soup-mlx ▸ serving  chat {base}/   api {base}/v1   (Ctrl-C to stop)", flush=True)
    uvicorn.run(app, host=args.serve_host, port=args.serve_port, log_level="warning")


if __name__ == "__main__":
    main()
