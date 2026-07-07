"""Provision machines for the MLX backend — the MLX counterpart to `soup
provision`. Creates a `~/mlxenv` with the MLX runtime on each host so `mlx_run.py`
can just launch. Idempotent: re-running is a fast "ready ✓".

    python scripts/mlx_provision.py \
        --driver --host you@other-mac=<ip> --identity ~/.ssh/key

`--driver` provisions this machine (needs the serving deps too); each `--host`
(``ssh_target=ip``) provisions a remote worker. Each machine still needs the model
in its Hugging Face cache — that part is unchanged from the PyTorch path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# The worker only needs the MLX runtime; the driver additionally serves HTTP.
WORKER_PKGS = "mlx mlx-lm 'transformers<5'"
WORKER_IMPORTS = "import mlx.core as mx, mlx_lm, transformers"
DRIVER_PKGS = WORKER_PKGS + " fastapi uvicorn"
DRIVER_IMPORTS = WORKER_IMPORTS + ", fastapi, uvicorn"


def setup_script(pkgs: str, imports: str) -> str:
    # Idempotent: make the venv if missing, install deps only if not already
    # importable, then verify. Uses the system python3 (MLX supports 3.9+).
    return (
        "set -e\n"
        'if [ ! -x "$HOME/mlxenv/bin/python" ]; then\n'
        '  echo "[mlx-provision] creating ~/mlxenv"\n'
        "  /usr/bin/python3 -m venv \"$HOME/mlxenv\"\n"
        "fi\n"
        f'if "$HOME/mlxenv/bin/python" -c "{imports}" >/dev/null 2>&1; then\n'
        '  echo "[mlx-provision] deps already present"\n'
        "else\n"
        '  echo "[mlx-provision] installing MLX runtime (first time, ~1-2 min) ..."\n'
        '  "$HOME/mlxenv/bin/pip" install -q --upgrade pip\n'
        f'  "$HOME/mlxenv/bin/pip" install -q {pkgs}\n'
        "fi\n"
        f'"$HOME/mlxenv/bin/python" -c "{imports}; print(\'[mlx-provision] ready — mlx\', mx.__version__)"\n'
    )


def ssh_args(identity):
    args = ["-o", "ConnectTimeout=25", "-o", "StrictHostKeyChecking=accept-new"]
    if identity:
        import os

        args += ["-i", os.path.expanduser(identity)]
    return args


def provision(label: str, script: str, ssh_target: str | None, identity) -> bool:
    print(f"soup-mlx ▸ provision {label} …", flush=True)
    if ssh_target is None:
        cmd = ["bash", "-c", script]
    else:
        cmd = ["ssh", *ssh_args(identity), ssh_target, script]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    for line in (result.stdout + result.stderr).splitlines():
        if "[mlx-provision]" in line:
            print("  " + line.split("[mlx-provision]", 1)[1].strip(), flush=True)
    if result.returncode != 0:
        print(f"  ✗ failed:\n{result.stderr[-800:]}", flush=True)
        return False
    print(f"  ✓ {label} ready", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", action="store_true", help="also provision this machine (serving deps)")
    ap.add_argument("--host", action="append", default=[], help="remote worker ssh_target=ip")
    ap.add_argument("--identity", default=None)
    args = ap.parse_args()

    ok = True
    if args.driver:
        ok &= provision("localhost (driver)", setup_script(DRIVER_PKGS, DRIVER_IMPORTS), None, args.identity)
    for spec in args.host:
        ssh_target = spec.split("=", 1)[0]
        ok &= provision(ssh_target, setup_script(WORKER_PKGS, WORKER_IMPORTS), ssh_target, args.identity)

    if not ok:
        sys.exit(1)
    print("soup-mlx ▸ all hosts ready. Now: mlx_run.py --model … --host … [--serve]", flush=True)


if __name__ == "__main__":
    main()
