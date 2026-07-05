"""Run and launch things on a host — local via subprocess, remote via SSH.

One code path for both so the launcher never special-cases "is this my machine."
SSH is always non-interactive (`BatchMode=yes`), fails fast (`ConnectTimeout`),
and keeps long links alive (`ServerAliveInterval`) — the flags that turned prior
flaky-link hangs into clean errors.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from somatic.cluster.hosts import Host


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def ssh_prefix(host: Host) -> list[str]:
    args = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "ServerAliveInterval=15",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    identity = host.resolved_identity()
    if identity:
        args += ["-i", identity, "-o", "IdentitiesOnly=yes"]
    args.append(host.ssh)
    return args


def shell_quote_path(path: str) -> str:
    """Shell-quote a path but preserve a leading ``~/`` so the remote shell still
    expands it. Quoting the whole ``~/x`` would make ``~`` literal and break cd."""

    if path == "~":
        return "~"
    if path.startswith("~/"):
        rest = path[2:]
        return "~/" + shlex.quote(rest) if rest else "~/"
    return shlex.quote(path)


def _repo_prefixed(host: Host, command: str) -> str:
    repo = str(Path(host.repo).expanduser()) if host.is_local else host.repo
    return f"cd {shell_quote_path(repo)} && {command}"


def run(host: Host, command: str, *, timeout: float = 30.0, cd: bool = True) -> RunResult:
    """Run a shell `command` on `host`. With `cd` (default), runs in the repo dir;
    set `cd=False` for commands that must run before the repo exists (e.g. mkdir)."""

    full = _repo_prefixed(host, command) if cd else command
    if host.is_local:
        argv = ["/bin/sh", "-c", full]
    else:
        argv = ssh_prefix(host) + [full]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return RunResult(returncode=124, stdout="", stderr=f"timeout after {timeout}s")
    return RunResult(proc.returncode, proc.stdout.strip(), proc.stderr.strip())


def rsync_push(
    host: Host,
    sources: list[str],
    dest: str,
    *,
    delete: bool = False,
    excludes: tuple[str, ...] = (),
    timeout: float = 900.0,
) -> RunResult:
    """rsync local `sources` to `host:dest` over SSH (runs on the driver)."""

    ssh_e = "ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new"
    identity = host.resolved_identity()
    if identity:
        ssh_e += f" -i {shlex.quote(identity)} -o IdentitiesOnly=yes"
    args = ["rsync", "-az"]
    if delete:
        args.append("--delete")
    for pattern in excludes:
        args += ["--exclude", pattern]
    args += ["-e", ssh_e, *sources, f"{host.ssh}:{dest}"]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return RunResult(returncode=124, stdout="", stderr=f"rsync timeout after {timeout}s")
    return RunResult(proc.returncode, proc.stdout.strip(), proc.stderr.strip())


def find_executable(host: Host, candidates: list[str]) -> str | None:
    """Return the first existing executable path on `host` from `candidates`
    (absolute paths), or a bare name if it's on the non-interactive PATH."""

    checks = " || ".join(
        f"[ -x {shell_quote_path(c)} ] && echo {shell_quote_path(c)}" for c in candidates
    )
    result = run(host, f"({checks}) 2>/dev/null | head -1", cd=False, timeout=12.0)
    path = result.stdout.strip().splitlines()
    return path[0] if path and path[0] else None


def launch_worker(host: Host, worker_command: str, log_path: str) -> int | None:
    """Start a long-lived worker on `host`, detached, and return its PID.

    Local: Popen with a new session (survives this process' group signals until
    we choose to kill it). Remote: caffeinate + setsid + nohup so lid-close /
    idle-sleep can't reap it, capturing the remote PID from `echo PID:$!`.
    """

    if host.is_local:
        repo = str(Path(host.repo).expanduser())
        log = open(log_path, "wb")
        proc = subprocess.Popen(
            ["/bin/sh", "-c", f"cd {shlex.quote(repo)} && exec {worker_command}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc.pid

    # caffeinate -i keeps the host awake (lid-close/idle-sleep reaped prior runs);
    # nohup detaches from the SSH session's HUP. (setsid is absent on macOS.)
    remote = (
        f"caffeinate -i nohup {worker_command} "
        f"> {shlex.quote(log_path)} 2>&1 & echo PID:$!"
    )
    result = run(host, remote, timeout=20.0)
    for line in result.stdout.splitlines():
        if line.startswith("PID:"):
            try:
                return int(line[4:].strip())
            except ValueError:
                return None
    return None


def kill_pid(host: Host, pid: int, *, log_path: str | None = None) -> None:
    """SIGTERM then SIGKILL a pid on `host` (best-effort, idempotent)."""

    if host.is_local:
        import os
        import signal

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(pid), sig)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError):
                    return
            time.sleep(0.5)
        return
    run(host, f"kill {pid} 2>/dev/null; sleep 1; kill -9 {pid} 2>/dev/null; true", timeout=15.0)


def sweep(host: Host, pattern: str) -> None:
    """pkill any worker matching `pattern` on `host` — the orphan recourse."""

    run(host, f"pkill -f '{pattern}' 2>/dev/null; true", timeout=15.0)


def health_url(host: Host, port: int) -> str:
    return f"http://{host.ip}:{port}/health"
