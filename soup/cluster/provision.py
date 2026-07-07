"""`soup provision` — make a host ready to hold a slice of the model.

Automates the finicky setup that otherwise has to be done by hand on every
machine before `soup run` will work: push the current code, make sure the
Python env has the deps, and warm the model cache. It is the inverse of the
`doctor` preflight checks — run it once per machine and preflight goes green.

Each step is idempotent: re-running on an already-provisioned host is fast and
reports "ok" without redoing work. Nothing here touches the remote `.venv`
directory (it is never rsynced or deleted).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

from soup.cluster import doctor, ssh
from soup.cluster.hosts import Host, coerce_hosts, validate_model_id
from soup.cluster.ssh import shell_quote_path

# uv is commonly installed off the non-interactive SSH PATH.
_UV_CANDIDATES = [
    "~/.local/bin/uv", "~/.cargo/bin/uv", "/opt/homebrew/bin/uv", "/usr/local/bin/uv", "uv",
]

# Code needed to run a worker. We never sync .venv/.git/reports/checkpoints/etc.
_CODE_DIRS = ["soup", "scripts"]
_CODE_FILES = ["pyproject.toml", "uv.lock"]
_RSYNC_EXCLUDES = ("__pycache__", "*.pyc", ".DS_Store")


@dataclass
class ProvisionResult:
    host: str
    steps: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(ok for _, ok, _ in self.steps)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.steps.append((name, ok, detail))


def _detect_repo_root() -> Path:
    import soup

    return Path(soup.__file__).resolve().parent.parent


def _remote_repo(host: Host) -> str:
    # host.repo may contain ~; leave it for the remote shell to expand.
    return host.repo


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(f"soup ▸ {msg}", flush=True)


def _sync_code(host: Host, repo_root: Path) -> tuple[bool, str]:
    repo = _remote_repo(host)
    mk = ssh.run(host, f"mkdir -p {shell_quote_path(repo)}", cd=False, timeout=15.0)
    if not mk.ok:
        return False, f"could not create {repo}: {mk.stderr or mk.stdout}"
    # Directories with --delete (remove stale modules); loose files without.
    for name in _CODE_DIRS:
        src = repo_root / name
        if not src.is_dir():
            continue
        res = ssh.rsync_push(
            host, [f"{src}/"], f"{repo}/{name}/", delete=True, excludes=_RSYNC_EXCLUDES
        )
        if not res.ok:
            return False, f"rsync {name} failed: {res.stderr or res.stdout}"
    files = [str(repo_root / f) for f in _CODE_FILES if (repo_root / f).is_file()]
    if files:
        res = ssh.rsync_push(host, files, f"{repo}/")
        if not res.ok:
            return False, f"rsync files failed: {res.stderr or res.stdout}"
    return True, "code synced"


def _deps_ok(host: Host) -> bool:
    check = ssh.run(
        host,
        f"{host.remote_python()} -c 'import torch, transformers, safetensors' 2>/dev/null && echo ok",
        timeout=40.0,
    )
    return "ok" in check.stdout


def _ensure_env(host: Host, quiet: bool) -> tuple[bool, str]:
    if _deps_ok(host):
        return True, "deps already present"
    uv = ssh.find_executable(host, _UV_CANDIDATES)
    if not uv:
        return False, (
            "deps missing and uv not found — install uv on this host "
            "(https://astral.sh/uv) or create .venv with the deps, then re-provision"
        )
    _log(quiet, f"{host.display}: syncing deps with {uv} (first time is slow) …")
    sync = ssh.run(host, f"{shell_quote_path(uv)} sync", timeout=1800.0)
    if not sync.ok:
        return False, f"uv sync failed: {(sync.stderr or sync.stdout)[-300:]}"
    if not _deps_ok(host):
        return False, "uv sync ran but torch/transformers still not importable"
    return True, "deps installed via uv sync"


def _hf_cache_dirname(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _model_cached(host: Host, model_id: str) -> bool:
    # model_id is validated to a safe charset (no quotes/metachars), so it can be
    # embedded directly as a Python string literal. shlex.quote is WRONG here: it
    # quotes for the shell, leaving 'Qwen/Qwen3-1.7B' bare -> a Python SyntaxError.
    check = ssh.run(
        host,
        f"{host.remote_python()} -c \"from huggingface_hub import try_to_load_from_cache as t; "
        f"import sys; sys.exit(0 if t('{model_id}', 'config.json') else 3)\"",
        timeout=25.0,
    )
    return check.returncode == 0


def _driver_main_revision(local_dir: Path) -> str:
    """The authoritative main-revision hash from the driver's own cache."""

    ref = local_dir / "refs" / "main"
    if ref.is_file():
        content = ref.read_text().strip()
        if content:
            return content
    # Fall back to the single snapshot the driver has (only if unambiguous).
    snapshots = [p for p in (local_dir / "snapshots").glob("*") if p.is_dir()]
    return snapshots[0].name if len(snapshots) == 1 else ""


def _push_model(host: Host, model_id: str, quiet: bool) -> tuple[bool, str]:
    """rsync the driver's HF cache for this model to the host, then set refs/main
    to the driver's KNOWN revision and verify the write actually landed."""

    dirname = _hf_cache_dirname(model_id)
    local_dir = Path("~/.cache/huggingface/hub").expanduser() / dirname
    if not local_dir.is_dir():
        return False, f"model not in the driver's cache either ({local_dir}); download it here first"
    want_hash = _driver_main_revision(local_dir)
    if not want_hash:
        return False, "cannot determine the model's main revision on the driver (ambiguous/empty refs)"

    remote_hub = "~/.cache/huggingface/hub"
    mk = ssh.run(host, f"mkdir -p {remote_hub}", cd=False, timeout=15.0)
    if not mk.ok:
        return False, f"could not create the HF cache dir on the host: {mk.stderr or mk.stdout}"
    _log(quiet, f"{host.display}: pushing {dirname} (this can take a while) …")
    res = ssh.rsync_push(host, [f"{local_dir}"], f"{remote_hub}/", excludes=_RSYNC_EXCLUDES)
    if not res.ok:
        return False, f"model rsync failed: {res.stderr or res.stdout}"

    # HF-CACHE-RSYNC GOTCHA: a raw copy can leave refs/main empty -> local_files_only
    # fails. Write the driver's known hash, then VERIFY it landed and the snapshot
    # exists — a silently failed write (read-only/missing dir) must NOT report ready.
    remote_model = f"{remote_hub}/{dirname}"
    repair = (
        f"D={shell_quote_path(remote_model)}; H={shlex.quote(want_hash)}; "
        f'mkdir -p "$D/refs" && rm -f "$D/refs/main" && printf %s "$H" > "$D/refs/main" '
        f'&& [ "$(cat "$D/refs/main")" = "$H" ] && [ -d "$D/snapshots/$H" ] '
        f'&& echo "refs-ok=$H" || echo "refs-failed"'
    )
    fix = ssh.run(host, repair, cd=False, timeout=20.0)
    if "refs-ok=" not in fix.stdout:
        return False, f"pushed but refs/main repair failed: {fix.stdout or fix.stderr}"
    return True, f"model pushed + refs/main={want_hash}"


def _remote_download(host: Host, model_id: str, quiet: bool) -> tuple[bool, str]:
    _log(quiet, f"{host.display}: downloading {model_id} on the host …")
    dl = ssh.run(
        host,
        f"{host.remote_python()} -c \"from huggingface_hub import snapshot_download as s; "
        f"s('{model_id}')\"",
        timeout=3600.0,
    )
    if not dl.ok:
        return False, f"remote download failed: {(dl.stderr or dl.stdout)[-300:]}"
    return True, "model downloaded on host"


def _ensure_model(host: Host, model_id: str, push_model: bool, quiet: bool) -> tuple[bool, str]:
    if _model_cached(host, model_id):
        return True, f"{model_id} already cached"
    if push_model:
        return _push_model(host, model_id, quiet)
    ok, detail = _remote_download(host, model_id, quiet)
    if ok:
        return ok, detail
    # Fall back to a push if the host can't reach the hub but the driver has it.
    pushed_ok, pushed_detail = _push_model(host, model_id, quiet)
    if pushed_ok:
        return True, pushed_detail + " (host download failed; pushed from driver)"
    return False, f"{detail}; and {pushed_detail}"


def provision(
    hosts: list["Host | str"],
    *,
    model: str | None = None,
    push_model: bool = False,
    source_repo: str | None = None,
    sync_env: bool = True,
    quiet: bool = False,
) -> list[ProvisionResult]:
    if model is not None:
        validate_model_id(model)
    repo_root = Path(source_repo).expanduser() if source_repo else _detect_repo_root()
    results: list[ProvisionResult] = []
    for host in coerce_hosts(hosts):
        result = ProvisionResult(host=host.display)
        if host.is_local:
            result.add("local", True, "source host — nothing to provision")
            results.append(result)
            continue

        _log(quiet, f"provisioning {host.display} …")
        reach = ssh.run(host, "echo ok", cd=False, timeout=12.0)
        if not (reach.ok and "ok" in reach.stdout):
            result.add("ssh", False, f"unreachable: {reach.stderr or reach.stdout}")
            results.append(result)
            continue
        result.add("ssh", True, "reachable")

        result.add("code", *_sync_code(host, repo_root))
        if sync_env:
            result.add("deps", *_ensure_env(host, quiet))
        if model:
            result.add("model", *_ensure_model(host, model, push_model, quiet))
        results.append(result)
    return results
