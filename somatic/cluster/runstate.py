"""On-disk record of a launched cluster, so `soup down` works from any shell.

Every run writes `~/.somatic/run/<run_id>/plan.json` (and a `latest` pointer)
BEFORE workers are health-waited, so a worker that starts then dies is still
teardownable. The record holds enough to kill every process: each host, its
SSH target, the remote PID, the port, and the runtime-id tag for pkill fallback.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_ROOT = Path("~/.somatic/run").expanduser()


@dataclass
class WorkerRecord:
    ssh: str
    is_local: bool
    ip: str
    port: int
    pid: int | None
    layer_start: int
    layer_end: int
    runtime_id: str
    log_path: str
    identity: str | None = None
    repo: str = "~/somatic"


@dataclass
class RunRecord:
    run_id: str
    model_id: str
    precision: str
    mode: str
    serve_host: str
    serve_port: int
    workers: list[WorkerRecord] = field(default_factory=list)

    def dir(self) -> Path:
        return STATE_ROOT / self.run_id

    def save(self) -> None:
        d = self.dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / "plan.json").write_text(json.dumps(_to_dict(self), indent=2))
        latest = STATE_ROOT / "latest"
        latest.unlink(missing_ok=True)
        latest.write_text(self.run_id)

    @classmethod
    def load(cls, run_id: str) -> "RunRecord":
        payload = json.loads((STATE_ROOT / run_id / "plan.json").read_text())
        workers = [WorkerRecord(**w) for w in payload.pop("workers", [])]
        return cls(workers=workers, **payload)

    @classmethod
    def load_latest(cls) -> "RunRecord | None":
        latest = STATE_ROOT / "latest"
        if not latest.exists():
            return None
        run_id = latest.read_text().strip()
        if not (STATE_ROOT / run_id / "plan.json").exists():
            return None
        return cls.load(run_id)

    @classmethod
    def list_runs(cls) -> list[str]:
        if not STATE_ROOT.exists():
            return []
        return sorted(
            p.name for p in STATE_ROOT.iterdir() if p.is_dir() and (p / "plan.json").exists()
        )

    def remove(self) -> None:
        import shutil

        shutil.rmtree(self.dir(), ignore_errors=True)


def _to_dict(record: RunRecord) -> dict:
    payload = asdict(record)
    return payload
