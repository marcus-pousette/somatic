"""The run ledger round-trips and drives teardown by pid."""

from __future__ import annotations

import soup.cluster.runstate as runstate
from soup.cluster.runstate import RunRecord, WorkerRecord


def _rewire_state_root(tmp_path, monkeypatch):
    root = tmp_path / "run"
    monkeypatch.setattr(runstate, "STATE_ROOT", root)
    return root


def _record(run_id: str = "demo-abc123") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        model_id="Qwen/Qwen3-1.7B",
        precision="bf16",
        mode="relay",
        serve_host="127.0.0.1",
        serve_port=8000,
        workers=[
            WorkerRecord(ssh="localhost", is_local=True, ip="127.0.0.1", port=8801,
                         pid=111, layer_start=0, layer_end=14, runtime_id="demo-w0", log_path="/tmp/w0.log"),
            WorkerRecord(ssh="user@host", is_local=False, ip="host", port=8801,
                         pid=222, layer_start=14, layer_end=28, runtime_id="demo-w1", log_path="w1.log"),
        ],
    )


def test_save_load_roundtrip(tmp_path, monkeypatch) -> None:
    _rewire_state_root(tmp_path, monkeypatch)
    rec = _record()
    rec.save()
    again = RunRecord.load("demo-abc123")
    assert again.model_id == "Qwen/Qwen3-1.7B"
    assert [w.layer_end for w in again.workers] == [14, 28]
    assert again.workers[1].ssh == "user@host"


def test_latest_pointer(tmp_path, monkeypatch) -> None:
    _rewire_state_root(tmp_path, monkeypatch)
    _record("run-1").save()
    _record("run-2").save()
    latest = RunRecord.load_latest()
    assert latest is not None and latest.run_id == "run-2"
    assert set(RunRecord.list_runs()) == {"run-1", "run-2"}


def test_remove(tmp_path, monkeypatch) -> None:
    _rewire_state_root(tmp_path, monkeypatch)
    rec = _record()
    rec.save()
    rec.remove()
    assert RunRecord.list_runs() == []
