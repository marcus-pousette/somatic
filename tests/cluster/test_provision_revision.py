"""_driver_main_revision reads the authoritative hash, not a guessed snapshot."""

from __future__ import annotations

from pathlib import Path

from somatic.cluster.provision import _driver_main_revision

HASH = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
OTHER = "aaaabbbbccccddddeeeeffff0000111122223333"


def _make_cache(root: Path, *, refs: str | None, snapshots: list[str]) -> Path:
    model = root / "models--Qwen--Qwen3-1.7B"
    (model / "refs").mkdir(parents=True, exist_ok=True)
    if refs is not None:
        (model / "refs" / "main").write_text(refs)
    for snap in snapshots:
        (model / "snapshots" / snap).mkdir(parents=True, exist_ok=True)
    return model


def test_uses_refs_main_when_present(tmp_path: Path) -> None:
    # refs/main is authoritative even if several snapshots exist.
    model = _make_cache(tmp_path, refs=HASH, snapshots=[HASH, OTHER])
    assert _driver_main_revision(model) == HASH


def test_falls_back_to_single_snapshot(tmp_path: Path) -> None:
    model = _make_cache(tmp_path, refs="", snapshots=[HASH])
    assert _driver_main_revision(model) == HASH


def test_ambiguous_snapshots_without_refs_returns_empty(tmp_path: Path) -> None:
    # No refs and >1 snapshot -> can't guess -> empty (push will refuse).
    model = _make_cache(tmp_path, refs=None, snapshots=[HASH, OTHER])
    assert _driver_main_revision(model) == ""
