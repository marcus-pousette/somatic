"""The layer-fit is contiguous, capacity-proportional, and refuses when it can't fit."""

from __future__ import annotations

import pytest

from somatic.cluster.capacity import DRIVER_RUNTIME_OVERHEAD, GIB, fit_ranges
from somatic.cluster.errors import (
    InsufficientCapacityError,
    InvalidLayerRangeError,
    LayerTooLargeError,
)
from somatic.cluster.footprint import ModelFootprint
from somatic.cluster.hosts import Host


def _footprint(layers: int, layer_gib: float, head_gib: float) -> ModelFootprint:
    return ModelFootprint(
        model_id="fixture",
        total_layers=layers,
        per_layer_bytes=[int(layer_gib * GIB)] * layers,
        head_bytes=int(head_gib * GIB),
        disk_dtype="BF16",
        precision="bf16",
    )


def test_contiguous_and_covers_all_layers() -> None:
    fp = _footprint(20, 1.0, 2.0)
    hosts = [Host(ssh="a", driver=True), Host(ssh="b")]
    free = {"a": int(40 * GIB), "b": int(40 * GIB)}
    plan = fit_ranges(fp, hosts, free, headroom=0.9)
    # every layer assigned exactly once, in order
    covered = []
    for e in plan.entries:
        covered.extend(range(e.layer_start, e.layer_end))
    assert covered == list(range(20))
    assert plan.entries[0].layer_start == 0
    assert plan.entries[-1].layer_end == 20


def test_capacity_proportional() -> None:
    # Driver too small to hold everything, so both hosts get layers and the
    # roomier one takes more. (usable: driver 22-1.5-0.5=20 -> 20 layers;
    # small 14-1.5=12.5 -> the 10 that remain.)
    fp = _footprint(30, 1.0, 0.5)
    hosts = [Host(ssh="big", driver=True), Host(ssh="small")]
    free = {"big": int(22 * GIB), "small": int(14 * GIB)}
    plan = fit_ranges(fp, hosts, free, headroom=1.0)
    assert len(plan.entries) == 2
    big, small = plan.entries[0], plan.entries[1]
    assert big.num_layers > small.num_layers
    assert big.num_layers + small.num_layers == 30


def test_driver_head_is_reserved() -> None:
    # Same raw RAM on both, but the driver's 8 GiB head steals ~8 layers from it.
    # usable: driver 25.5-1.5-8=16 -> 16 layers; wrk 25.5-1.5=24 -> the other 24.
    fp = _footprint(40, 1.0, 8.0)
    hosts = [Host(ssh="drv", driver=True), Host(ssh="wrk")]
    free = {"drv": int(25.5 * GIB), "wrk": int(25.5 * GIB)}
    plan = fit_ranges(fp, hosts, free, headroom=1.0)
    drv, wrk = plan.entries[0], plan.entries[1]
    assert wrk.num_layers - drv.num_layers >= 7


def test_insufficient_capacity_raises() -> None:
    fp = _footprint(40, 1.0, 2.0)
    hosts = [Host(ssh="a", driver=True), Host(ssh="b")]
    free = {"a": int(8 * GIB), "b": int(8 * GIB)}
    with pytest.raises(InsufficientCapacityError):
        fit_ranges(fp, hosts, free, headroom=0.8)


def test_layer_too_large_raises() -> None:
    # a single layer bigger than any host's usable RAM
    fp = _footprint(4, 30.0, 1.0)
    hosts = [Host(ssh="a", driver=True), Host(ssh="b")]
    free = {"a": int(16 * GIB), "b": int(16 * GIB)}
    with pytest.raises(LayerTooLargeError):
        fit_ranges(fp, hosts, free, headroom=0.9)


def test_pinned_ranges_are_honoured() -> None:
    fp = _footprint(28, 0.1, 0.5)
    hosts = [Host(ssh="a", driver=True, layers=(0, 14)), Host(ssh="b", layers=(14, 28))]
    free = {"a": int(40 * GIB), "b": int(40 * GIB)}
    plan = fit_ranges(fp, hosts, free)
    assert [(e.layer_start, e.layer_end) for e in plan.entries] == [(0, 14), (14, 28)]


def test_distinct_ports_per_worker() -> None:
    fp = _footprint(28, 0.1, 0.5)
    hosts = [Host(ssh="a", driver=True, layers=(0, 14)), Host(ssh="b", layers=(14, 28))]
    free = {"a": int(40 * GIB), "b": int(40 * GIB)}
    plan = fit_ranges(fp, hosts, free, base_port=8801)
    ports = [e.port for e in plan.entries]
    assert ports == [8801, 8802]
    assert len(set(ports)) == len(ports)


def test_pinned_gap_rejected() -> None:
    fp = _footprint(28, 0.1, 0.5)
    hosts = [Host(ssh="a", driver=True, layers=(0, 10)), Host(ssh="b", layers=(20, 28))]
    free = {"a": int(40 * GIB), "b": int(40 * GIB)}
    with pytest.raises(InvalidLayerRangeError):
        fit_ranges(fp, hosts, free)


def test_pinned_overlap_rejected() -> None:
    fp = _footprint(28, 0.1, 0.5)
    hosts = [Host(ssh="a", driver=True, layers=(0, 20)), Host(ssh="b", layers=(10, 28))]
    free = {"a": int(40 * GIB), "b": int(40 * GIB)}
    with pytest.raises(InvalidLayerRangeError):
        fit_ranges(fp, hosts, free)


def test_pinned_out_of_bounds_rejected() -> None:
    fp = _footprint(28, 0.1, 0.5)
    hosts = [Host(ssh="a", driver=True, layers=(0, 30))]
    free = {"a": int(40 * GIB)}
    with pytest.raises(InvalidLayerRangeError):
        fit_ranges(fp, hosts, free)
