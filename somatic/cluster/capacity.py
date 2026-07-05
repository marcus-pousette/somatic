"""Probe each host's free RAM and fit a contiguous layer split onto the cluster.

Two jobs:
  * `probe_free_ram(host)` — ask a host how many bytes are actually free, using
    the host's own Python (psutil if present, else a stdlib vm_stat / /proc read),
    so no extra dependency is required on any machine.
  * `fit_ranges(...)` — assign each host a contiguous `[start, end)` slice of the
    model's layers, largest-first in host order, reserving the driver's head and a
    per-worker runtime floor, and refusing (loudly) to launch if it doesn't fit.

The engine chains hidden states in layer order, so a contiguous greedy fill in
host order is optimal for the single chain — no bin-packer needed. Ranges come
out capacity-proportional: a 32 GB machine takes more layers than an 8 GB one.
"""

from __future__ import annotations

from dataclasses import dataclass

from somatic.cluster.errors import (
    HostUnreachableError,
    InsufficientCapacityError,
    InvalidLayerRangeError,
    LayerTooLargeError,
)
from somatic.cluster.footprint import ModelFootprint
from somatic.cluster.hosts import Host
from somatic.cluster import ssh

GIB = 1024 ** 3

# Reserved on every worker host for torch + activations + KV cache growth, on top
# of the shard weights. bf16 shards are mmap-backed (RSS undercounts until pages
# fault in), so this is deliberately generous rather than trusting live RSS.
DRIVER_RUNTIME_OVERHEAD = int(1.5 * GIB)

# A self-contained probe: try psutil, else parse the OS. Prints "FREE=<bytes>".
_PROBE_SNIPPET = (
    "import sys\n"
    "def free():\n"
    "    try:\n"
    "        import psutil\n"
    "        return psutil.virtual_memory().available\n"
    "    except Exception:\n"
    "        pass\n"
    "    import platform, subprocess\n"
    "    if platform.system() == 'Darwin':\n"
    "        ps = int(subprocess.check_output(['sysctl','-n','hw.pagesize']).split()[0])\n"
    "        out = subprocess.check_output(['vm_stat']).decode()\n"
    "        pages = 0\n"
    "        for key in ('Pages free','Pages inactive','Pages speculative'):\n"
    "            for line in out.splitlines():\n"
    "                if line.startswith(key):\n"
    "                    pages += int(line.split(':')[1].strip().rstrip('.'))\n"
    "        return pages * ps\n"
    "    for line in open('/proc/meminfo'):\n"
    "        if line.startswith('MemAvailable'):\n"
    "            return int(line.split()[1]) * 1024\n"
    "    return 0\n"
    "print('FREE=%d' % free())\n"
)


@dataclass(frozen=True)
class PlanEntry:
    host: Host
    layer_start: int
    layer_end: int
    port: int
    free_bytes: int
    is_driver: bool

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start


@dataclass(frozen=True)
class LaunchPlan:
    model_id: str
    precision: str
    footprint: ModelFootprint
    entries: list[PlanEntry]

    @property
    def layout(self) -> str:
        return " -> ".join(
            f"[{e.layer_start},{e.layer_end})@{e.host.display}" for e in self.entries
        )

    def bytes_on(self, entry: PlanEntry) -> int:
        weights = sum(self.footprint.per_layer_bytes[entry.layer_start:entry.layer_end])
        if entry.is_driver:
            weights += self.footprint.head_bytes
        return weights


def probe_free_ram(host: Host, *, timeout: float = 12.0) -> int:
    """Free bytes on `host`, via the host's own Python."""

    command = f"{host.remote_python()} -c \"$(printf '%s' {_shquote(_PROBE_SNIPPET)})\""
    result = ssh.run(host, command, timeout=timeout)
    if not result.ok:
        raise HostUnreachableError(
            f"could not probe free RAM on {host.display}: {result.stderr or result.stdout or 'no response'}"
        )
    for line in result.stdout.splitlines():
        if line.startswith("FREE="):
            return int(line[5:])
    raise HostUnreachableError(f"probe on {host.display} returned no FREE= line: {result.stdout!r}")


def _shquote(snippet: str) -> str:
    # Single-quote for the shell, escaping embedded single quotes.
    return "'" + snippet.replace("'", "'\\''") + "'"


def fit_ranges(
    footprint: ModelFootprint,
    hosts: list[Host],
    free_ram: dict[str, int],
    *,
    headroom: float = 0.80,
    base_port: int = 8801,
) -> LaunchPlan:
    """Assign contiguous layer ranges to hosts by capacity. Raises if it can't fit."""

    per_layer = footprint.per_layer_bytes
    total = footprint.total_layers

    usable: dict[str, int] = {}
    for host in hosts:
        free = free_ram[host.ssh]
        room = int(free * headroom) - DRIVER_RUNTIME_OVERHEAD
        if host.driver:
            room -= footprint.head_bytes
        usable[host.ssh] = max(room, 0)

    # A layer that fits on no host means the model can't be split at this precision.
    max_usable = max(usable.values()) if usable else 0
    biggest_layer = max(per_layer) if per_layer else 0
    if biggest_layer > max_usable:
        raise LayerTooLargeError(
            f"a single layer needs {biggest_layer / GIB:.2f} GiB but the roomiest host "
            f"has {max_usable / GIB:.2f} GiB usable — lower --precision or use a bigger machine"
        )

    entries: list[PlanEntry] = []
    cursor = 0
    for host in hosts:
        if cursor >= total:
            break
        budget = usable[host.ssh]
        if host.layers is not None:  # user pinned this host's range explicitly
            start, end = host.layers
            # Pins must tile the chain: in bounds, non-empty, and gapless/non-overlapping.
            if not (0 <= start < end <= total):
                raise InvalidLayerRangeError(
                    f"pinned range [{start},{end}) on {host.display} is out of bounds for "
                    f"a {total}-layer model (need 0 <= start < end <= {total})"
                )
            if start != cursor:
                raise InvalidLayerRangeError(
                    f"pinned range [{start},{end}) on {host.display} leaves a gap or overlaps: "
                    f"the previous host covered up to layer {cursor}, so this must start at {cursor}"
                )
            entries.append(_entry(host, start, end, base_port, len(entries), free_ram))
            cursor = end
            continue
        end = cursor
        used = 0
        while end < total and used + per_layer[end] <= budget:
            used += per_layer[end]
            end += 1
        if end == cursor:
            continue  # host too small to hold even one layer at this cursor; skip
        entries.append(_entry(host, cursor, end, base_port, len(entries), free_ram))
        cursor = end

    if cursor < total:
        shortfall_layers = total - cursor
        shortfall_bytes = sum(per_layer[cursor:])
        raise InsufficientCapacityError(
            f"the hosts hold {cursor}/{total} layers; {shortfall_layers} layers "
            f"({shortfall_bytes / GIB:.1f} GiB) don't fit. Add a host, free RAM, or lower --precision."
        )

    # Invariant: the entries must be a gapless, in-order tiling of [0, total).
    _assert_contiguous_tiling(entries, total)

    return LaunchPlan(
        model_id=footprint.model_id,
        precision=footprint.precision,
        footprint=footprint,
        entries=entries,
    )


def _assert_contiguous_tiling(entries: list[PlanEntry], total: int) -> None:
    expected = 0
    for entry in entries:
        if entry.layer_start != expected:
            raise InvalidLayerRangeError(
                f"layer ranges are not contiguous: expected next start {expected}, "
                f"got [{entry.layer_start},{entry.layer_end})"
            )
        expected = entry.layer_end
    if expected != total:
        raise InvalidLayerRangeError(
            f"layer ranges cover [0,{expected}) but the model has {total} layers"
        )


def _entry(host: Host, start: int, end: int, base_port: int, index: int, free_ram: dict[str, int]) -> PlanEntry:
    return PlanEntry(
        host=host,
        layer_start=start,
        layer_end=end,
        # Distinct port per worker so two workers pinned to the same host don't collide.
        port=base_port + index,
        free_bytes=free_ram[host.ssh],
        is_driver=host.driver,
    )
