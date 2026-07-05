"""Typed failures for the cluster launcher — each one carries an actionable message."""

from __future__ import annotations


class ClusterError(Exception):
    """Base for every launcher failure."""


class HostUnreachableError(ClusterError):
    """A host could not be reached / probed over SSH."""


class PreflightError(ClusterError):
    """A host failed a preflight check (missing deps, stale script, model not cached)."""


class InsufficientCapacityError(ClusterError):
    """The hosts' combined free RAM cannot hold the model at this precision."""


class LayerTooLargeError(ClusterError):
    """A single layer does not fit on any host — the model can't be split here."""


class InvalidLayerRangeError(ClusterError):
    """Pinned layer ranges are out of bounds, overlapping, or leave a gap."""


class WorkerFailedError(ClusterError):
    """A worker process died or never became healthy."""
