"""Somatic cluster: run a model too big for one machine, split across the machines you have.

Public surface:
    from somatic.cluster import Cluster, Host
    Cluster.launch(model, hosts) -> Cluster        # probe → fit → spawn → serve
    Cluster.plan(model, hosts)   -> LaunchPlan      # dry run, launches nothing
"""

from somatic.cluster.capacity import LaunchPlan, PlanEntry
from somatic.cluster.errors import (
    ClusterError,
    HostUnreachableError,
    InsufficientCapacityError,
    InvalidLayerRangeError,
    LayerTooLargeError,
    PreflightError,
    WorkerFailedError,
)
from somatic.cluster.footprint import ModelFootprint, model_footprint
from somatic.cluster.hosts import Host
from somatic.cluster.launcher import Cluster, teardown_run
from somatic.cluster.provision import ProvisionResult, provision
from somatic.cluster.verify import VerifyReport, verify

__all__ = [
    "Cluster",
    "Host",
    "LaunchPlan",
    "PlanEntry",
    "ModelFootprint",
    "model_footprint",
    "teardown_run",
    "verify",
    "VerifyReport",
    "provision",
    "ProvisionResult",
    "ClusterError",
    "HostUnreachableError",
    "PreflightError",
    "InsufficientCapacityError",
    "InvalidLayerRangeError",
    "LayerTooLargeError",
    "WorkerFailedError",
]
