"""Computer Soup cluster: run a model too big for one machine, split across the machines you have.

Public surface:
    from soup.cluster import Cluster, Host
    Cluster.launch(model, hosts) -> Cluster        # probe → fit → spawn → serve
    Cluster.plan(model, hosts)   -> LaunchPlan      # dry run, launches nothing
"""

from soup.cluster.capacity import LaunchPlan, PlanEntry
from soup.cluster.errors import (
    ClusterError,
    HostUnreachableError,
    InsufficientCapacityError,
    InvalidLayerRangeError,
    LayerTooLargeError,
    PreflightError,
    WorkerFailedError,
)
from soup.cluster.footprint import ModelFootprint, model_footprint
from soup.cluster.hosts import Host
from soup.cluster.launcher import Cluster, teardown_run
from soup.cluster.provision import ProvisionResult, provision
from soup.cluster.verify import VerifyReport, verify

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
