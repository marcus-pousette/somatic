from __future__ import annotations

from datetime import datetime
import os
import platform
import resource
import time
from typing import Any

from pydantic import BaseModel, Field

from soup.runtime.schemas import utc_now


class SequenceWorkerMetrics(BaseModel):
    """App-ready point-in-time worker process telemetry."""

    runtime_id: str
    collected_at: datetime = Field(default_factory=utc_now)
    process_cpu_seconds: float = Field(ge=0.0)
    rss_bytes: int | None = Field(default=None, ge=0)
    max_rss_bytes: int | None = Field(default=None, ge=0)
    thread_count: int | None = Field(default=None, ge=0)
    open_file_descriptor_count: int | None = Field(default=None, ge=0)
    qwen_shard_id: str | None = None
    qwen_layer_start: int | None = Field(default=None, ge=0)
    qwen_layer_end: int | None = Field(default=None, ge=0)
    qwen_weight_loading_mode: str | None = None
    qwen_full_model_parameter_bytes: int | None = Field(default=None, ge=0)
    qwen_assigned_layer_parameter_bytes: int | None = Field(default=None, ge=0)
    qwen_loaded_parameter_bytes: int | None = Field(default=None, ge=0)
    qwen_unassigned_loaded_parameter_bytes: int | None = Field(default=None, ge=0)
    qwen_shard_only_weight_loading_claimed: bool = False
    qwen_per_machine_ram_reduction_claimed: bool = False
    active_qwen_cache_count: int = Field(default=0, ge=0)
    total_qwen_cache_count: int = Field(default=0, ge=0)
    released_qwen_cache_count: int = Field(default=0, ge=0)
    expired_qwen_cache_count: int = Field(default=0, ge=0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


def collect_sequence_worker_metrics(
    *,
    runtime_id: str,
    qwen_shard_runtime: Any | None = None,
) -> SequenceWorkerMetrics:
    shard_id = None
    layer_start = None
    layer_end = None
    active_cache_count = 0
    released_cache_count = 0
    expired_cache_count = 0
    total_cache_count = 0
    loading_mode = None
    full_model_parameter_bytes = None
    assigned_layer_parameter_bytes = None
    loaded_parameter_bytes = None
    unassigned_loaded_parameter_bytes = None
    shard_only_weight_loading_claimed = False
    per_machine_ram_reduction_claimed = False
    if qwen_shard_runtime is not None:
        shard_id = qwen_shard_runtime.manifest.shard_id
        layer_range = qwen_shard_runtime.manifest.layer_range
        if layer_range is not None:
            layer_start = layer_range.start
            layer_end = layer_range.end
        handles = qwen_shard_runtime.cache_handles()
        total_cache_count = len(handles)
        active_cache_count = sum(1 for handle in handles if handle.status == "active")
        released_cache_count = sum(1 for handle in handles if handle.status == "released")
        expired_cache_count = sum(1 for handle in handles if handle.status == "expired")
        loading_accounting = getattr(qwen_shard_runtime, "loading_accounting", None)
        if loading_accounting is not None:
            loading_mode = loading_accounting.loading_mode
            full_model_parameter_bytes = loading_accounting.full_model_parameter_bytes
            assigned_layer_parameter_bytes = loading_accounting.assigned_layer_parameter_bytes
            loaded_parameter_bytes = loading_accounting.loaded_parameter_bytes
            unassigned_loaded_parameter_bytes = loading_accounting.unassigned_loaded_parameter_bytes
            shard_only_weight_loading_claimed = loading_accounting.shard_only_weight_loading_claimed
            per_machine_ram_reduction_claimed = loading_accounting.per_machine_ram_reduction_claimed

    return SequenceWorkerMetrics(
        runtime_id=runtime_id,
        process_cpu_seconds=time.process_time(),
        rss_bytes=_current_rss_bytes(),
        max_rss_bytes=_max_rss_bytes(),
        thread_count=_thread_count(),
        open_file_descriptor_count=_open_file_descriptor_count(),
        qwen_shard_id=shard_id,
        qwen_layer_start=layer_start,
        qwen_layer_end=layer_end,
        qwen_weight_loading_mode=loading_mode,
        qwen_full_model_parameter_bytes=full_model_parameter_bytes,
        qwen_assigned_layer_parameter_bytes=assigned_layer_parameter_bytes,
        qwen_loaded_parameter_bytes=loaded_parameter_bytes,
        qwen_unassigned_loaded_parameter_bytes=unassigned_loaded_parameter_bytes,
        qwen_shard_only_weight_loading_claimed=shard_only_weight_loading_claimed,
        qwen_per_machine_ram_reduction_claimed=per_machine_ram_reduction_claimed,
        active_qwen_cache_count=active_cache_count,
        total_qwen_cache_count=total_cache_count,
        released_qwen_cache_count=released_cache_count,
        expired_qwen_cache_count=expired_cache_count,
        metadata={
            "platform": platform.system(),
            "process_id": os.getpid(),
            "telemetry_schema": "sequence-worker-metrics-v0",
        },
    )


def _current_rss_bytes() -> int | None:
    statm_path = "/proc/self/statm"
    try:
        with open(statm_path, encoding="utf-8") as handle:
            parts = handle.read().split()
        if len(parts) >= 2:
            return int(parts[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        pass
    return _max_rss_bytes()


def _max_rss_bytes() -> int | None:
    try:
        max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    if max_rss <= 0:
        return None
    # Linux reports KiB; Darwin reports bytes.
    if platform.system() == "Darwin":
        return max_rss
    return max_rss * 1024


def _thread_count() -> int | None:
    task_path = "/proc/self/task"
    try:
        return len(os.listdir(task_path))
    except OSError:
        return None


def _open_file_descriptor_count() -> int | None:
    fd_path = "/proc/self/fd"
    try:
        return len(os.listdir(fd_path))
    except OSError:
        return None
