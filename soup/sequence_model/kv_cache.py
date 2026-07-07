from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from soup.runtime.schemas import utc_now
from soup.sequence_model.interfaces import Precision
from soup.sequence_model.qwen_partition import QwenLayerRange
from soup.sequence_model.tensors import TensorPayload


KVCacheStatus = Literal["active", "released", "expired"]
KVCacheEventKind = Literal["create", "prefill", "decode", "truncate", "release", "expire"]


class KVCacheHandle(BaseModel):
    cache_id: str = Field(default_factory=lambda: f"kv_{uuid4().hex}")
    sequence_id: str
    shard_id: str
    runtime_id: str
    layer_range: QwenLayerRange
    precision: Precision
    current_position: int = Field(default=0, ge=0)
    status: KVCacheStatus = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    def assert_active(self) -> None:
        if self.status != "active":
            raise ValueError(f"KV cache {self.cache_id} is {self.status}")


class KVCacheChunk(BaseModel):
    cache_id: str
    sequence_id: str
    shard_id: str
    position_start: int = Field(ge=0)
    position_end: int = Field(gt=0)
    key: TensorPayload
    value: TensorPayload

    @model_validator(mode="after")
    def _validate_chunk(self) -> "KVCacheChunk":
        if self.position_end <= self.position_start:
            raise ValueError("KV cache chunk end must be greater than start")
        if self.key.shape != self.value.shape:
            raise ValueError("KV cache key/value tensors must have matching shape")
        if self.key.dtype != self.value.dtype:
            raise ValueError("KV cache key/value tensors must have matching dtype")
        return self


class KVCacheLifecycleEvent(BaseModel):
    kind: KVCacheEventKind
    cache_id: str
    sequence_id: str
    shard_id: str
    position_start: int = Field(ge=0)
    position_end: int = Field(ge=0)
    occurred_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class KVCacheRegistry:
    """In-memory lifecycle registry used by tests and local workers before durable remote storage exists."""

    def __init__(self) -> None:
        self._handles: dict[str, KVCacheHandle] = {}
        self._events: list[KVCacheLifecycleEvent] = []

    def create(
        self,
        *,
        sequence_id: str,
        shard_id: str,
        runtime_id: str,
        layer_range: QwenLayerRange,
        precision: Precision,
        ttl_seconds: float | None = None,
    ) -> KVCacheHandle:
        now = utc_now()
        handle = KVCacheHandle(
            sequence_id=sequence_id,
            shard_id=shard_id,
            runtime_id=runtime_id,
            layer_range=layer_range,
            precision=precision,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds) if ttl_seconds is not None else None,
        )
        self._handles[handle.cache_id] = handle
        self._events.append(_event("create", handle, 0, 0))
        return handle

    def require(self, cache_id: str, *, sequence_id: str, shard_id: str) -> KVCacheHandle:
        handle = self._handles.get(cache_id)
        if handle is None:
            raise KeyError(cache_id)
        if handle.sequence_id != sequence_id:
            raise ValueError(f"KV cache {cache_id} belongs to sequence {handle.sequence_id}, not {sequence_id}")
        if handle.shard_id != shard_id:
            raise ValueError(f"KV cache {cache_id} belongs to shard {handle.shard_id}, not {shard_id}")
        if handle.expires_at is not None and utc_now() >= handle.expires_at and handle.status == "active":
            handle.status = "expired"
            handle.updated_at = utc_now()
            self._events.append(_event("expire", handle, handle.current_position, handle.current_position))
        handle.assert_active()
        return handle

    def prefill(self, chunk: KVCacheChunk) -> KVCacheHandle:
        handle = self.require(chunk.cache_id, sequence_id=chunk.sequence_id, shard_id=chunk.shard_id)
        if chunk.position_start != 0:
            raise ValueError("prefill must start at position 0")
        return self._advance(handle, "prefill", chunk.position_start, chunk.position_end)

    def decode(self, chunk: KVCacheChunk) -> KVCacheHandle:
        handle = self.require(chunk.cache_id, sequence_id=chunk.sequence_id, shard_id=chunk.shard_id)
        if chunk.position_start != handle.current_position:
            raise ValueError(
                f"decode position mismatch for {chunk.cache_id}: expected {handle.current_position}, got {chunk.position_start}"
            )
        return self._advance(handle, "decode", chunk.position_start, chunk.position_end)

    def release(self, cache_id: str, *, sequence_id: str, shard_id: str) -> KVCacheHandle:
        handle = self.require(cache_id, sequence_id=sequence_id, shard_id=shard_id)
        handle.status = "released"
        handle.updated_at = utc_now()
        self._events.append(_event("release", handle, handle.current_position, handle.current_position))
        return handle

    def truncate(self, cache_id: str, *, sequence_id: str, shard_id: str, length: int) -> KVCacheHandle:
        handle = self.require(cache_id, sequence_id=sequence_id, shard_id=shard_id)
        if length < 0 or length > handle.current_position:
            raise ValueError(
                f"KV cache truncate length {length} outside [0, {handle.current_position}] for {cache_id}"
            )
        previous_position = handle.current_position
        handle.current_position = length
        handle.updated_at = utc_now()
        self._events.append(_event("truncate", handle, previous_position, length))
        return handle

    def expire(self, *, now: datetime | None = None) -> list[KVCacheHandle]:
        resolved_now = now or utc_now()
        expired: list[KVCacheHandle] = []
        for handle in self._handles.values():
            if handle.status == "active" and handle.expires_at is not None and resolved_now >= handle.expires_at:
                handle.status = "expired"
                handle.updated_at = resolved_now
                self._events.append(_event("expire", handle, handle.current_position, handle.current_position))
                expired.append(handle)
        return expired

    def events(self) -> list[KVCacheLifecycleEvent]:
        return list(self._events)

    def handles(self) -> list[KVCacheHandle]:
        return list(self._handles.values())

    def _advance(self, handle: KVCacheHandle, kind: Literal["prefill", "decode"], position_start: int, position_end: int) -> KVCacheHandle:
        if position_end <= position_start:
            raise ValueError("KV cache update must advance position")
        if position_start < handle.current_position:
            raise ValueError(
                f"KV cache update overlaps existing state for {handle.cache_id}: current {handle.current_position}, got {position_start}"
            )
        handle.current_position = position_end
        handle.updated_at = utc_now()
        self._events.append(_event(kind, handle, position_start, position_end))
        return handle


def _event(kind: KVCacheEventKind, handle: KVCacheHandle, position_start: int, position_end: int) -> KVCacheLifecycleEvent:
    return KVCacheLifecycleEvent(
        kind=kind,
        cache_id=handle.cache_id,
        sequence_id=handle.sequence_id,
        shard_id=handle.shard_id,
        position_start=position_start,
        position_end=position_end,
    )
