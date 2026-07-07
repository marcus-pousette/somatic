from __future__ import annotations

import time
from typing import Any, Callable, Literal, Sequence

import httpx
import numpy as np

from soup.sequence_model.execution import SequenceExecutionResult, UnitExecutionTrace, payload_digest
from soup.sequence_model.boundary_adapters import (
    BOUNDARY_POSITION_SCOPED_TRANSPORT_METADATA_KEY,
    BoundaryAdapterStrategy,
    boundary_adapter_summary_from_metadata,
    decode_payload_from_boundary,
    encode_payload_for_boundary,
    expand_compact_boundary_adapter_metrics,
    normalize_boundary_adapter_strategy,
)
from soup.sequence_model.boundary_compression.trainable_autoencoder_wire import (
    load_trainable_autoencoder_basis_artifact,
)
from soup.sequence_model.boundary_compression.wire_codec_runtime import (
    TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
    TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND,
    decode_boundary_wire_payload,
    encode_boundary_wire_payload,
)
from soup.sequence_model.interfaces import (
    CalibrationProfile,
    ModelManifest,
    ModelUnit,
    Planner,
    Precision,
    ResourceProfile,
)
from soup.sequence_model.kv_cache import KVCacheHandle
from soup.sequence_model.qwen_real import QwenCacheForwardResult, QwenShardTrace
from soup.sequence_model.tensor_execution import TensorSequenceExecutionResult, TensorUnitExecutionTrace, prompt_to_tensor
from soup.sequence_model.telemetry import SequenceWorkerMetrics
from soup.sequence_model.tensors import TENSOR_FRAME_MEDIA_TYPE, TensorPayload, decode_tensor_frame, encode_tensor_frame
from soup.sequence_model.tensors import encode_tensor_frame_from_raw
from soup.sequence_model.transformers_qwen import QwenGenerationResult


BoundaryTransportStrategyCallback = Callable[[dict[str, Any], TensorPayload | None], str]
TensorTransformCallback = Callable[[dict[str, Any], TensorPayload], tuple[TensorPayload, dict[str, Any]]]


class SequenceWorkerClient:
    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        auth_token: str | None = None,
        boundary_context_base_url: str | None = None,
        trace_capture_callback: Callable[[dict[str, Any]], None] | None = None,
        tensor_transform_callback: TensorTransformCallback | None = None,
        boundary_transport_strategy_callback: BoundaryTransportStrategyCallback | None = None,
        trainable_autoencoder_basis_artifact: dict[str, Any] | None = None,
        trainable_autoencoder_basis_artifact_path: str | None = None,
        trainable_autoencoder_source_strategy: str = "int8_symmetric",
        trainable_autoencoder_allow_source_only_fallback: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.auth_token = auth_token
        self.boundary_context_base_url = (
            boundary_context_base_url.rstrip("/")
            if boundary_context_base_url is not None
            else self.base_url
        )
        self._trace_capture_callback = trace_capture_callback
        self._tensor_transform_callback = tensor_transform_callback
        self._boundary_transport_strategy_callback = boundary_transport_strategy_callback
        self._trainable_autoencoder_source_strategy = normalize_boundary_adapter_strategy(
            trainable_autoencoder_source_strategy
        )
        self._trainable_autoencoder_allow_source_only_fallback = bool(
            trainable_autoencoder_allow_source_only_fallback
        )
        if trainable_autoencoder_basis_artifact is not None:
            self._trainable_autoencoder_basis_artifact = trainable_autoencoder_basis_artifact
        elif trainable_autoencoder_basis_artifact_path is not None:
            self._trainable_autoencoder_basis_artifact = load_trainable_autoencoder_basis_artifact(
                trainable_autoencoder_basis_artifact_path
            )
        else:
            self._trainable_autoencoder_basis_artifact = None

    async def profile(self) -> ResourceProfile:
        data = await self._request_json("GET", "/sequence/profile")
        return ResourceProfile.model_validate(data["resource_profile"])

    async def metrics(self) -> SequenceWorkerMetrics:
        data = await self._request_json("GET", "/sequence/metrics")
        return SequenceWorkerMetrics.model_validate(data["metrics"])

    async def calibrate(self, manifest: ModelManifest) -> CalibrationProfile:
        data = await self._request_json("POST", "/sequence/calibrate", json={"manifest": manifest.model_dump(mode="json")})
        return CalibrationProfile.model_validate(data["calibration"])

    async def execute_unit(self, unit: ModelUnit, payload: dict[str, Any], *, precision: Precision) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        data = await self._request_json(
            "POST",
            "/sequence/execute-unit",
            json={"unit": unit.model_dump(mode="json"), "payload": payload, "precision": precision},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response_payload = data["payload"]
        if not isinstance(response_payload, dict):
            raise ValueError("sequence worker returned a non-object payload")
        return response_payload, elapsed_ms

    async def execute_tensor_unit(self, unit: ModelUnit, tensor: TensorPayload, *, precision: Precision) -> tuple[TensorPayload, float]:
        started = time.perf_counter()
        data = await self._request_json(
            "POST",
            "/sequence/execute-tensor-unit",
            json={"unit": unit.model_dump(mode="json"), "tensor": tensor.model_dump(mode="json"), "precision": precision},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return TensorPayload.model_validate(data["tensor"]), elapsed_ms

    async def execute_tensor_unit_binary(self, unit: ModelUnit, tensor: TensorPayload, *, precision: Precision) -> tuple[TensorPayload, float]:
        frame = encode_tensor_frame(
            tensor,
            extra_header={
                "unit": unit.model_dump(mode="json"),
                "precision": precision,
                "transport": "binary-tensor-frame-v1",
            },
        )
        headers = {
            "Accept": TENSOR_FRAME_MEDIA_TYPE,
            "Content-Type": TENSOR_FRAME_MEDIA_TYPE,
        }
        if self.auth_token is not None:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        started = time.perf_counter()
        if self._client is not None:
            response = await self._client.request("POST", "/sequence/execute-tensor-unit-binary", content=frame, headers=headers)
        else:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
                response = await client.request("POST", "/sequence/execute-tensor-unit-binary", content=frame, headers=headers)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        payload, _ = decode_tensor_frame(response.content)
        return payload, elapsed_ms

    async def generate_qwen(
        self,
        *,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 24,
        device: str = "cpu",
        local_files_only: bool = False,
        precision: Precision = "bf16",
    ) -> QwenGenerationResult:
        data = await self._request_json(
            "POST",
            "/sequence/generate",
            json={
                "adapter": "transformers-qwen",
                "model_id": model_id,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "device": device,
                "local_files_only": local_files_only,
                "precision": precision,
            },
        )
        return QwenGenerationResult.model_validate(data["result"])

    async def forward_qwen_shard(self, tensor: TensorPayload) -> tuple[TensorPayload, QwenShardTrace, float]:
        started = time.perf_counter()
        data = await self._request_json(
            "POST",
            "/sequence/qwen/forward-shard",
            json={"tensor": tensor.model_dump(mode="json")},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return TensorPayload.model_validate(data["tensor"]), QwenShardTrace.model_validate(data["trace"]), elapsed_ms

    async def forward_qwen_shard_binary(
        self,
        tensor: TensorPayload,
        *,
        boundary_adapter_strategy: str = "identity",
    ) -> tuple[TensorPayload, QwenShardTrace, float]:
        response_tensor, header, elapsed_ms = await self._request_tensor_frame(
            "/sequence/qwen/forward-shard-binary",
            tensor,
            extra_header={
                "transport": "qwen-binary-tensor-frame-v1",
            },
            boundary_adapter_strategy=boundary_adapter_strategy,
        )
        return response_tensor, _qwen_trace_from_header(header), elapsed_ms

    async def create_qwen_cache(self, *, sequence_id: str, ttl_seconds: float | None = None) -> KVCacheHandle:
        data = await self._request_json(
            "POST",
            "/sequence/qwen/cache/create",
            json={"sequence_id": sequence_id, "ttl_seconds": ttl_seconds},
        )
        return KVCacheHandle.model_validate(data["cache"])

    async def truncate_qwen_cache(
        self, *, cache_id: str, sequence_id: str, length: int
    ) -> KVCacheHandle:
        data = await self._request_json(
            "POST",
            "/sequence/qwen/cache/truncate",
            json={
                "cache_id": cache_id,
                "sequence_id": sequence_id,
                "length": int(length),
            },
        )
        return KVCacheHandle.model_validate(data["cache"])

    async def prefill_qwen_shard(
        self,
        *,
        tensor: TensorPayload,
        cache_id: str,
        sequence_id: str,
        position_start: int = 0,
    ) -> tuple[QwenCacheForwardResult, float]:
        started = time.perf_counter()
        data = await self._request_json(
            "POST",
            "/sequence/qwen/cache/prefill",
            json={
                "tensor": tensor.model_dump(mode="json"),
                "cache_id": cache_id,
                "sequence_id": sequence_id,
                "position_start": position_start,
            },
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return QwenCacheForwardResult.model_validate(data["result"]), elapsed_ms

    async def prefill_qwen_shard_binary(
        self,
        *,
        tensor: TensorPayload,
        cache_id: str,
        sequence_id: str,
        position_start: int = 0,
        generation_step: int | None = None,
        boundary_adapter_strategy: str = "identity",
        sequence_metadata: dict[str, Any] | None = None,
    ) -> tuple[QwenCacheForwardResult, float]:
        response_tensor, header, elapsed_ms = await self._request_tensor_frame(
            "/sequence/qwen/cache/prefill-binary",
            tensor,
            extra_header={
                "cache_id": cache_id,
                "sequence_id": sequence_id,
                **(sequence_metadata or {}),
                "position_start": position_start,
                "generation_step": 0 if generation_step is None else generation_step,
                "generation_phase": "prefill",
                "transport": "qwen-kv-cache-binary-tensor-frame-v1",
            },
            boundary_adapter_strategy=boundary_adapter_strategy,
        )
        return (
            QwenCacheForwardResult(
                tensor=response_tensor,
                trace=_qwen_trace_from_header(header),
                cache=KVCacheHandle.model_validate(header["cache"]),
            ),
            elapsed_ms,
        )

    async def decode_qwen_shard(
        self,
        *,
        tensor: TensorPayload,
        cache_id: str,
        sequence_id: str,
        position_start: int,
    ) -> tuple[QwenCacheForwardResult, float]:
        started = time.perf_counter()
        data = await self._request_json(
            "POST",
            "/sequence/qwen/cache/decode",
            json={
                "tensor": tensor.model_dump(mode="json"),
                "cache_id": cache_id,
                "sequence_id": sequence_id,
                "position_start": position_start,
            },
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return QwenCacheForwardResult.model_validate(data["result"]), elapsed_ms

    async def decode_qwen_shard_binary(
        self,
        *,
        tensor: TensorPayload,
        cache_id: str,
        sequence_id: str,
        position_start: int,
        generation_step: int | None = None,
        boundary_adapter_strategy: str = "identity",
        sequence_metadata: dict[str, Any] | None = None,
    ) -> tuple[QwenCacheForwardResult, float]:
        extra_header = {
            "cache_id": cache_id,
            "sequence_id": sequence_id,
            **(sequence_metadata or {}),
            "position_start": position_start,
            "generation_phase": "decode",
            "transport": "qwen-kv-cache-binary-tensor-frame-v1",
        }
        if generation_step is not None:
            extra_header["generation_step"] = generation_step
        response_tensor, header, elapsed_ms = await self._request_tensor_frame(
            "/sequence/qwen/cache/decode-binary",
            tensor,
            extra_header=extra_header,
            boundary_adapter_strategy=boundary_adapter_strategy,
        )
        return (
            QwenCacheForwardResult(
                tensor=response_tensor,
                trace=_qwen_trace_from_header(header),
                cache=KVCacheHandle.model_validate(header["cache"]),
            ),
            elapsed_ms,
        )

    async def decode_qwen_shard_binary_coalesced(
        self,
        *,
        tensors: Sequence[TensorPayload],
        members: Sequence[dict[str, Any]],
        generation_step: int | None = None,
        boundary_adapter_strategy: str = "identity",
        sequence_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[QwenCacheForwardResult], float]:
        if not tensors:
            raise ValueError("coalesced Qwen decode requires at least one tensor")
        if len(tensors) != len(members):
            raise ValueError("coalesced Qwen decode tensor/member counts must match")
        arrays = [tensor.to_numpy() for tensor in tensors]
        first = arrays[0]
        if len(first.shape) < 1 or int(first.shape[0]) != 1:
            raise ValueError("coalesced Qwen decode tensors must have a single-member batch axis")
        for array in arrays[1:]:
            if tuple(array.shape[1:]) != tuple(first.shape[1:]):
                raise ValueError("coalesced Qwen decode tensors must share non-batch shape")
            if int(array.shape[0]) != 1:
                raise ValueError("coalesced Qwen decode tensors must have a single-member batch axis")
        first_tensor = tensors[0]
        if any(tensor.name != first_tensor.name for tensor in tensors):
            raise ValueError("coalesced Qwen decode tensors must share tensor name")
        if any(tensor.dtype != first_tensor.dtype for tensor in tensors):
            raise ValueError("coalesced Qwen decode tensors must share dtype")

        normalized_members = [
            _normalize_coalesced_member(member, index)
            for index, member in enumerate(members)
        ]
        stacked = TensorPayload.from_numpy(
            np.concatenate(arrays, axis=0),
            name=first_tensor.name,
            metadata={
                **first_tensor.metadata,
                "coalesced_transport": "qwen-kv-cache-binary-coalesced-tensor-frame-v1",
                "coalesced_member_count": len(tensors),
            },
        )
        extra_header = {
            **(sequence_metadata or {}),
            "coalesced_members": normalized_members,
            "coalesced_member_count": len(members),
            "generation_phase": "decode",
            "transport": "qwen-kv-cache-binary-coalesced-tensor-frame-v1",
        }
        if generation_step is not None:
            extra_header["generation_step"] = generation_step
        strategy_groups = self._select_coalesced_boundary_transport_strategy_groups(
            path="/sequence/qwen/cache/decode-binary-coalesced",
            tensor=stacked,
            extra_header=extra_header,
            default_strategy=boundary_adapter_strategy,
        )
        if _coalesced_strategy_groups_require_partition(
            strategy_groups,
            member_count=len(tensors),
        ):
            result_slots: list[QwenCacheForwardResult | None] = [
                None for _ in tensors
            ]
            total_elapsed_ms = 0.0
            for group in strategy_groups:
                indices = [int(index) for index in group.get("member_indices") or []]
                if not indices:
                    continue
                group_results, group_elapsed_ms = (
                    await self.decode_qwen_shard_binary_coalesced(
                        tensors=[tensors[index] for index in indices],
                        members=[normalized_members[index] for index in indices],
                        generation_step=generation_step,
                        boundary_adapter_strategy=_normalize_boundary_wire_strategy(
                            str(
                                group.get("boundary_adapter_strategy")
                                or boundary_adapter_strategy
                            )
                        ),
                        sequence_metadata=sequence_metadata,
                    )
                )
                total_elapsed_ms += group_elapsed_ms
                for index, result in zip(indices, group_results, strict=True):
                    result_slots[index] = result
            if any(result is None for result in result_slots):
                raise ValueError("coalesced Qwen decode strategy partition lost a member")
            return [result for result in result_slots if result is not None], total_elapsed_ms
        if strategy_groups:
            selected_group = strategy_groups[0]
            boundary_adapter_strategy = _normalize_boundary_wire_strategy(
                str(
                    selected_group.get("boundary_adapter_strategy")
                    or boundary_adapter_strategy
                )
            )
            request_boundary_adapter_strategy = _normalize_boundary_wire_strategy(
                str(
                    selected_group.get("request_boundary_adapter_strategy")
                    or boundary_adapter_strategy
                )
            )
            response_boundary_adapter_strategy = _normalize_boundary_wire_strategy(
                str(
                    selected_group.get("response_boundary_adapter_strategy")
                    or boundary_adapter_strategy
                )
            )
        else:
            request_boundary_adapter_strategy = boundary_adapter_strategy
            response_boundary_adapter_strategy = boundary_adapter_strategy
        response_tensor, header, elapsed_ms = await self._request_tensor_frame(
            "/sequence/qwen/cache/decode-binary-coalesced",
            stacked,
            extra_header=extra_header,
            boundary_adapter_strategy=boundary_adapter_strategy,
            request_boundary_adapter_strategy=request_boundary_adapter_strategy,
            response_boundary_adapter_strategy=response_boundary_adapter_strategy,
        )
        coalesced_results = header.get("coalesced_results")
        if not isinstance(coalesced_results, list) or len(coalesced_results) != len(members):
            raise ValueError("coalesced Qwen decode response did not include one result per member")
        response_array = response_tensor.to_numpy()
        if int(response_array.shape[0]) != len(coalesced_results):
            raise ValueError("coalesced Qwen decode response tensor/member counts do not match")
        member_traces = [
            QwenShardTrace.model_validate(row.get("trace"))
            for row in coalesced_results
            if isinstance(row, dict)
        ]
        if len(member_traces) != len(coalesced_results):
            raise ValueError("coalesced Qwen decode response traces must be objects")
        client_transport = header.get("_client_transport")
        client_transport = client_transport if isinstance(client_transport, dict) else {}
        aggregate_trace = _qwen_trace_from_header(header)
        request_frame_allocations = _allocate_integer_total(
            int(client_transport.get("request_frame_bytes") or 0),
            [trace.input_bytes for trace in member_traces],
        )
        response_frame_allocations = _allocate_integer_total(
            int(client_transport.get("response_frame_bytes") or 0),
            [trace.output_bytes for trace in member_traces],
        )
        results: list[QwenCacheForwardResult] = []
        for index, row in enumerate(coalesced_results):
            if not isinstance(row, dict):
                raise ValueError("coalesced Qwen decode response rows must be objects")
            member_tensor = TensorPayload.from_numpy(
                response_array[index : index + 1],
                name=response_tensor.name,
                metadata={
                    **response_tensor.metadata,
                    "coalesced_member_index": index,
                    "coalesced_member_count": len(coalesced_results),
                },
            )
            trace = member_traces[index].model_copy(
                update={
                    "request_frame_bytes": request_frame_allocations[index],
                    "response_frame_bytes": response_frame_allocations[index],
                    "boundary_adapter_id": aggregate_trace.boundary_adapter_id,
                    "boundary_adapter_applied": aggregate_trace.boundary_adapter_applied,
                    "boundary_adapter_input": aggregate_trace.boundary_adapter_input,
                    "boundary_adapter_output": aggregate_trace.boundary_adapter_output,
                }
            )
            results.append(
                QwenCacheForwardResult(
                    tensor=member_tensor,
                    trace=trace,
                    cache=KVCacheHandle.model_validate(row.get("cache")),
                )
            )
        return results, elapsed_ms

    async def release_qwen_cache(self, *, cache_id: str, sequence_id: str) -> KVCacheHandle:
        data = await self._request_json(
            "POST",
            "/sequence/qwen/cache/release",
            json={"cache_id": cache_id, "sequence_id": sequence_id},
        )
        return KVCacheHandle.model_validate(data["cache"])

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if self.auth_token is not None:
            headers = dict(kwargs.pop("headers", {}) or {})
            headers.setdefault("Authorization", f"Bearer {self.auth_token}")
            kwargs["headers"] = headers
        if self._client is not None:
            response = await self._client.request(method, path, **kwargs)
        else:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
                response = await client.request(method, path, **kwargs)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("sequence worker response was not a JSON object")
        return data

    async def _request_tensor_frame(
        self,
        path: str,
        tensor: TensorPayload,
        *,
        extra_header: dict[str, Any],
        boundary_adapter_strategy: str = "identity",
        request_boundary_adapter_strategy: str | None = None,
        response_boundary_adapter_strategy: str | None = None,
    ) -> tuple[TensorPayload, dict[str, Any], float]:
        strategy = _normalize_boundary_wire_strategy(boundary_adapter_strategy)
        request_context = {
            "schema_version": "sequence-worker-client-boundary-transform-context-v0",
            "base_url": self.boundary_context_base_url,
            "physical_base_url": self.base_url,
            "path": path,
            "role": "request",
            "shape": list(tensor.shape),
            "boundary_adapter_strategy": strategy,
            "default_boundary_adapter_strategy": strategy,
            "extra_header": dict(extra_header),
            "position_start": extra_header.get("position_start"),
            "generation_step": extra_header.get("generation_step"),
            "generation_phase": extra_header.get("generation_phase"),
        }
        response_context = {
            "schema_version": "sequence-worker-client-boundary-transform-context-v0",
            "base_url": self.boundary_context_base_url,
            "physical_base_url": self.base_url,
            "path": path,
            "role": "response",
            "shape": list(tensor.shape),
            "boundary_adapter_strategy": strategy,
            "default_boundary_adapter_strategy": strategy,
            "extra_header": dict(extra_header),
            "position_start": extra_header.get("position_start"),
            "generation_step": extra_header.get("generation_step"),
            "generation_phase": extra_header.get("generation_phase"),
        }
        request_strategy = (
            _normalize_boundary_wire_strategy(request_boundary_adapter_strategy)
            if request_boundary_adapter_strategy is not None
            else self._select_boundary_transport_strategy(
                request_context,
                tensor,
                default_strategy=strategy,
            )
        )
        response_strategy = (
            _normalize_boundary_wire_strategy(response_boundary_adapter_strategy)
            if response_boundary_adapter_strategy is not None
            else self._select_boundary_transport_strategy(
                response_context,
                None,
                default_strategy=strategy,
            )
        )
        if request_strategy == "learned_int8" and response_strategy == "learned_int8":
            response_strategy = "identity"
        if (
            request_strategy == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID
            and response_strategy == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID
        ):
            response_strategy = "identity"
        request_context["boundary_adapter_strategy"] = request_strategy
        request_context["request_boundary_adapter_strategy"] = request_strategy
        request_context["response_boundary_adapter_strategy"] = response_strategy
        response_context["boundary_adapter_strategy"] = response_strategy
        response_context["request_boundary_adapter_strategy"] = request_strategy
        response_context["response_boundary_adapter_strategy"] = response_strategy
        request_tensor = tensor
        request_tensor, request_transform_summary = self._transform_tensor_payload(
            request_context,
            tensor,
        )
        request_tensor = _tensor_with_position_scoped_transport_metadata(
            request_tensor,
            request_context,
        )
        response_position_scope = response_context.get(
            BOUNDARY_POSITION_SCOPED_TRANSPORT_METADATA_KEY
        )
        response_extra_header = (
            {
                "response_boundary_position_scoped_transport": response_position_scope,
            }
            if isinstance(response_position_scope, dict)
            else {}
        )
        if request_strategy == "learned_int8":
            frame, encoded_tensor, request_boundary_adapter = _encode_learned_int8_request_frame(
                request_tensor,
                extra_header={
                    **extra_header,
                    **response_extra_header,
                    "compact_qwen_response_trace": True,
                    "response_boundary_adapter_strategy": response_strategy,
                },
            )
        elif request_strategy == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID:
            frame, encoded_tensor, request_boundary_adapter = _encode_trainable_autoencoder_request_frame(
                request_tensor,
                basis_artifact=self._required_trainable_autoencoder_basis_artifact(),
                context=request_context,
                source_strategy=self._trainable_autoencoder_source_strategy,
                allow_source_only_fallback=(
                    self._trainable_autoencoder_allow_source_only_fallback
                ),
                extra_header={
                    **extra_header,
                    **response_extra_header,
                    "compact_qwen_response_trace": True,
                    "response_boundary_adapter_strategy": response_strategy,
                },
            )
        else:
            encoded_tensor, request_boundary_adapter = encode_payload_for_boundary(
                request_tensor,
                strategy=normalize_boundary_adapter_strategy(request_strategy),
            )
            frame = encode_tensor_frame(
                encoded_tensor,
                extra_header={
                    **extra_header,
                    **response_extra_header,
                    "compact_qwen_response_trace": True,
                    "response_boundary_adapter_strategy": response_strategy,
                },
            )
        _annotate_boundary_transform_frame_metrics(
            request_transform_summary,
            encoded_tensor=encoded_tensor,
            boundary_adapter=request_boundary_adapter,
            frame_bytes=len(frame),
            frame_role="request",
        )
        headers = {
            "Accept": TENSOR_FRAME_MEDIA_TYPE,
            "Content-Type": TENSOR_FRAME_MEDIA_TYPE,
        }
        if self.auth_token is not None:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        started = time.perf_counter()
        if self._client is not None:
            response = await self._client.request("POST", path, content=frame, headers=headers)
        else:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
                response = await client.request("POST", path, content=frame, headers=headers)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        response_encoded_tensor, response_header = decode_tensor_frame(response.content)
        if _is_trainable_autoencoder_wire_payload(
            response_header,
            response_encoded_tensor.metadata,
        ):
            decoded_array = decode_boundary_wire_payload(
                {
                    **response_header,
                    "shape": list(response_encoded_tensor.shape),
                    "dtype": response_encoded_tensor.dtype,
                    "tensor_name": response_encoded_tensor.name,
                    "tensor_metadata": response_encoded_tensor.metadata,
                },
                response_encoded_tensor.raw_bytes(),
                trainable_autoencoder_basis_artifact=(
                    self._required_trainable_autoencoder_basis_artifact()
                ),
            )
            response_encoded_metadata = dict(response_encoded_tensor.metadata)
            response_source_metadata = (
                response_encoded_metadata.get("source_metadata")
                if isinstance(response_encoded_metadata.get("source_metadata"), dict)
                else {}
            )
            response_tensor = TensorPayload.from_numpy(
                decoded_array,
                name=response_encoded_tensor.name,
                metadata={
                    **dict(response_source_metadata),
                    "trainable_autoencoder_source_latent_client_decoded": True,
                    "trainable_autoencoder_source_latent_response": {
                        "encoded_payload_kind": response_encoded_metadata.get(
                            "encoded_payload_kind"
                        ),
                        "basis_hash": response_encoded_metadata.get("basis_hash"),
                        "basis_selection_key": response_encoded_metadata.get(
                            "basis_selection_key"
                        ),
                        "basis_selection_kind": response_encoded_metadata.get(
                            "basis_selection_kind"
                        ),
                        "source_strategy": response_encoded_metadata.get(
                            "source_strategy"
                        ),
                        "source_only_fallback": response_encoded_metadata.get(
                            "source_only_fallback"
                        ),
                        "source_only_fallback_reason": response_encoded_metadata.get(
                            "source_only_fallback_reason"
                        ),
                        "basis_correction_applied": response_encoded_metadata.get(
                            "basis_correction_applied"
                        ),
                        "source_raw_byte_count": response_encoded_metadata.get(
                            "source_raw_byte_count"
                        ),
                        "latent_byte_count": response_encoded_metadata.get(
                            "latent_byte_count"
                        ),
                        "decoded_on_receive": True,
                        "production_runtime_claimed": False,
                    },
                },
            )
            response_boundary_adapter = {
                **dict(response_header.get("boundary_adapter") or {}),
                "adapter_id": TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
                "adapter_kind": "learned",
                "encoded_payload_kind": TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND,
                "decoded_on_receive": True,
                "source_strategy": response_encoded_metadata.get("source_strategy"),
                "source_raw_byte_count": response_encoded_metadata.get(
                    "source_raw_byte_count"
                ),
                "latent_byte_count": response_encoded_metadata.get("latent_byte_count"),
                "source_only_fallback": bool(
                    response_encoded_metadata.get("source_only_fallback")
                ),
                "source_only_fallback_reason": response_encoded_metadata.get(
                    "source_only_fallback_reason"
                ),
                "basis_correction_applied": response_encoded_metadata.get(
                    "basis_correction_applied"
                ),
                "position_scoped_mixed_frame": bool(
                    response_encoded_metadata.get("position_scoped_mixed_frame")
                ),
                "mixed_frame_layout": response_encoded_metadata.get(
                    "mixed_frame_layout"
                ),
                "scoped_correction": bool(
                    response_encoded_metadata.get("scoped_correction")
                ),
                "scope_row_count": response_encoded_metadata.get("scope_row_count"),
                "encoded_raw_bytes": response_encoded_tensor.byte_size(),
                "original_raw_bytes": response_tensor.byte_size(),
                "raw_byte_ratio": (
                    response_encoded_tensor.byte_size() / response_tensor.byte_size()
                    if response_tensor.byte_size()
                    else 1.0
                ),
                "production_runtime_claimed": False,
            }
        else:
            response_tensor, response_boundary_adapter = decode_payload_from_boundary(response_encoded_tensor)
        decoded_response_tensor = response_tensor
        response_context["response_header"] = response_header
        response_context["shape"] = list(response_tensor.shape)
        response_tensor, response_transform_summary = self._transform_tensor_payload(
            response_context,
            response_tensor,
        )
        response_boundary_adapter = {
            **response_boundary_adapter,
            **expand_compact_boundary_adapter_metrics(response_header.get("bo")),
        }
        _annotate_boundary_transform_frame_metrics(
            response_transform_summary,
            encoded_tensor=response_encoded_tensor,
            boundary_adapter=response_boundary_adapter,
            frame_bytes=len(response.content),
            frame_role="response",
        )
        if self._trace_capture_callback is not None:
            self._trace_capture_callback(
                {
                    "schema_version": "sequence-worker-client-tensor-frame-capture-v0",
                    "base_url": self.base_url,
                    "boundary_context_base_url": self.boundary_context_base_url,
                    "path": path,
                    "boundary_adapter_strategy": strategy,
                    "request_boundary_adapter_strategy": request_strategy,
                    "response_boundary_adapter_strategy": response_strategy,
                    "extra_header": dict(extra_header),
                    "request_context": dict(request_context),
                    "response_context": dict(response_context),
                    "request_tensor": tensor,
                    "transformed_request_tensor": request_tensor,
                    "encoded_request_tensor": encoded_tensor,
                    "request_boundary_adapter": request_boundary_adapter,
                    "request_transform": request_transform_summary,
                    "request_frame_bytes": len(frame),
                    "response_encoded_tensor": response_encoded_tensor,
                    "decoded_response_tensor": decoded_response_tensor,
                    "transformed_response_tensor": response_tensor,
                    "response_tensor": response_tensor,
                    "response_boundary_adapter": response_boundary_adapter,
                    "response_transform": response_transform_summary,
                    "response_frame_bytes": len(response.content),
                    "elapsed_ms": elapsed_ms,
                    "response_header": response_header,
                }
            )
        response_header["_client_transport"] = {
            "request_frame_bytes": len(frame),
            "response_frame_bytes": len(response.content),
            "boundary_adapter_strategy": strategy,
            "request_boundary_adapter_strategy": request_strategy,
            "response_boundary_adapter_strategy": response_strategy,
            "request_boundary_adapter": request_boundary_adapter,
            "response_boundary_adapter": response_boundary_adapter,
            "request_transform": request_transform_summary,
            "response_transform": response_transform_summary,
        }
        return response_tensor, response_header, elapsed_ms

    def _select_boundary_transport_strategy(
        self,
        context: dict[str, Any],
        payload: TensorPayload | None,
        *,
        default_strategy: str,
    ) -> str:
        if self._boundary_transport_strategy_callback is None:
            return default_strategy
        if _context_has_coalesced_members(context) and _bound_callback_owner_method(
            self._boundary_transport_strategy_callback,
            "transport_strategy_groups_for_coalesced_context",
        ) is not None:
            return default_strategy
        return _normalize_boundary_wire_strategy(
            self._boundary_transport_strategy_callback(context, payload)
        )

    def _select_coalesced_boundary_transport_strategy_groups(
        self,
        *,
        path: str,
        tensor: TensorPayload,
        extra_header: dict[str, Any],
        default_strategy: str,
    ) -> list[dict[str, Any]]:
        normalized_default = _normalize_boundary_wire_strategy(default_strategy)
        member_count = int(extra_header.get("coalesced_member_count") or 0)
        grouping_callback = _bound_callback_owner_method(
            self._boundary_transport_strategy_callback,
            "transport_strategy_groups_for_coalesced_context",
        )
        if grouping_callback is None:
            return _single_coalesced_strategy_group(
                member_count=member_count,
                strategy=normalized_default,
            )
        request_context = {
            "schema_version": "sequence-worker-client-boundary-transform-context-v0",
            "base_url": self.boundary_context_base_url,
            "physical_base_url": self.base_url,
            "path": path,
            "role": "request",
            "shape": list(tensor.shape),
            "boundary_adapter_strategy": normalized_default,
            "default_boundary_adapter_strategy": normalized_default,
            "extra_header": dict(extra_header),
            "position_start": extra_header.get("position_start"),
            "generation_step": extra_header.get("generation_step"),
            "generation_phase": extra_header.get("generation_phase"),
        }
        response_context = {
            **request_context,
            "role": "response",
        }
        raw_request_groups = grouping_callback(
            request_context,
            tensor,
            default_strategy=normalized_default,
        )
        raw_response_groups = grouping_callback(
            response_context,
            tensor,
            default_strategy=normalized_default,
        )
        request_groups = _normalize_coalesced_strategy_groups(
            raw_request_groups,
            member_count=member_count,
            default_strategy=normalized_default,
        )
        response_groups = _normalize_coalesced_strategy_groups(
            raw_response_groups,
            member_count=member_count,
            default_strategy=normalized_default,
        )
        return _coalesced_strategy_pair_groups(
            request_groups=request_groups,
            response_groups=response_groups,
            member_count=member_count,
        )

    def _transform_tensor_payload(
        self,
        context: dict[str, Any],
        payload: TensorPayload,
    ) -> tuple[TensorPayload, dict[str, Any]]:
        if self._tensor_transform_callback is None:
            return payload, {}
        coalesced_callback = _bound_callback_owner_method(
            self._tensor_transform_callback,
            "transform_coalesced_payload",
        )
        if coalesced_callback is not None and _context_has_coalesced_members(context):
            return coalesced_callback(context, payload)
        return self._tensor_transform_callback(context, payload)

    def _required_trainable_autoencoder_basis_artifact(self) -> dict[str, Any]:
        if self._trainable_autoencoder_basis_artifact is None:
            raise ValueError("trainable autoencoder wire transport requires a basis artifact")
        return self._trainable_autoencoder_basis_artifact


def _bound_callback_owner_method(
    callback: Callable[..., Any] | None,
    method_name: str,
) -> Callable[..., Any] | None:
    owner = getattr(callback, "__self__", None)
    method = getattr(owner, method_name, None)
    return method if callable(method) else None


def _context_has_coalesced_members(context: dict[str, Any]) -> bool:
    extra_header = context.get("extra_header")
    members = (
        extra_header.get("coalesced_members")
        if isinstance(extra_header, dict)
        else context.get("coalesced_members")
    )
    return isinstance(members, list) and bool(members)


def _normalize_boundary_wire_strategy(value: str | None) -> str:
    if value == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID:
        return TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID
    return normalize_boundary_adapter_strategy(str(value or "identity"))


def _is_trainable_autoencoder_wire_payload(
    header: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    return (
        header.get("boundary_adapter_strategy")
        == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID
        or metadata.get("boundary_adapter_strategy")
        == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID
        or metadata.get("encoded_payload_kind")
        == TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND
    )


def _single_coalesced_strategy_group(
    *,
    member_count: int,
    strategy: str,
) -> list[dict[str, Any]]:
    normalized_strategy = _normalize_boundary_wire_strategy(strategy)
    return [
        {
            "schema_version": "sequence-worker-client-coalesced-strategy-group-v0",
            "member_indices": list(range(member_count)),
            "member_count": member_count,
            "boundary_adapter_strategy": normalized_strategy,
            "request_boundary_adapter_strategy": normalized_strategy,
            "response_boundary_adapter_strategy": normalized_strategy,
            "strategy_group_index": 0,
        }
    ]


def _coalesced_strategy_pair_groups(
    *,
    request_groups: list[dict[str, Any]],
    response_groups: list[dict[str, Any]],
    member_count: int,
) -> list[dict[str, Any]]:
    request_by_member = _coalesced_member_strategy_map(request_groups)
    response_by_member = _coalesced_member_strategy_map(response_groups)
    groups: list[dict[str, Any]] = []
    pair_to_group: dict[tuple[str, str], dict[str, Any]] = {}
    for member_index in range(member_count):
        request_strategy = request_by_member.get(member_index, "identity")
        response_strategy = response_by_member.get(member_index, "identity")
        key = (str(request_strategy), str(response_strategy))
        group = pair_to_group.get(key)
        if group is None:
            boundary_strategy = _coalesced_boundary_strategy_for_pair(
                request_strategy,
                response_strategy,
            )
            group = {
                "schema_version": "sequence-worker-client-coalesced-strategy-group-v0",
                "member_indices": [],
                "member_count": 0,
                "boundary_adapter_strategy": boundary_strategy,
                "request_boundary_adapter_strategy": request_strategy,
                "response_boundary_adapter_strategy": response_strategy,
                "strategy_group_index": len(groups),
            }
            pair_to_group[key] = group
            groups.append(group)
        group["member_indices"].append(member_index)
        group["member_count"] = len(group["member_indices"])
    return groups


def _coalesced_member_strategy_map(
    groups: list[dict[str, Any]],
) -> dict[int, str]:
    strategies: dict[int, str] = {}
    for group in groups:
        strategy = _normalize_boundary_wire_strategy(
            str(group.get("boundary_adapter_strategy") or "identity")
        )
        for raw_index in group.get("member_indices") or []:
            strategies[int(raw_index)] = strategy
    return strategies


def _coalesced_boundary_strategy_for_pair(
    request_strategy: str,
    response_strategy: str,
) -> str:
    normalized_response = _normalize_boundary_wire_strategy(response_strategy)
    if normalized_response != "identity":
        return normalized_response
    return _normalize_boundary_wire_strategy(request_strategy)


def _normalize_coalesced_strategy_groups(
    raw_groups: Any,
    *,
    member_count: int,
    default_strategy: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_groups, list) or not raw_groups:
        return _single_coalesced_strategy_group(
            member_count=member_count,
            strategy=default_strategy,
        )
    groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            raise ValueError("coalesced strategy groups must be objects")
        raw_indices = raw_group.get("member_indices")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError("coalesced strategy group requires member_indices")
        member_indices: list[int] = []
        for raw_index in raw_indices:
            index = int(raw_index)
            if index < 0 or index >= member_count:
                raise ValueError(
                    "coalesced strategy group member index is outside member count: "
                    f"{index} for {member_count}"
                )
            if index in seen:
                raise ValueError(
                    f"coalesced strategy group duplicated member index {index}"
                )
            seen.add(index)
            member_indices.append(index)
        strategy = _normalize_boundary_wire_strategy(
            str(raw_group.get("boundary_adapter_strategy") or default_strategy)
        )
        request_strategy = _normalize_boundary_wire_strategy(
            str(raw_group.get("request_boundary_adapter_strategy") or strategy)
        )
        response_strategy = _normalize_boundary_wire_strategy(
            str(raw_group.get("response_boundary_adapter_strategy") or strategy)
        )
        normalized = dict(raw_group)
        normalized.update(
            {
                "schema_version": (
                    "sequence-worker-client-coalesced-strategy-group-v0"
                ),
                "member_indices": member_indices,
                "member_count": len(member_indices),
                "boundary_adapter_strategy": strategy,
                "request_boundary_adapter_strategy": request_strategy,
                "response_boundary_adapter_strategy": response_strategy,
                "strategy_group_index": int(
                    raw_group.get("strategy_group_index", group_index)
                ),
            }
        )
        groups.append(normalized)
    missing = [index for index in range(member_count) if index not in seen]
    if missing:
        raise ValueError(
            "coalesced strategy groups did not cover all members: "
            + ", ".join(str(index) for index in missing)
        )
    return groups


def _coalesced_strategy_groups_require_partition(
    groups: list[dict[str, Any]],
    *,
    member_count: int,
) -> bool:
    if len(groups) != 1:
        return True
    indices = [int(index) for index in groups[0].get("member_indices") or []]
    return sorted(indices) != list(range(member_count))


def _annotate_boundary_transform_frame_metrics(
    transform_summary: dict[str, Any],
    *,
    encoded_tensor: TensorPayload,
    boundary_adapter: dict[str, Any],
    frame_bytes: int,
    frame_role: str,
) -> None:
    if not transform_summary:
        return
    encoded_raw_bytes = int(
        boundary_adapter.get("encoded_raw_bytes") or encoded_tensor.byte_size()
    )
    original_raw_bytes = int(
        boundary_adapter.get("original_raw_bytes")
        or transform_summary.get("original_bytes")
        or encoded_raw_bytes
    )
    transform_summary.update(
        {
            "boundary_frame_role": frame_role,
            "boundary_frame_adapter_id": boundary_adapter.get("adapter_id"),
            "boundary_frame_adapter_kind": boundary_adapter.get("adapter_kind"),
            "boundary_frame_encoded_dtype": boundary_adapter.get("encoded_dtype")
            or encoded_tensor.dtype,
            "boundary_frame_encoded_shape": list(encoded_tensor.shape),
            "boundary_frame_encoded_raw_bytes": encoded_raw_bytes,
            "boundary_frame_original_raw_bytes": original_raw_bytes,
            "boundary_frame_raw_byte_ratio": (
                encoded_raw_bytes / original_raw_bytes if original_raw_bytes else 1.0
            ),
            "boundary_frame_raw_byte_savings_ratio": max(
                0.0,
                1.0 - (encoded_raw_bytes / original_raw_bytes)
                if original_raw_bytes
                else 0.0,
            ),
            "boundary_frame_wire_bytes": int(frame_bytes),
            "boundary_frame_tensor_payload_byte_size": encoded_tensor.byte_size(),
            "boundary_frame_payload_byte_delta_vs_original": (
                encoded_raw_bytes - original_raw_bytes
            ),
            "boundary_frame_payload_bytes_below_original": (
                encoded_raw_bytes < original_raw_bytes
            ),
            "boundary_frame_payload_bytes_unchanged": (
                encoded_raw_bytes == original_raw_bytes
            ),
            "boundary_frame_position_scoped_mixed_frame": bool(
                boundary_adapter.get("position_scoped_mixed_frame")
            ),
            "boundary_frame_mixed_frame_layout": boundary_adapter.get(
                "mixed_frame_layout"
            ),
            "boundary_frame_mixed_frame_axis": boundary_adapter.get(
                "mixed_frame_axis"
            ),
            "boundary_frame_mixed_frame_position": boundary_adapter.get(
                "mixed_frame_position"
            ),
            "boundary_frame_source_strategy": boundary_adapter.get("source_strategy"),
            "boundary_frame_source_raw_byte_count": boundary_adapter.get(
                "source_raw_byte_count"
            ),
            "boundary_frame_latent_byte_count": boundary_adapter.get(
                "latent_byte_count"
            ),
            "boundary_frame_source_only_fallback": bool(
                boundary_adapter.get("source_only_fallback")
            ),
            "boundary_frame_source_only_fallback_reason": boundary_adapter.get(
                "source_only_fallback_reason"
            ),
            "boundary_frame_basis_correction_applied": boundary_adapter.get(
                "basis_correction_applied"
            ),
        }
    )
    _annotate_coalesced_member_transform_frame_metrics(
        transform_summary,
        encoded_tensor=encoded_tensor,
        boundary_adapter=boundary_adapter,
        encoded_raw_bytes=encoded_raw_bytes,
        original_raw_bytes=original_raw_bytes,
        frame_bytes=int(frame_bytes),
        frame_role=frame_role,
    )


def _annotate_coalesced_member_transform_frame_metrics(
    transform_summary: dict[str, Any],
    *,
    encoded_tensor: TensorPayload,
    boundary_adapter: dict[str, Any],
    encoded_raw_bytes: int,
    original_raw_bytes: int,
    frame_bytes: int,
    frame_role: str,
) -> None:
    member_summaries = transform_summary.get("member_summaries")
    if not isinstance(member_summaries, list) or not member_summaries:
        return
    members = [row for row in member_summaries if isinstance(row, dict)]
    if not members:
        return
    original_weights = [int(row.get("original_bytes") or 0) for row in members]
    if not any(original_weights):
        original_weights = [1 for _ in members]
    encoded_allocations = _allocate_integer_total(encoded_raw_bytes, original_weights)
    frame_allocations = _allocate_integer_total(frame_bytes, original_weights)
    original_allocations = _allocate_integer_total(original_raw_bytes, original_weights)
    encoded_shape = list(encoded_tensor.shape)
    if encoded_shape and int(encoded_shape[0]) == len(members):
        encoded_shape = [1, *encoded_shape[1:]]
    for index, member in enumerate(members):
        member_original = int(
            member.get("original_bytes") or original_allocations[index] or 0
        )
        member_encoded = int(encoded_allocations[index])
        member_frame_bytes = int(frame_allocations[index])
        member.update(
            {
                "boundary_frame_role": frame_role,
                "boundary_frame_adapter_id": boundary_adapter.get("adapter_id"),
                "boundary_frame_adapter_kind": boundary_adapter.get("adapter_kind"),
                "boundary_frame_encoded_dtype": boundary_adapter.get("encoded_dtype")
                or encoded_tensor.dtype,
                "boundary_frame_encoded_shape": encoded_shape,
                "boundary_frame_encoded_raw_bytes": member_encoded,
                "boundary_frame_original_raw_bytes": member_original,
                "boundary_frame_raw_byte_ratio": (
                    member_encoded / member_original if member_original else 1.0
                ),
                "boundary_frame_raw_byte_savings_ratio": max(
                    0.0,
                    1.0 - (member_encoded / member_original)
                    if member_original
                    else 0.0,
                ),
                "boundary_frame_wire_bytes": member_frame_bytes,
                "boundary_frame_tensor_payload_byte_size": member_encoded,
                "boundary_frame_payload_byte_delta_vs_original": (
                    member_encoded - member_original
                ),
                "boundary_frame_payload_bytes_below_original": (
                    member_encoded < member_original
                ),
                "boundary_frame_payload_bytes_unchanged": (
                    member_encoded == member_original
                ),
                "boundary_frame_position_scoped_mixed_frame": bool(
                    boundary_adapter.get("position_scoped_mixed_frame")
                ),
                "boundary_frame_mixed_frame_layout": boundary_adapter.get(
                    "mixed_frame_layout"
                ),
                "boundary_frame_mixed_frame_axis": boundary_adapter.get(
                    "mixed_frame_axis"
                ),
                    "boundary_frame_mixed_frame_position": boundary_adapter.get(
                        "mixed_frame_position"
                    ),
                    "boundary_frame_source_strategy": boundary_adapter.get(
                        "source_strategy"
                    ),
                    "boundary_frame_source_raw_byte_count": boundary_adapter.get(
                        "source_raw_byte_count"
                    ),
                    "boundary_frame_latent_byte_count": boundary_adapter.get(
                        "latent_byte_count"
                    ),
                    "boundary_frame_source_only_fallback": bool(
                        boundary_adapter.get("source_only_fallback")
                    ),
                    "boundary_frame_source_only_fallback_reason": boundary_adapter.get(
                        "source_only_fallback_reason"
                    ),
                    "boundary_frame_basis_correction_applied": boundary_adapter.get(
                        "basis_correction_applied"
                    ),
                    "boundary_frame_coalesced_parent_encoded_raw_bytes": encoded_raw_bytes,
                    "boundary_frame_coalesced_parent_original_raw_bytes": original_raw_bytes,
                    "boundary_frame_coalesced_parent_wire_bytes": frame_bytes,
                "boundary_frame_coalesced_member_index": index,
                "boundary_frame_coalesced_member_count": len(members),
            }
        )


def _tensor_with_position_scoped_transport_metadata(
    tensor: TensorPayload,
    context: dict[str, Any],
) -> TensorPayload:
    scoped_transport = context.get(BOUNDARY_POSITION_SCOPED_TRANSPORT_METADATA_KEY)
    if not isinstance(scoped_transport, dict):
        return tensor
    metadata = dict(tensor.metadata)
    metadata[BOUNDARY_POSITION_SCOPED_TRANSPORT_METADATA_KEY] = dict(scoped_transport)
    return tensor.model_copy(update={"metadata": metadata})


def _normalize_coalesced_member(member: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(member, dict):
        raise ValueError("coalesced Qwen decode members must be objects")
    cache_id = member.get("cache_id")
    sequence_id = member.get("sequence_id")
    if not isinstance(cache_id, str) or not cache_id:
        raise ValueError("coalesced Qwen decode member requires non-empty `cache_id`")
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("coalesced Qwen decode member requires non-empty `sequence_id`")
    if "position_start" not in member:
        raise ValueError("coalesced Qwen decode member requires `position_start`")
    try:
        position_start = int(member["position_start"])
    except (TypeError, ValueError) as exc:
        raise ValueError("coalesced Qwen decode member `position_start` must be an integer") from exc
    if position_start < 0:
        raise ValueError("coalesced Qwen decode member `position_start` must be non-negative")
    raw_metadata = member.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    return {
        "member_index": int(member.get("member_index", index)),
        "cache_id": cache_id,
        "sequence_id": sequence_id,
        "position_start": position_start,
        "metadata": metadata,
    }


def _allocate_integer_total(total: int, weights: Sequence[int]) -> list[int]:
    if not weights:
        return []
    if total <= 0:
        return [0 for _ in weights]
    normalized_weights = [max(int(weight), 0) for weight in weights]
    weight_total = sum(normalized_weights)
    if weight_total <= 0:
        base = total // len(weights)
        allocations = [base for _ in weights]
        for index in range(total - sum(allocations)):
            allocations[index] += 1
        return allocations
    allocations: list[int] = []
    used = 0
    for index, weight in enumerate(normalized_weights):
        if index == len(normalized_weights) - 1:
            allocation = total - used
        else:
            allocation = (total * weight) // weight_total
            used += allocation
        allocations.append(allocation)
    return allocations


def _encode_learned_int8_request_frame(
    tensor: TensorPayload,
    *,
    extra_header: dict[str, Any],
) -> tuple[bytes, TensorPayload, dict[str, Any]]:
    from soup.sequence_model.boundary_compression.wire_codec_runtime import (
        encode_learned_int8_wire_payload,
    )

    encoded = encode_learned_int8_wire_payload(tensor.to_numpy(), tensor_name=tensor.name)
    tensor_metadata = dict(tensor.metadata)
    header_fields = dict(encoded.header_fields)
    frame_shape = [len(encoded.payload)]
    encoded_tensor = TensorPayload.from_raw_bytes(
        encoded.payload,
        dtype=header_fields["dtype"],  # type: ignore[arg-type]
        shape=frame_shape,
        name=str(header_fields.get("tensor_name") or tensor.name),
        metadata=tensor_metadata,
    )
    frame = encode_tensor_frame_from_raw(
        encoded.payload,
        dtype=header_fields["dtype"],  # type: ignore[arg-type]
        shape=frame_shape,
        name=encoded_tensor.name,
        metadata=tensor_metadata,
        extra_header={
            **extra_header,
            "boundary_adapter_strategy": "learned_int8",
            "learned_wire_payload_kind": header_fields["tensor_metadata"]["encoded_payload_kind"],
            "original_shape": header_fields["original_shape"],
        },
    )
    request_boundary_adapter = dict(header_fields.get("boundary_adapter") or {})
    request_boundary_adapter["compact_stable_frame"] = True
    request_boundary_adapter["stable_sequence_worker_client_encode"] = True
    return frame, encoded_tensor, request_boundary_adapter


def _encode_trainable_autoencoder_request_frame(
    tensor: TensorPayload,
    *,
    basis_artifact: dict[str, Any],
    context: dict[str, Any],
    source_strategy: BoundaryAdapterStrategy,
    allow_source_only_fallback: bool,
    extra_header: dict[str, Any],
) -> tuple[bytes, TensorPayload, dict[str, Any]]:
    encoded = encode_boundary_wire_payload(
        tensor.to_numpy(),
        boundary_adapter_strategy=TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
        tensor_name=tensor.name,
        tensor_metadata=tensor.metadata,
        trainable_autoencoder_basis_artifact=basis_artifact,
        trainable_autoencoder_source_strategy=source_strategy,
        trainable_autoencoder_context=context,
        trainable_autoencoder_allow_source_only_fallback=allow_source_only_fallback,
    )
    header_fields = dict(encoded.header_fields)
    tensor_metadata = dict(header_fields.get("tensor_metadata") or {})
    frame_shape = [len(encoded.payload)]
    encoded_tensor = TensorPayload.from_raw_bytes(
        encoded.payload,
        dtype=header_fields["dtype"],  # type: ignore[arg-type]
        shape=frame_shape,
        name=str(header_fields.get("tensor_name") or tensor.name),
        metadata=tensor_metadata,
    )
    frame = encode_tensor_frame_from_raw(
        encoded.payload,
        dtype=header_fields["dtype"],  # type: ignore[arg-type]
        shape=frame_shape,
        name=encoded_tensor.name,
        metadata=tensor_metadata,
        extra_header={
            **extra_header,
            "boundary_adapter_strategy": TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
            "trainable_autoencoder_payload_kind": TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND,
            "trainable_autoencoder_basis_hash": header_fields.get(
                "trainable_autoencoder_basis_hash"
            ),
            "original_shape": header_fields["original_shape"],
        },
    )
    request_boundary_adapter = dict(header_fields.get("boundary_adapter") or {})
    request_boundary_adapter["stable_sequence_worker_client_encode"] = True
    request_boundary_adapter["runtime_transport_implemented_claimed"] = False
    request_boundary_adapter["worker_route_transport_exercised"] = True
    request_boundary_adapter["production_runtime_claimed"] = False
    return frame, encoded_tensor, request_boundary_adapter


def _qwen_trace_from_header(header: dict[str, Any]) -> QwenShardTrace:
    trace_payload = header.get("trace")
    if trace_payload is None:
        trace_payload = _expand_compact_qwen_trace(header.get("qt"))
    trace = QwenShardTrace.model_validate(trace_payload)
    client_transport = header.get("_client_transport")
    if not isinstance(client_transport, dict):
        return trace
    server_request_boundary = boundary_adapter_summary_from_metadata(trace.boundary_adapter_input)
    server_response_boundary = boundary_adapter_summary_from_metadata(trace.boundary_adapter_output)
    request_boundary = _merge_boundary_adapter_summaries(
        server_request_boundary,
        boundary_adapter_summary_from_metadata(client_transport.get("request_boundary_adapter")),
    )
    response_boundary = _merge_boundary_adapter_summaries(
        server_response_boundary,
        boundary_adapter_summary_from_metadata(client_transport.get("response_boundary_adapter")),
    )
    request_adapter_id = str(request_boundary.get("adapter_id") or "")
    response_adapter_id = str(response_boundary.get("adapter_id") or "")
    adapter_id = (
        response_adapter_id
        if response_adapter_id and response_adapter_id != "identity"
        else request_adapter_id
        if request_adapter_id and request_adapter_id != "identity"
        else response_adapter_id
        or request_adapter_id
        or trace.boundary_adapter_id
        or "identity"
    )
    return trace.model_copy(
        update={
            "request_frame_bytes": int(client_transport.get("request_frame_bytes") or 0),
            "response_frame_bytes": int(client_transport.get("response_frame_bytes") or 0),
            "input_original_bytes": request_boundary.get("original_raw_bytes") or trace.input_original_bytes,
            "input_encoded_bytes": request_boundary.get("encoded_raw_bytes") or trace.input_encoded_bytes,
            "output_original_bytes": response_boundary.get("original_raw_bytes") or trace.output_original_bytes,
            "output_encoded_bytes": response_boundary.get("encoded_raw_bytes") or trace.output_encoded_bytes,
            "boundary_adapter_id": adapter_id,
            "boundary_adapter_applied": adapter_id != "identity",
            "boundary_adapter_input": request_boundary or trace.boundary_adapter_input,
            "boundary_adapter_output": response_boundary or trace.boundary_adapter_output,
        }
    )


def _expand_compact_qwen_trace(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Qwen tensor frame header requires `trace` or compact `qt`")
    return {
        "shard_id": payload["sid"],
        "runtime_id": payload["rid"],
        "layer_start": payload["ls"],
        "layer_end": payload["le"],
        "input_shape": payload["is"],
        "output_shape": payload["os"],
        "input_bytes": payload.get("ib", 0),
        "output_bytes": payload.get("ob", 0),
        "input_original_bytes": payload.get("iob"),
        "output_original_bytes": payload.get("oob"),
        "input_encoded_bytes": payload.get("ieb"),
        "output_encoded_bytes": payload.get("oeb"),
        "request_frame_bytes": payload.get("rfb"),
        "response_frame_bytes": payload.get("sfb"),
        "boundary_adapter_id": payload.get("ba", "identity"),
        "boundary_adapter_applied": bool(payload.get("baa", False)),
        "boundary_adapter_input": payload.get("bi") or {},
        "boundary_adapter_output": payload.get("bo") or {},
        "elapsed_ms": payload.get("em", 0.0),
        "route_elapsed_ms": payload.get("rem"),
    }


def _merge_boundary_adapter_summaries(
    server_summary: dict[str, Any],
    client_summary: dict[str, Any],
) -> dict[str, Any]:
    if not server_summary:
        return client_summary
    if not client_summary:
        return server_summary
    merged = {**server_summary, **client_summary}
    for key in ("encode_ms", "decode_ms"):
        client_value = float(client_summary.get(key) or 0.0)
        server_value = float(server_summary.get(key) or 0.0)
        if client_value <= 0.0 and server_value > 0.0:
            merged[key] = server_value
    return merged


class RemoteSequenceRuntimeCoordinator:
    def __init__(self, workers: Sequence[SequenceWorkerClient], *, planner: Planner | None = None) -> None:
        if not workers:
            raise ValueError("remote coordinator requires at least one worker")
        self.workers = list(workers)
        self.planner = planner or Planner()

    async def discover_resources(self) -> list[ResourceProfile]:
        return [await worker.profile() for worker in self.workers]

    async def calibrate(self, manifest: ModelManifest) -> list[CalibrationProfile]:
        return [await worker.calibrate(manifest) for worker in self.workers]

    async def execute(
        self,
        manifest: ModelManifest,
        *,
        prompt: str,
        resources: Sequence[ResourceProfile] | None = None,
        calibrations: Sequence[CalibrationProfile] | None = None,
    ) -> SequenceExecutionResult:
        resolved_resources = list(resources or await self.discover_resources())
        resolved_calibrations = list(calibrations or await self.calibrate(manifest))
        plan = self.planner.plan(manifest, resolved_resources, resolved_calibrations)
        worker_by_runtime = {
            resource.runtime_id: worker
            for resource, worker in zip(resolved_resources, self.workers, strict=False)
        }
        payload: dict[str, Any] = {"prompt": prompt, "state": []}
        traces: list[UnitExecutionTrace] = []

        for unit in manifest.graph.ordered_units():
            placement = plan.placement_for(unit.unit_id)
            worker = worker_by_runtime.get(placement.runtime_id)
            if worker is None:
                raise RuntimeError(f"no worker client for runtime {placement.runtime_id}")
            before = payload_digest(payload)
            payload, observed_latency_ms = await worker.execute_unit(unit, payload, precision=plan.precision)
            after = payload_digest(payload)
            traces.append(
                UnitExecutionTrace(
                    unit_id=unit.unit_id,
                    unit_kind=unit.kind,
                    runtime_id=placement.runtime_id,
                    input_digest=before,
                    output_digest=after,
                    simulated_latency_ms=observed_latency_ms,
                )
            )

        final_state = [float(value) for value in payload["state"]]
        output_digest = payload_digest({"model_id": manifest.model_id, "state": final_state})
        architecture_neutral = bool(manifest.graph.unit_kinds() - {"embedding", "attention_block", "mlp", "norm", "lm_head"})
        return SequenceExecutionResult(
            model_id=manifest.model_id,
            prompt=prompt,
            plan=plan,
            traces=traces,
            final_state=final_state,
            output_text=f"remote-simulated:{manifest.model_id}:{output_digest[:12]}",
            architecture_neutral=architecture_neutral,
            metadata={
                "adapter": "remote-sequence-workers",
                "unit_count": len(traces),
                "unit_kinds": sorted(manifest.graph.unit_kinds()),
                "runtimes_used": sorted(plan.runtime_ids()),
                "worker_count": len(self.workers),
            },
        )

    async def execute_tensor_pipeline(
        self,
        manifest: ModelManifest,
        *,
        prompt: str,
        initial_tensor: TensorPayload | None = None,
        hidden_size: int = 16,
        tensor_transport: Literal["json", "binary"] = "json",
        resources: Sequence[ResourceProfile] | None = None,
        calibrations: Sequence[CalibrationProfile] | None = None,
    ) -> TensorSequenceExecutionResult:
        resolved_resources = list(resources or await self.discover_resources())
        resolved_calibrations = list(calibrations or await self.calibrate(manifest))
        plan = self.planner.plan(manifest, resolved_resources, resolved_calibrations)
        worker_by_runtime = {
            resource.runtime_id: worker
            for resource, worker in zip(resolved_resources, self.workers, strict=False)
        }
        tensor = initial_tensor or prompt_to_tensor(prompt, hidden_size=hidden_size)
        traces: list[TensorUnitExecutionTrace] = []

        for unit in manifest.graph.ordered_units():
            placement = plan.placement_for(unit.unit_id)
            worker = worker_by_runtime.get(placement.runtime_id)
            if worker is None:
                raise RuntimeError(f"no worker client for runtime {placement.runtime_id}")
            before = tensor.digest()
            input_shape = list(tensor.shape)
            if tensor_transport == "binary":
                tensor, observed_latency_ms = await worker.execute_tensor_unit_binary(unit, tensor, precision=plan.precision)
            else:
                tensor, observed_latency_ms = await worker.execute_tensor_unit(unit, tensor, precision=plan.precision)
            after = tensor.digest()
            traces.append(
                TensorUnitExecutionTrace(
                    unit_id=unit.unit_id,
                    unit_kind=unit.kind,
                    runtime_id=placement.runtime_id,
                    input_digest=before,
                    output_digest=after,
                    observed_latency_ms=observed_latency_ms,
                    input_shape=input_shape,
                    output_shape=list(tensor.shape),
                    dtype=tensor.dtype,
                )
            )

        output_digest = tensor.digest()
        architecture_neutral = bool(manifest.graph.unit_kinds() - {"embedding", "attention_block", "mlp", "norm", "lm_head"})
        return TensorSequenceExecutionResult(
            model_id=manifest.model_id,
            prompt=prompt,
            plan=plan,
            traces=traces,
            final_tensor=tensor,
            output_text=f"remote-tensor-pipeline:{manifest.model_id}:{output_digest[:12]}",
            architecture_neutral=architecture_neutral,
            metadata={
                "adapter": "remote-tensor-sequence-workers",
                "transport": "sequence-worker-http",
                "payload": "typed-tensor-frame-binary" if tensor_transport == "binary" else "typed-tensor-base64-json",
                "unit_count": len(traces),
                "unit_kinds": sorted(manifest.graph.unit_kinds()),
                "runtimes_used": sorted(plan.runtime_ids()),
                "weight_source": "deterministic-prototype-weights",
                "proof_boundary": "real tensor payloads and layer ops; not pretrained distributed Qwen weights",
            },
        )
