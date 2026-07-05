from __future__ import annotations

import math
import time
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from somatic.sequence_model.boundary_codec_backend import (
    BOUNDARY_CODEC_BACKEND_CONTRACT_METADATA_KEY,
    BOUNDARY_CODEC_BACKEND_REQUEST_METADATA_KEY,
    BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY,
    BOUNDARY_CODEC_RAW_FRAME_RECEIVE_METADATA_KEY,
    BoundaryCodecBackendContract,
    BoundaryCodecBackendId,
    BoundaryCodecBackendRuntimeDecision,
    coerce_boundary_codec_backend_contract,
    resolve_boundary_codec_backend_runtime_decision,
)
from somatic.sequence_model.boundary_codec_native import (
    BoundaryCodecNativeBinding,
    load_boundary_codec_native_binding,
)
from somatic.sequence_model.interfaces import BoundaryAdapterKind, BoundaryAdapterSpec, UnitKind
from somatic.sequence_model.tensors import TensorDType, TensorFrameParts, TensorPayload, encode_tensor_frame


BoundaryAdapterStrategy = Literal[
    "identity",
    "fp16",
    "int8_symmetric",
    "learned_int8",
    "learned_per_boundary_int8",
    "learned_residual4_int8",
    "learned_residual4_sparse24_int8",
    "learned_residual5_int8",
    "learned_residual6_int8",
    "learned_residual8_int8",
]
BOUNDARY_FRAME_ADAPTER_SCHEMA_VERSION = "sequence-boundary-frame-adapter-v0"
BOUNDARY_ADAPTER_METADATA_KEY = "boundary_adapter"
BOUNDARY_ADAPTER_COMPACT_METADATA_KEY = "ba"
BOUNDARY_RAW_RESPONSE_ENCODE_TIMING_METADATA_KEY = "raw_response_encode_timing"
BOUNDARY_POSITION_SCOPED_TRANSPORT_METADATA_KEY = "boundary_position_scoped_transport"


class BoundaryEncodedTensor(BaseModel):
    adapter_id: str
    tensor: TensorPayload
    original_dtype: str
    original_shape: list[int]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BoundaryEncodedRawFrame(BaseModel):
    adapter_id: str
    name: str
    dtype: TensorDType
    shape: list[int]
    raw: bytes
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_encode_path: str
    tensor_payload_constructed_for_frame: bool

    def byte_size(self) -> int:
        return len(self.raw)


class BoundaryTransferProfile(BaseModel):
    schema_version: str = "sequence-boundary-transfer-profile-v0"
    boundary_id: str
    adapter_id: str
    adapter_kind: BoundaryAdapterKind
    source_unit_id: str
    target_unit_id: str
    source_unit_kind: UnitKind
    target_unit_kind: UnitKind
    original_dtype: str
    encoded_dtype: str
    shape: list[int]
    original_raw_bytes: int = Field(ge=0)
    encoded_raw_bytes: int = Field(ge=0)
    original_frame_bytes: int = Field(ge=0)
    encoded_frame_bytes: int = Field(ge=0)
    raw_byte_ratio: float = Field(ge=0.0)
    frame_byte_ratio: float = Field(ge=0.0)
    raw_byte_savings_ratio: float = Field(ge=0.0)
    frame_byte_savings_ratio: float = Field(ge=0.0)
    max_abs_error: float = Field(ge=0.0)
    mean_abs_error: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    cosine_similarity: float
    reversible: bool
    encode_ms: float = Field(ge=0.0)
    decode_ms: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_adapter_spec(self) -> BoundaryAdapterSpec:
        return BoundaryAdapterSpec(
            adapter_id=self.adapter_id,
            kind=self.adapter_kind,
            display_name=self.metadata.get("display_name", self.adapter_id),
            estimated_raw_byte_ratio=self.raw_byte_ratio,
            estimated_frame_byte_ratio=self.frame_byte_ratio,
            estimated_mean_abs_error=self.mean_abs_error,
            estimated_max_abs_error=self.max_abs_error,
            reversible=self.reversible,
            metadata={
                "profile_schema_version": self.schema_version,
                "profile_boundary_id": self.boundary_id,
                "profile_shape": self.shape,
                "profile_original_dtype": self.original_dtype,
                "profile_encoded_dtype": self.encoded_dtype,
                "architecture_neutral_boundary_adapter": True,
            },
        )


def normalize_boundary_adapter_strategy(value: str | None) -> BoundaryAdapterStrategy:
    if value is None or value == "":
        return "identity"
    if value in {
        "identity",
        "fp16",
        "int8_symmetric",
        "learned_int8",
        "learned_per_boundary_int8",
        "learned_residual4_int8",
        "learned_residual4_sparse24_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
    }:
        return value  # type: ignore[return-value]
    raise ValueError(f"unsupported boundary adapter strategy: {value}")


def encode_boundary_tensor(
    array: np.ndarray,
    *,
    strategy: BoundaryAdapterStrategy,
    name: str = "activation",
    metadata: dict[str, Any] | None = None,
) -> BoundaryEncodedTensor:
    original = np.ascontiguousarray(array)
    base_metadata = dict(metadata or {})
    base_metadata["boundary_adapter_strategy"] = strategy
    if strategy == "identity":
        encoded_array = original.copy()
        adapter_metadata: dict[str, Any] = {"reversible": True}
    elif strategy == "fp16":
        encoded_array = original.astype(np.float16)
        adapter_metadata = {"restore_dtype": str(original.dtype), "bounded_loss": True}
    elif strategy == "int8_symmetric" and _position_scoped_mixed_transport_request(base_metadata) is not None:
        encoded_array, adapter_metadata = _encode_position_scoped_mixed_int8_fp16(
            original,
            request=_position_scoped_mixed_transport_request(base_metadata) or {},
        )
    elif strategy == "int8_symmetric":
        float_array = original.astype(np.float32, copy=False)
        max_abs = float(np.max(np.abs(float_array))) if float_array.size else 0.0
        scale = max(max_abs / 127.0, 1e-12)
        encoded_array = np.clip(np.rint(float_array / scale), -127, 127).astype(np.int8)
        adapter_metadata = {
            "restore_dtype": str(original.dtype),
            "scale": scale,
            "zero_point": 0,
            "bounded_loss": True,
        }
    elif strategy == "learned_per_boundary_int8":
        float_array = original.astype(np.float32, copy=False)
        flat = _flatten_boundary_rows(float_array)
        scales = np.maximum(np.max(np.abs(flat), axis=1, keepdims=True) / 127.0, 1e-12).astype(np.float32)
        encoded_array = np.clip(np.rint(flat / scales), -127, 127).astype(np.int8).reshape(original.shape)
        adapter_metadata = {
            "restore_dtype": str(original.dtype),
            "scale_mode": "per_boundary",
            "scale_shape": list(scales.shape),
            "scale_values": [float(value) for value in scales.reshape(-1)],
            "zero_point": 0,
            "bounded_loss": True,
            "research_candidate": True,
        }
    elif strategy in {
        "learned_residual4_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
    }:
        float_array = original.astype(np.float32, copy=False)
        flat = _flatten_boundary_rows(float_array)
        base_scales = np.maximum(np.max(np.abs(flat), axis=1, keepdims=True) / 127.0, 1e-12).astype(np.float32)
        base_quantized = np.clip(np.rint(flat / base_scales), -127, 127).astype(np.int8)
        base_restored = base_quantized.astype(np.float32) * base_scales
        residual = flat - base_restored
        residual_bits = _residual_bits_for_strategy(strategy)
        residual_levels = float((2 ** (residual_bits - 1)) - 1)
        residual_scales = np.maximum(np.max(np.abs(residual), axis=1, keepdims=True) / residual_levels, 1e-12).astype(
            np.float32
        )
        residual_quantized = np.clip(
            np.rint(residual / residual_scales),
            -int(residual_levels),
            int(residual_levels),
        ).astype(np.int8)
        if residual_bits == 8:
            residual_payload = residual_quantized.reshape(-1)
            residual_packing = "signed_int8_dense"
        else:
            residual_payload = _pack_signed_int_bits(residual_quantized.reshape(-1), bits=residual_bits).view(np.int8)
            residual_packing = f"signed_int{residual_bits}_bitstream_lsb"
        base_bytes = base_quantized.reshape(-1)
        encoded_array = np.concatenate([base_bytes, residual_payload]).astype(np.int8, copy=False)
        scales = np.concatenate([base_scales, residual_scales], axis=1).astype(np.float32)
        adapter_metadata = {
            "restore_dtype": str(original.dtype),
            "scale_mode": f"per_boundary_residual{residual_bits}",
            "scale_shape": list(scales.shape),
            "scale_values": [float(value) for value in scales.reshape(-1)],
            "zero_point": 0,
            "residual_bits": residual_bits,
            "residual_packing": residual_packing,
            "original_element_count": int(flat.size),
            "base_byte_count": int(base_bytes.size),
            "residual_packed_byte_count": int(residual_payload.size),
            "encoded_shape": list(encoded_array.shape),
            "bounded_loss": True,
            "research_candidate": True,
        }
    elif strategy == "learned_residual4_sparse24_int8":
        float_array = original.astype(np.float32, copy=False)
        flat = _flatten_boundary_rows(float_array)
        base_scales = np.maximum(np.max(np.abs(flat), axis=1, keepdims=True) / 127.0, 1e-12).astype(np.float32)
        base_quantized = np.clip(np.rint(flat / base_scales), -127, 127).astype(np.int8)
        base_restored = base_quantized.astype(np.float32) * base_scales
        residual = flat - base_restored
        residual_bits = 4
        residual_levels = float((2 ** (residual_bits - 1)) - 1)
        residual_scales = np.maximum(np.max(np.abs(residual), axis=1, keepdims=True) / residual_levels, 1e-12).astype(
            np.float32
        )
        residual_quantized = np.clip(
            np.rint(residual / residual_scales),
            -int(residual_levels),
            int(residual_levels),
        ).astype(np.int8)
        residual_payload = _pack_signed_int_bits(residual_quantized.reshape(-1), bits=residual_bits).view(np.int8)
        sparse_per_row = min(_sparse_corrections_for_strategy(strategy), flat.shape[1])
        sparse_indices = np.zeros((flat.shape[0], sparse_per_row), dtype="<u2")
        sparse_values = np.zeros((flat.shape[0], sparse_per_row), dtype=np.int8)
        sparse_scales = np.full((flat.shape[0], 1), 1e-12, dtype=np.float32)
        if sparse_per_row:
            residual4_restored = residual_quantized.astype(np.float32) * residual_scales
            residual_error = residual - residual4_restored
            for row_index, row_error in enumerate(residual_error):
                if sparse_per_row == flat.shape[1]:
                    selected = np.arange(flat.shape[1])
                else:
                    selected = np.argpartition(np.abs(row_error), -sparse_per_row)[-sparse_per_row:]
                    selected = np.sort(selected)
                selected_error = row_error[selected]
                sparse_scale = max(float(np.max(np.abs(selected_error))) / 127.0, 1e-12)
                sparse_scales[row_index, 0] = sparse_scale
                sparse_indices[row_index] = selected.astype("<u2", copy=False)
                sparse_values[row_index] = np.clip(
                    np.rint(selected_error / sparse_scale),
                    -127,
                    127,
                ).astype(np.int8)
        base_bytes = base_quantized.reshape(-1)
        sparse_index_payload = sparse_indices.reshape(-1).view(np.uint8).view(np.int8)
        sparse_value_payload = sparse_values.reshape(-1)
        encoded_array = np.concatenate(
            [base_bytes, residual_payload, sparse_index_payload, sparse_value_payload]
        ).astype(np.int8, copy=False)
        scales = np.concatenate([base_scales, residual_scales, sparse_scales], axis=1).astype(np.float32)
        adapter_metadata = {
            "restore_dtype": str(original.dtype),
            "scale_mode": "per_boundary_residual4_sparse24",
            "scale_shape": list(scales.shape),
            "scale_values": [float(value) for value in scales.reshape(-1)],
            "zero_point": 0,
            "residual_bits": residual_bits,
            "residual_packing": "signed_int4_bitstream_lsb",
            "original_element_count": int(flat.size),
            "base_byte_count": int(base_bytes.size),
            "residual_packed_byte_count": int(residual_payload.size),
            "sparse_corrections_per_row": int(sparse_per_row),
            "sparse_correction_count": int(sparse_values.size),
            "sparse_index_dtype": "uint16_le",
            "sparse_index_byte_count": int(sparse_index_payload.size),
            "sparse_value_dtype": "int8",
            "sparse_value_byte_count": int(sparse_value_payload.size),
            "encoded_shape": list(encoded_array.shape),
            "bounded_loss": True,
            "research_candidate": True,
        }
    else:
        raise ValueError(f"unsupported boundary adapter strategy: {strategy}")

    base_metadata.update(adapter_metadata)
    payload = TensorPayload.from_numpy(encoded_array, name=name, metadata=base_metadata)
    return BoundaryEncodedTensor(
        adapter_id=strategy,
        tensor=payload,
        original_dtype=str(original.dtype),
        original_shape=list(original.shape),
        metadata={
            "adapter_strategy": strategy,
            **adapter_metadata,
        },
    )


def _encode_position_scoped_mixed_int8_fp16(
    original: np.ndarray,
    *,
    request: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    float_array = original.astype(np.float32, copy=False)
    axis, position = _normalized_position_scope(
        shape=list(float_array.shape),
        request=request,
    )
    axis_length = int(float_array.shape[axis])
    target_slice = [slice(None)] * float_array.ndim
    target_slice[axis] = slice(position, position + 1)
    target = np.ascontiguousarray(float_array[tuple(target_slice)])
    before_slice = [slice(None)] * float_array.ndim
    before_slice[axis] = slice(0, position)
    after_slice = [slice(None)] * float_array.ndim
    after_slice[axis] = slice(position + 1, axis_length)
    rest_parts = [
        np.ascontiguousarray(float_array[tuple(before_slice)]),
        np.ascontiguousarray(float_array[tuple(after_slice)]),
    ]
    rest = (
        np.concatenate(rest_parts, axis=axis)
        if any(int(part.shape[axis]) > 0 for part in rest_parts)
        else np.empty(
            [
                int(dim) if dim_index != axis else 0
                for dim_index, dim in enumerate(float_array.shape)
            ],
            dtype=np.float32,
        )
    )
    rest_fp16 = np.ascontiguousarray(rest.astype(np.float16))
    max_abs = float(np.max(np.abs(target))) if target.size else 0.0
    scale = max(max_abs / 127.0, 1e-12)
    target_int8 = np.ascontiguousarray(
        np.clip(np.rint(target / scale), -127, 127).astype(np.int8)
    )
    raw = rest_fp16.tobytes(order="C") + target_int8.tobytes(order="C")
    encoded_array = np.frombuffer(raw, dtype=np.int8).copy()
    return encoded_array, {
        "restore_dtype": str(original.dtype),
        "scale": scale,
        "zero_point": 0,
        "bounded_loss": True,
        "position_scoped_mixed_frame": True,
        "mixed_frame_layout": "fp16_except_int8_position_v0",
        "mixed_frame_axis": axis,
        "mixed_frame_position": position,
        "mixed_frame_original_shape": [int(dim) for dim in float_array.shape],
        "mixed_frame_rest_shape": [int(dim) for dim in rest_fp16.shape],
        "mixed_frame_rest_dtype": "float16",
        "mixed_frame_rest_byte_count": int(rest_fp16.nbytes),
        "mixed_frame_target_shape": [int(dim) for dim in target_int8.shape],
        "mixed_frame_target_dtype": "int8",
        "mixed_frame_target_byte_count": int(target_int8.nbytes),
        "mixed_frame_target_scale": scale,
        "mixed_frame_target_adapter_id": "int8_symmetric",
        "research_candidate": True,
    }


def _decode_position_scoped_mixed_int8_fp16(encoded: BoundaryEncodedTensor) -> np.ndarray:
    metadata = {**encoded.tensor.metadata, **encoded.metadata}
    original_shape = [int(dim) for dim in metadata.get("mixed_frame_original_shape") or encoded.original_shape]
    axis = int(metadata.get("mixed_frame_axis"))
    position = int(metadata.get("mixed_frame_position"))
    if axis < 0:
        axis += len(original_shape)
    if axis < 0 or axis >= len(original_shape):
        raise ValueError("position-scoped mixed frame axis is out of range")
    if position < 0:
        position += int(original_shape[axis])
    if position < 0 or position >= int(original_shape[axis]):
        raise ValueError("position-scoped mixed frame position is out of range")
    rest_shape = [int(dim) for dim in metadata.get("mixed_frame_rest_shape") or []]
    target_shape = [int(dim) for dim in metadata.get("mixed_frame_target_shape") or []]
    if not rest_shape or not target_shape:
        raise ValueError("position-scoped mixed frame is missing rest/target shapes")
    rest_byte_count = int(metadata.get("mixed_frame_rest_byte_count") or 0)
    target_byte_count = int(metadata.get("mixed_frame_target_byte_count") or 0)
    raw = encoded.tensor.to_numpy().astype(np.int8, copy=False).reshape(-1).tobytes(order="C")
    if len(raw) < rest_byte_count + target_byte_count:
        raise ValueError("position-scoped mixed frame payload is shorter than declared")
    rest_raw = raw[:rest_byte_count]
    target_raw = raw[rest_byte_count : rest_byte_count + target_byte_count]
    rest = (
        np.frombuffer(rest_raw, dtype=np.float16).reshape(tuple(rest_shape)).astype(np.float32)
        if rest_byte_count
        else np.empty(tuple(rest_shape), dtype=np.float32)
    )
    scale = float(metadata.get("mixed_frame_target_scale") or metadata.get("scale") or 1.0)
    target = (
        np.frombuffer(target_raw, dtype=np.int8)
        .reshape(tuple(target_shape))
        .astype(np.float32)
        * np.float32(scale)
    )
    restored = np.empty(tuple(original_shape), dtype=np.float32)
    before_count = position
    after_count = int(original_shape[axis]) - position - 1
    target_slice = [slice(None)] * restored.ndim
    target_slice[axis] = slice(position, position + 1)
    restored[tuple(target_slice)] = target
    if before_count:
        restored_before_slice = [slice(None)] * restored.ndim
        restored_before_slice[axis] = slice(0, position)
        rest_before_slice = [slice(None)] * rest.ndim
        rest_before_slice[axis] = slice(0, before_count)
        restored[tuple(restored_before_slice)] = rest[tuple(rest_before_slice)]
    if after_count:
        restored_after_slice = [slice(None)] * restored.ndim
        restored_after_slice[axis] = slice(position + 1, None)
        rest_after_slice = [slice(None)] * rest.ndim
        rest_after_slice[axis] = slice(before_count, None)
        restored[tuple(restored_after_slice)] = rest[tuple(rest_after_slice)]
    return restored.astype(np.dtype(encoded.original_dtype), copy=False)


def _position_scoped_mixed_transport_request(metadata: dict[str, Any]) -> dict[str, Any] | None:
    raw = metadata.get(BOUNDARY_POSITION_SCOPED_TRANSPORT_METADATA_KEY)
    if not isinstance(raw, dict):
        return None
    scope = str(raw.get("position_scope") or "")
    if scope in {"", "whole_tensor"}:
        return None
    return dict(raw)


def _normalized_position_scope(
    *,
    shape: list[int],
    request: dict[str, Any],
) -> tuple[int, int]:
    if len(shape) < 1:
        raise ValueError("position-scoped mixed frame requires a non-empty shape")
    axis = int(request.get("target_position_axis") if request.get("target_position_axis") is not None else 1)
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError("position-scoped mixed frame target axis is out of range")
    raw_position = request.get("target_position")
    position = int(raw_position if raw_position is not None else int(shape[axis]) - 1)
    if position < 0:
        position += int(shape[axis])
    if position < 0 or position >= int(shape[axis]):
        raise ValueError("position-scoped mixed frame target position is out of range")
    return axis, position


def _is_position_scoped_mixed_metadata(*metadatas: dict[str, Any]) -> bool:
    return any(bool(metadata.get("position_scoped_mixed_frame")) for metadata in metadatas)


def decode_boundary_tensor(encoded: BoundaryEncodedTensor) -> np.ndarray:
    array = encoded.tensor.to_numpy()
    if encoded.adapter_id == "identity":
        return array.astype(np.dtype(encoded.original_dtype), copy=False)
    if encoded.adapter_id == "fp16":
        return array.astype(np.dtype(encoded.original_dtype), copy=False)
    if encoded.adapter_id == "int8_symmetric":
        if _is_position_scoped_mixed_metadata(encoded.metadata, encoded.tensor.metadata):
            return _decode_position_scoped_mixed_int8_fp16(encoded)
        scale = float(encoded.metadata.get("scale", encoded.tensor.metadata.get("scale", 1.0)))
        restored = array.astype(np.float32) * scale
        return restored.astype(np.dtype(encoded.original_dtype), copy=False)
    if encoded.adapter_id == "learned_per_boundary_int8":
        scales = _boundary_scale_array(
            scale_values=encoded.metadata.get("scale_values", encoded.tensor.metadata.get("scale_values")),
            scale_shape=encoded.metadata.get("scale_shape", encoded.tensor.metadata.get("scale_shape")),
            expected_columns=1,
            adapter_id=encoded.adapter_id,
        )
        restored = (array.astype(np.float32).reshape(scales.shape[0], -1) * scales).reshape(encoded.original_shape)
        return restored.astype(np.dtype(encoded.original_dtype), copy=False)
    if encoded.adapter_id in {
        "learned_residual4_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
    }:
        scales = _boundary_scale_array(
            scale_values=encoded.metadata.get("scale_values", encoded.tensor.metadata.get("scale_values")),
            scale_shape=encoded.metadata.get("scale_shape", encoded.tensor.metadata.get("scale_shape")),
            expected_columns=2,
            adapter_id=encoded.adapter_id,
        )
        element_count = int(encoded.metadata.get("original_element_count") or math.prod(encoded.original_shape))
        base_byte_count = int(encoded.metadata.get("base_byte_count") or element_count)
        residual_bits = int(
            encoded.metadata.get("residual_bits") or _residual_bits_for_strategy(encoded.adapter_id)
        )
        default_residual_byte_count = element_count if residual_bits == 8 else math.ceil(element_count * residual_bits / 8)
        residual_byte_count = int(
            encoded.metadata.get("residual_packed_byte_count") or default_residual_byte_count
        )
        raw = array.astype(np.int8, copy=False).reshape(-1)
        if raw.size < base_byte_count + residual_byte_count:
            raise ValueError(f"{encoded.adapter_id} tensor is shorter than declared packed payload")
        base_quantized = raw[:base_byte_count].reshape(scales.shape[0], -1).astype(np.float32)
        residual_payload = raw[base_byte_count : base_byte_count + residual_byte_count]
        residual_quantized = (
            residual_payload[:element_count].astype(np.int8)
            if residual_bits == 8
            else _unpack_signed_int_bits(residual_payload.view(np.uint8), count=element_count, bits=residual_bits)
        ).reshape(scales.shape[0], -1)
        base_scales = scales[:, 0:1]
        residual_scales = scales[:, 1:2]
        restored = (base_quantized * base_scales) + (residual_quantized.astype(np.float32) * residual_scales)
        return restored.reshape(encoded.original_shape).astype(np.dtype(encoded.original_dtype), copy=False)
    if encoded.adapter_id == "learned_residual4_sparse24_int8":
        scales = _boundary_scale_array(
            scale_values=encoded.metadata.get("scale_values", encoded.tensor.metadata.get("scale_values")),
            scale_shape=encoded.metadata.get("scale_shape", encoded.tensor.metadata.get("scale_shape")),
            expected_columns=3,
            adapter_id=encoded.adapter_id,
        )
        element_count = int(encoded.metadata.get("original_element_count") or math.prod(encoded.original_shape))
        base_byte_count = int(encoded.metadata.get("base_byte_count") or element_count)
        residual_bits = int(encoded.metadata.get("residual_bits") or 4)
        residual_byte_count = int(
            encoded.metadata.get("residual_packed_byte_count") or math.ceil(element_count * residual_bits / 8)
        )
        sparse_per_row = int(
            encoded.metadata.get("sparse_corrections_per_row")
            or min(_sparse_corrections_for_strategy(encoded.adapter_id), element_count)
        )
        sparse_index_byte_count = int(
            encoded.metadata.get("sparse_index_byte_count") or scales.shape[0] * sparse_per_row * 2
        )
        sparse_value_byte_count = int(
            encoded.metadata.get("sparse_value_byte_count") or scales.shape[0] * sparse_per_row
        )
        raw = array.astype(np.int8, copy=False).reshape(-1)
        declared_size = base_byte_count + residual_byte_count + sparse_index_byte_count + sparse_value_byte_count
        if raw.size < declared_size:
            raise ValueError(f"{encoded.adapter_id} tensor is shorter than declared sparse payload")
        base_quantized = raw[:base_byte_count].reshape(scales.shape[0], -1).astype(np.float32)
        residual_payload_start = base_byte_count
        residual_payload_end = residual_payload_start + residual_byte_count
        residual_payload = raw[residual_payload_start:residual_payload_end]
        residual_quantized = _unpack_signed_int_bits(
            residual_payload.view(np.uint8),
            count=element_count,
            bits=residual_bits,
        ).reshape(scales.shape[0], -1)
        base_scales = scales[:, 0:1]
        residual_scales = scales[:, 1:2]
        sparse_scales = scales[:, 2:3]
        restored = (base_quantized * base_scales) + (residual_quantized.astype(np.float32) * residual_scales)
        if sparse_per_row:
            index_start = residual_payload_end
            index_end = index_start + sparse_index_byte_count
            value_end = index_end + sparse_value_byte_count
            index_bytes = np.ascontiguousarray(raw[index_start:index_end]).view(np.uint8)
            sparse_indices = index_bytes.view("<u2").reshape(scales.shape[0], sparse_per_row)
            sparse_values = raw[index_end:value_end].reshape(scales.shape[0], sparse_per_row).astype(np.float32)
            row_ids = np.arange(scales.shape[0])[:, None]
            restored[row_ids, sparse_indices.astype(np.int64)] += sparse_values * sparse_scales
        return restored.reshape(encoded.original_shape).astype(np.dtype(encoded.original_dtype), copy=False)
    raise ValueError(f"unsupported boundary adapter id: {encoded.adapter_id}")


def encode_payload_for_boundary(
    tensor: TensorPayload,
    *,
    strategy: BoundaryAdapterStrategy,
    backend_contract: BoundaryCodecBackendContract | dict[str, Any] | None = None,
    requested_backend_id: BoundaryCodecBackendId | None = None,
    rust_native_runtime_binding_available: bool | None = None,
) -> tuple[TensorPayload, dict[str, Any]]:
    original = tensor.to_numpy()
    native_binding = _load_rust_native_boundary_codec_binding(
        rust_native_runtime_binding_available=rust_native_runtime_binding_available
    )
    backend_decision = _boundary_codec_backend_runtime_decision(
        metadata=tensor.metadata,
        backend_contract=backend_contract,
        requested_backend_id=requested_backend_id,
        adapter_id=strategy,
        operation="encode",
        rust_native_runtime_binding_available=native_binding is not None,
        tensor_shape=[int(dim) for dim in original.shape],
    )
    started = time.perf_counter()
    native_error: str | None = None
    if backend_decision is not None and backend_decision.executed_backend_id == "rust_native":
        try:
            encoded = _encode_boundary_tensor_with_native_binding(
                native_binding,
                original,
                strategy=strategy,
                name=tensor.name,
                metadata=tensor.metadata,
            )
        except Exception as exc:
            native_error = type(exc).__name__
            encoded = encode_boundary_tensor(
                original,
                strategy=strategy,
                name=tensor.name,
                metadata=tensor.metadata,
            )
            backend_decision = _fallback_backend_decision_after_native_error(backend_decision, native_error)
    else:
        encoded = encode_boundary_tensor(
            original,
            strategy=strategy,
            name=tensor.name,
            metadata=tensor.metadata,
        )
    encode_ms = (time.perf_counter() - started) * 1000.0
    restored = decode_boundary_tensor(encoded)
    diff = restored.astype(np.float32, copy=False) - original.astype(np.float32, copy=False)
    abs_diff = np.abs(diff)
    encoded_raw_bytes = _encoded_raw_bytes_with_adapter_overhead(encoded)
    metadata = _boundary_adapter_metadata(
        adapter_id=strategy,
        adapter_kind=_adapter_kind(strategy),
        original_dtype=str(original.dtype),
        encoded_dtype=encoded.tensor.dtype,
        shape=list(original.shape),
        original_raw_bytes=tensor.byte_size(),
        encoded_raw_bytes=encoded_raw_bytes,
        reversible=strategy == "identity",
        max_abs_error=float(abs_diff.max()) if abs_diff.size else 0.0,
        mean_abs_error=float(abs_diff.mean()) if abs_diff.size else 0.0,
        rmse=float(math.sqrt(float(np.mean(diff * diff)))) if diff.size else 0.0,
        cosine_similarity=_cosine_similarity(original.astype(np.float32, copy=False), restored.astype(np.float32, copy=False)),
        encode_ms=encode_ms,
        decode_ms=0.0,
    )
    if backend_decision is not None:
        metadata[BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY] = backend_decision.model_dump(mode="json")
    _copy_adapter_specific_metadata(metadata, encoded.metadata, strategy)
    payload_metadata = _boundary_payload_metadata_for_frame(
        source_metadata=encoded.tensor.metadata,
        boundary_metadata=metadata,
        strategy=strategy,
        backend_decision=backend_decision,
    )
    encoded_payload = encoded.tensor.model_copy(update={"metadata": payload_metadata})
    return encoded_payload, metadata


def encode_payload_for_boundary_raw_frame(
    tensor: TensorPayload,
    *,
    strategy: BoundaryAdapterStrategy,
    backend_contract: BoundaryCodecBackendContract | dict[str, Any] | None = None,
    requested_backend_id: BoundaryCodecBackendId | None = None,
    rust_native_runtime_binding_available: bool | None = None,
) -> tuple[BoundaryEncodedRawFrame, dict[str, Any]]:
    native_binding = _load_rust_native_boundary_codec_binding(
        rust_native_runtime_binding_available=rust_native_runtime_binding_available
    )
    backend_decision = _boundary_codec_backend_runtime_decision(
        metadata=tensor.metadata,
        backend_contract=backend_contract,
        requested_backend_id=requested_backend_id,
        adapter_id=strategy,
        operation="encode",
        rust_native_runtime_binding_available=native_binding is not None,
        tensor_shape=[int(dim) for dim in tensor.shape],
    )
    if (
        strategy != "identity"
        and backend_decision is not None
        and backend_decision.executed_backend_id == "rust_native"
        and _position_scoped_mixed_transport_request(tensor.metadata) is None
    ):
        tensor_to_numpy_started = time.perf_counter()
        original = tensor.to_numpy()
        tensor_to_numpy_ms = (time.perf_counter() - tensor_to_numpy_started) * 1000.0
        native_encode_started = time.perf_counter()
        native_error: str | None = None
        try:
            raw_encoded = _encode_boundary_tensor_raw_with_native_binding(
                native_binding,
                original,
                strategy=strategy,
                name=tensor.name,
                metadata=tensor.metadata,
            )
        except Exception as exc:
            native_error = type(exc).__name__
            raw_encoded = None
            backend_decision = _fallback_backend_decision_after_native_error(backend_decision, native_error)
        if raw_encoded is not None:
            native_binding_encode_ms = (time.perf_counter() - native_encode_started) * 1000.0
            encode_ms = tensor_to_numpy_ms + native_binding_encode_ms
            metadata_assembly_started = time.perf_counter()
            adapter_metadata = dict(raw_encoded["metadata"])
            encoded_raw_bytes = _encoded_raw_bytes_with_adapter_metadata(
                len(raw_encoded["raw"]),
                adapter_id=strategy,
                metadata=adapter_metadata,
            )
            boundary_metadata = _boundary_adapter_metadata(
                adapter_id=strategy,
                adapter_kind=_adapter_kind(strategy),
                original_dtype=str(original.dtype),
                encoded_dtype=str(raw_encoded["dtype"]),
                shape=list(original.shape),
                original_raw_bytes=tensor.byte_size(),
                encoded_raw_bytes=encoded_raw_bytes,
                reversible=False,
                max_abs_error=None,
                mean_abs_error=None,
                rmse=None,
                cosine_similarity=None,
                encode_ms=encode_ms,
                decode_ms=0.0,
            )
            boundary_metadata["quality_probe_mode"] = "skipped_raw_response_encode_path"
            boundary_metadata["tensor_payload_constructed_for_frame"] = False
            boundary_metadata["raw_encode_path"] = "rust_native_raw_frame"
            boundary_metadata[BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY] = backend_decision.model_dump(mode="json")
            _copy_adapter_specific_metadata(boundary_metadata, adapter_metadata, strategy)
            payload_metadata = _boundary_payload_metadata_for_frame(
                source_metadata=tensor.metadata,
                boundary_metadata=boundary_metadata,
                strategy=strategy,
                backend_decision=backend_decision,
            )
            metadata_assembly_ms = (time.perf_counter() - metadata_assembly_started) * 1000.0
            boundary_metadata["encode_ms"] = tensor_to_numpy_ms + native_binding_encode_ms + metadata_assembly_ms
            raw_response_encode_timing = {
                "schema_version": "boundary-raw-response-encode-timing-v0",
                "tensor_to_numpy_ms": tensor_to_numpy_ms,
                "native_binding_encode_ms": native_binding_encode_ms,
                "metadata_assembly_ms": metadata_assembly_ms,
                "native_raw_bytes": len(raw_encoded["raw"]),
                "metadata_scale_count": len(adapter_metadata.get("scale_values") or []),
                "native_binding_available": native_binding is not None,
            }
            boundary_metadata[BOUNDARY_RAW_RESPONSE_ENCODE_TIMING_METADATA_KEY] = raw_response_encode_timing
            return (
                BoundaryEncodedRawFrame(
                    adapter_id=strategy,
                    name=tensor.name,
                    dtype=raw_encoded["dtype"],  # type: ignore[arg-type]
                    shape=[int(dim) for dim in raw_encoded["shape"]],
                    raw=raw_encoded["raw"],
                    metadata=payload_metadata,
                    raw_encode_path="rust_native_raw_frame",
                    tensor_payload_constructed_for_frame=False,
                ),
                boundary_metadata,
            )

    encoded_payload, boundary_metadata = encode_payload_for_boundary(
        tensor,
        strategy=strategy,
        backend_contract=backend_contract,
        requested_backend_id=requested_backend_id,
        rust_native_runtime_binding_available=rust_native_runtime_binding_available,
    )
    fallback_decision = boundary_metadata.get(BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY)
    if isinstance(fallback_decision, dict) and fallback_decision.get("executed_backend_id") == "python_reference":
        raw_encode_path = "python_reference_tensor_payload_frame"
    else:
        raw_encode_path = "tensor_payload_frame"
    boundary_metadata["quality_probe_mode"] = "response_encode_roundtrip_probe"
    boundary_metadata["tensor_payload_constructed_for_frame"] = True
    boundary_metadata["raw_encode_path"] = raw_encode_path
    boundary_metadata[BOUNDARY_RAW_RESPONSE_ENCODE_TIMING_METADATA_KEY] = {
        "schema_version": "boundary-raw-response-encode-timing-v0",
        "tensor_to_numpy_ms": None,
        "native_binding_encode_ms": None,
        "metadata_assembly_ms": None,
        "native_raw_bytes": encoded_payload.byte_size(),
        "metadata_scale_count": None,
        "native_binding_available": False,
    }
    return (
        BoundaryEncodedRawFrame(
            adapter_id=strategy,
            name=encoded_payload.name,
            dtype=encoded_payload.dtype,
            shape=encoded_payload.shape,
            raw=encoded_payload.raw_bytes(),
            metadata=encoded_payload.metadata,
            raw_encode_path=raw_encode_path,
            tensor_payload_constructed_for_frame=True,
        ),
        boundary_metadata,
    )


def decode_payload_from_boundary(
    tensor: TensorPayload,
    *,
    backend_contract: BoundaryCodecBackendContract | dict[str, Any] | None = None,
    requested_backend_id: BoundaryCodecBackendId | None = None,
    rust_native_runtime_binding_available: bool | None = None,
) -> tuple[TensorPayload, dict[str, Any]]:
    metadata = _boundary_adapter_metadata_from_payload(tensor.metadata, tensor=tensor)
    if not isinstance(metadata, dict):
        identity_metadata = _boundary_adapter_metadata(
            adapter_id="identity",
            adapter_kind="identity",
            original_dtype=tensor.dtype,
            encoded_dtype=tensor.dtype,
            shape=tensor.shape,
            original_raw_bytes=tensor.byte_size(),
            encoded_raw_bytes=tensor.byte_size(),
            reversible=True,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            rmse=0.0,
            cosine_similarity=1.0,
            encode_ms=0.0,
            decode_ms=0.0,
        )
        native_binding = _load_rust_native_boundary_codec_binding(
            rust_native_runtime_binding_available=rust_native_runtime_binding_available
        )
        backend_decision = _boundary_codec_backend_runtime_decision(
            metadata=tensor.metadata,
            backend_contract=backend_contract,
            requested_backend_id=requested_backend_id,
            adapter_id="identity",
            operation="decode",
            rust_native_runtime_binding_available=native_binding is not None,
            tensor_shape=[int(dim) for dim in tensor.shape],
        )
        if backend_decision is not None:
            identity_metadata[BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY] = backend_decision.model_dump(mode="json")
        return tensor, identity_metadata
    adapter_id = normalize_boundary_adapter_strategy(str(metadata.get("adapter_id") or "identity"))
    native_binding = _load_rust_native_boundary_codec_binding(
        rust_native_runtime_binding_available=rust_native_runtime_binding_available
    )
    backend_decision = _boundary_codec_backend_runtime_decision(
        metadata=tensor.metadata,
        backend_contract=backend_contract,
        requested_backend_id=requested_backend_id,
        adapter_id=adapter_id,
        operation="decode",
        rust_native_runtime_binding_available=native_binding is not None,
        tensor_shape=[int(dim) for dim in metadata.get("shape", tensor.shape)],
    )
    if adapter_id == "identity":
        identity_metadata = dict(metadata)
        if backend_decision is not None:
            identity_metadata[BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY] = backend_decision.model_dump(mode="json")
        return tensor.model_copy(update={"metadata": {**tensor.metadata, **identity_metadata}}), identity_metadata
    started = time.perf_counter()
    encoded = BoundaryEncodedTensor(
        adapter_id=adapter_id,
        tensor=tensor,
        original_dtype=str(metadata.get("original_dtype") or tensor.dtype),
        original_shape=[int(dim) for dim in metadata.get("shape", tensor.shape)],
        metadata={**tensor.metadata, **metadata},
    )
    if backend_decision is not None and backend_decision.executed_backend_id == "rust_native":
        try:
            decoded_array = _decode_boundary_tensor_with_native_binding(native_binding, encoded)
        except Exception as exc:
            decoded_array = decode_boundary_tensor(encoded)
            backend_decision = _fallback_backend_decision_after_native_error(backend_decision, type(exc).__name__)
    else:
        decoded_array = decode_boundary_tensor(encoded)
    decode_ms = (time.perf_counter() - started) * 1000.0
    decoded_metadata = dict(tensor.metadata)
    decoded_metadata[BOUNDARY_ADAPTER_METADATA_KEY] = {
        **metadata,
        "decoded_on_receive": True,
        "decode_ms": decode_ms,
    }
    if backend_decision is not None:
        decision_payload = backend_decision.model_dump(mode="json")
        decoded_metadata[BOUNDARY_ADAPTER_METADATA_KEY][BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY] = decision_payload
        decoded_metadata[BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY] = decision_payload
    decoded_metadata.pop(BOUNDARY_ADAPTER_COMPACT_METADATA_KEY, None)
    decoded = TensorPayload.from_numpy(decoded_array, name=tensor.name, metadata=decoded_metadata)
    return decoded, decoded_metadata[BOUNDARY_ADAPTER_METADATA_KEY]


def _load_rust_native_boundary_codec_binding(
    *,
    rust_native_runtime_binding_available: bool | None,
) -> BoundaryCodecNativeBinding | None:
    if rust_native_runtime_binding_available is False:
        return None
    return load_boundary_codec_native_binding()


def _encode_boundary_tensor_with_native_binding(
    binding: BoundaryCodecNativeBinding | None,
    array: np.ndarray,
    *,
    strategy: BoundaryAdapterStrategy,
    name: str,
    metadata: dict[str, Any] | None,
) -> BoundaryEncodedTensor:
    if binding is None:
        raise RuntimeError("rust native boundary codec binding is not loaded")
    if getattr(binding, "lower_copy_ffi_available", False):
        response = binding.encode_array_lower_copy_raw(array, adapter_id=strategy)
        encoded_raw = bytes(response["bytes"])
        encoded_dtype = response["dtype"]
        encoded_shape = response["shape"]
    elif getattr(binding, "binary_ffi_available", False):
        response = binding.encode_array_binary_raw(array, adapter_id=strategy)
        encoded_raw = bytes(response["bytes"])
        encoded_dtype = response["dtype"]
        encoded_shape = response["shape"]
    else:
        response = binding.encode_array(array, adapter_id=strategy)
        encoded_raw = base64_decode_ascii(str(response["encoded_bytes_b64"]))
        encoded_dtype = response["encoded_dtype"]
        encoded_shape = response["encoded_shape"]
    adapter_metadata = dict(response.get("metadata") or {})
    tensor = TensorPayload.from_raw_bytes(
        encoded_raw,
        dtype=encoded_dtype,  # type: ignore[arg-type]
        shape=[int(dim) for dim in encoded_shape],
        name=name,
        metadata={**(metadata or {}), **adapter_metadata, "boundary_adapter_strategy": strategy},
    )
    return BoundaryEncodedTensor(
        adapter_id=strategy,
        tensor=tensor,
        original_dtype=str(np.asarray(array).dtype),
        original_shape=list(np.asarray(array).shape),
        metadata={"adapter_strategy": strategy, **adapter_metadata},
    )


def _encode_boundary_tensor_raw_with_native_binding(
    binding: BoundaryCodecNativeBinding | None,
    array: np.ndarray,
    *,
    strategy: BoundaryAdapterStrategy,
    name: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    del name, metadata
    if binding is None:
        raise RuntimeError("rust native boundary codec binding is not loaded")
    if getattr(binding, "lower_copy_ffi_available", False):
        response = binding.encode_array_lower_copy_raw(array, adapter_id=strategy)
    elif getattr(binding, "binary_ffi_available", False):
        response = binding.encode_array_binary_raw(array, adapter_id=strategy)
    else:
        json_response = binding.encode_array(array, adapter_id=strategy)
        response = {
            "bytes": base64_decode_ascii(str(json_response["encoded_bytes_b64"])),
            "dtype": json_response["encoded_dtype"],
            "shape": json_response["encoded_shape"],
            "metadata": json_response.get("metadata") or {},
        }
    return {
        "raw": bytes(response["bytes"]),
        "dtype": str(response["dtype"]),
        "shape": [int(dim) for dim in response["shape"]],
        "metadata": dict(response.get("metadata") or {}),
    }


def _decode_boundary_tensor_with_native_binding(
    binding: BoundaryCodecNativeBinding | None,
    encoded: BoundaryEncodedTensor,
) -> np.ndarray:
    if binding is None:
        raise RuntimeError("rust native boundary codec binding is not loaded")
    decoded = binding.decode_array(
        adapter_id=encoded.adapter_id,
        original_dtype=encoded.original_dtype,
        original_shape=encoded.original_shape,
        encoded_dtype=encoded.tensor.dtype,
        encoded_shape=encoded.tensor.shape,
        encoded_bytes=encoded.tensor.raw_bytes(),
        metadata=encoded.metadata,
    )
    return decoded.astype(np.dtype(encoded.original_dtype), copy=False)


def _fallback_backend_decision_after_native_error(
    decision: BoundaryCodecBackendRuntimeDecision,
    error_type: str,
) -> BoundaryCodecBackendRuntimeDecision:
    reason_codes = [
        *decision.reason_codes,
        "rust_native_runtime_execution_failed",
        f"rust_native_runtime_error_type:{error_type}",
        "python_reference_fallback_executed",
    ]
    return decision.model_copy(
        update={
            "executed_backend_id": "python_reference",
            "status": "fallback",
            "reason_codes": reason_codes,
        }
    )


def base64_decode_ascii(value: str) -> bytes:
    import base64

    return base64.b64decode(value.encode("ascii"), validate=True)


def _boundary_codec_backend_runtime_decision(
    *,
    metadata: dict[str, Any],
    backend_contract: BoundaryCodecBackendContract | dict[str, Any] | None,
    requested_backend_id: BoundaryCodecBackendId | None,
    adapter_id: str,
    operation: str,
    rust_native_runtime_binding_available: bool,
    tensor_shape: list[int] | None = None,
) -> BoundaryCodecBackendRuntimeDecision | None:
    resolved_contract = coerce_boundary_codec_backend_contract(backend_contract)
    if resolved_contract is None:
        resolved_contract = coerce_boundary_codec_backend_contract(
            metadata.get(BOUNDARY_CODEC_BACKEND_CONTRACT_METADATA_KEY)
        )
    resolved_requested_backend_id = requested_backend_id
    if resolved_requested_backend_id is None:
        candidate = metadata.get(BOUNDARY_CODEC_BACKEND_REQUEST_METADATA_KEY)
        if candidate in {"python_reference", "rust_native"}:
            resolved_requested_backend_id = candidate
    if resolved_contract is None and resolved_requested_backend_id is None:
        return None
    return resolve_boundary_codec_backend_runtime_decision(
        contract=resolved_contract,
        requested_backend_id=resolved_requested_backend_id,
        adapter_id=adapter_id,
        operation=operation,  # type: ignore[arg-type]
        rust_native_runtime_binding_available=rust_native_runtime_binding_available,
        tensor_shape=tensor_shape,
        allow_shape_aware_policy=True,
    )


def boundary_adapter_summary_from_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    if "a" in metadata or "od" in metadata or "eb" in metadata:
        metadata = _expand_compact_boundary_adapter_metadata(metadata)
    return {
        "schema_version": metadata.get("schema_version"),
        "adapter_id": metadata.get("adapter_id"),
        "adapter_kind": metadata.get("adapter_kind"),
        "original_dtype": metadata.get("original_dtype"),
        "encoded_dtype": metadata.get("encoded_dtype"),
        "shape": metadata.get("shape"),
        "original_raw_bytes": metadata.get("original_raw_bytes"),
        "encoded_raw_bytes": metadata.get("encoded_raw_bytes"),
        "raw_byte_ratio": metadata.get("raw_byte_ratio"),
        "raw_byte_savings_ratio": metadata.get("raw_byte_savings_ratio"),
        "max_abs_error": metadata.get("max_abs_error"),
        "mean_abs_error": metadata.get("mean_abs_error"),
        "rmse": metadata.get("rmse"),
        "cosine_similarity": metadata.get("cosine_similarity"),
        "reversible": metadata.get("reversible"),
        "scale_mode": metadata.get("scale_mode"),
        "position_scoped_mixed_frame": metadata.get("position_scoped_mixed_frame"),
        "mixed_frame_layout": metadata.get("mixed_frame_layout"),
        "mixed_frame_axis": metadata.get("mixed_frame_axis"),
        "mixed_frame_position": metadata.get("mixed_frame_position"),
        "mixed_frame_original_shape": metadata.get("mixed_frame_original_shape"),
        "mixed_frame_rest_byte_count": metadata.get("mixed_frame_rest_byte_count"),
        "mixed_frame_target_byte_count": metadata.get("mixed_frame_target_byte_count"),
        "scale_shape": metadata.get("scale_shape"),
        "scale_count": metadata.get("scale_count"),
        "residual_bits": metadata.get("residual_bits"),
        "residual_packed_byte_count": metadata.get("residual_packed_byte_count"),
        "sparse_corrections_per_row": metadata.get("sparse_corrections_per_row"),
        "sparse_index_byte_count": metadata.get("sparse_index_byte_count"),
        "sparse_value_byte_count": metadata.get("sparse_value_byte_count"),
        "artifact_id": metadata.get("artifact_id"),
        "artifact_hash_validated": metadata.get("artifact_hash_validated"),
        "learned_payload_family": metadata.get("learned_payload_family"),
        "boundary_compression_strategy": metadata.get("boundary_compression_strategy"),
        "encoded_payload_kind": metadata.get("encoded_payload_kind"),
        "basis_hash": metadata.get("basis_hash"),
        "source_strategy": metadata.get("source_strategy"),
        "worker_route_transport_exercised": metadata.get("worker_route_transport_exercised"),
        "runtime_transport_implemented_claimed": metadata.get("runtime_transport_implemented_claimed"),
        "planner_selectable_claimed": metadata.get("planner_selectable_claimed"),
        "production_runtime_claimed": metadata.get("production_runtime_claimed"),
        "encode_ms": metadata.get("encode_ms"),
        "decode_ms": metadata.get("decode_ms"),
        BOUNDARY_CODEC_RAW_FRAME_RECEIVE_METADATA_KEY: metadata.get(BOUNDARY_CODEC_RAW_FRAME_RECEIVE_METADATA_KEY),
        BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY: metadata.get(BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY),
    }


def boundary_adapter_metadata_from_payload_metadata(
    payload_metadata: dict[str, Any],
    *,
    tensor_dtype: str,
    tensor_shape: list[int],
    tensor_byte_size: int,
) -> dict[str, Any] | None:
    metadata = payload_metadata.get(BOUNDARY_ADAPTER_METADATA_KEY)
    if isinstance(metadata, dict):
        return metadata
    compact = payload_metadata.get(BOUNDARY_ADAPTER_COMPACT_METADATA_KEY)
    if isinstance(compact, dict):
        expanded = _expand_compact_boundary_adapter_metadata(compact)
        expanded.setdefault("encoded_dtype", tensor_dtype)
        expanded.setdefault("shape", tensor_shape)
        expanded.setdefault("encoded_raw_bytes", tensor_byte_size)
        encoded_raw_bytes = int(expanded.get("encoded_raw_bytes") or tensor_byte_size)
        original_raw_bytes = int(expanded.get("original_raw_bytes") or encoded_raw_bytes)
        expanded["raw_byte_ratio"] = encoded_raw_bytes / original_raw_bytes if original_raw_bytes else 1.0
        expanded["raw_byte_savings_ratio"] = max(0.0, 1.0 - expanded["raw_byte_ratio"])
        return expanded
    return None


def decode_boundary_frame_parts_to_numpy(
    parts: TensorFrameParts,
    *,
    backend_contract: BoundaryCodecBackendContract | dict[str, Any] | None = None,
    requested_backend_id: BoundaryCodecBackendId | None = None,
    rust_native_runtime_binding_available: bool | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    metadata = boundary_adapter_metadata_from_payload_metadata(
        parts.metadata,
        tensor_dtype=parts.dtype,
        tensor_shape=parts.shape,
        tensor_byte_size=len(parts.raw),
    )
    if metadata is None:
        metadata = _boundary_adapter_metadata(
            adapter_id="identity",
            adapter_kind="identity",
            original_dtype=parts.dtype,
            encoded_dtype=parts.dtype,
            shape=parts.shape,
            original_raw_bytes=len(parts.raw),
            encoded_raw_bytes=len(parts.raw),
            reversible=True,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            rmse=0.0,
            cosine_similarity=1.0,
            encode_ms=0.0,
            decode_ms=0.0,
        )
    adapter_id = normalize_boundary_adapter_strategy(str(metadata.get("adapter_id") or "identity"))
    native_binding = _load_rust_native_boundary_codec_binding(
        rust_native_runtime_binding_available=rust_native_runtime_binding_available
    )
    backend_decision = _boundary_codec_backend_runtime_decision(
        metadata=parts.metadata,
        backend_contract=backend_contract,
        requested_backend_id=requested_backend_id,
        adapter_id=adapter_id,
        operation="decode",
        rust_native_runtime_binding_available=native_binding is not None,
        tensor_shape=[int(dim) for dim in metadata.get("shape", parts.shape)],
    )
    started = time.perf_counter()
    decoded_array: np.ndarray
    raw_frame_decode_path = "identity_raw_frame"
    tensor_payload_constructed = False
    if adapter_id == "identity":
        decoded_array = np.frombuffer(parts.raw, dtype=np.dtype(parts.dtype)).reshape(tuple(parts.shape)).copy()
    elif backend_decision is not None and backend_decision.executed_backend_id == "rust_native":
        try:
            decoded_array = _decode_raw_boundary_frame_with_native_binding(
                native_binding,
                parts=parts,
                metadata=metadata,
                adapter_id=adapter_id,
            )
            raw_frame_decode_path = "rust_native_raw_frame"
        except Exception as exc:
            decoded_array = _decode_raw_boundary_frame_with_python_reference(
                parts=parts,
                metadata=metadata,
                adapter_id=adapter_id,
            )
            tensor_payload_constructed = True
            raw_frame_decode_path = "python_reference_payload_fallback"
            backend_decision = _fallback_backend_decision_after_native_error(backend_decision, type(exc).__name__)
    else:
        decoded_array = _decode_raw_boundary_frame_with_python_reference(
            parts=parts,
            metadata=metadata,
            adapter_id=adapter_id,
        )
        tensor_payload_constructed = True
        raw_frame_decode_path = "python_reference_payload_fallback" if adapter_id != "identity" else "identity_raw_frame"
    decode_ms = (time.perf_counter() - started) * 1000.0
    decoded_metadata = dict(parts.metadata)
    boundary_metadata = {
        **metadata,
        "decoded_on_receive": True,
        "decode_ms": decode_ms,
        BOUNDARY_CODEC_RAW_FRAME_RECEIVE_METADATA_KEY: {
            "enabled": True,
            "decode_path": raw_frame_decode_path,
            "tensor_payload_constructed_for_decode": tensor_payload_constructed,
            "native_binding_available": native_binding is not None,
        },
    }
    if backend_decision is not None:
        decision_payload = backend_decision.model_dump(mode="json")
        boundary_metadata[BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY] = decision_payload
        decoded_metadata[BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY] = decision_payload
    decoded_metadata[BOUNDARY_ADAPTER_METADATA_KEY] = boundary_metadata
    decoded_metadata.pop(BOUNDARY_ADAPTER_COMPACT_METADATA_KEY, None)
    return decoded_array.astype(np.dtype(str(metadata.get("original_dtype") or parts.dtype)), copy=False), decoded_metadata, boundary_metadata


def _decode_raw_boundary_frame_with_native_binding(
    binding: BoundaryCodecNativeBinding | None,
    *,
    parts: TensorFrameParts,
    metadata: dict[str, Any],
    adapter_id: str,
) -> np.ndarray:
    if binding is None:
        raise RuntimeError("rust native boundary codec binding is not loaded")
    decoded = binding.decode_array(
        adapter_id=adapter_id,
        original_dtype=str(metadata.get("original_dtype") or parts.dtype),
        original_shape=[int(dim) for dim in metadata.get("shape", parts.shape)],
        encoded_dtype=parts.dtype,
        encoded_shape=parts.shape,
        encoded_bytes=parts.raw,
        metadata={**parts.metadata, **metadata},
    )
    return decoded.astype(np.dtype(str(metadata.get("original_dtype") or parts.dtype)), copy=False)


def _decode_raw_boundary_frame_with_python_reference(
    *,
    parts: TensorFrameParts,
    metadata: dict[str, Any],
    adapter_id: str,
) -> np.ndarray:
    tensor = TensorPayload.from_raw_bytes(
        parts.raw,
        dtype=parts.dtype,
        shape=parts.shape,
        name=parts.name,
        metadata=parts.metadata,
    )
    encoded = BoundaryEncodedTensor(
        adapter_id=adapter_id,
        tensor=tensor,
        original_dtype=str(metadata.get("original_dtype") or parts.dtype),
        original_shape=[int(dim) for dim in metadata.get("shape", parts.shape)],
        metadata={**parts.metadata, **metadata},
    )
    return decode_boundary_tensor(encoded)


def profile_boundary_transfer(
    array: np.ndarray,
    *,
    strategy: BoundaryAdapterStrategy,
    boundary_id: str,
    source_unit_id: str,
    target_unit_id: str,
    source_unit_kind: UnitKind,
    target_unit_kind: UnitKind,
    name: str = "activation",
    metadata: dict[str, Any] | None = None,
) -> BoundaryTransferProfile:
    original = np.ascontiguousarray(array)
    original_payload = TensorPayload.from_numpy(original, name=name, metadata=metadata or {})
    original_frame = encode_tensor_frame(
        original_payload,
        extra_header={"boundary_id": boundary_id, "boundary_adapter_strategy": "identity"},
    )
    encode_started = time.perf_counter()
    encoded_payload, adapter_metadata = encode_payload_for_boundary(original_payload, strategy=strategy)
    encode_ms = (time.perf_counter() - encode_started) * 1000.0
    encoded_frame = encode_tensor_frame(
        encoded_payload,
        extra_header={"boundary_id": boundary_id, "boundary_adapter_strategy": strategy},
    )
    decode_started = time.perf_counter()
    restored_payload, _ = decode_payload_from_boundary(encoded_payload)
    restored = restored_payload.to_numpy()
    decode_ms = (time.perf_counter() - decode_started) * 1000.0

    original_float = original.astype(np.float32, copy=False)
    restored_float = restored.astype(np.float32, copy=False)
    diff = restored_float - original_float
    abs_diff = np.abs(diff)
    original_raw_bytes = original_payload.byte_size()
    encoded_raw_bytes = int(adapter_metadata.get("encoded_raw_bytes") or encoded_payload.byte_size())
    original_frame_bytes = len(original_frame)
    encoded_frame_bytes = len(encoded_frame)
    raw_byte_ratio = encoded_raw_bytes / original_raw_bytes if original_raw_bytes else 1.0
    frame_byte_ratio = encoded_frame_bytes / original_frame_bytes if original_frame_bytes else 1.0
    adapter_kind = _adapter_kind(strategy)
    return BoundaryTransferProfile(
        boundary_id=boundary_id,
        adapter_id=strategy,
        adapter_kind=adapter_kind,
        source_unit_id=source_unit_id,
        target_unit_id=target_unit_id,
        source_unit_kind=source_unit_kind,
        target_unit_kind=target_unit_kind,
        original_dtype=str(original.dtype),
        encoded_dtype=encoded_payload.dtype,
        shape=list(original.shape),
        original_raw_bytes=original_raw_bytes,
        encoded_raw_bytes=encoded_raw_bytes,
        original_frame_bytes=original_frame_bytes,
        encoded_frame_bytes=encoded_frame_bytes,
        raw_byte_ratio=raw_byte_ratio,
        frame_byte_ratio=frame_byte_ratio,
        raw_byte_savings_ratio=max(0.0, 1.0 - raw_byte_ratio),
        frame_byte_savings_ratio=max(0.0, 1.0 - frame_byte_ratio),
        max_abs_error=float(abs_diff.max()) if abs_diff.size else 0.0,
        mean_abs_error=float(abs_diff.mean()) if abs_diff.size else 0.0,
        rmse=float(math.sqrt(float(np.mean(diff * diff)))) if diff.size else 0.0,
        cosine_similarity=_cosine_similarity(original_float, restored_float),
        reversible=bool(adapter_metadata.get("reversible", strategy == "identity")),
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        metadata={
            "display_name": _strategy_display_name(strategy),
            "strategy": strategy,
            "architecture_neutral_boundary_adapter": True,
            **(metadata or {}),
        },
    )


def summarize_boundary_profiles(profiles: list[BoundaryTransferProfile]) -> dict[str, Any]:
    by_strategy: dict[str, list[BoundaryTransferProfile]] = {}
    for profile in profiles:
        by_strategy.setdefault(profile.adapter_id, []).append(profile)

    summaries: dict[str, Any] = {}
    for strategy, strategy_profiles in sorted(by_strategy.items()):
        original_raw = sum(profile.original_raw_bytes for profile in strategy_profiles)
        encoded_raw = sum(profile.encoded_raw_bytes for profile in strategy_profiles)
        original_frame = sum(profile.original_frame_bytes for profile in strategy_profiles)
        encoded_frame = sum(profile.encoded_frame_bytes for profile in strategy_profiles)
        summaries[strategy] = {
            "adapter_id": strategy,
            "profile_count": len(strategy_profiles),
            "original_raw_bytes": original_raw,
            "encoded_raw_bytes": encoded_raw,
            "original_frame_bytes": original_frame,
            "encoded_frame_bytes": encoded_frame,
            "raw_byte_ratio": encoded_raw / original_raw if original_raw else 1.0,
            "frame_byte_ratio": encoded_frame / original_frame if original_frame else 1.0,
            "raw_byte_savings_ratio": max(0.0, 1.0 - encoded_raw / original_raw) if original_raw else 0.0,
            "frame_byte_savings_ratio": max(0.0, 1.0 - encoded_frame / original_frame) if original_frame else 0.0,
            "max_abs_error": max(profile.max_abs_error for profile in strategy_profiles),
            "mean_abs_error": sum(profile.mean_abs_error for profile in strategy_profiles) / len(strategy_profiles),
            "max_rmse": max(profile.rmse for profile in strategy_profiles),
            "min_cosine_similarity": min(profile.cosine_similarity for profile in strategy_profiles),
            "reversible": all(profile.reversible for profile in strategy_profiles),
            "adapter_kind": strategy_profiles[0].adapter_kind,
        }
    return summaries


def adapter_spec_from_summary(summary: dict[str, Any]) -> BoundaryAdapterSpec:
    return BoundaryAdapterSpec(
        adapter_id=str(summary["adapter_id"]),
        kind=summary["adapter_kind"],
        display_name=_strategy_display_name(str(summary["adapter_id"])),
        estimated_raw_byte_ratio=float(summary["raw_byte_ratio"]),
        estimated_frame_byte_ratio=float(summary["frame_byte_ratio"]),
        estimated_mean_abs_error=float(summary["mean_abs_error"]),
        estimated_max_abs_error=float(summary["max_abs_error"]),
        reversible=bool(summary["reversible"]),
        metadata={
            "source": "boundary-transfer-profile-summary",
            "architecture_neutral_boundary_adapter": True,
            "profile_count": int(summary["profile_count"]),
        },
    )


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator <= 0.0:
        return 1.0
    return float(np.dot(left_flat, right_flat) / denominator)


def _flatten_boundary_rows(array: np.ndarray) -> np.ndarray:
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.shape[-1] == 0:
        raise ValueError("learned_per_boundary_int8 requires a non-empty final tensor dimension")
    return array.reshape(-1, array.shape[-1])


def _boundary_scale_array(
    *,
    scale_values: Any,
    scale_shape: Any,
    expected_columns: int | None = None,
    adapter_id: str = "learned boundary adapter",
) -> np.ndarray:
    if not isinstance(scale_values, list) or not scale_values:
        raise ValueError(f"{adapter_id} metadata requires non-empty scale_values")
    if not isinstance(scale_shape, list) or len(scale_shape) != 2:
        raise ValueError(f"{adapter_id} metadata requires scale_shape [rows, columns]")
    rows, columns = (int(scale_shape[0]), int(scale_shape[1]))
    if rows < 1 or columns < 1:
        raise ValueError(f"{adapter_id} scale_shape must have positive row and column counts")
    if expected_columns is not None and columns != expected_columns:
        raise ValueError(f"{adapter_id} scale_shape must have {expected_columns} columns")
    if len(scale_values) != rows * columns:
        raise ValueError(f"{adapter_id} scale_values length must match scale_shape")
    return np.asarray(scale_values, dtype=np.float32).reshape(rows, columns)


def _pack_signed_int4(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    if np.any(flat < -8) or np.any(flat > 7):
        raise ValueError("signed int4 values must be in [-8, 7]")
    nibbles = (flat.astype(np.int16) & 0x0F).astype(np.uint8)
    if nibbles.size % 2:
        nibbles = np.concatenate([nibbles, np.zeros(1, dtype=np.uint8)])
    return (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8)


def _unpack_signed_int4(packed: np.ndarray, count: int) -> np.ndarray:
    bytes_array = np.asarray(packed, dtype=np.uint8).reshape(-1)
    low = bytes_array & 0x0F
    high = (bytes_array >> 4) & 0x0F
    nibbles = np.empty(bytes_array.size * 2, dtype=np.uint8)
    nibbles[0::2] = low
    nibbles[1::2] = high
    signed = nibbles.astype(np.int16)
    signed[signed >= 8] -= 16
    return signed[:count].astype(np.int8)


def _pack_signed_int_bits(values: np.ndarray, *, bits: int) -> np.ndarray:
    if bits == 4:
        return _pack_signed_int4(values)
    if bits < 2 or bits > 7:
        raise ValueError("signed bit packing supports bit widths from 2 to 7")
    flat = np.asarray(values, dtype=np.int16).reshape(-1)
    minimum = -(2 ** (bits - 1))
    maximum = (2 ** (bits - 1)) - 1
    if np.any(flat < minimum) or np.any(flat > maximum):
        raise ValueError(f"signed int{bits} values must be in [{minimum}, {maximum}]")
    mask = (1 << bits) - 1
    encoded = (flat & mask).astype(np.uint16)
    out = np.zeros(math.ceil(encoded.size * bits / 8), dtype=np.uint8)
    bit_offset = 0
    for value in encoded:
        byte_index = bit_offset // 8
        inner_offset = bit_offset % 8
        shifted = int(value) << inner_offset
        out[byte_index] |= shifted & 0xFF
        if inner_offset + bits > 8:
            out[byte_index + 1] |= (shifted >> 8) & 0xFF
        bit_offset += bits
    return out


def _unpack_signed_int_bits(packed: np.ndarray, *, count: int, bits: int) -> np.ndarray:
    if bits == 4:
        return _unpack_signed_int4(packed, count)
    if bits < 2 or bits > 7:
        raise ValueError("signed bit unpacking supports bit widths from 2 to 7")
    bytes_array = np.asarray(packed, dtype=np.uint8).reshape(-1)
    mask = (1 << bits) - 1
    values = np.zeros(count, dtype=np.int16)
    bit_offset = 0
    for index in range(count):
        byte_index = bit_offset // 8
        inner_offset = bit_offset % 8
        raw = int(bytes_array[byte_index]) >> inner_offset
        if inner_offset + bits > 8:
            raw |= int(bytes_array[byte_index + 1]) << (8 - inner_offset)
        values[index] = raw & mask
        bit_offset += bits
    sign_bit = 1 << (bits - 1)
    values[values & sign_bit != 0] -= 1 << bits
    return values.astype(np.int8)


def _encoded_raw_bytes_with_adapter_overhead(encoded: BoundaryEncodedTensor) -> int:
    return _encoded_raw_bytes_with_adapter_metadata(
        encoded.tensor.byte_size(),
        adapter_id=encoded.adapter_id,
        metadata={**encoded.tensor.metadata, **encoded.metadata},
    )


def _encoded_raw_bytes_with_adapter_metadata(raw_bytes: int, *, adapter_id: str, metadata: dict[str, Any]) -> int:
    if adapter_id in {
        "learned_per_boundary_int8",
        "learned_residual4_int8",
        "learned_residual4_sparse24_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
    }:
        scale_values = metadata.get("scale_values")
        if isinstance(scale_values, list):
            raw_bytes += len(scale_values) * np.dtype(np.float32).itemsize
    return raw_bytes


def _strategy_display_name(strategy: str) -> str:
    names = {
        "identity": "Full hidden-state transfer",
        "fp16": "FP16 boundary transfer",
        "int8_symmetric": "Symmetric int8 boundary transfer",
        "learned_per_boundary_int8": "Learned per-boundary int8 transfer",
        "learned_residual4_int8": "Learned int8 plus residual4 boundary transfer",
        "learned_residual4_sparse24_int8": "Learned int8 plus residual4 sparse24 boundary transfer",
        "learned_residual5_int8": "Learned int8 plus residual5 boundary transfer",
        "learned_residual6_int8": "Learned int8 plus residual6 boundary transfer",
        "learned_residual8_int8": "Learned int8 plus residual8 boundary transfer",
    }
    return names.get(strategy, strategy)


def _adapter_kind(strategy: str) -> BoundaryAdapterKind:
    if strategy == "identity":
        return "identity"
    if strategy == "fp16":
        return "precision_cast"
    if strategy in {
        "learned_per_boundary_int8",
        "learned_residual4_int8",
        "learned_residual4_sparse24_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
    }:
        return "learned"
    return "quantized"


def _boundary_adapter_metadata(
    *,
    adapter_id: BoundaryAdapterStrategy,
    adapter_kind: BoundaryAdapterKind,
    original_dtype: str,
    encoded_dtype: str,
    shape: list[int],
    original_raw_bytes: int,
    encoded_raw_bytes: int,
    reversible: bool,
    max_abs_error: float | None,
    mean_abs_error: float | None,
    rmse: float | None,
    cosine_similarity: float | None,
    encode_ms: float,
    decode_ms: float,
) -> dict[str, Any]:
    raw_byte_ratio = encoded_raw_bytes / original_raw_bytes if original_raw_bytes else 1.0
    return {
        "schema_version": BOUNDARY_FRAME_ADAPTER_SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "adapter_kind": adapter_kind,
        "display_name": _strategy_display_name(adapter_id),
        "original_dtype": original_dtype,
        "encoded_dtype": encoded_dtype,
        "shape": shape,
        "original_raw_bytes": original_raw_bytes,
        "encoded_raw_bytes": encoded_raw_bytes,
        "raw_byte_ratio": raw_byte_ratio,
        "raw_byte_savings_ratio": max(0.0, 1.0 - raw_byte_ratio),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "rmse": rmse,
        "cosine_similarity": cosine_similarity,
        "reversible": reversible,
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
        "architecture_neutral_boundary_adapter": True,
    }


def _copy_adapter_specific_metadata(
    metadata: dict[str, Any],
    encoded_metadata: dict[str, Any],
    strategy: BoundaryAdapterStrategy,
) -> None:
    if strategy == "int8_symmetric" and "scale" in encoded_metadata:
        metadata["scale"] = encoded_metadata["scale"]
        for key in (
            "position_scoped_mixed_frame",
            "mixed_frame_layout",
            "mixed_frame_axis",
            "mixed_frame_position",
            "mixed_frame_original_shape",
            "mixed_frame_rest_shape",
            "mixed_frame_rest_dtype",
            "mixed_frame_rest_byte_count",
            "mixed_frame_target_shape",
            "mixed_frame_target_dtype",
            "mixed_frame_target_byte_count",
            "mixed_frame_target_scale",
            "mixed_frame_target_adapter_id",
        ):
            if key in encoded_metadata:
                metadata[key] = encoded_metadata[key]
    if strategy in {
        "learned_per_boundary_int8",
        "learned_residual4_int8",
        "learned_residual4_sparse24_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
    }:
        metadata["scale_mode"] = encoded_metadata.get("scale_mode")
        metadata["scale_shape"] = encoded_metadata.get("scale_shape")
        metadata["scale_values"] = encoded_metadata.get("scale_values")
        metadata["scale_count"] = len(encoded_metadata.get("scale_values") or [])
    if strategy in {
        "learned_residual4_int8",
        "learned_residual4_sparse24_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
    }:
        metadata["residual_bits"] = encoded_metadata.get("residual_bits")
        metadata["residual_packing"] = encoded_metadata.get("residual_packing")
        metadata["original_element_count"] = encoded_metadata.get("original_element_count")
        metadata["base_byte_count"] = encoded_metadata.get("base_byte_count")
        metadata["residual_packed_byte_count"] = encoded_metadata.get("residual_packed_byte_count")
        metadata["encoded_shape"] = encoded_metadata.get("encoded_shape")
    if strategy == "learned_residual4_sparse24_int8":
        metadata["sparse_corrections_per_row"] = encoded_metadata.get("sparse_corrections_per_row")
        metadata["sparse_correction_count"] = encoded_metadata.get("sparse_correction_count")
        metadata["sparse_index_dtype"] = encoded_metadata.get("sparse_index_dtype")
        metadata["sparse_index_byte_count"] = encoded_metadata.get("sparse_index_byte_count")
        metadata["sparse_value_dtype"] = encoded_metadata.get("sparse_value_dtype")
        metadata["sparse_value_byte_count"] = encoded_metadata.get("sparse_value_byte_count")


def _boundary_payload_metadata_for_frame(
    *,
    source_metadata: dict[str, Any],
    boundary_metadata: dict[str, Any],
    strategy: BoundaryAdapterStrategy,
    backend_decision: BoundaryCodecBackendRuntimeDecision | None,
) -> dict[str, Any]:
    payload_metadata = dict(source_metadata)
    for stale_key in (
        "boundary_adapter_strategy",
        "restore_dtype",
        "bounded_loss",
        "reversible",
        "zero_point",
        "scale",
        "scale_mode",
        "scale_shape",
        "scale_values",
        "residual_bits",
        "residual_packing",
        "original_element_count",
        "base_byte_count",
        "residual_packed_byte_count",
        "sparse_corrections_per_row",
        "sparse_correction_count",
        "sparse_index_dtype",
        "sparse_index_byte_count",
        "sparse_value_dtype",
        "sparse_value_byte_count",
        "encoded_shape",
        "research_candidate",
        BOUNDARY_POSITION_SCOPED_TRANSPORT_METADATA_KEY,
        "position_scoped_mixed_frame",
        "mixed_frame_layout",
        "mixed_frame_axis",
        "mixed_frame_position",
        "mixed_frame_original_shape",
        "mixed_frame_rest_shape",
        "mixed_frame_rest_dtype",
        "mixed_frame_rest_byte_count",
        "mixed_frame_target_shape",
        "mixed_frame_target_dtype",
        "mixed_frame_target_byte_count",
        "mixed_frame_target_scale",
        "mixed_frame_target_adapter_id",
    ):
        payload_metadata.pop(stale_key, None)
    if strategy == "identity":
        payload_metadata.pop(BOUNDARY_ADAPTER_METADATA_KEY, None)
        payload_metadata.pop(BOUNDARY_ADAPTER_COMPACT_METADATA_KEY, None)
    else:
        payload_metadata.pop(BOUNDARY_ADAPTER_METADATA_KEY, None)
        payload_metadata[BOUNDARY_ADAPTER_COMPACT_METADATA_KEY] = _compact_boundary_adapter_metadata(boundary_metadata)
    if backend_decision is not None:
        payload_metadata[BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY] = backend_decision.model_dump(mode="json")
    return payload_metadata


def _boundary_adapter_metadata_from_payload(
    payload_metadata: dict[str, Any],
    *,
    tensor: TensorPayload,
) -> dict[str, Any] | None:
    return boundary_adapter_metadata_from_payload_metadata(
        payload_metadata,
        tensor_dtype=tensor.dtype,
        tensor_shape=tensor.shape,
        tensor_byte_size=tensor.byte_size(),
    )


def _compact_boundary_adapter_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "v": 0,
        "a": metadata.get("adapter_id"),
        "od": metadata.get("original_dtype"),
        "or": metadata.get("original_raw_bytes"),
        "er": metadata.get("encoded_raw_bytes"),
    }
    adapter_id = metadata.get("adapter_id")
    if adapter_id == "int8_symmetric":
        compact["s"] = metadata.get("scale")
        if metadata.get("position_scoped_mixed_frame"):
            compact["pm"] = True
            compact["pl"] = metadata.get("mixed_frame_layout")
            compact["pa"] = metadata.get("mixed_frame_axis")
            compact["pp"] = metadata.get("mixed_frame_position")
            compact["osh"] = metadata.get("mixed_frame_original_shape") or metadata.get("shape")
            compact["prs"] = metadata.get("mixed_frame_rest_shape")
            compact["prb"] = metadata.get("mixed_frame_rest_byte_count")
            compact["pts"] = metadata.get("mixed_frame_target_shape")
            compact["ptb"] = metadata.get("mixed_frame_target_byte_count")
            compact["ptscl"] = metadata.get("mixed_frame_target_scale")
    if adapter_id in {
        "learned_per_boundary_int8",
        "learned_residual4_int8",
        "learned_residual4_sparse24_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
    }:
        if adapter_id != "learned_residual4_sparse24_int8":
            compact["sm"] = metadata.get("scale_mode")
        compact["sh"] = metadata.get("scale_shape")
        compact["sv"] = metadata.get("scale_values")
        compact["sc"] = metadata.get("scale_count")
    if adapter_id in {
        "learned_residual4_int8",
        "learned_residual4_sparse24_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
    }:
        compact["osh"] = metadata.get("shape")
        compact["oe"] = metadata.get("original_element_count")
        if adapter_id == "learned_residual4_sparse24_int8":
            compact["sk"] = metadata.get("sparse_corrections_per_row")
        else:
            compact["rbit"] = metadata.get("residual_bits")
            compact["rp"] = metadata.get("residual_packing")
            compact["bc"] = metadata.get("base_byte_count")
            compact["rc"] = metadata.get("residual_packed_byte_count")
    return {key: value for key, value in compact.items() if value is not None}


def _expand_compact_boundary_adapter_metadata(compact: dict[str, Any]) -> dict[str, Any]:
    adapter_id = normalize_boundary_adapter_strategy(str(compact.get("a") or "identity"))
    original_raw_bytes = int(compact.get("or") or 0)
    encoded_raw_bytes = int(compact.get("er") or 0)
    adapter_kind = _adapter_kind(adapter_id)
    expanded: dict[str, Any] = {
        "schema_version": BOUNDARY_FRAME_ADAPTER_SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "adapter_kind": adapter_kind,
        "display_name": _strategy_display_name(adapter_id),
        "original_dtype": compact.get("od") or "float32",
        "original_raw_bytes": original_raw_bytes,
        "encoded_raw_bytes": encoded_raw_bytes,
        "reversible": adapter_id == "identity",
        "architecture_neutral_boundary_adapter": True,
    }
    if "s" in compact:
        expanded["scale"] = compact["s"]
    if adapter_id == "int8_symmetric" and bool(compact.get("pm")):
        expanded["position_scoped_mixed_frame"] = True
        expanded["mixed_frame_layout"] = compact.get("pl") or "fp16_except_int8_position_v0"
        expanded["mixed_frame_axis"] = compact.get("pa")
        expanded["mixed_frame_position"] = compact.get("pp")
        expanded["mixed_frame_original_shape"] = compact.get("osh")
        expanded["mixed_frame_rest_shape"] = compact.get("prs")
        expanded["mixed_frame_rest_dtype"] = "float16"
        expanded["mixed_frame_rest_byte_count"] = compact.get("prb")
        expanded["mixed_frame_target_shape"] = compact.get("pts")
        expanded["mixed_frame_target_dtype"] = "int8"
        expanded["mixed_frame_target_byte_count"] = compact.get("ptb")
        expanded["mixed_frame_target_scale"] = compact.get("ptscl") or compact.get("s")
        if compact.get("osh") is not None:
            expanded["shape"] = compact.get("osh")
    if adapter_id in {
        "learned_per_boundary_int8",
        "learned_residual4_int8",
        "learned_residual4_sparse24_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
    }:
        expanded["scale_mode"] = compact.get("sm") or (
            "per_boundary_residual4_sparse24"
            if adapter_id == "learned_residual4_sparse24_int8"
            else (
                f"per_boundary_residual{_residual_bits_for_strategy(adapter_id)}"
                if adapter_id
                in {
                    "learned_residual4_int8",
                    "learned_residual5_int8",
                    "learned_residual6_int8",
                    "learned_residual8_int8",
                }
                else "per_boundary"
            )
        )
        expanded["scale_shape"] = compact.get("sh")
        expanded["scale_values"] = compact.get("sv")
        expanded["scale_count"] = compact.get("sc")
    if adapter_id in {
        "learned_residual4_int8",
        "learned_residual4_sparse24_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
    }:
        if compact.get("osh") is not None:
            expanded["shape"] = compact.get("osh")
        residual_bits = _residual_bits_for_strategy(adapter_id)
        expanded["residual_bits"] = compact.get("rbit") or residual_bits
        expanded["residual_packing"] = compact.get("rp") or (
            "signed_int8_dense" if residual_bits == 8 else f"signed_int{residual_bits}_bitstream_lsb"
        )
        expanded["original_element_count"] = compact.get("oe")
        expanded["base_byte_count"] = compact.get("bc")
        expanded["residual_packed_byte_count"] = compact.get("rc")
        if adapter_id == "learned_residual4_sparse24_int8":
            expanded["sparse_corrections_per_row"] = compact.get("sk")
            expanded["sparse_index_dtype"] = compact.get("sid") or "uint16_le"
            expanded["sparse_index_byte_count"] = compact.get("sib")
            expanded["sparse_value_dtype"] = compact.get("svd") or "int8"
            expanded["sparse_value_byte_count"] = compact.get("svb")
    return expanded


def _residual_bits_for_strategy(strategy: str) -> int:
    if strategy == "learned_residual4_int8":
        return 4
    if strategy == "learned_residual4_sparse24_int8":
        return 4
    if strategy == "learned_residual5_int8":
        return 5
    if strategy == "learned_residual6_int8":
        return 6
    if strategy == "learned_residual8_int8":
        return 8
    raise ValueError(f"{strategy} is not a residual learned adapter strategy")


def _sparse_corrections_for_strategy(strategy: str) -> int:
    if strategy == "learned_residual4_sparse24_int8":
        return 24
    raise ValueError(f"{strategy} is not a sparse residual learned adapter strategy")


def compact_boundary_adapter_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "ma": metadata.get("mean_abs_error"),
        "xa": metadata.get("max_abs_error"),
        "rm": metadata.get("rmse"),
        "cs": metadata.get("cosine_similarity"),
    }
    return {key: value for key, value in compact.items() if value is not None}


def expand_compact_boundary_adapter_metrics(compact: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(compact, dict):
        return {}
    expanded: dict[str, Any] = {}
    if "ma" in compact:
        expanded["mean_abs_error"] = float(compact["ma"])
    if "xa" in compact:
        expanded["max_abs_error"] = float(compact["xa"])
    if "rm" in compact:
        expanded["rmse"] = float(compact["rm"])
    if "cs" in compact:
        expanded["cosine_similarity"] = float(compact["cs"])
    return expanded
