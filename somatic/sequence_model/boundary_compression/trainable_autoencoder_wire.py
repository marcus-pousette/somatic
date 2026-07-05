from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

import numpy as np

from somatic.sequence_model.boundary_adapters import (
    BoundaryAdapterStrategy,
    _normalized_position_scope,
    _position_scoped_mixed_transport_request,
    decode_payload_from_boundary,
    encode_payload_for_boundary,
)
from somatic.sequence_model.tensors import TensorPayload


TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID = "trainable_autoencoder_source_latent"
TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND = "source_payload_plus_latent_coefficients_v0"


def encode_trainable_autoencoder_source_latent_payload(
    tensor: TensorPayload,
    *,
    basis_artifact: dict[str, Any],
    source_strategy: BoundaryAdapterStrategy,
    context: dict[str, Any] | None = None,
    allow_source_only_fallback: bool = False,
) -> tuple[TensorPayload, dict[str, Any]]:
    original = tensor.to_numpy().astype(np.float32, copy=False)
    _require_basis_source_strategy_pairing(basis_artifact, source_strategy=source_strategy)
    source_payload, source_adapter = encode_payload_for_boundary(
        tensor,
        strategy=source_strategy,
    )
    scope: dict[str, Any] | None = None
    try:
        basis = _select_basis_for_tensor(
            basis_artifact,
            shape=list(original.shape),
            context=context or {},
        )
    except ValueError as exc:
        scoped_selection = _select_scoped_basis_for_tensor(
            basis_artifact,
            shape=list(original.shape),
            tensor_metadata=dict(tensor.metadata),
            context=context or {},
        )
        if scoped_selection is None:
            if allow_source_only_fallback and _basis_miss_is_source_only_fallback_eligible(exc):
                return _encode_source_only_fallback_payload(
                    tensor=tensor,
                    source_payload=source_payload,
                    source_adapter=source_adapter,
                    basis_artifact=basis_artifact,
                    source_strategy=source_strategy,
                    reason=str(exc),
                )
            raise
        basis, scope = scoped_selection
    mean = _array_from_serialized_payload(basis["mean"]).astype(np.float32, copy=False)
    components = _array_from_serialized_payload(basis["components"]).astype(
        np.float32,
        copy=False,
    )
    flat = original.reshape(-1)
    if scope is not None:
        axis_length = int(original.shape[scope["scope_axis"]])
        expected_slice_size = (
            (int(flat.size) // axis_length) * int(scope["scope_count"])
            if axis_length > 0
            else -1
        )
        if int(mean.size) != expected_slice_size:
            if allow_source_only_fallback:
                return _encode_source_only_fallback_payload(
                    tensor=tensor,
                    source_payload=source_payload,
                    source_adapter=source_adapter,
                    basis_artifact=basis_artifact,
                    source_strategy=source_strategy,
                    reason="trainable autoencoder scoped basis width does not match target slice",
                )
            raise ValueError(
                "trainable autoencoder scoped basis width does not match target slice"
            )
    if scope is None and int(mean.size) != int(flat.size):
        scope = _resolve_scoped_transport(
            basis_artifact,
            mean_size=int(mean.size),
            original_shape=list(original.shape),
            flat_size=int(flat.size),
            tensor_metadata=dict(tensor.metadata),
            context=context or {},
        )
        if scope is None:
            if allow_source_only_fallback:
                return _encode_source_only_fallback_payload(
                    tensor=tensor,
                    source_payload=source_payload,
                    source_adapter=source_adapter,
                    basis_artifact=basis_artifact,
                    source_strategy=source_strategy,
                    reason="trainable autoencoder basis width does not match tensor width",
                )
            raise ValueError("trainable autoencoder basis width does not match tensor width")
    if components.ndim != 2 or int(components.shape[-1]) != int(mean.size):
        if allow_source_only_fallback:
            return _encode_source_only_fallback_payload(
                tensor=tensor,
                source_payload=source_payload,
                source_adapter=source_adapter,
                basis_artifact=basis_artifact,
                source_strategy=source_strategy,
                reason="trainable autoencoder components do not match tensor width",
            )
        raise ValueError("trainable autoencoder components do not match tensor width")
    source_restored, _ = decode_payload_from_boundary(source_payload)
    source_array = source_restored.to_numpy().astype(np.float32, copy=False)
    if scope is not None:
        residual = (
            _scoped_slice(original, scope)
            - _scoped_slice(source_array.reshape(original.shape), scope)
        ).reshape(-1)
    else:
        residual = (original - source_array).reshape(-1)
    coefficients = (
        ((residual - mean) @ components.T).astype(np.float16, copy=False)
        if components.size
        else np.asarray([], dtype=np.float16)
    )
    source_raw = source_payload.raw_bytes()
    latent_raw = np.ascontiguousarray(coefficients).tobytes(order="C")
    raw = source_raw + latent_raw
    metadata = {
        "adapter_id": TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
        "encoded_payload_kind": TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND,
        "basis_hash": basis_artifact.get("basis_hash"),
        "basis_kind": _basis_kind(basis_artifact),
        "basis_selection_key": basis.get("basis_selection_key"),
        "basis_selection_kind": basis.get("basis_selection_kind"),
        "source_strategy": source_strategy,
        "source_name": source_payload.name,
        "source_dtype": source_payload.dtype,
        "source_shape": list(source_payload.shape),
        "source_metadata": source_payload.metadata,
        "source_raw_byte_count": len(source_raw),
        "source_adapter": source_adapter,
        "latent_dtype": "float16",
        "latent_shape": [int(coefficients.size)],
        "latent_byte_count": len(latent_raw),
        "original_dtype": tensor.dtype,
        "original_shape": list(tensor.shape),
        "residual_gain": float(basis_artifact.get("residual_gain") or 1.0),
        "source_only_fallback": False,
        "source_only_fallback_reason": None,
        "basis_correction_applied": True,
        "position_scoped_mixed_frame": bool(
            (source_adapter or {}).get("position_scoped_mixed_frame")
        ),
        "mixed_frame_layout": (source_adapter or {}).get("mixed_frame_layout"),
        "scoped_correction": scope is not None,
        "scope_kind": scope["scope_kind"] if scope is not None else None,
        "scope_axis": scope["scope_axis"] if scope is not None else None,
        "scope_start": scope["scope_start"] if scope is not None else None,
        "scope_count": scope["scope_count"] if scope is not None else None,
        "position_scoped_mixed_frame": scope is not None,
        "mixed_frame_layout": (
            "source_frame_plus_latent_target_slice_v0" if scope is not None else None
        ),
        "mixed_frame_axis": scope["scope_axis"] if scope is not None else None,
        "mixed_frame_position": scope["scope_start"] if scope is not None else None,
        "runtime_transport_implemented_claimed": False,
        "production_runtime_claimed": False,
    }
    encoded = TensorPayload.from_raw_bytes(
        raw,
        dtype="int8",
        shape=[len(raw)],
        name=tensor.name,
        metadata=metadata,
    )
    return encoded, _wire_metadata(
        metadata,
        original_raw_bytes=tensor.byte_size(),
        encoded_raw_bytes=len(raw),
    )


def decode_trainable_autoencoder_source_latent_payload(
    tensor: TensorPayload,
    *,
    basis_artifact: dict[str, Any],
) -> tuple[TensorPayload, dict[str, Any]]:
    metadata = dict(tensor.metadata)
    if metadata.get("encoded_payload_kind") != TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND:
        raise ValueError("tensor is not a trainable autoencoder source-latent payload")
    expected_hash = metadata.get("basis_hash")
    actual_hash = basis_artifact.get("basis_hash")
    if expected_hash is not None and actual_hash is not None and expected_hash != actual_hash:
        raise ValueError("trainable autoencoder basis hash mismatch")
    wire_source_strategy = metadata.get("source_strategy")
    if wire_source_strategy is not None:
        _require_basis_source_strategy_pairing(
            basis_artifact,
            source_strategy=str(wire_source_strategy),
        )
    raw = tensor.raw_bytes()
    source_count = _metadata_int(metadata, "source_raw_byte_count")
    latent_count = _metadata_int(metadata, "latent_byte_count")
    if len(raw) < source_count + latent_count:
        raise ValueError("trainable autoencoder payload is shorter than declared")
    source_payload = TensorPayload.from_raw_bytes(
        raw[:source_count],
        dtype=metadata["source_dtype"],
        shape=[int(dim) for dim in metadata["source_shape"]],
        name=str(metadata.get("source_name") or tensor.name),
        metadata=dict(metadata.get("source_metadata") or {}),
    )
    source_restored, source_decode_metadata = decode_payload_from_boundary(source_payload)
    source_array = source_restored.to_numpy().astype(np.float32, copy=False)
    original_dtype = str(metadata.get("original_dtype") or source_restored.dtype)
    if bool(metadata.get("source_only_fallback")):
        decoded = TensorPayload.from_numpy(
            source_array.astype(np.dtype(original_dtype), copy=False),
            name=tensor.name,
            metadata={
                **source_restored.metadata,
                "trainable_autoencoder_source_latent": {
                    "adapter_id": TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
                    "basis_hash": actual_hash,
                    "source_decode_metadata": source_decode_metadata,
                    "decoded_on_receive": True,
                    "source_only_fallback": True,
                    "source_only_fallback_reason": metadata.get(
                        "source_only_fallback_reason"
                    ),
                    "basis_correction_applied": False,
                    "runtime_transport_implemented_claimed": False,
                    "production_runtime_claimed": False,
                },
            },
        )
        return decoded, _wire_metadata(
            metadata,
            original_raw_bytes=decoded.byte_size(),
            encoded_raw_bytes=len(raw),
        )
    basis = _select_basis_by_metadata(
        basis_artifact,
        metadata=metadata,
        fallback_shape=list(source_array.shape),
    )
    mean = _array_from_serialized_payload(basis["mean"]).astype(np.float32, copy=False)
    components = _array_from_serialized_payload(basis["components"]).astype(
        np.float32,
        copy=False,
    )
    coefficients = np.frombuffer(
        raw[source_count : source_count + latent_count],
        dtype=np.float16,
    ).astype(np.float32)
    if components.size and int(coefficients.size) != int(components.shape[0]):
        raise ValueError("trainable autoencoder latent coefficient count mismatch")
    reconstructed_residual = (
        mean + (coefficients @ components)
        if components.size
        else mean
    )
    residual_gain = np.float32(float(metadata.get("residual_gain") or 1.0))
    if bool(metadata.get("scoped_correction")):
        original_shape = [int(dim) for dim in metadata.get("original_shape") or []]
        if not original_shape or int(np.prod(original_shape)) != int(source_array.size):
            raise ValueError(
                "trainable autoencoder scoped correction does not match frame size"
            )
        scope = {
            "scope_axis": _metadata_int(metadata, "scope_axis"),
            "scope_start": _metadata_int(metadata, "scope_start"),
            "scope_count": _metadata_int(metadata, "scope_count"),
        }
        restored = source_array.reshape(original_shape).copy()
        target_slice = _scoped_slice(restored, scope)
        if int(reconstructed_residual.size) != int(target_slice.size):
            raise ValueError(
                "trainable autoencoder scoped correction does not match basis width"
            )
        target_slice += residual_gain * reconstructed_residual.reshape(
            target_slice.shape
        )
        restored = restored.reshape(source_array.shape)
    else:
        restored = source_array + (
            residual_gain * reconstructed_residual.reshape(source_array.shape)
        )
    decoded = TensorPayload.from_numpy(
        restored.astype(np.dtype(original_dtype), copy=False),
        name=tensor.name,
        metadata={
            **source_restored.metadata,
            "trainable_autoencoder_source_latent": {
                "adapter_id": TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
                "basis_hash": actual_hash,
                "source_decode_metadata": source_decode_metadata,
                "decoded_on_receive": True,
                "source_only_fallback": False,
                "basis_correction_applied": True,
                "runtime_transport_implemented_claimed": False,
                "production_runtime_claimed": False,
            },
        },
    )
    return decoded, _wire_metadata(
        metadata,
        original_raw_bytes=decoded.byte_size(),
        encoded_raw_bytes=len(raw),
    )


def load_trainable_autoencoder_basis_artifact(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "ready":
        raise ValueError("trainable autoencoder basis artifact is not ready")
    if payload.get("basis_hash") and payload.get("basis_hash") != _basis_payload_hash(payload):
        raise ValueError("trainable autoencoder basis artifact hash mismatch")
    return payload


def _wire_metadata(
    metadata: dict[str, Any],
    *,
    original_raw_bytes: int,
    encoded_raw_bytes: int,
) -> dict[str, Any]:
    return {
        "adapter_id": TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
        "adapter_kind": "learned",
        "encoded_payload_kind": TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND,
        "basis_hash": metadata.get("basis_hash"),
        "source_strategy": metadata.get("source_strategy"),
        "source_raw_byte_count": metadata.get("source_raw_byte_count"),
        "latent_byte_count": metadata.get("latent_byte_count"),
        "source_only_fallback": bool(metadata.get("source_only_fallback")),
        "source_only_fallback_reason": metadata.get("source_only_fallback_reason"),
        "basis_correction_applied": bool(metadata.get("basis_correction_applied")),
        "position_scoped_mixed_frame": bool(
            metadata.get("position_scoped_mixed_frame")
        ),
        "mixed_frame_layout": metadata.get("mixed_frame_layout"),
        "scoped_correction": bool(metadata.get("scoped_correction")),
        "scope_kind": metadata.get("scope_kind"),
        "scope_axis": metadata.get("scope_axis"),
        "scope_start": metadata.get("scope_start"),
        "scope_count": metadata.get("scope_count"),
        "position_scoped_mixed_frame": bool(
            metadata.get("position_scoped_mixed_frame")
        ),
        "mixed_frame_layout": metadata.get("mixed_frame_layout"),
        "mixed_frame_axis": metadata.get("mixed_frame_axis"),
        "mixed_frame_position": metadata.get("mixed_frame_position"),
        "original_dtype": metadata.get("original_dtype"),
        "encoded_dtype": "int8",
        "shape": metadata.get("original_shape"),
        "original_raw_bytes": int(original_raw_bytes),
        "encoded_raw_bytes": int(encoded_raw_bytes),
        "raw_byte_ratio": (
            encoded_raw_bytes / float(original_raw_bytes)
            if original_raw_bytes > 0
            else 1.0
        ),
        "raw_byte_savings_ratio": (
            max(0.0, 1.0 - (encoded_raw_bytes / float(original_raw_bytes)))
            if original_raw_bytes > 0
            else 0.0
        ),
        "reversible": False,
        "architecture_neutral_boundary_adapter": True,
        "runtime_transport_implemented_claimed": False,
        "production_runtime_claimed": False,
    }


def _encode_source_only_fallback_payload(
    *,
    tensor: TensorPayload,
    source_payload: TensorPayload,
    source_adapter: dict[str, Any],
    basis_artifact: dict[str, Any],
    source_strategy: BoundaryAdapterStrategy,
    reason: str,
) -> tuple[TensorPayload, dict[str, Any]]:
    source_raw = source_payload.raw_bytes()
    metadata = {
        "adapter_id": TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
        "encoded_payload_kind": TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND,
        "basis_hash": basis_artifact.get("basis_hash"),
        "basis_kind": _basis_kind(basis_artifact),
        "basis_selection_key": None,
        "basis_selection_kind": "source_only_fallback",
        "source_strategy": source_strategy,
        "source_name": source_payload.name,
        "source_dtype": source_payload.dtype,
        "source_shape": list(source_payload.shape),
        "source_metadata": source_payload.metadata,
        "source_raw_byte_count": len(source_raw),
        "source_adapter": source_adapter,
        "latent_dtype": "float16",
        "latent_shape": [0],
        "latent_byte_count": 0,
        "original_dtype": tensor.dtype,
        "original_shape": list(tensor.shape),
        "residual_gain": float(basis_artifact.get("residual_gain") or 1.0),
        "source_only_fallback": True,
        "source_only_fallback_reason": reason,
        "basis_correction_applied": False,
        "scoped_correction": False,
        "scope_kind": None,
        "scope_axis": None,
        "scope_start": None,
        "scope_count": None,
        "position_scoped_mixed_frame": False,
        "mixed_frame_layout": None,
        "mixed_frame_axis": None,
        "mixed_frame_position": None,
        "runtime_transport_implemented_claimed": False,
        "production_runtime_claimed": False,
    }
    encoded = TensorPayload.from_raw_bytes(
        source_raw,
        dtype="int8",
        shape=[len(source_raw)],
        name=tensor.name,
        metadata=metadata,
    )
    return encoded, _wire_metadata(
        metadata,
        original_raw_bytes=tensor.byte_size(),
        encoded_raw_bytes=len(source_raw),
    )


def _require_basis_source_strategy_pairing(
    basis_artifact: dict[str, Any],
    *,
    source_strategy: str,
) -> None:
    declared = basis_artifact.get("source_payload_transport_strategy")
    if declared is None:
        return
    if str(declared) != str(source_strategy):
        raise ValueError(
            "trainable autoencoder basis was authored against source strategy "
            f"`{declared}` but the runtime requested `{source_strategy}` — "
            "mixed basis/source pairings are fail-closed"
        )


def _scoped_slice(array: np.ndarray, scope: dict[str, Any]) -> np.ndarray:
    moved = np.moveaxis(array, int(scope["scope_axis"]), 0)
    start = int(scope["scope_start"])
    return moved[start : start + int(scope["scope_count"])]


def _scoped_location_authorized(
    basis_artifact: dict[str, Any],
    *,
    context: dict[str, Any],
    shape: list[int],
) -> bool:
    locations = basis_artifact.get("target_boundary_locations")
    if not isinstance(locations, list) or not locations:
        return False
    if not isinstance(context, dict) or not context:
        return False
    return any(
        _context_matches_target_boundary_location(
            location,
            context=context,
            shape=shape,
        )
        for location in locations
        if isinstance(location, dict)
    )


def _parse_shape_key(key: str) -> list[int]:
    try:
        return [int(dim) for dim in str(key).split("x")]
    except ValueError:
        return []


def _select_scoped_basis_for_tensor(
    basis_artifact: dict[str, Any],
    *,
    shape: list[int],
    tensor_metadata: dict[str, Any],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Scoped selection for multi-shape bases narrower than the frame.

    A multi-shape basis keyed by a slice shape (e.g. `1x1x1024`) can serve a
    wider frame (e.g. `[1, 4, 1024]`) by correcting only the target slice
    along the sequence axis. Same fail-closed authorization as
    `_resolve_scoped_transport`.
    """
    if str(basis_artifact.get("correction_scope") or "") != "target_boundary":
        return None
    basis = basis_artifact.get("basis")
    if not isinstance(basis, dict) or basis.get("basis_kind") != "multi_shape":
        return None
    ndim = len(shape)
    if ndim < 2:
        return None
    request = _position_scoped_mixed_transport_request(tensor_metadata)
    if request is None and not _scoped_location_authorized(
        basis_artifact,
        context=context,
        shape=shape,
    ):
        return None
    axis = ndim - 2
    position: int | None = None
    if request is not None:
        axis, position = _normalized_position_scope(shape=shape, request=request)
    candidates: list[tuple[str, str, list[int], dict[str, Any]]] = []
    by_context = basis.get("bases_by_context_shape")
    if isinstance(by_context, dict):
        for key, entry in by_context.items():
            parts = str(key).split("|")
            if len(parts) != 3:
                continue
            if parts[0] != str(context.get("path") or "") or parts[1] != str(
                context.get("role") or ""
            ):
                continue
            candidates.append((str(key), "context_shape", _parse_shape_key(parts[2]), entry))
    by_shape = basis.get("bases_by_shape")
    if isinstance(by_shape, dict):
        for key, entry in by_shape.items():
            candidates.append((str(key), "shape", _parse_shape_key(str(key)), entry))
    for key, kind, candidate_shape, entry in candidates:
        if not isinstance(entry, dict) or len(candidate_shape) != ndim:
            continue
        if any(
            int(candidate_shape[index]) != int(shape[index])
            for index in range(ndim)
            if index != axis
        ):
            continue
        count = int(candidate_shape[axis])
        frame_length = int(shape[axis])
        if count < 1 or count > frame_length:
            continue
        if position is not None:
            if count != 1:
                continue
            start = position
        else:
            start = frame_length - count
        return (
            {**entry, "basis_selection_kind": kind, "basis_selection_key": key},
            {
                "scope_kind": (
                    "position_scoped" if position is not None else "trailing_positions"
                ),
                "scope_axis": axis,
                "scope_start": start,
                "scope_count": count,
            },
        )
    return None


def _resolve_scoped_transport(
    basis_artifact: dict[str, Any],
    *,
    mean_size: int,
    original_shape: list[int],
    flat_size: int,
    tensor_metadata: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any] | None:
    """Scoped/mixed-frame eligibility for a basis narrower than the frame.

    The latent correction is applied to a target slice of the frame (the
    boundary positions the basis was trained against) while the rest of the
    frame ships under the plain source strategy. Scoped mode is fail-closed
    and artifact-driven: it engages only when the basis artifact declares
    `correction_scope == "target_boundary"`, and either the sanctioned
    `boundary_position_scoped_transport` request rides the tensor metadata
    (single target position along the declared axis) or the encode context
    (path, role) matches one of the artifact's `target_boundary_locations`
    (trailing positions along the sequence axis, sized by the basis width).
    """
    if str(basis_artifact.get("correction_scope") or "") != "target_boundary":
        return None
    if not original_shape:
        return None
    request = _position_scoped_mixed_transport_request(tensor_metadata)
    if request is not None:
        axis, position = _normalized_position_scope(
            shape=original_shape,
            request=request,
        )
        axis_length = int(original_shape[axis])
        if axis_length > 0 and flat_size % axis_length == 0:
            slice_width = flat_size // axis_length
            if slice_width == mean_size:
                return {
                    "scope_kind": "position_scoped",
                    "scope_axis": axis,
                    "scope_start": position,
                    "scope_count": 1,
                }
    if not _scoped_location_authorized(
        basis_artifact,
        context=context,
        shape=original_shape,
    ):
        return None
    if len(original_shape) < 2:
        return None
    axis = len(original_shape) - 2
    axis_length = int(original_shape[axis])
    if axis_length <= 0 or flat_size % axis_length != 0:
        return None
    slice_width = flat_size // axis_length
    if slice_width <= 0 or mean_size % slice_width != 0:
        return None
    count = mean_size // slice_width
    if count < 1 or count > axis_length:
        return None
    return {
        "scope_kind": "trailing_positions",
        "scope_axis": axis,
        "scope_start": axis_length - count,
        "scope_count": count,
    }


def _context_matches_target_boundary_location(
    location: dict[str, Any],
    *,
    context: dict[str, Any],
    shape: list[int],
) -> bool:
    if str(location.get("path") or "") != str(context.get("path") or ""):
        return False
    if str(location.get("role") or "") != str(context.get("role") or ""):
        return False
    if bool(location.get("match_shape")):
        location_shape = [int(dim) for dim in (location.get("shape") or [])]
        if location_shape != [int(dim) for dim in shape]:
            return False
    return True


def _basis_miss_is_source_only_fallback_eligible(exc: ValueError) -> bool:
    message = str(exc)
    return message.startswith("no trainable autoencoder basis matches")


def _select_basis_for_tensor(
    basis_artifact: dict[str, Any],
    *,
    shape: list[int],
    context: dict[str, Any],
) -> dict[str, Any]:
    basis = basis_artifact.get("basis")
    if not isinstance(basis, dict):
        raise ValueError("trainable autoencoder basis artifact is missing basis")
    if basis.get("basis_kind") == "single_shape":
        return {
            **basis,
            "basis_selection_kind": "single_shape",
            "basis_selection_key": _shape_key(shape),
        }
    if basis.get("basis_kind") != "multi_shape":
        raise ValueError("unsupported trainable autoencoder basis kind")
    context_key = _context_shape_key(
        path=context.get("path"),
        role=context.get("role"),
        shape=shape,
    )
    by_context = basis.get("bases_by_context_shape")
    if isinstance(by_context, dict) and isinstance(by_context.get(context_key), dict):
        return {
            **by_context[context_key],
            "basis_selection_kind": "context_shape",
            "basis_selection_key": context_key,
        }
    shape_key = _shape_key(shape)
    by_shape = basis.get("bases_by_shape")
    if isinstance(by_shape, dict) and isinstance(by_shape.get(shape_key), dict):
        return {
            **by_shape[shape_key],
            "basis_selection_kind": "shape",
            "basis_selection_key": shape_key,
        }
    raise ValueError("no trainable autoencoder basis matches tensor shape/context")


def _select_basis_by_metadata(
    basis_artifact: dict[str, Any],
    *,
    metadata: dict[str, Any],
    fallback_shape: list[int],
) -> dict[str, Any]:
    basis = basis_artifact.get("basis")
    if not isinstance(basis, dict):
        raise ValueError("trainable autoencoder basis artifact is missing basis")
    key = metadata.get("basis_selection_key")
    kind = metadata.get("basis_selection_kind")
    if basis.get("basis_kind") == "single_shape":
        return basis
    if kind == "context_shape":
        selected = (basis.get("bases_by_context_shape") or {}).get(key)
        if isinstance(selected, dict):
            return selected
    if kind == "shape":
        selected = (basis.get("bases_by_shape") or {}).get(key)
        if isinstance(selected, dict):
            return selected
    return _select_basis_for_tensor(
        basis_artifact,
        shape=fallback_shape,
        context={},
    )


def _array_from_serialized_payload(payload: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(str(payload.get("data_b64") or "").encode("ascii"), validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    if payload.get("sha256") and payload.get("sha256") != digest:
        raise ValueError("serialized trainable autoencoder array hash mismatch")
    dtype = np.dtype(str(payload.get("dtype") or "float16"))
    shape = [int(dim) for dim in payload.get("shape") or []]
    return np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy()


def _metadata_int(metadata: dict[str, Any], key: str) -> int:
    try:
        value = int(metadata.get(key))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trainable autoencoder metadata `{key}` must be an integer") from exc
    if value < 0:
        raise ValueError(f"trainable autoencoder metadata `{key}` must be non-negative")
    return value


def _basis_kind(basis_artifact: dict[str, Any]) -> str | None:
    basis = basis_artifact.get("basis")
    return str(basis.get("basis_kind")) if isinstance(basis, dict) else None


def _shape_key(shape: list[int]) -> str:
    return "x".join(str(int(dim)) for dim in shape)


def _context_shape_key(*, path: Any, role: Any, shape: list[int]) -> str:
    return f"{path}|{role}|{_shape_key(shape)}"


def _basis_payload_hash(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("basis_hash", None)
    encoded = json.dumps(
        clone,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
