from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID = "sub_int8_sparse_codebook"
SPARSE_CODEBOOK_WIRE_CODEC_SCHEMA_VERSION = "boundary-sparse-codebook-wire-codec-v0"
SPARSE_CODEBOOK_WIRE_PAYLOAD_KIND = (
    "base_packed_bits_scale_float16_sparse_indices_codebook_ids"
)


@dataclass(frozen=True)
class SparseCodebookWirePayload:
    payload: bytes
    metadata: dict[str, Any]


def encode_sparse_codebook_components(
    array: np.ndarray,
    *,
    base_bits: int,
    sparse_corrections_per_row: int,
    codebook_values: Sequence[float] | np.ndarray,
) -> SparseCodebookWirePayload:
    original = np.ascontiguousarray(array, dtype=np.float32)
    rows = _flatten_hidden_rows(original)
    row_count = int(rows.shape[0])
    hidden_size = int(rows.shape[1])
    _validate_config(
        hidden_size=hidden_size,
        base_bits=base_bits,
        sparse_corrections_per_row=sparse_corrections_per_row,
        codebook_values=codebook_values,
    )
    codebook = _normalized_codebook_values(codebook_values)
    sparse_per_row = min(int(sparse_corrections_per_row), hidden_size)
    levels = int((2 ** (int(base_bits) - 1)) - 1)
    base_packed_bytes_per_row = int(math.ceil(hidden_size * int(base_bits) / 8))
    row_payload_bytes = base_packed_bytes_per_row + 2 + (sparse_per_row * 3)
    payload = bytearray()
    max_abs_error = 0.0
    mean_abs_error_total = 0.0

    for row in rows:
        scale = max(float(np.max(np.abs(row))) / float(levels), 1.0e-12)
        quantized = np.clip(np.rint(row / scale), -levels, levels).astype(np.int16)
        restored = quantized.astype(np.float32) * np.float32(scale)
        residual = row - restored
        indices = _top_residual_indices(residual, sparse_per_row)
        codebook_ids = _nearest_codebook_ids(residual[indices], codebook)
        if sparse_per_row:
            restored[indices] += codebook[codebook_ids]
        abs_error = np.abs(row - restored)
        max_abs_error = max(max_abs_error, float(np.max(abs_error)))
        mean_abs_error_total += float(np.mean(abs_error))

        unsigned = (quantized + levels).astype(np.uint8)
        packed = _pack_unsigned_values(unsigned, bits=int(base_bits))
        payload.extend(packed)
        payload.extend(np.asarray([scale], dtype="<f2").tobytes())
        payload.extend(np.asarray(indices, dtype="<u2").tobytes())
        payload.extend(codebook_ids.astype(np.uint8, copy=False).tobytes())

    return SparseCodebookWirePayload(
        payload=bytes(payload),
        metadata={
            "schema_version": SPARSE_CODEBOOK_WIRE_CODEC_SCHEMA_VERSION,
            "adapter_id": SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID,
            "encoded_payload_kind": SPARSE_CODEBOOK_WIRE_PAYLOAD_KIND,
            "original_shape": [int(dim) for dim in original.shape],
            "row_count": row_count,
            "hidden_size": hidden_size,
            "base_bits": int(base_bits),
            "base_levels": levels,
            "base_scale_dtype": "float16_le",
            "base_packed_bytes_per_row": base_packed_bytes_per_row,
            "sparse_corrections_per_row": sparse_per_row,
            "sparse_index_dtype": "uint16_le",
            "sparse_codebook_id_dtype": "uint8",
            "codebook_size": int(codebook.size),
            "codebook_hash": sparse_codebook_hash(codebook),
            "row_payload_bytes": row_payload_bytes,
            "payload_byte_count": len(payload),
            "mean_abs_error": mean_abs_error_total / float(max(row_count, 1)),
            "max_abs_error": max_abs_error,
            "architecture_neutral_boundary_codec": True,
            "exact_replay_passed_claimed": False,
            "planner_selectable_claimed": False,
            "production_runtime_claimed": False,
        },
    )


def decode_sparse_codebook_components(
    header_metadata: dict[str, Any],
    payload: bytes,
    *,
    codebook_values: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    if codebook_values is None:
        raise ValueError(
            f"{SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID} payload received but no "
            "artifact-resident codebook is configured"
        )
    codebook = _normalized_codebook_values(codebook_values)
    expected_hash = str(header_metadata.get("codebook_hash") or "")
    observed_hash = sparse_codebook_hash(codebook)
    if expected_hash and expected_hash != observed_hash:
        raise ValueError(
            f"{SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID} codebook hash mismatch: "
            f"expected {expected_hash}, observed {observed_hash}"
        )
    if str(header_metadata.get("encoded_payload_kind") or "") != (
        SPARSE_CODEBOOK_WIRE_PAYLOAD_KIND
    ):
        raise ValueError("sparse-codebook payload kind mismatch")

    original_shape = [int(dim) for dim in header_metadata["original_shape"]]
    row_count = int(header_metadata["row_count"])
    hidden_size = int(header_metadata["hidden_size"])
    if int(np.prod(original_shape)) != row_count * hidden_size:
        raise ValueError("sparse-codebook payload shape metadata is inconsistent")

    base_bits = int(header_metadata["base_bits"])
    levels = int(header_metadata.get("base_levels") or ((2 ** (base_bits - 1)) - 1))
    sparse_per_row = int(header_metadata["sparse_corrections_per_row"])
    base_packed_bytes_per_row = int(header_metadata["base_packed_bytes_per_row"])
    row_payload_bytes = base_packed_bytes_per_row + 2 + (sparse_per_row * 3)
    expected_payload_bytes = row_payload_bytes * row_count
    if len(payload) != expected_payload_bytes:
        raise ValueError(
            f"{SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID} payload length {len(payload)} "
            f"does not match expected {expected_payload_bytes}"
        )

    rows = np.zeros((row_count, hidden_size), dtype=np.float32)
    offset = 0
    for row_index in range(row_count):
        packed = payload[offset : offset + base_packed_bytes_per_row]
        offset += base_packed_bytes_per_row
        scale = float(np.frombuffer(payload[offset : offset + 2], dtype="<f2", count=1)[0])
        offset += 2
        indices = np.frombuffer(
            payload[offset : offset + sparse_per_row * 2],
            dtype="<u2",
            count=sparse_per_row,
        ).astype(np.int64)
        offset += sparse_per_row * 2
        codebook_ids = np.frombuffer(
            payload[offset : offset + sparse_per_row],
            dtype=np.uint8,
            count=sparse_per_row,
        )
        offset += sparse_per_row
        if np.any(indices >= hidden_size):
            raise ValueError("sparse-codebook correction index exceeds hidden width")
        if np.any(codebook_ids >= codebook.size):
            raise ValueError("sparse-codebook correction id exceeds codebook size")

        unsigned = _unpack_unsigned_values(
            packed,
            bits=base_bits,
            count=hidden_size,
        ).astype(np.int16)
        quantized = unsigned - levels
        row = quantized.astype(np.float32) * np.float32(scale)
        if sparse_per_row:
            row[indices] += codebook[codebook_ids]
        rows[row_index] = row

    return np.ascontiguousarray(rows.reshape(original_shape), dtype=np.float32)


def sparse_codebook_hash(codebook_values: Sequence[float] | np.ndarray) -> str:
    codebook = _normalized_codebook_values(codebook_values)
    digest = hashlib.sha256()
    digest.update(np.asarray([codebook.size], dtype="<u4").tobytes())
    digest.update(np.ascontiguousarray(codebook, dtype="<f4").tobytes())
    return digest.hexdigest()


def _validate_config(
    *,
    hidden_size: int,
    base_bits: int,
    sparse_corrections_per_row: int,
    codebook_values: Sequence[float] | np.ndarray,
) -> None:
    if int(base_bits) < 2 or int(base_bits) > 7:
        raise ValueError("sparse-codebook base_bits must be between 2 and 7")
    if int(sparse_corrections_per_row) < 0:
        raise ValueError("sparse-codebook sparse corrections must be non-negative")
    if hidden_size > 65535:
        raise ValueError("sparse-codebook uint16 indices require hidden width <= 65535")
    codebook = _normalized_codebook_values(codebook_values)
    if codebook.size < 1 or codebook.size > 256:
        raise ValueError("sparse-codebook codebook size must be between 1 and 256")


def _normalized_codebook_values(
    codebook_values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    codebook = np.asarray(codebook_values, dtype=np.float32).reshape(-1)
    if codebook.size == 0:
        raise ValueError("sparse-codebook requires at least one codebook value")
    if not np.all(np.isfinite(codebook)):
        raise ValueError("sparse-codebook codebook values must be finite")
    return np.ascontiguousarray(codebook, dtype=np.float32)


def _flatten_hidden_rows(array: np.ndarray) -> np.ndarray:
    if array.ndim < 1:
        raise ValueError("sparse-codebook payload requires at least one dimension")
    hidden_size = int(array.shape[-1])
    if hidden_size < 1:
        raise ValueError("sparse-codebook payload requires non-empty hidden dimension")
    return np.ascontiguousarray(array.reshape(-1, hidden_size), dtype=np.float32)


def _top_residual_indices(residual: np.ndarray, sparse_per_row: int) -> np.ndarray:
    if sparse_per_row <= 0:
        return np.zeros((0,), dtype=np.uint16)
    if sparse_per_row >= residual.size:
        indices = np.arange(residual.size, dtype=np.int64)
    else:
        indices = np.argpartition(np.abs(residual), -sparse_per_row)[-sparse_per_row:]
        indices = np.sort(indices)
    return indices.astype(np.uint16, copy=False)


def _nearest_codebook_ids(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.uint8)
    nearest = np.argmin(np.abs(values[:, None] - codebook[None, :]), axis=1)
    return nearest.astype(np.uint8, copy=False)


def _pack_unsigned_values(values: np.ndarray, *, bits: int) -> bytes:
    limit = 1 << int(bits)
    if np.any(values >= limit):
        raise ValueError("value exceeds packed bit width")
    out = bytearray()
    bit_buffer = 0
    bit_count = 0
    for value in values.astype(np.uint16, copy=False):
        bit_buffer |= int(value) << bit_count
        bit_count += int(bits)
        while bit_count >= 8:
            out.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8
    if bit_count:
        out.append(bit_buffer & 0xFF)
    return bytes(out)


def _unpack_unsigned_values(payload: bytes, *, bits: int, count: int) -> np.ndarray:
    out = np.zeros(int(count), dtype=np.uint8)
    bit_buffer = 0
    bit_count = 0
    byte_index = 0
    mask = (1 << int(bits)) - 1
    for value_index in range(int(count)):
        while bit_count < int(bits):
            if byte_index >= len(payload):
                raise ValueError("not enough bytes for packed sparse-codebook base")
            bit_buffer |= int(payload[byte_index]) << bit_count
            byte_index += 1
            bit_count += 8
        out[value_index] = bit_buffer & mask
        bit_buffer >>= int(bits)
        bit_count -= int(bits)
    return out
