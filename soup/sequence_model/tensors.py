from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field, model_validator


TensorDType = Literal["float32", "float16", "int32", "int8"]
TensorEncoding = Literal["base64"]
TensorFrameHeader = dict[str, Any]
TENSOR_FRAME_MAGIC = b"SMTENSOR1"
TENSOR_FRAME_MEDIA_TYPE = "application/vnd.soup.tensor-frame"


_NUMPY_DTYPES: dict[str, np.dtype] = {
    "float32": np.dtype("float32"),
    "float16": np.dtype("float16"),
    "int32": np.dtype("int32"),
    "int8": np.dtype("int8"),
}


@dataclass(frozen=True)
class TensorFrameParts:
    """Validated binary tensor-frame view that keeps tensor bytes out of base64 payloads."""

    name: str
    dtype: TensorDType
    shape: list[int]
    layout: str
    metadata: dict[str, Any]
    extra: dict[str, Any]
    raw: bytes


class TensorPayload(BaseModel):
    """Portable typed tensor payload for crossing sequence-worker boundaries."""

    name: str = "activation"
    dtype: TensorDType = "float32"
    shape: list[int] = Field(min_length=1)
    data: str
    encoding: TensorEncoding = "base64"
    layout: Literal["row_major"] = "row_major"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw_bytes(
        cls,
        raw: bytes,
        *,
        dtype: TensorDType,
        shape: list[int],
        name: str = "activation",
        metadata: dict[str, Any] | None = None,
    ) -> "TensorPayload":
        expected_bytes = math.prod(shape) * _numpy_dtype(dtype).itemsize
        if len(raw) != expected_bytes:
            raise ValueError(f"tensor byte length {len(raw)} does not match shape/dtype byte length {expected_bytes}")
        return cls(
            name=name,
            dtype=dtype,
            shape=shape,
            data=base64.b64encode(raw).decode("ascii"),
            metadata=metadata or {},
        )

    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray,
        *,
        name: str = "activation",
        metadata: dict[str, Any] | None = None,
    ) -> "TensorPayload":
        normalized = np.ascontiguousarray(array)
        dtype = str(normalized.dtype)
        if dtype not in _NUMPY_DTYPES:
            raise ValueError(f"unsupported tensor dtype {dtype}")
        return cls(
            name=name,
            dtype=dtype,  # type: ignore[arg-type]
            shape=list(normalized.shape),
            data=base64.b64encode(normalized.tobytes(order="C")).decode("ascii"),
            metadata=metadata or {},
        )

    def to_numpy(self) -> np.ndarray:
        raw = base64.b64decode(self.data.encode("ascii"), validate=True)
        array = np.frombuffer(raw, dtype=_numpy_dtype(self.dtype))
        return array.reshape(tuple(self.shape)).copy()

    def raw_bytes(self) -> bytes:
        return base64.b64decode(self.data.encode("ascii"), validate=True)

    def byte_size(self) -> int:
        return math.prod(self.shape) * _numpy_dtype(self.dtype).itemsize

    def digest(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.name.encode("utf-8"))
        hasher.update(self.dtype.encode("utf-8"))
        hasher.update(",".join(str(dim) for dim in self.shape).encode("utf-8"))
        hasher.update(self.data.encode("ascii"))
        return hasher.hexdigest()

    @model_validator(mode="after")
    def _validate_tensor_storage(self) -> "TensorPayload":
        if any(dim <= 0 for dim in self.shape):
            raise ValueError("tensor shape dimensions must be positive")
        raw = base64.b64decode(self.data.encode("ascii"), validate=True)
        expected_bytes = self.byte_size()
        if len(raw) != expected_bytes:
            raise ValueError(f"tensor byte length {len(raw)} does not match shape/dtype byte length {expected_bytes}")
        return self


def _numpy_dtype(dtype: TensorDType) -> np.dtype:
    return _NUMPY_DTYPES[dtype]


def encode_tensor_frame(payload: TensorPayload, *, extra_header: dict[str, Any] | None = None) -> bytes:
    return _encode_tensor_frame(
        raw=payload.raw_bytes(),
        name=payload.name,
        dtype=payload.dtype,
        shape=payload.shape,
        layout=payload.layout,
        metadata=payload.metadata,
        extra_header=extra_header,
    )


def encode_tensor_frame_from_numpy(
    array: np.ndarray,
    *,
    name: str = "activation",
    metadata: dict[str, Any] | None = None,
    extra_header: dict[str, Any] | None = None,
) -> bytes:
    normalized = np.ascontiguousarray(array)
    dtype = str(normalized.dtype)
    if dtype not in _NUMPY_DTYPES:
        raise ValueError(f"unsupported tensor dtype {dtype}")
    return _encode_tensor_frame(
        raw=normalized.tobytes(order="C"),
        name=name,
        dtype=dtype,  # type: ignore[arg-type]
        shape=list(normalized.shape),
        layout="row_major",
        metadata=metadata or {},
        extra_header=extra_header,
    )


def encode_tensor_frame_from_raw(
    raw: bytes,
    *,
    dtype: TensorDType,
    shape: list[int],
    name: str = "activation",
    metadata: dict[str, Any] | None = None,
    extra_header: dict[str, Any] | None = None,
    layout: str = "row_major",
) -> bytes:
    return _encode_tensor_frame(
        raw=raw,
        name=name,
        dtype=dtype,
        shape=shape,
        layout=layout,
        metadata=metadata or {},
        extra_header=extra_header,
    )


def decode_tensor_frame(frame: bytes) -> tuple[TensorPayload, dict[str, Any]]:
    tensor_header, extra, raw = _decode_tensor_frame_parts(frame)
    payload = TensorPayload.from_raw_bytes(
        raw,
        dtype=cast(TensorDType, tensor_header["dtype"]),
        shape=[int(dim) for dim in tensor_header["shape"]],
        name=str(tensor_header.get("name") or "activation"),
        metadata=tensor_header.get("metadata") or {},
    )
    return payload, extra


def decode_tensor_frame_to_raw(frame: bytes) -> TensorFrameParts:
    tensor_header, extra, raw = _decode_tensor_frame_parts(frame)
    return TensorFrameParts(
        name=str(tensor_header.get("name") or "activation"),
        dtype=cast(TensorDType, tensor_header["dtype"]),
        shape=[int(dim) for dim in tensor_header["shape"]],
        layout=str(tensor_header.get("layout") or "row_major"),
        metadata=dict(tensor_header.get("metadata") or {}),
        extra=extra,
        raw=raw,
    )


def decode_tensor_frame_to_numpy(frame: bytes) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    tensor_header, extra, raw = _decode_tensor_frame_parts(frame)
    dtype = cast(TensorDType, tensor_header["dtype"])
    shape = [int(dim) for dim in tensor_header["shape"]]
    array = np.frombuffer(raw, dtype=_numpy_dtype(dtype)).reshape(tuple(shape)).copy()
    return array, tensor_header, extra


def _decode_tensor_frame_parts(frame: bytes) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    if len(frame) < len(TENSOR_FRAME_MAGIC) + 4:
        raise ValueError("tensor frame is too short")
    if not frame.startswith(TENSOR_FRAME_MAGIC):
        raise ValueError("tensor frame magic mismatch")
    header_length_offset = len(TENSOR_FRAME_MAGIC)
    header_length = struct.unpack(">I", frame[header_length_offset : header_length_offset + 4])[0]
    header_start = header_length_offset + 4
    header_end = header_start + header_length
    if header_end > len(frame):
        raise ValueError("tensor frame header length exceeds frame length")
    header = json.loads(frame[header_start:header_end].decode("utf-8"))
    tensor_header = header.get("tensor")
    if not isinstance(tensor_header, dict):
        raise ValueError("tensor frame header is missing tensor metadata")
    dtype = str(tensor_header.get("dtype", ""))
    if dtype not in _NUMPY_DTYPES:
        raise ValueError(f"unsupported tensor dtype {dtype}")
    shape = tensor_header.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError("tensor frame shape must be a non-empty list")
    raw = frame[header_end:]
    byte_length = int(tensor_header.get("byte_length", -1))
    if len(raw) != byte_length:
        raise ValueError(f"tensor frame byte length {len(raw)} does not match header byte length {byte_length}")
    expected_bytes = math.prod(int(dim) for dim in shape) * _numpy_dtype(cast(TensorDType, dtype)).itemsize
    if len(raw) != expected_bytes:
        raise ValueError(f"tensor frame byte length {len(raw)} does not match shape/dtype byte length {expected_bytes}")
    tensor_header["dtype"] = dtype
    tensor_header["shape"] = [int(dim) for dim in shape]
    return tensor_header, dict(header.get("extra") or {}), raw


def _encode_tensor_frame(
    *,
    raw: bytes,
    name: str,
    dtype: TensorDType,
    shape: list[int],
    layout: str,
    metadata: dict[str, Any],
    extra_header: dict[str, Any] | None,
) -> bytes:
    expected_bytes = math.prod(shape) * _numpy_dtype(dtype).itemsize
    if len(raw) != expected_bytes:
        raise ValueError(f"tensor byte length {len(raw)} does not match shape/dtype byte length {expected_bytes}")
    header = {
        "format": "soup.tensor-frame.v1",
        "tensor": {
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "layout": layout,
            "metadata": metadata,
            "byte_length": len(raw),
        },
        "extra": extra_header or {},
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return TENSOR_FRAME_MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + raw
