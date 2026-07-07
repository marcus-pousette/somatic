from __future__ import annotations

import base64
import ctypes
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


BOUNDARY_CODEC_NATIVE_BINDING_SCHEMA_VERSION = "boundary-codec-native-binding-v0"
BOUNDARY_CODEC_NATIVE_LIBRARY_ENV = "SOUP_BOUNDARY_CODEC_RUST_DYLIB"
_ADAPTER_ID_TO_CODE = {
    "fp16": 1,
    "learned_residual4_sparse24_int8": 2,
}
_DTYPE_ID_TO_CODE = {
    "float32": 1,
    "float16": 2,
    "int32": 3,
    "int8": 4,
}
_DTYPE_CODE_TO_ID = {value: key for key, value in _DTYPE_ID_TO_CODE.items()}

_BINDING_CACHE: BoundaryCodecNativeBinding | None = None
_BINDING_CACHE_ATTEMPTED = False


class _BoundaryCodecBinaryResult(ctypes.Structure):
    _fields_ = [
        ("ok", ctypes.c_uint8),
        ("dtype_code", ctypes.c_uint32),
        ("shape_ptr", ctypes.POINTER(ctypes.c_size_t)),
        ("shape_len", ctypes.c_size_t),
        ("shape_capacity", ctypes.c_size_t),
        ("bytes_ptr", ctypes.POINTER(ctypes.c_uint8)),
        ("bytes_len", ctypes.c_size_t),
        ("bytes_capacity", ctypes.c_size_t),
        ("metadata_json_ptr", ctypes.c_void_p),
    ]


class _BoundaryCodecEncodeIntoResult(ctypes.Structure):
    _fields_ = [
        ("ok", ctypes.c_uint8),
        ("dtype_code", ctypes.c_uint32),
        ("shape_ptr", ctypes.POINTER(ctypes.c_size_t)),
        ("shape_len", ctypes.c_size_t),
        ("shape_capacity", ctypes.c_size_t),
        ("bytes_len", ctypes.c_size_t),
        ("metadata_json_ptr", ctypes.c_void_p),
    ]


class _BoundaryCodecDecodeIntoResult(ctypes.Structure):
    _fields_ = [
        ("ok", ctypes.c_uint8),
        ("decoded_value_count", ctypes.c_size_t),
        ("error_json_ptr", ctypes.c_void_p),
    ]


class BoundaryCodecNativeBinding:
    def __init__(self, library_path: Path) -> None:
        self.library_path = library_path
        self._library = ctypes.CDLL(str(library_path))
        self._library.boundary_codec_encode_json.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._library.boundary_codec_encode_json.restype = ctypes.c_void_p
        self._library.boundary_codec_decode_json.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._library.boundary_codec_decode_json.restype = ctypes.c_void_p
        self._library.boundary_codec_free_string.argtypes = [ctypes.c_void_p]
        self._library.boundary_codec_free_string.restype = None
        self.binary_ffi_available = self._configure_binary_ffi()
        self.lower_copy_ffi_available = self._configure_lower_copy_ffi()
        self.typed_metadata_decode_ffi_available = self._configure_typed_metadata_decode_ffi()

    def encode_array(self, array: np.ndarray, *, adapter_id: str) -> dict[str, Any]:
        if self.lower_copy_ffi_available:
            return self.encode_array_lower_copy(array, adapter_id=adapter_id)
        if self.binary_ffi_available:
            return self.encode_array_binary(array, adapter_id=adapter_id)
        return self.encode_array_json(array, adapter_id=adapter_id)

    def encode_array_lower_copy(self, array: np.ndarray, *, adapter_id: str) -> dict[str, Any]:
        decoded_result = self.encode_array_lower_copy_raw(array, adapter_id=adapter_id)
        return {
            "schema_version": BOUNDARY_CODEC_NATIVE_BINDING_SCHEMA_VERSION,
            "ok": True,
            "operation": "encode",
            "adapter_id": adapter_id,
            "encoded_dtype": decoded_result["dtype"],
            "encoded_shape": decoded_result["shape"],
            "encoded_bytes_b64": base64.b64encode(decoded_result["bytes"]).decode("ascii"),
            "metadata": decoded_result["metadata"],
            "ffi_mode": "lower_copy",
        }

    def encode_array_lower_copy_raw(self, array: np.ndarray, *, adapter_id: str) -> dict[str, Any]:
        adapter_code = _adapter_code(adapter_id)
        normalized = np.ascontiguousarray(array, dtype=np.float32)
        shape = [int(dim) for dim in normalized.shape]
        shape_buffer = _size_t_array(shape)
        flat = np.ascontiguousarray(normalized.reshape(-1), dtype=np.float32)
        output_capacity = _encoded_capacity(adapter_id=adapter_id, shape=shape)
        output_buffer = _uint8_zeroed_array(output_capacity)
        result = self._library.boundary_codec_encode_f32_into(
            adapter_code,
            shape_buffer,
            len(shape),
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(flat.size),
            output_buffer,
            output_capacity,
        )
        decoded_result = self._consume_encode_into_result(result)
        decoded_result["bytes"] = (
            ctypes.string_at(output_buffer, int(decoded_result["bytes_len"]))
            if decoded_result["bytes_len"]
            else b""
        )
        return decoded_result

    def encode_array_binary(self, array: np.ndarray, *, adapter_id: str) -> dict[str, Any]:
        decoded_result = self.encode_array_binary_raw(array, adapter_id=adapter_id)
        return {
            "schema_version": BOUNDARY_CODEC_NATIVE_BINDING_SCHEMA_VERSION,
            "ok": True,
            "operation": "encode",
            "adapter_id": adapter_id,
            "encoded_dtype": decoded_result["dtype"],
            "encoded_shape": decoded_result["shape"],
            "encoded_bytes_b64": base64.b64encode(decoded_result["bytes"]).decode("ascii"),
            "metadata": decoded_result["metadata"],
            "ffi_mode": "binary",
        }

    def encode_array_binary_raw(self, array: np.ndarray, *, adapter_id: str) -> dict[str, Any]:
        adapter_code = _adapter_code(adapter_id)
        normalized = np.ascontiguousarray(array, dtype=np.float32)
        shape = [int(dim) for dim in normalized.shape]
        shape_buffer = _size_t_array(shape)
        flat = np.ascontiguousarray(normalized.reshape(-1), dtype=np.float32)
        result = self._library.boundary_codec_encode_f32_binary(
            adapter_code,
            shape_buffer,
            len(shape),
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(flat.size),
        )
        return self._consume_binary_result(result)

    def encode_array_json(self, array: np.ndarray, *, adapter_id: str) -> dict[str, Any]:
        normalized = np.ascontiguousarray(array, dtype=np.float32)
        response = self._call_json(
            self._library.boundary_codec_encode_json,
            {
                "adapter_id": adapter_id,
                "original_dtype": str(normalized.dtype),
                "input_shape": list(normalized.shape),
                "input_values_f32": [float(value) for value in normalized.reshape(-1)],
            },
        )
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error") or "native encode failed"))
        response["ffi_mode"] = "json"
        return response

    def decode_array(
        self,
        *,
        adapter_id: str,
        original_dtype: str,
        original_shape: list[int],
        encoded_dtype: str,
        encoded_shape: list[int],
        encoded_bytes: bytes,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        if self.lower_copy_ffi_available:
            return self.decode_array_lower_copy(
                adapter_id=adapter_id,
                original_dtype=original_dtype,
                original_shape=original_shape,
                encoded_dtype=encoded_dtype,
                encoded_shape=encoded_shape,
                encoded_bytes=encoded_bytes,
                metadata=metadata,
            )
        if self.binary_ffi_available:
            return self.decode_array_binary(
                adapter_id=adapter_id,
                original_dtype=original_dtype,
                original_shape=original_shape,
                encoded_dtype=encoded_dtype,
                encoded_shape=encoded_shape,
                encoded_bytes=encoded_bytes,
                metadata=metadata,
            )
        return self.decode_array_json(
            adapter_id=adapter_id,
            original_dtype=original_dtype,
            original_shape=original_shape,
            encoded_dtype=encoded_dtype,
            encoded_shape=encoded_shape,
            encoded_bytes=encoded_bytes,
            metadata=metadata,
        )

    def decode_array_lower_copy(
        self,
        *,
        adapter_id: str,
        original_dtype: str,
        original_shape: list[int],
        encoded_dtype: str,
        encoded_shape: list[int],
        encoded_bytes: bytes,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        del original_dtype
        adapter_code = _adapter_code(adapter_id)
        encoded_dtype_code = _dtype_code(encoded_dtype)
        original_shape_values = [int(dim) for dim in original_shape]
        encoded_shape_values = [int(dim) for dim in encoded_shape]
        original_shape_buffer = _size_t_array(original_shape_values)
        encoded_shape_buffer = _size_t_array(encoded_shape_values)
        encoded_bytes_buffer = _uint8_array(encoded_bytes)
        output = np.empty(tuple(original_shape_values), dtype=np.float32)
        native_metadata = _native_metadata(metadata)
        if self.typed_metadata_decode_ffi_available:
            scale_values = _float32_array(native_metadata["scale_values"])
            scale_shape = _size_t_array(native_metadata["scale_shape"])
            result = self._library.boundary_codec_decode_f32_into_metadata(
                adapter_code,
                original_shape_buffer,
                len(original_shape_values),
                encoded_dtype_code,
                encoded_shape_buffer,
                len(encoded_shape_values),
                encoded_bytes_buffer,
                len(encoded_bytes),
                scale_values,
                len(native_metadata["scale_values"]),
                scale_shape,
                len(native_metadata["scale_shape"]),
                int(native_metadata["original_element_count"]),
                int(native_metadata["base_byte_count"]),
                int(native_metadata["residual_packed_byte_count"]),
                int(native_metadata["sparse_corrections_per_row"]),
                int(native_metadata["sparse_index_byte_count"]),
                int(native_metadata["sparse_value_byte_count"]),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(output.size),
            )
            self._consume_decode_into_result(result)
        else:
            metadata_json = json.dumps(native_metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
            metadata_buffer = ctypes.create_string_buffer(metadata_json)
            pointer = self._library.boundary_codec_decode_f32_into(
                adapter_code,
                original_shape_buffer,
                len(original_shape_values),
                encoded_dtype_code,
                encoded_shape_buffer,
                len(encoded_shape_values),
                encoded_bytes_buffer,
                len(encoded_bytes),
                ctypes.cast(metadata_buffer, ctypes.c_void_p),
                len(metadata_json),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(output.size),
            )
            self._consume_status_json(pointer)
        return output

    def decode_array_binary(
        self,
        *,
        adapter_id: str,
        original_dtype: str,
        original_shape: list[int],
        encoded_dtype: str,
        encoded_shape: list[int],
        encoded_bytes: bytes,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        return self.decode_array_binary_raw(
            adapter_id=adapter_id,
            original_dtype=original_dtype,
            original_shape=original_shape,
            encoded_dtype=encoded_dtype,
            encoded_shape=encoded_shape,
            encoded_bytes=encoded_bytes,
            metadata=metadata,
        )

    def decode_array_binary_raw(
        self,
        *,
        adapter_id: str,
        original_dtype: str,
        original_shape: list[int],
        encoded_dtype: str,
        encoded_shape: list[int],
        encoded_bytes: bytes,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        del original_dtype
        adapter_code = _adapter_code(adapter_id)
        encoded_dtype_code = _dtype_code(encoded_dtype)
        original_shape_values = [int(dim) for dim in original_shape]
        encoded_shape_values = [int(dim) for dim in encoded_shape]
        original_shape_buffer = _size_t_array(original_shape_values)
        encoded_shape_buffer = _size_t_array(encoded_shape_values)
        encoded_bytes_buffer = _uint8_array(encoded_bytes)
        metadata_json = json.dumps(_native_metadata(metadata), separators=(",", ":"), sort_keys=True).encode("utf-8")
        metadata_buffer = ctypes.create_string_buffer(metadata_json)
        result = self._library.boundary_codec_decode_f32_binary(
            adapter_code,
            original_shape_buffer,
            len(original_shape_values),
            encoded_dtype_code,
            encoded_shape_buffer,
            len(encoded_shape_values),
            encoded_bytes_buffer,
            len(encoded_bytes),
            ctypes.cast(metadata_buffer, ctypes.c_void_p),
            len(metadata_json),
        )
        decoded_result = self._consume_binary_result(result)
        if decoded_result["dtype"] != "float32":
            raise RuntimeError(f"native decode returned unexpected dtype {decoded_result['dtype']}")
        return np.frombuffer(decoded_result["bytes"], dtype="<f4").reshape(tuple(original_shape_values)).copy()

    def decode_array_json(
        self,
        *,
        adapter_id: str,
        original_dtype: str,
        original_shape: list[int],
        encoded_dtype: str,
        encoded_shape: list[int],
        encoded_bytes: bytes,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        response = self._call_json(
            self._library.boundary_codec_decode_json,
            {
                "adapter_id": adapter_id,
                "original_dtype": original_dtype,
                "original_shape": original_shape,
                "encoded_dtype": encoded_dtype,
                "encoded_shape": encoded_shape,
                "encoded_bytes_b64": base64.b64encode(encoded_bytes).decode("ascii"),
                "metadata": _native_metadata(metadata),
            },
        )
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error") or "native decode failed"))
        return np.asarray(response["decoded_values_f32"], dtype=np.float32).reshape(tuple(original_shape))

    def _configure_binary_ffi(self) -> bool:
        try:
            self._library.boundary_codec_encode_f32_binary.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            ]
            self._library.boundary_codec_encode_f32_binary.restype = _BoundaryCodecBinaryResult
            self._library.boundary_codec_decode_f32_binary.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            self._library.boundary_codec_decode_f32_binary.restype = _BoundaryCodecBinaryResult
            self._library.boundary_codec_free_binary_result.argtypes = [_BoundaryCodecBinaryResult]
            self._library.boundary_codec_free_binary_result.restype = None
        except AttributeError:
            return False
        return True

    def _configure_lower_copy_ffi(self) -> bool:
        try:
            self._library.boundary_codec_encode_f32_into.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
            ]
            self._library.boundary_codec_encode_f32_into.restype = _BoundaryCodecEncodeIntoResult
            self._library.boundary_codec_decode_f32_into.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            ]
            self._library.boundary_codec_decode_f32_into.restype = ctypes.c_void_p
            self._library.boundary_codec_free_encode_into_result.argtypes = [_BoundaryCodecEncodeIntoResult]
            self._library.boundary_codec_free_encode_into_result.restype = None
        except AttributeError:
            return False
        return True

    def _configure_typed_metadata_decode_ffi(self) -> bool:
        try:
            self._library.boundary_codec_decode_f32_into_metadata.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            ]
            self._library.boundary_codec_decode_f32_into_metadata.restype = _BoundaryCodecDecodeIntoResult
            self._library.boundary_codec_free_decode_into_result.argtypes = [_BoundaryCodecDecodeIntoResult]
            self._library.boundary_codec_free_decode_into_result.restype = None
        except AttributeError:
            return False
        return True

    def _consume_encode_into_result(self, result: _BoundaryCodecEncodeIntoResult) -> dict[str, Any]:
        try:
            metadata: dict[str, Any] = {}
            if result.metadata_json_ptr:
                metadata_bytes = ctypes.string_at(result.metadata_json_ptr)
                parsed_metadata = json.loads(metadata_bytes.decode("utf-8"))
                if isinstance(parsed_metadata, dict):
                    metadata = parsed_metadata
            if result.ok != 1:
                raise RuntimeError(str(metadata.get("error") or "native lower-copy boundary codec failed"))
            return {
                "dtype": _dtype_from_code(int(result.dtype_code)),
                "shape": _copy_size_t_pointer(result.shape_ptr, int(result.shape_len)),
                "bytes_len": int(result.bytes_len),
                "metadata": metadata,
            }
        finally:
            self._library.boundary_codec_free_encode_into_result(result)

    def _consume_decode_into_result(self, result: _BoundaryCodecDecodeIntoResult) -> dict[str, Any]:
        try:
            error: dict[str, Any] = {}
            if result.error_json_ptr:
                error_bytes = ctypes.string_at(result.error_json_ptr)
                parsed_error = json.loads(error_bytes.decode("utf-8"))
                if isinstance(parsed_error, dict):
                    error = parsed_error
            if result.ok != 1:
                raise RuntimeError(str(error.get("error") or "native typed-metadata decode failed"))
            return {
                "decoded_value_count": int(result.decoded_value_count),
            }
        finally:
            self._library.boundary_codec_free_decode_into_result(result)

    def _consume_binary_result(self, result: _BoundaryCodecBinaryResult) -> dict[str, Any]:
        try:
            metadata: dict[str, Any] = {}
            if result.metadata_json_ptr:
                metadata_bytes = ctypes.string_at(result.metadata_json_ptr)
                parsed_metadata = json.loads(metadata_bytes.decode("utf-8"))
                if isinstance(parsed_metadata, dict):
                    metadata = parsed_metadata
            if result.ok != 1:
                raise RuntimeError(str(metadata.get("error") or "native binary boundary codec failed"))
            dtype = _dtype_from_code(int(result.dtype_code))
            shape = _copy_size_t_pointer(result.shape_ptr, int(result.shape_len))
            raw = (
                ctypes.string_at(result.bytes_ptr, int(result.bytes_len))
                if result.bytes_ptr and result.bytes_len
                else b""
            )
            return {"dtype": dtype, "shape": shape, "bytes": raw, "metadata": metadata}
        finally:
            self._library.boundary_codec_free_binary_result(result)

    def _consume_status_json(self, pointer: int) -> dict[str, Any]:
        if not pointer:
            raise RuntimeError("native boundary codec returned a null status pointer")
        try:
            response = ctypes.string_at(pointer)
        finally:
            self._library.boundary_codec_free_string(pointer)
        data = json.loads(response.decode("utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError("native boundary codec status response was not an object")
        if data.get("ok") is not True:
            raise RuntimeError(str(data.get("error") or "native boundary codec status failed"))
        return data

    def _call_json(self, function: Any, payload: dict[str, Any]) -> dict[str, Any]:
        request = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        buffer = ctypes.create_string_buffer(request)
        pointer = function(ctypes.cast(buffer, ctypes.c_void_p), len(request))
        if not pointer:
            raise RuntimeError("native boundary codec returned a null response pointer")
        try:
            response = ctypes.string_at(pointer)
        finally:
            self._library.boundary_codec_free_string(pointer)
        data = json.loads(response.decode("utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError("native boundary codec response was not an object")
        return data


def load_boundary_codec_native_binding(
    *,
    library_path: Path | str | None = None,
    use_cache: bool = True,
) -> BoundaryCodecNativeBinding | None:
    global _BINDING_CACHE, _BINDING_CACHE_ATTEMPTED
    if use_cache and _BINDING_CACHE_ATTEMPTED:
        return _BINDING_CACHE
    _BINDING_CACHE_ATTEMPTED = True
    resolved_path = _resolve_library_path(library_path)
    if resolved_path is None or not resolved_path.exists():
        _BINDING_CACHE = None
        return None
    try:
        _BINDING_CACHE = BoundaryCodecNativeBinding(resolved_path)
    except OSError:
        _BINDING_CACHE = None
    return _BINDING_CACHE


def reset_boundary_codec_native_binding_cache() -> None:
    global _BINDING_CACHE, _BINDING_CACHE_ATTEMPTED
    _BINDING_CACHE = None
    _BINDING_CACHE_ATTEMPTED = False


def boundary_codec_native_binding_status(*, library_path: Path | str | None = None) -> dict[str, Any]:
    resolved_path = _resolve_library_path(library_path)
    binding = load_boundary_codec_native_binding(library_path=library_path, use_cache=False)
    return {
        "schema_version": BOUNDARY_CODEC_NATIVE_BINDING_SCHEMA_VERSION,
        "library_path": str(resolved_path) if resolved_path is not None else None,
        "library_exists": bool(resolved_path and resolved_path.exists()),
        "available": binding is not None,
        "binding_kind": (
            _binding_kind(binding)
            if binding is not None
            else "ctypes-cdylib-unavailable"
        ),
        "lower_copy_ffi_available": bool(binding is not None and binding.lower_copy_ffi_available),
        "typed_metadata_decode_ffi_available": bool(
            binding is not None and binding.typed_metadata_decode_ffi_available
        ),
        "binary_ffi_available": bool(binding is not None and binding.binary_ffi_available),
        "json_ffi_available": binding is not None,
    }


def _resolve_library_path(library_path: Path | str | None) -> Path | None:
    if library_path is not None:
        return Path(library_path)
    env_path = os.environ.get(BOUNDARY_CODEC_NATIVE_LIBRARY_ENV)
    if env_path:
        return Path(env_path)
    root = Path(__file__).resolve().parents[2]
    return root / "crates" / "boundary-codec" / "target" / "debug" / _library_filename()


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libboundary_codec.dylib"
    if sys.platform.startswith("linux"):
        return "libboundary_codec.so"
    if sys.platform.startswith("win"):
        return "boundary_codec.dll"
    return "libboundary_codec.so"


def _native_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    scale_shape = [int(value) for value in metadata.get("scale_shape") or []]
    element_count = int(metadata.get("original_element_count") or 0)
    residual_bits = int(metadata.get("residual_bits") or 4)
    rows = int(scale_shape[0]) if scale_shape else 0
    sparse_per_row = int(metadata.get("sparse_corrections_per_row") or 0)
    return {
        "scale_values": [float(value) for value in metadata.get("scale_values") or []],
        "scale_shape": scale_shape,
        "original_element_count": element_count,
        "base_byte_count": int(metadata.get("base_byte_count") or element_count),
        "residual_packed_byte_count": int(
            metadata.get("residual_packed_byte_count") or math.ceil(element_count * residual_bits / 8)
        ),
        "sparse_corrections_per_row": sparse_per_row,
        "sparse_index_byte_count": int(metadata.get("sparse_index_byte_count") or rows * sparse_per_row * 2),
        "sparse_value_byte_count": int(metadata.get("sparse_value_byte_count") or rows * sparse_per_row),
    }


def _binding_kind(binding: BoundaryCodecNativeBinding) -> str:
    if binding.lower_copy_ffi_available and binding.typed_metadata_decode_ffi_available:
        return "ctypes-cdylib-lower-copy-typed-metadata-ffi"
    if binding.lower_copy_ffi_available:
        return "ctypes-cdylib-lower-copy-ffi"
    if binding.binary_ffi_available:
        return "ctypes-cdylib-binary-ffi"
    return "ctypes-cdylib-json-ffi"


def _adapter_code(adapter_id: str) -> int:
    try:
        return _ADAPTER_ID_TO_CODE[adapter_id]
    except KeyError as exc:
        raise RuntimeError(f"unsupported native boundary adapter {adapter_id}") from exc


def _dtype_code(dtype: str) -> int:
    try:
        return _DTYPE_ID_TO_CODE[dtype]
    except KeyError as exc:
        raise RuntimeError(f"unsupported native tensor dtype {dtype}") from exc


def _dtype_from_code(dtype_code: int) -> str:
    try:
        return _DTYPE_CODE_TO_ID[dtype_code]
    except KeyError as exc:
        raise RuntimeError(f"unsupported native tensor dtype code {dtype_code}") from exc


def _size_t_array(values: list[int]) -> Any:
    if not values:
        return ctypes.POINTER(ctypes.c_size_t)()
    return (ctypes.c_size_t * len(values))(*values)


def _uint8_array(values: bytes) -> Any:
    if not values:
        return ctypes.POINTER(ctypes.c_uint8)()
    return (ctypes.c_uint8 * len(values)).from_buffer_copy(values)


def _uint8_zeroed_array(length: int) -> Any:
    if length <= 0:
        return ctypes.POINTER(ctypes.c_uint8)()
    return (ctypes.c_uint8 * length)()


def _float32_array(values: list[float]) -> Any:
    if not values:
        return ctypes.POINTER(ctypes.c_float)()
    return (ctypes.c_float * len(values))(*values)


def _copy_size_t_pointer(pointer: Any, length: int) -> list[int]:
    if length == 0:
        return []
    if not pointer:
        raise RuntimeError("native boundary codec returned null shape pointer")
    return [int(pointer[index]) for index in range(length)]


def _encoded_capacity(*, adapter_id: str, shape: list[int]) -> int:
    element_count = math.prod(shape)
    if adapter_id == "fp16":
        return int(element_count * np.dtype(np.float16).itemsize)
    if adapter_id == "learned_residual4_sparse24_int8":
        if not shape:
            raise RuntimeError("sparse24 native encode requires a non-empty shape")
        columns = int(shape[-1])
        if columns <= 0:
            raise RuntimeError("sparse24 native encode requires a positive final dimension")
        rows = int(element_count // columns)
        sparse_per_row = min(24, columns)
        residual_packed_byte_count = math.ceil(element_count * 4 / 8)
        return int(element_count + residual_packed_byte_count + rows * sparse_per_row * 2 + rows * sparse_per_row)
    raise RuntimeError(f"unsupported native boundary adapter {adapter_id}")
