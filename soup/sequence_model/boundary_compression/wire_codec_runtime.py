from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soup.sequence_model.boundary_adapters import (
    decode_payload_from_boundary,
    encode_payload_for_boundary,
    normalize_boundary_adapter_strategy,
)
from soup.sequence_model.boundary_compression.sparse_codebook_wire import (
    SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID,
    SPARSE_CODEBOOK_WIRE_CODEC_SCHEMA_VERSION,
    SPARSE_CODEBOOK_WIRE_PAYLOAD_KIND,
    decode_sparse_codebook_components,
    encode_sparse_codebook_components,
)
from soup.sequence_model.boundary_compression.boundary_wire_lossless import (
    compress_wire_payload,
    decompress_wire_payload,
    strategy_is_lossless_eligible,
)
from soup.sequence_model.boundary_compression.trainable_autoencoder_wire import (
    TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
    TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND,
    decode_trainable_autoencoder_source_latent_payload,
    encode_trainable_autoencoder_source_latent_payload,
)
from soup.sequence_model.tensors import TensorPayload


LEARNED_INT8_BOUNDARY_ADAPTER_ID = "learned_int8"
LEARNED_INT8_WIRE_CODEC_SCHEMA_VERSION = "boundary-learned-int8-wire-codec-v0"
LEARNED_INT8_WIRE_CODEC_ARTIFACT_SCHEMA_VERSION = (
    "boundary-learned-int8-wire-codec-artifact-v0"
)
LEARNED_INT8_WIRE_PAYLOAD_KIND = "scale_float32_le_plus_int8_vector"
BOUNDARY_WIRE_ENCODED_PAYLOAD_SCHEMA_VERSION = "boundary-wire-encoded-payload-v0"


@dataclass(frozen=True)
class BoundaryWireEncodedPayload:
    payload: bytes
    header_fields: dict[str, Any]
    accounting_strategy: str


@dataclass(frozen=True)
class LearnedInt8WireCodec:
    """Runtime decoder for a trained learned-int8 boundary codec payload.

    The codec is architecture-neutral at the transport boundary: it accepts a
    dequantized activation vector and maps it back onto the downstream model
    slice's expected activation surface. The source artifact may be Qwen today,
    but the wire/runtime contract is only shape + learned parameters.
    """

    source_path: str
    region_id: str
    parameter_hash: str | None
    parameter_count: int | None
    low_rank_rank: int | None
    gain: float
    bias: np.ndarray
    low_rank_down: np.ndarray | None = None
    low_rank_up: np.ndarray | None = None
    residual_scale: float = 1.0
    low_rank_residual_scale: float | None = None
    mlp_residual_scale: float | None = None
    boundary_compression_strategy: str | None = None
    mlp_rank: int | None = None
    mlp_encoder: np.ndarray | None = None
    mlp_latent_bias: np.ndarray | None = None
    mlp_decoder: np.ndarray | None = None
    ladder_rank: int | None = None
    ladder_expert_count: int | None = None
    ladder_low_rank_downs: tuple[np.ndarray, ...] = ()
    ladder_low_rank_ups: tuple[np.ndarray, ...] = ()
    ladder_gate_weights: np.ndarray | None = None
    sparse_count: int | None = None
    sparse_indices: np.ndarray | None = None
    sparse_values: np.ndarray | None = None

    @property
    def width(self) -> int:
        return int(self.bias.shape[0])

    def decode_dequantized(self, dequantized: np.ndarray) -> np.ndarray:
        vector = np.ascontiguousarray(dequantized, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.width:
            raise ValueError(
                f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} vector width "
                f"{vector.shape[0]} does not match codec width {self.width}"
            )
        restored = (self.gain * vector + self.bias).astype(np.float32, copy=False)
        residual = self._residual_from_dequantized(vector)
        if residual is not None:
            restored = restored + (np.float32(self.residual_scale) * residual)
        return restored.astype(np.float32, copy=False)

    def _residual_from_dequantized(self, vector: np.ndarray) -> np.ndarray | None:
        has_low_rank = self.low_rank_down is not None or self.low_rank_up is not None
        has_mlp = (
            self.mlp_encoder is not None
            or self.mlp_latent_bias is not None
            or self.mlp_decoder is not None
        )
        has_ladder = bool(
            self.ladder_low_rank_downs
            or self.ladder_low_rank_ups
            or self.ladder_gate_weights is not None
        )
        has_sparse = self.sparse_indices is not None or self.sparse_values is not None
        hybrid_payload = (
            self.boundary_compression_strategy
            == "int8_symmetric_last_token_hybrid_residual_decode"
        )
        sparse_payload = (
            self.boundary_compression_strategy
            == "int8_symmetric_last_token_sparse_residual_decode"
        )
        if sum(bool(item) for item in (has_low_rank, has_mlp, has_ladder, has_sparse)) > 1 and not (
            hybrid_payload and has_low_rank and has_mlp and not has_ladder and not has_sparse
        ):
            raise ValueError(
                f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} payload mixes residual families"
            )
        if has_sparse and not sparse_payload:
            raise ValueError(
                f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} sparse payload requires sparse strategy"
            )
        if hybrid_payload:
            low_rank = _low_rank_residual(
                vector,
                down=self.low_rank_down,
                up=self.low_rank_up,
                width=self.width,
            )
            mlp = _mlp_residual(
                vector,
                encoder=self.mlp_encoder,
                latent_bias=self.mlp_latent_bias,
                decoder=self.mlp_decoder,
                width=self.width,
            )
            return (
                np.float32(
                    self.low_rank_residual_scale
                    if self.low_rank_residual_scale is not None
                    else self.residual_scale
                )
                * low_rank
            ) + (
                np.float32(
                    self.mlp_residual_scale
                    if self.mlp_residual_scale is not None
                    else self.residual_scale
                )
                * mlp
            )
        if has_low_rank:
            return _low_rank_residual(
                vector,
                down=self.low_rank_down,
                up=self.low_rank_up,
                width=self.width,
            )
        if has_mlp:
            return _mlp_residual(
                vector,
                encoder=self.mlp_encoder,
                latent_bias=self.mlp_latent_bias,
                decoder=self.mlp_decoder,
                width=self.width,
            )
        if has_ladder:
            return _ladder_residual(
                vector,
                downs=self.ladder_low_rank_downs,
                ups=self.ladder_low_rank_ups,
                gate_weights=self.ladder_gate_weights,
                width=self.width,
            )
        if has_sparse:
            return _sparse_residual(
                indices=self.sparse_indices,
                values=self.sparse_values,
                width=self.width,
            )
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": LEARNED_INT8_WIRE_CODEC_SCHEMA_VERSION,
            "adapter_id": LEARNED_INT8_BOUNDARY_ADAPTER_ID,
            "source_path": self.source_path,
            "region_id": self.region_id,
            "parameter_hash": self.parameter_hash,
            "parameter_count": self.parameter_count,
            "low_rank_rank": self.low_rank_rank,
            "mlp_rank": self.mlp_rank,
            "ladder_rank": self.ladder_rank,
            "ladder_expert_count": self.ladder_expert_count,
            "sparse_count": self.sparse_count,
            "boundary_compression_strategy": self.boundary_compression_strategy,
            "learned_payload_family": self.learned_payload_family,
            "width": self.width,
            "architecture_neutral_boundary_codec": True,
            "production_runtime_claimed": False,
        }

    @property
    def learned_payload_family(self) -> str:
        if (
            self.boundary_compression_strategy
            == "int8_symmetric_last_token_hybrid_residual_decode"
        ):
            return "hybrid_residual"
        if self.low_rank_down is not None or self.low_rank_up is not None:
            return "low_rank_residual"
        if (
            self.mlp_encoder is not None
            or self.mlp_latent_bias is not None
            or self.mlp_decoder is not None
        ):
            return "mlp_residual"
        if (
            self.ladder_low_rank_downs
            or self.ladder_low_rank_ups
            or self.ladder_gate_weights is not None
        ):
            return "residual_ladder"
        if self.sparse_indices is not None or self.sparse_values is not None:
            return "sparse_residual"
        return "affine"


@dataclass(frozen=True)
class LearnedInt8WireCodecArtifact:
    schema_version: str
    artifact_id: str
    adapter_id: str
    codec_results_path: str
    codec_results_sha256: str
    region_id: str
    parameter_hash: str | None
    parameter_count: int | None
    low_rank_rank: int | None
    learned_payload_family: str
    boundary_compression_strategy: str | None
    width: int
    fallback_adapter_id: str
    payload_kind: str
    claims: dict[str, bool]
    metadata: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "adapter_id": self.adapter_id,
            "codec_results_path": self.codec_results_path,
            "codec_results_sha256": self.codec_results_sha256,
            "region_id": self.region_id,
            "parameter_hash": self.parameter_hash,
            "parameter_count": self.parameter_count,
            "low_rank_rank": self.low_rank_rank,
            "learned_payload_family": self.learned_payload_family,
            "boundary_compression_strategy": self.boundary_compression_strategy,
            "width": self.width,
            "fallback_adapter_id": self.fallback_adapter_id,
            "payload_kind": self.payload_kind,
            "claims": dict(self.claims),
            "metadata": dict(self.metadata),
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(), indent=2) + "\n",
            encoding="utf-8",
        )


def build_learned_int8_wire_codec_artifact(
    codec_results_path: str | Path,
    *,
    region_id: str = "shared",
    fallback_adapter_id: str = "fp16",
    artifact_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LearnedInt8WireCodecArtifact:
    codec = load_learned_int8_wire_codec(codec_results_path, region_id=region_id)
    codec_results_sha256 = _sha256_file(_resolve_path(codec_results_path))
    parameter_prefix = (codec.parameter_hash or codec_results_sha256)[:12]
    artifact_id = artifact_id or (
        f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID}:{region_id}:"
        f"{codec.width}:{parameter_prefix}"
    )
    return LearnedInt8WireCodecArtifact(
        schema_version=LEARNED_INT8_WIRE_CODEC_ARTIFACT_SCHEMA_VERSION,
        artifact_id=artifact_id,
        adapter_id=LEARNED_INT8_BOUNDARY_ADAPTER_ID,
        codec_results_path=str(codec_results_path),
        codec_results_sha256=codec_results_sha256,
        region_id=region_id,
        parameter_hash=codec.parameter_hash,
        parameter_count=codec.parameter_count,
        low_rank_rank=codec.low_rank_rank,
        learned_payload_family=codec.learned_payload_family,
        boundary_compression_strategy=codec.boundary_compression_strategy,
        width=codec.width,
        fallback_adapter_id=fallback_adapter_id,
        payload_kind=LEARNED_INT8_WIRE_PAYLOAD_KIND,
        claims={
            "architecture_neutral_boundary_codec": True,
            "runtime_artifact_hash_validated_claimed": True,
            "planner_selectable_claimed": False,
            "production_runtime_claimed": False,
            "rust_native_execution_claimed": False,
            "public_worker_security_claimed": False,
            "distributed_training_claimed": False,
            "token_launch_claimed": False,
        },
        metadata={
            "source_codec_metadata": codec.metadata(),
            "artifact_runtime_scope": "bounded_live_wire_codec_loader",
            "fallback_adapter_id": fallback_adapter_id,
            **(metadata or {}),
        },
    )


def load_learned_int8_wire_codec_artifact(
    artifact_path: str | Path,
) -> LearnedInt8WireCodecArtifact:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != LEARNED_INT8_WIRE_CODEC_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported learned-int8 wire codec artifact schema")
    if payload.get("adapter_id") != LEARNED_INT8_BOUNDARY_ADAPTER_ID:
        raise ValueError("learned-int8 wire codec artifact has wrong adapter_id")
    if payload.get("payload_kind") != LEARNED_INT8_WIRE_PAYLOAD_KIND:
        raise ValueError("learned-int8 wire codec artifact has wrong payload kind")
    claims = payload.get("claims") if isinstance(payload.get("claims"), dict) else {}
    metadata = dict(payload.get("metadata") or {})
    source_codec_metadata = (
        metadata.get("source_codec_metadata")
        if isinstance(metadata.get("source_codec_metadata"), dict)
        else {}
    )
    forbidden_true = [
        claim
        for claim in (
            "planner_selectable_claimed",
            "production_runtime_claimed",
            "rust_native_execution_claimed",
            "public_worker_security_claimed",
            "distributed_training_claimed",
            "token_launch_claimed",
        )
        if claims.get(claim) is True
    ]
    if forbidden_true:
        raise ValueError(
            "learned-int8 wire codec artifact overclaims: "
            + ", ".join(forbidden_true)
        )
    return LearnedInt8WireCodecArtifact(
        schema_version=str(payload["schema_version"]),
        artifact_id=str(payload["artifact_id"]),
        adapter_id=str(payload["adapter_id"]),
        codec_results_path=str(payload["codec_results_path"]),
        codec_results_sha256=str(payload["codec_results_sha256"]),
        region_id=str(payload.get("region_id") or "shared"),
        parameter_hash=payload.get("parameter_hash"),
        parameter_count=_optional_int(payload.get("parameter_count")),
        low_rank_rank=_optional_int(payload.get("low_rank_rank")),
        learned_payload_family=str(
            payload.get("learned_payload_family")
            or source_codec_metadata.get("learned_payload_family")
            or (
                "low_rank_residual"
                if payload.get("low_rank_rank") is not None
                else "unknown"
            )
        ),
        boundary_compression_strategy=_optional_str(
            payload.get("boundary_compression_strategy")
            or source_codec_metadata.get("boundary_compression_strategy")
        ),
        width=int(payload["width"]),
        fallback_adapter_id=str(payload.get("fallback_adapter_id") or "fp16"),
        payload_kind=str(payload["payload_kind"]),
        claims={str(key): bool(value) for key, value in claims.items()},
        metadata=metadata,
    )


def load_learned_int8_wire_codec_from_artifact(
    artifact_path: str | Path,
) -> LearnedInt8WireCodec:
    artifact = load_learned_int8_wire_codec_artifact(artifact_path)
    codec_results_path = _resolve_path(artifact.codec_results_path)
    observed_sha = _sha256_file(codec_results_path)
    if observed_sha != artifact.codec_results_sha256:
        raise ValueError(
            "learned-int8 wire codec artifact source hash mismatch: "
            f"expected {artifact.codec_results_sha256}, observed {observed_sha}"
        )
    codec = load_learned_int8_wire_codec(
        codec_results_path,
        region_id=artifact.region_id,
    )
    if codec.parameter_hash != artifact.parameter_hash:
        raise ValueError(
            "learned-int8 wire codec artifact parameter hash mismatch: "
            f"expected {artifact.parameter_hash}, observed {codec.parameter_hash}"
        )
    if (
        artifact.learned_payload_family != "unknown"
        and codec.learned_payload_family != artifact.learned_payload_family
    ):
        raise ValueError(
            "learned-int8 wire codec artifact payload family mismatch: "
            f"expected {artifact.learned_payload_family}, observed "
            f"{codec.learned_payload_family}"
        )
    if (
        artifact.boundary_compression_strategy is not None
        and codec.boundary_compression_strategy is not None
        and codec.boundary_compression_strategy
        != artifact.boundary_compression_strategy
    ):
        raise ValueError(
            "learned-int8 wire codec artifact boundary strategy mismatch: "
            f"expected {artifact.boundary_compression_strategy}, observed "
            f"{codec.boundary_compression_strategy}"
        )
    if codec.width != artifact.width:
        raise ValueError(
            "learned-int8 wire codec artifact width mismatch: "
            f"expected {artifact.width}, observed {codec.width}"
        )
    return codec


def write_learned_int8_wire_codec_artifact(
    codec_results_path: str | Path,
    output_path: str | Path,
    *,
    region_id: str = "shared",
    fallback_adapter_id: str = "fp16",
    artifact_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LearnedInt8WireCodecArtifact:
    artifact = build_learned_int8_wire_codec_artifact(
        codec_results_path,
        region_id=region_id,
        fallback_adapter_id=fallback_adapter_id,
        artifact_id=artifact_id,
        metadata=metadata,
    )
    artifact.write_json(output_path)
    return artifact


def load_learned_int8_wire_codec(
    codec_results_path: str | Path,
    *,
    region_id: str = "shared",
) -> LearnedInt8WireCodec:
    source_path = str(codec_results_path)
    codec_results = json.loads(Path(codec_results_path).read_text(encoding="utf-8"))
    shared_payload = None
    for row in codec_results["metadata"]["learned_region_payloads"]:
        if row.get("region_id") == region_id:
            shared_payload = row["learned_parameter_payload"]
            break
    if shared_payload is None:
        raise RuntimeError(f"no {region_id!r} codec payload in codec results")
    return LearnedInt8WireCodec(
        source_path=source_path,
        region_id=region_id,
        parameter_hash=shared_payload.get("parameter_hash"),
        parameter_count=_optional_int(shared_payload.get("parameter_count")),
        low_rank_rank=_optional_int(shared_payload.get("low_rank_rank")),
        gain=float(shared_payload["gain"]),
        bias=np.asarray(shared_payload["bias"], dtype=np.float32),
        low_rank_down=_optional_float_array(shared_payload.get("low_rank_down")),
        low_rank_up=_optional_float_array(shared_payload.get("low_rank_up")),
        residual_scale=float(shared_payload.get("residual_scale", 1.0)),
        low_rank_residual_scale=_optional_float(
            shared_payload.get("low_rank_residual_scale")
        ),
        mlp_residual_scale=_optional_float(shared_payload.get("mlp_residual_scale")),
        boundary_compression_strategy=shared_payload.get("boundary_compression_strategy"),
        mlp_rank=_optional_int(shared_payload.get("mlp_rank")),
        mlp_encoder=_optional_float_array(shared_payload.get("mlp_encoder")),
        mlp_latent_bias=_optional_float_array(shared_payload.get("mlp_latent_bias")),
        mlp_decoder=_optional_float_array(shared_payload.get("mlp_decoder")),
        ladder_rank=_optional_int(shared_payload.get("ladder_rank")),
        ladder_expert_count=_optional_int(shared_payload.get("ladder_expert_count")),
        ladder_low_rank_downs=tuple(
            np.asarray(item, dtype=np.float32)
            for item in shared_payload.get("ladder_low_rank_downs") or []
        ),
        ladder_low_rank_ups=tuple(
            np.asarray(item, dtype=np.float32)
            for item in shared_payload.get("ladder_low_rank_ups") or []
        ),
        ladder_gate_weights=_ladder_gate_weights_from_payload(shared_payload),
        sparse_count=_optional_int(shared_payload.get("sparse_count")),
        sparse_indices=_optional_int_array(shared_payload.get("sparse_indices")),
        sparse_values=_optional_float_array(shared_payload.get("sparse_values")),
    )


def _maybe_apply_lossless_payload_layer(
    result: BoundaryWireEncodedPayload,
    *,
    lossless_payload_codec: str | None,
) -> BoundaryWireEncodedPayload:
    """Losslessly recompress an encoded payload when requested and eligible.

    Byte-exact: the decode side inflates before its own decoder runs, so no
    fidelity gate is touched. Skips ineligible (fp16/identity) strategies and
    fails closed to the original bytes when compression does not shrink.
    """
    if lossless_payload_codec is None:
        return result
    if not strategy_is_lossless_eligible(result.accounting_strategy):
        return result
    compressed, header_delta = compress_wire_payload(
        result.payload,
        codec=lossless_payload_codec,
    )
    header_fields = dict(result.header_fields)
    header_fields.update(header_delta)
    header_fields["nbytes"] = len(compressed)
    header_fields["lossless_pre_compression_nbytes"] = len(result.payload)
    return BoundaryWireEncodedPayload(
        payload=compressed,
        header_fields=header_fields,
        accounting_strategy=result.accounting_strategy,
    )


def encode_boundary_wire_payload(
    array: np.ndarray,
    *,
    boundary_adapter_strategy: str = "identity",
    tensor_name: str = "hidden_states",
    tensor_metadata: dict[str, Any] | None = None,
    sparse_codebook_base_bits: int | None = None,
    sparse_codebook_corrections_per_row: int | None = None,
    sparse_codebook_values: list[float] | np.ndarray | None = None,
    trainable_autoencoder_basis_artifact: dict[str, Any] | None = None,
    trainable_autoencoder_source_strategy: str = "int8_symmetric",
    trainable_autoencoder_context: dict[str, Any] | None = None,
    trainable_autoencoder_allow_source_only_fallback: bool = False,
    lossless_payload_codec: str | None = None,
) -> BoundaryWireEncodedPayload:
    if boundary_adapter_strategy == LEARNED_INT8_BOUNDARY_ADAPTER_ID:
        result = encode_learned_int8_wire_payload(array, tensor_name=tensor_name)
    elif boundary_adapter_strategy == SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID:
        if sparse_codebook_base_bits is None:
            raise ValueError("sparse-codebook payload requires base bits")
        if sparse_codebook_corrections_per_row is None:
            raise ValueError("sparse-codebook payload requires corrections per row")
        if sparse_codebook_values is None:
            raise ValueError("sparse-codebook payload requires codebook values")
        result = encode_sparse_codebook_wire_payload(
            array,
            tensor_name=tensor_name,
            base_bits=sparse_codebook_base_bits,
            sparse_corrections_per_row=sparse_codebook_corrections_per_row,
            codebook_values=sparse_codebook_values,
        )
    elif boundary_adapter_strategy == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID:
        if trainable_autoencoder_basis_artifact is None:
            raise ValueError("trainable autoencoder wire payload requires a basis artifact")
        result = encode_trainable_autoencoder_wire_payload(
            array,
            basis_artifact=trainable_autoencoder_basis_artifact,
            source_strategy=trainable_autoencoder_source_strategy,
            context=trainable_autoencoder_context,
            allow_source_only_fallback=trainable_autoencoder_allow_source_only_fallback,
            tensor_name=tensor_name,
            tensor_metadata=tensor_metadata,
        )
    else:
        strategy = normalize_boundary_adapter_strategy(boundary_adapter_strategy)
        tensor = TensorPayload.from_numpy(
            np.ascontiguousarray(array, dtype=np.float32),
            name=tensor_name,
            metadata=dict(tensor_metadata or {}),
        )
        encoded_tensor, boundary_adapter = encode_payload_for_boundary(
            tensor,
            strategy=strategy,
        )
        payload = encoded_tensor.raw_bytes()
        result = BoundaryWireEncodedPayload(
            payload=payload,
            header_fields={
                "schema_version": BOUNDARY_WIRE_ENCODED_PAYLOAD_SCHEMA_VERSION,
                "shape": list(encoded_tensor.shape),
                "dtype": encoded_tensor.dtype,
                "tensor_name": encoded_tensor.name,
                "tensor_metadata": encoded_tensor.metadata,
                "nbytes": len(payload),
                "boundary_adapter_strategy": strategy,
                "boundary_adapter": boundary_adapter,
            },
            accounting_strategy=strategy,
        )
    return _maybe_apply_lossless_payload_layer(
        result,
        lossless_payload_codec=lossless_payload_codec,
    )


def encode_learned_int8_wire_payload(
    array: np.ndarray,
    *,
    tensor_name: str = "hidden_states",
) -> BoundaryWireEncodedPayload:
    original = np.ascontiguousarray(array, dtype=np.float32)
    vector = original.reshape(-1)
    max_abs = max(float(np.abs(vector).max()), 1.0e-6)
    scale = max_abs / 127.0
    quantized = np.clip(np.round(vector / scale), -127, 127).astype(np.int8)
    payload = np.asarray([scale], dtype="<f4").tobytes() + quantized.tobytes(
        order="C"
    )
    original_raw_bytes = int(original.nbytes)
    encoded_raw_bytes = int(len(payload))
    return BoundaryWireEncodedPayload(
        payload=payload,
        header_fields={
            "schema_version": BOUNDARY_WIRE_ENCODED_PAYLOAD_SCHEMA_VERSION,
            "shape": [int(quantized.size)],
            "original_shape": [int(dim) for dim in original.shape],
            "dtype": "int8",
            "tensor_name": tensor_name,
            "tensor_metadata": {
                "boundary_adapter_strategy": LEARNED_INT8_BOUNDARY_ADAPTER_ID,
                "encoded_payload_kind": LEARNED_INT8_WIRE_PAYLOAD_KIND,
                "scale_dtype": "float32_le",
                "architecture_neutral_boundary_codec": True,
            },
            "nbytes": encoded_raw_bytes,
            "boundary_adapter_strategy": LEARNED_INT8_BOUNDARY_ADAPTER_ID,
            "boundary_adapter": {
                "adapter_id": LEARNED_INT8_BOUNDARY_ADAPTER_ID,
                "adapter_kind": "learned_int8_wire_codec",
                "original_raw_bytes": original_raw_bytes,
                "encoded_raw_bytes": encoded_raw_bytes,
                "raw_byte_ratio": (
                    encoded_raw_bytes / original_raw_bytes
                    if original_raw_bytes
                    else 1.0
                ),
                "encoded_payload_kind": LEARNED_INT8_WIRE_PAYLOAD_KIND,
                "production_runtime_claimed": False,
            },
        },
        accounting_strategy=LEARNED_INT8_BOUNDARY_ADAPTER_ID,
    )


def encode_sparse_codebook_wire_payload(
    array: np.ndarray,
    *,
    base_bits: int,
    sparse_corrections_per_row: int,
    codebook_values: list[float] | np.ndarray,
    tensor_name: str = "hidden_states",
) -> BoundaryWireEncodedPayload:
    original = np.ascontiguousarray(array, dtype=np.float32)
    encoded = encode_sparse_codebook_components(
        original,
        base_bits=base_bits,
        sparse_corrections_per_row=sparse_corrections_per_row,
        codebook_values=codebook_values,
    )
    metadata = dict(encoded.metadata)
    original_raw_bytes = int(original.nbytes)
    encoded_raw_bytes = int(len(encoded.payload))
    return BoundaryWireEncodedPayload(
        payload=encoded.payload,
        header_fields={
            "schema_version": BOUNDARY_WIRE_ENCODED_PAYLOAD_SCHEMA_VERSION,
            "shape": [encoded_raw_bytes],
            "original_shape": metadata["original_shape"],
            "dtype": "uint8",
            "tensor_name": tensor_name,
            "tensor_metadata": {
                "boundary_adapter_strategy": SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID,
                "encoded_payload_kind": SPARSE_CODEBOOK_WIRE_PAYLOAD_KIND,
                "wire_codec_schema_version": SPARSE_CODEBOOK_WIRE_CODEC_SCHEMA_VERSION,
                "architecture_neutral_boundary_codec": True,
                **metadata,
            },
            "nbytes": encoded_raw_bytes,
            "boundary_adapter_strategy": SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID,
            "boundary_adapter": {
                "adapter_id": SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID,
                "adapter_kind": "sparse_codebook_wire_codec",
                "original_raw_bytes": original_raw_bytes,
                "encoded_raw_bytes": encoded_raw_bytes,
                "raw_byte_ratio": (
                    encoded_raw_bytes / original_raw_bytes
                    if original_raw_bytes
                    else 1.0
                ),
                "encoded_payload_kind": SPARSE_CODEBOOK_WIRE_PAYLOAD_KIND,
                "base_bits": metadata["base_bits"],
                "sparse_corrections_per_row": metadata[
                    "sparse_corrections_per_row"
                ],
                "codebook_size": metadata["codebook_size"],
                "codebook_hash": metadata["codebook_hash"],
                "exact_replay_passed_claimed": False,
                "planner_selectable_claimed": False,
                "production_runtime_claimed": False,
            },
        },
        accounting_strategy=SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID,
    )


def encode_trainable_autoencoder_wire_payload(
    array: np.ndarray,
    *,
    basis_artifact: dict[str, Any],
    source_strategy: str = "int8_symmetric",
    context: dict[str, Any] | None = None,
    allow_source_only_fallback: bool = False,
    tensor_name: str = "hidden_states",
    tensor_metadata: dict[str, Any] | None = None,
) -> BoundaryWireEncodedPayload:
    original = np.ascontiguousarray(array, dtype=np.float32)
    tensor = TensorPayload.from_numpy(
        original,
        name=tensor_name,
        metadata=dict(tensor_metadata or {}),
    )
    encoded_tensor, boundary_adapter = encode_trainable_autoencoder_source_latent_payload(
        tensor,
        basis_artifact=basis_artifact,
        source_strategy=normalize_boundary_adapter_strategy(source_strategy),
        context=context,
        allow_source_only_fallback=allow_source_only_fallback,
    )
    payload = encoded_tensor.raw_bytes()
    return BoundaryWireEncodedPayload(
        payload=payload,
        header_fields={
            "schema_version": BOUNDARY_WIRE_ENCODED_PAYLOAD_SCHEMA_VERSION,
            "shape": list(encoded_tensor.shape),
            "original_shape": [int(dim) for dim in original.shape],
            "dtype": encoded_tensor.dtype,
            "tensor_name": encoded_tensor.name,
            "tensor_metadata": encoded_tensor.metadata,
            "nbytes": len(payload),
            "boundary_adapter_strategy": TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
            "boundary_adapter": boundary_adapter,
            "trainable_autoencoder_basis_hash": basis_artifact.get("basis_hash"),
            "trainable_autoencoder_source_strategy": source_strategy,
            "trainable_autoencoder_source_only_fallback_allowed": bool(
                allow_source_only_fallback
            ),
        },
        accounting_strategy=TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
    )


def decode_boundary_wire_payload(
    header: dict[str, Any],
    payload: bytes,
    *,
    learned_int8_codec: LearnedInt8WireCodec | None = None,
    sparse_codebook_values: list[float] | np.ndarray | None = None,
    trainable_autoencoder_basis_artifact: dict[str, Any] | None = None,
) -> np.ndarray:
    # Inflate a lossless-recompressed payload BEFORE any strategy decoder runs,
    # so the decoders receive the exact pre-lossless bytes (byte-identical).
    payload = decompress_wire_payload(payload, header=header)
    if header.get("boundary_adapter_strategy") == LEARNED_INT8_BOUNDARY_ADAPTER_ID:
        return decode_learned_int8_wire_payload(
            header,
            payload,
            codec=learned_int8_codec,
        )
    tensor_metadata = (
        header.get("tensor_metadata")
        if isinstance(header.get("tensor_metadata"), dict)
        else {}
    )
    if (
        header.get("boundary_adapter_strategy") == SPARSE_CODEBOOK_BOUNDARY_ADAPTER_ID
        or tensor_metadata.get("encoded_payload_kind") == SPARSE_CODEBOOK_WIRE_PAYLOAD_KIND
    ):
        return decode_sparse_codebook_wire_payload(
            header,
            payload,
            codebook_values=sparse_codebook_values,
        )
    if (
        header.get("boundary_adapter_strategy") == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID
        or tensor_metadata.get("encoded_payload_kind")
        == TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND
    ):
        return decode_trainable_autoencoder_wire_payload(
            header,
            payload,
            basis_artifact=trainable_autoencoder_basis_artifact,
        )

    dtype = str(header.get("dtype") or "float32")
    shape = [int(dim) for dim in header["shape"]]
    tensor = TensorPayload.from_raw_bytes(
        payload,
        dtype=dtype,  # type: ignore[arg-type]
        shape=shape,
        name=str(header.get("tensor_name") or "hidden_states"),
        metadata=dict(header.get("tensor_metadata") or {}),
    )
    decoded, _ = decode_payload_from_boundary(tensor)
    return np.ascontiguousarray(decoded.to_numpy(), dtype=np.float32)


def decode_trainable_autoencoder_wire_payload(
    header: dict[str, Any],
    payload: bytes,
    *,
    basis_artifact: dict[str, Any] | None,
) -> np.ndarray:
    if basis_artifact is None:
        raise ValueError("trainable autoencoder wire payload requires a basis artifact")
    dtype = str(header.get("dtype") or "int8")
    shape = [int(dim) for dim in header["shape"]]
    tensor = TensorPayload.from_raw_bytes(
        payload,
        dtype=dtype,  # type: ignore[arg-type]
        shape=shape,
        name=str(header.get("tensor_name") or "hidden_states"),
        metadata=dict(header.get("tensor_metadata") or {}),
    )
    decoded, _ = decode_trainable_autoencoder_source_latent_payload(
        tensor,
        basis_artifact=basis_artifact,
    )
    return np.ascontiguousarray(decoded.to_numpy(), dtype=np.float32)


def decode_sparse_codebook_wire_payload(
    header: dict[str, Any],
    payload: bytes,
    *,
    codebook_values: list[float] | np.ndarray | None,
) -> np.ndarray:
    metadata = (
        header.get("tensor_metadata")
        if isinstance(header.get("tensor_metadata"), dict)
        else {}
    )
    return decode_sparse_codebook_components(
        metadata,
        payload,
        codebook_values=codebook_values,
    )


def apply_learned_int8_wire_codec_to_last_position(
    array: np.ndarray,
    *,
    codec: LearnedInt8WireCodec,
) -> np.ndarray:
    original = np.ascontiguousarray(array, dtype=np.float32)
    if original.ndim != 3 or original.shape[0] != 1:
        raise ValueError(
            "learned_int8 local fallback expects a single-stream "
            "[1, positions, hidden] activation tensor"
        )
    out = original.copy()
    last_position = out[:, -1:, :]
    encoded = encode_learned_int8_wire_payload(last_position)
    decoded = decode_learned_int8_wire_payload(
        encoded.header_fields,
        encoded.payload,
        codec=codec,
    )
    out[:, -1:, :] = decoded
    return out


def decode_learned_int8_wire_payload(
    header: dict[str, Any],
    payload: bytes,
    *,
    codec: LearnedInt8WireCodec | None,
) -> np.ndarray:
    if codec is None:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} payload received but no "
            "wire codec is configured"
        )
    if len(payload) < 5:
        raise ValueError(f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} payload is too short")
    original_shape = [
        int(dim) for dim in header.get("original_shape") or header["shape"]
    ]
    element_count = int(np.prod(original_shape))
    expected = 4 + element_count
    if len(payload) != expected:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} payload length {len(payload)} "
            f"does not match expected {expected}"
        )
    scale = float(np.frombuffer(payload[:4], dtype="<f4", count=1)[0])
    quantized = np.frombuffer(payload[4:], dtype=np.int8, count=element_count)
    dequantized = quantized.astype(np.float32) * scale
    decoded = codec.decode_dequantized(dequantized)
    return np.ascontiguousarray(decoded.reshape(original_shape), dtype=np.float32)


def _low_rank_residual(
    vector: np.ndarray,
    *,
    down: np.ndarray | None,
    up: np.ndarray | None,
    width: int,
) -> np.ndarray:
    if down is None or up is None:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} low-rank payload is incomplete"
        )
    if down.ndim != 2 or up.ndim != 2:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} low-rank payload must be 2D"
        )
    if down.shape[0] != width or up.shape[1] != width:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} low-rank width mismatch"
        )
    if down.shape[1] != up.shape[0]:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} low-rank rank mismatch"
        )
    return np.asarray((vector @ down) @ up, dtype=np.float32)


def _mlp_residual(
    vector: np.ndarray,
    *,
    encoder: np.ndarray | None,
    latent_bias: np.ndarray | None,
    decoder: np.ndarray | None,
    width: int,
) -> np.ndarray:
    if encoder is None or latent_bias is None or decoder is None:
        raise ValueError(f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} MLP payload is incomplete")
    if encoder.ndim != 2 or decoder.ndim != 2 or latent_bias.ndim != 1:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} MLP payload has invalid rank"
        )
    if encoder.shape[0] != width or decoder.shape[1] != width:
        raise ValueError(f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} MLP width mismatch")
    if encoder.shape[1] != decoder.shape[0] or encoder.shape[1] != latent_bias.shape[0]:
        raise ValueError(f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} MLP latent mismatch")
    latent = np.tanh((vector @ encoder) + latent_bias).astype(np.float32)
    return np.asarray(latent @ decoder, dtype=np.float32)


def _ladder_residual(
    vector: np.ndarray,
    *,
    downs: tuple[np.ndarray, ...],
    ups: tuple[np.ndarray, ...],
    gate_weights: np.ndarray | None,
    width: int,
) -> np.ndarray:
    if not downs or not ups or gate_weights is None:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} residual-ladder payload is incomplete"
        )
    if len(downs) != len(ups) or len(downs) != int(gate_weights.shape[0]):
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} residual-ladder expert mismatch"
        )
    residual = np.zeros(width, dtype=np.float32)
    for gate_weight, down, up in zip(gate_weights, downs, ups, strict=True):
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError(
                f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} residual-ladder experts must be 2D"
            )
        if down.shape[0] != width or up.shape[1] != width:
            raise ValueError(
                f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} residual-ladder width mismatch"
            )
        if down.shape[1] != up.shape[0]:
            raise ValueError(
                f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} residual-ladder rank mismatch"
            )
        residual = residual + np.float32(gate_weight) * np.asarray(
            (vector @ down) @ up,
            dtype=np.float32,
        )
    return residual


def _sparse_residual(
    *,
    indices: np.ndarray | None,
    values: np.ndarray | None,
    width: int,
) -> np.ndarray:
    if indices is None or values is None:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} sparse payload is incomplete"
        )
    if indices.ndim != 1 or values.ndim != 1:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} sparse payload must be 1D"
        )
    if indices.shape[0] != values.shape[0]:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} sparse index/value count mismatch"
        )
    if np.any(indices < 0) or np.any(indices >= width):
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} sparse index out of range"
        )
    residual = np.zeros(width, dtype=np.float32)
    residual[indices.astype(np.int64, copy=False)] = values.astype(
        np.float32,
        copy=False,
    )
    return residual


def _optional_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return None
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} payload has non-finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _optional_int_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.int64)
    if array.size == 0:
        return None
    return np.ascontiguousarray(array, dtype=np.int64)


def _ladder_gate_weights_from_payload(payload: dict[str, Any]) -> np.ndarray | None:
    weights = _optional_float_array(payload.get("ladder_gate_weights"))
    if weights is not None:
        if weights.ndim != 1:
            raise ValueError(
                f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} residual-ladder gates must be 1D"
            )
        return weights
    logits = _optional_float_array(payload.get("ladder_gate_logits"))
    if logits is None:
        return None
    if logits.ndim != 1:
        raise ValueError(
            f"{LEARNED_INT8_BOUNDARY_ADAPTER_ID} residual-ladder logits must be 1D"
        )
    shifted = logits - np.max(logits)
    exp = np.exp(shifted).astype(np.float32)
    return np.ascontiguousarray(exp / np.sum(exp), dtype=np.float32)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
