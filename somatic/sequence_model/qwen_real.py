from __future__ import annotations

import importlib.util
import time
from typing import Any, Awaitable, Callable, Sequence

from pydantic import BaseModel, Field

from somatic.sequence_model.boundary_adapters import normalize_boundary_adapter_strategy
from somatic.sequence_model.boundary_compression.wire_codec_runtime import (
    TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
)
from somatic.sequence_model.interfaces import Precision, ResourceProfile
from somatic.sequence_model.kv_cache import KVCacheChunk, KVCacheHandle, KVCacheRegistry
from somatic.sequence_model.qwen import qwen_manifest_from_config
from somatic.sequence_model.qwen_partition import QwenDistributedPlan, QwenLayerRange, QwenShardManifest, build_qwen_distributed_plan
from somatic.sequence_model.tensors import TensorPayload
from somatic.sequence_model.transformers_qwen import _torch_dtype


class QwenShardTrace(BaseModel):
    shard_id: str
    runtime_id: str
    layer_start: int
    layer_end: int
    input_shape: list[int]
    output_shape: list[int]
    input_bytes: int = Field(default=0, ge=0)
    output_bytes: int = Field(default=0, ge=0)
    input_original_bytes: int | None = Field(default=None, ge=0)
    output_original_bytes: int | None = Field(default=None, ge=0)
    input_encoded_bytes: int | None = Field(default=None, ge=0)
    output_encoded_bytes: int | None = Field(default=None, ge=0)
    request_frame_bytes: int | None = Field(default=None, ge=0)
    response_frame_bytes: int | None = Field(default=None, ge=0)
    boundary_adapter_id: str = "identity"
    boundary_adapter_applied: bool = False
    boundary_adapter_input: dict[str, Any] = Field(default_factory=dict)
    boundary_adapter_output: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: float = Field(ge=0.0)
    route_elapsed_ms: float | None = Field(default=None, ge=0.0)


class QwenLocalForwardResult(BaseModel):
    model_id: str
    logits: TensorPayload
    traces: list[QwenShardTrace]
    generated_token_ids: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QwenCacheForwardResult(BaseModel):
    tensor: TensorPayload
    trace: QwenShardTrace
    cache: KVCacheHandle


class QwenWorkerCacheGenerationResult(BaseModel):
    model_id: str
    sequence_id: str
    prompt_length: int
    generated_token_ids: list[int]
    caches: list[KVCacheHandle]
    traces: list[QwenShardTrace]
    metadata: dict[str, Any] = Field(default_factory=dict)


class QwenWorkerLoadingAccounting(BaseModel):
    """Explicitly separates execution ownership from resident weight ownership."""

    schema_version: str = "qwen-worker-loading-accounting-v0"
    loading_mode: str
    model_id: str
    runtime_id: str
    layer_start: int = Field(ge=0)
    layer_end: int = Field(gt=0)
    total_layers: int = Field(ge=0)
    assigned_layer_count: int = Field(ge=0)
    full_model_parameter_count: int = Field(ge=0)
    full_model_parameter_bytes: int = Field(ge=0)
    decoder_layer_parameter_count: int = Field(ge=0)
    decoder_layer_parameter_bytes: int = Field(ge=0)
    assigned_layer_parameter_count: int = Field(ge=0)
    assigned_layer_parameter_bytes: int = Field(ge=0)
    loaded_parameter_count: int = Field(ge=0)
    loaded_parameter_bytes: int = Field(ge=0)
    unassigned_loaded_parameter_count: int = Field(ge=0)
    unassigned_loaded_parameter_bytes: int = Field(ge=0)
    shard_only_weight_loading_claimed: bool = False
    per_machine_ram_reduction_claimed: bool = False
    proof_boundary: str = (
        "This worker may execute only a layer range, but current accounting must not claim per-machine RAM "
        "reduction unless loaded_parameter_bytes is bounded to the assigned shard plus required edge modules."
    )

    @property
    def assigned_parameter_fraction(self) -> float:
        if self.full_model_parameter_bytes <= 0:
            return 0.0
        return self.assigned_layer_parameter_bytes / self.full_model_parameter_bytes

    @property
    def loaded_parameter_fraction(self) -> float:
        if self.full_model_parameter_bytes <= 0:
            return 0.0
        return self.loaded_parameter_bytes / self.full_model_parameter_bytes

    def summary(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["assigned_parameter_fraction"] = self.assigned_parameter_fraction
        payload["loaded_parameter_fraction"] = self.loaded_parameter_fraction
        return payload


def _qwen_boundary_sequence_metadata(
    sequence_metadata: dict[str, Any] | None,
    *,
    worker_index: int,
    cache: KVCacheHandle,
) -> dict[str, Any]:
    metadata = dict(sequence_metadata or {})
    layer_start = int(cache.layer_range.start)
    layer_end = int(cache.layer_range.end)
    metadata.update(
        {
            "boundary_slot": int(worker_index),
            "boundary_layer_index": layer_end - 1,
            "boundary_layer_start": layer_start,
            "boundary_layer_end": layer_end,
            "split_layer_end": layer_end,
            "qwen_layer_start": layer_start,
            "qwen_layer_end": layer_end,
        }
    )
    return metadata


def _required_member_string(member: dict[str, Any], name: str) -> str:
    value = member.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"coalesced Qwen decode member requires non-empty `{name}`")
    return value


def _required_member_int(member: dict[str, Any], name: str) -> int:
    if name not in member:
        raise ValueError(f"coalesced Qwen decode member requires `{name}`")
    try:
        resolved = int(member[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"coalesced Qwen decode member `{name}` must be an integer") from exc
    if resolved < 0:
        raise ValueError(f"coalesced Qwen decode member `{name}` must be non-negative")
    return resolved


class QwenLocalLayerShard:
    def __init__(self, *, manifest: QwenShardManifest, layers: Sequence[Any]) -> None:
        if manifest.layer_range is None:
            raise ValueError("local Qwen layer shard requires a layer range")
        if len(layers) != manifest.layer_range.count:
            raise ValueError("layer module count must match shard layer range")
        self.manifest = manifest
        self.layers = list(layers)

    def forward(
        self,
        hidden_states: Any,
        *,
        position_ids: Any,
        position_embeddings: Any,
        attention_mask: Any = None,
        use_cache: bool = False,
        past_key_values: Any = None,
    ) -> Any:
        for layer in self.layers:
            output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            hidden_states = output[0] if isinstance(output, tuple) else output
        return hidden_states


class QwenWorkerLayerRuntime:
    """Worker-owned Qwen layer range exposed behind the generic sequence-worker boundary."""

    def __init__(
        self,
        *,
        manifest: QwenShardManifest,
        layers: Sequence[Any],
        rotary_emb: Any,
        loading_accounting: QwenWorkerLoadingAccounting | None = None,
        device: str = "cpu",
    ) -> None:
        self.manifest = manifest
        self.shard = QwenLocalLayerShard(manifest=manifest, layers=layers)
        self.rotary_emb = rotary_emb
        self.loading_accounting = loading_accounting
        self.device = device
        self.cache_registry = KVCacheRegistry()
        self._hf_caches: dict[str, Any] = {}

    @classmethod
    def from_model(
        cls,
        *,
        model: Any,
        manifest: QwenShardManifest,
        device: str = "cpu",
        loading_mode: str = "shared_model_layer_slice",
    ) -> "QwenWorkerLayerRuntime":
        backbone = _qwen_backbone(model)
        if manifest.layer_range is None:
            raise ValueError("Qwen worker runtime requires a layer range")
        layers = list(backbone.layers[manifest.layer_range.start : manifest.layer_range.end])
        loading_accounting = _qwen_worker_loading_accounting(
            model=model,
            manifest=manifest,
            loading_mode=loading_mode,
        )
        manifest = manifest.model_copy(
            update={
                "metadata": {
                    **manifest.metadata,
                    "qwen_weight_loading_accounting_schema": loading_accounting.schema_version,
                    "qwen_weight_loading_mode": loading_accounting.loading_mode,
                    "qwen_full_model_parameter_count": loading_accounting.full_model_parameter_count,
                    "qwen_full_model_parameter_bytes": loading_accounting.full_model_parameter_bytes,
                    "qwen_assigned_layer_parameter_count": loading_accounting.assigned_layer_parameter_count,
                    "qwen_assigned_layer_parameter_bytes": loading_accounting.assigned_layer_parameter_bytes,
                    "qwen_loaded_parameter_count": loading_accounting.loaded_parameter_count,
                    "qwen_loaded_parameter_bytes": loading_accounting.loaded_parameter_bytes,
                    "qwen_unassigned_loaded_parameter_count": loading_accounting.unassigned_loaded_parameter_count,
                    "qwen_unassigned_loaded_parameter_bytes": loading_accounting.unassigned_loaded_parameter_bytes,
                    "qwen_shard_only_weight_loading_claimed": loading_accounting.shard_only_weight_loading_claimed,
                    "qwen_per_machine_ram_reduction_claimed": loading_accounting.per_machine_ram_reduction_claimed,
                },
            }
        )
        return cls(
            manifest=manifest,
            layers=layers,
            rotary_emb=backbone.rotary_emb,
            loading_accounting=loading_accounting,
            device=device,
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        runtime_id: str,
        layer_start: int,
        layer_end: int,
        precision: Precision = "fp32",
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> "QwenWorkerLayerRuntime":
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError("missing optional dependency: transformers")
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        if layer_end <= layer_start:
            raise ValueError("Qwen shard layer_end must be greater than layer_start")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
        total_layers = int(getattr(config, "num_hidden_layers", 0))
        hidden_size = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)) or 0)
        if layer_start < 0 or layer_end > total_layers:
            raise ValueError(f"Qwen shard layer range [{layer_start}, {layer_end}) is outside model layer count {total_layers}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=_torch_dtype(torch, device=device, precision=precision),
            device_map=None,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model.to(device)
        model.eval()
        manifest = QwenShardManifest(
            shard_id=f"{model_id}:shard-{layer_start}-{layer_end}",
            runtime_id=runtime_id,
            model_id=model_id,
            weight_uri=model_id,
            precision=precision,
            device=device,
            layer_range=QwenLayerRange(start=layer_start, end=layer_end),
            roles=["layer_range"],
            metadata={
                "loader": "qwen-worker-from-pretrained-v0",
                "total_layers": total_layers,
                "hidden_size": hidden_size,
                "local_files_only": local_files_only,
            },
        )
        return cls.from_model(
            model=model,
            manifest=manifest,
            device=device,
            loading_mode="full_model_then_layer_slice",
        )

    @classmethod
    def from_pretrained_shard(
        cls,
        *,
        model_id: str,
        runtime_id: str,
        layer_start: int,
        layer_end: int,
        precision: Precision = "fp32",
        device: str = "cpu",
        local_files_only: bool = True,
        include_embed: bool = False,
        include_final_norm: bool = False,
        include_lm_head: bool = False,
    ) -> "QwenWorkerLayerRuntime":
        """Load ONLY this worker's layer-slice weights into memory.

        The whole point of splitting a model across machines is to run a model
        too large for any single machine. `from_pretrained` loads the entire
        model on every worker (defeating that); this builds the architecture
        on the meta device (zero memory) and materialises only the parameters
        this worker actually runs — its transformer layers, plus optionally
        the embedding / final norm / lm_head when this worker owns them. Peak
        memory is the shard's weights, not the whole model.
        """
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError("missing optional dependency: transformers")
        import json as _json
        import os as _os

        import torch
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM

        if layer_end <= layer_start:
            raise ValueError("Qwen shard layer_end must be greater than layer_start")
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=local_files_only
        )
        total_layers = int(getattr(config, "num_hidden_layers", 0))
        hidden_size = int(
            getattr(config, "hidden_size", getattr(config, "n_embd", 0)) or 0
        )
        if layer_start < 0 or layer_end > total_layers:
            raise ValueError(
                f"Qwen shard layer range [{layer_start}, {layer_end}) is outside "
                f"model layer count {total_layers}"
            )
        snapshot_dir = snapshot_download(
            model_id,
            local_files_only=local_files_only,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"],
        )
        dtype = _torch_dtype(torch, device=device, precision=precision)

        # tensor-name -> shard-file weight map (single-file models have no index)
        index_path = _os.path.join(snapshot_dir, "model.safetensors.index.json")
        if _os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as handle:
                weight_map = _json.load(handle)["weight_map"]
        else:
            single = _os.path.join(snapshot_dir, "model.safetensors")
            if not _os.path.exists(single):
                raise FileNotFoundError(
                    f"no safetensors weights found for shard loading under {snapshot_dir}"
                )
            with safe_open(single, framework="pt", device="cpu") as handle:
                weight_map = {key: "model.safetensors" for key in handle.keys()}

        needed: set[str] = set()
        layer_prefixes = tuple(
            f"model.layers.{i}." for i in range(layer_start, layer_end)
        )
        for key in weight_map:
            if key.startswith(layer_prefixes):
                needed.add(key)
        if include_embed and "model.embed_tokens.weight" in weight_map:
            needed.add("model.embed_tokens.weight")
        if include_final_norm and "model.norm.weight" in weight_map:
            needed.add("model.norm.weight")
        if include_lm_head and "lm_head.weight" in weight_map:
            needed.add("lm_head.weight")
        if not needed:
            raise ValueError(
                "shard weight selection is empty — check the layer range and roles"
            )

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.eval()

        by_file: dict[str, list[str]] = {}
        for key in needed:
            by_file.setdefault(weight_map[key], []).append(key)
        for file_name, keys in by_file.items():
            with safe_open(
                _os.path.join(snapshot_dir, file_name),
                framework="pt",
                device="cpu",
            ) as handle:
                for key in keys:
                    set_module_tensor_to_device(
                        model,
                        key,
                        device,
                        value=handle.get_tensor(key).to(dtype),
                        dtype=dtype,
                    )

        manifest = QwenShardManifest(
            shard_id=f"{model_id}:shard-{layer_start}-{layer_end}",
            runtime_id=runtime_id,
            model_id=model_id,
            weight_uri=model_id,
            precision=precision,
            device=device,
            layer_range=QwenLayerRange(start=layer_start, end=layer_end),
            roles=["layer_range"],
            metadata={
                "loader": "qwen-worker-shard-only-v0",
                "total_layers": total_layers,
                "hidden_size": hidden_size,
                "local_files_only": local_files_only,
                "shard_only_weight_load": True,
                "materialized_tensor_count": len(needed),
            },
        )
        return cls.from_model(
            model=model,
            manifest=manifest,
            device=device,
            loading_mode="shard_only_weight_load",
        )

    def forward_tensor(self, tensor: TensorPayload) -> tuple[TensorPayload, QwenShardTrace]:
        return self.forward_array(tensor.to_numpy(), name=tensor.name, metadata=tensor.metadata)

    def forward_array(
        self,
        array: Any,
        *,
        name: str = "hidden_states",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TensorPayload, QwenShardTrace]:
        import torch

        self._eval()
        hidden_states = torch.from_numpy(array).to(self.device)
        layer_range = self.manifest.layer_range
        if layer_range is None:
            raise ValueError("Qwen worker runtime requires a layer range")
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        input_shape = list(hidden_states.shape)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.shard.forward(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=None,
                use_cache=False,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        output_tensor = TensorPayload.from_numpy(
            output.detach().cpu().float().numpy(),
            name=name,
            metadata={
                **(metadata or {}),
                "qwen_shard_id": self.manifest.shard_id,
                "qwen_runtime_id": self.manifest.runtime_id,
                "qwen_layer_start": layer_range.start,
                "qwen_layer_end": layer_range.end,
            },
        )
        trace = QwenShardTrace(
            shard_id=self.manifest.shard_id,
            runtime_id=self.manifest.runtime_id,
            layer_start=layer_range.start,
            layer_end=layer_range.end,
            input_shape=input_shape,
            output_shape=list(output.shape),
            input_bytes=int(getattr(array, "nbytes", 0)),
            output_bytes=output_tensor.byte_size(),
            elapsed_ms=elapsed_ms,
        )
        return output_tensor, trace

    def create_cache(self, *, sequence_id: str, ttl_seconds: float | None = None) -> KVCacheHandle:
        from transformers.cache_utils import DynamicCache

        layer_range = self.manifest.layer_range
        if layer_range is None:
            raise ValueError("Qwen worker runtime requires a layer range")
        handle = self.cache_registry.create(
            sequence_id=sequence_id,
            shard_id=self.manifest.shard_id,
            runtime_id=self.manifest.runtime_id,
            layer_range=layer_range,
            precision=self.manifest.precision,
            ttl_seconds=ttl_seconds,
        )
        self._hf_caches[handle.cache_id] = DynamicCache()
        return handle

    def prefill_tensor(self, *, tensor: TensorPayload, cache_id: str, sequence_id: str, position_start: int = 0) -> QwenCacheForwardResult:
        return self.prefill_array(
            tensor.to_numpy(),
            name=tensor.name,
            metadata=tensor.metadata,
            cache_id=cache_id,
            sequence_id=sequence_id,
            position_start=position_start,
        )

    def prefill_array(
        self,
        array: Any,
        *,
        name: str = "hidden_states",
        metadata: dict[str, Any] | None = None,
        cache_id: str,
        sequence_id: str,
        position_start: int = 0,
    ) -> QwenCacheForwardResult:
        handle = self.cache_registry.require(cache_id, sequence_id=sequence_id, shard_id=self.manifest.shard_id)
        if position_start != 0 or handle.current_position != 0:
            raise ValueError("Qwen shard prefill must start from an empty cache at position 0")
        output_tensor, trace = self._forward_cached_array(
            array,
            name=name,
            metadata=metadata,
            cache_id=cache_id,
            position_start=position_start,
        )
        chunk = self._kv_chunk(cache_id=cache_id, sequence_id=sequence_id, position_start=position_start, position_end=position_start + output_tensor.shape[1])
        handle = self.cache_registry.prefill(chunk)
        return QwenCacheForwardResult(tensor=output_tensor, trace=trace, cache=handle)

    def decode_tensor(self, *, tensor: TensorPayload, cache_id: str, sequence_id: str, position_start: int) -> QwenCacheForwardResult:
        return self.decode_array(
            tensor.to_numpy(),
            name=tensor.name,
            metadata=tensor.metadata,
            cache_id=cache_id,
            sequence_id=sequence_id,
            position_start=position_start,
        )

    def decode_array(
        self,
        array: Any,
        *,
        name: str = "hidden_states",
        metadata: dict[str, Any] | None = None,
        cache_id: str,
        sequence_id: str,
        position_start: int,
    ) -> QwenCacheForwardResult:
        handle = self.cache_registry.require(cache_id, sequence_id=sequence_id, shard_id=self.manifest.shard_id)
        if position_start != handle.current_position:
            raise ValueError(f"decode position mismatch for {cache_id}: expected {handle.current_position}, got {position_start}")
        output_tensor, trace = self._forward_cached_array(
            array,
            name=name,
            metadata=metadata,
            cache_id=cache_id,
            position_start=position_start,
        )
        position_end = position_start + output_tensor.shape[1]
        chunk = self._kv_chunk(cache_id=cache_id, sequence_id=sequence_id, position_start=position_start, position_end=position_end)
        handle = self.cache_registry.decode(chunk)
        return QwenCacheForwardResult(tensor=output_tensor, trace=trace, cache=handle)

    def decode_arrays_coalesced(
        self,
        array: Any,
        *,
        members: Sequence[dict[str, Any]],
        name: str = "hidden_states",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TensorPayload, list[QwenShardTrace], list[KVCacheHandle]]:
        """Decode one stacked transport frame while preserving one KV cache per member."""
        import numpy as np

        if not members:
            raise ValueError("coalesced Qwen decode requires at least one member")
        if len(getattr(array, "shape", [])) < 1:
            raise ValueError("coalesced Qwen decode tensor must include a member axis")
        if int(array.shape[0]) != len(members):
            raise ValueError(
                "coalesced Qwen decode member count must match tensor batch axis: "
                f"{len(members)} members for shape {list(array.shape)}"
            )
        outputs: list[Any] = []
        traces: list[QwenShardTrace] = []
        caches: list[KVCacheHandle] = []
        for member_index, member in enumerate(members):
            if not isinstance(member, dict):
                raise ValueError("coalesced Qwen decode members must be objects")
            member_metadata = dict(metadata or {})
            raw_member_metadata = member.get("metadata")
            if isinstance(raw_member_metadata, dict):
                member_metadata.update(raw_member_metadata)
            member_metadata.update(
                {
                    "coalesced_transport": "qwen-kv-cache-binary-coalesced-tensor-frame-v1",
                    "coalesced_member_index": member_index,
                    "coalesced_member_count": len(members),
                }
            )
            result = self.decode_array(
                array[member_index : member_index + 1],
                name=name,
                metadata=member_metadata,
                cache_id=_required_member_string(member, "cache_id"),
                sequence_id=_required_member_string(member, "sequence_id"),
                position_start=_required_member_int(member, "position_start"),
            )
            outputs.append(result.tensor.to_numpy())
            traces.append(
                result.trace.model_copy(
                    update={
                        "input_shape": list(array[member_index : member_index + 1].shape),
                    }
                )
            )
            caches.append(result.cache)
        output_array = np.concatenate(outputs, axis=0)
        output_tensor = TensorPayload.from_numpy(
            output_array,
            name=name,
            metadata={
                **(metadata or {}),
                "qwen_shard_id": self.manifest.shard_id,
                "qwen_runtime_id": self.manifest.runtime_id,
                "coalesced_transport": "qwen-kv-cache-binary-coalesced-tensor-frame-v1",
                "coalesced_member_count": len(members),
                "coalesced_cache_isolation_preserved": True,
                "coalesced_transformer_compute_claimed": False,
            },
        )
        return output_tensor, traces, caches

    def release_cache(self, *, cache_id: str, sequence_id: str) -> KVCacheHandle:
        handle = self.cache_registry.release(cache_id, sequence_id=sequence_id, shard_id=self.manifest.shard_id)
        self._hf_caches.pop(cache_id, None)
        return handle

    def truncate_cache(self, *, cache_id: str, sequence_id: str, length: int) -> KVCacheHandle:
        """Roll the KV cache back to `length` positions (fallback retransmission)."""
        hf_cache = self._hf_caches.get(cache_id)
        if hf_cache is None:
            raise KeyError(f"unknown Qwen KV cache {cache_id}")
        handle = self.cache_registry.truncate(
            cache_id,
            sequence_id=sequence_id,
            shard_id=self.manifest.shard_id,
            length=int(length),
        )
        hf_cache.crop(int(length))
        return handle

    def cache_handles(self) -> list[KVCacheHandle]:
        return self.cache_registry.handles()

    def _forward_cached_tensor(self, *, tensor: TensorPayload, cache_id: str, position_start: int) -> tuple[TensorPayload, QwenShardTrace]:
        return self._forward_cached_array(
            tensor.to_numpy(),
            name=tensor.name,
            metadata=tensor.metadata,
            cache_id=cache_id,
            position_start=position_start,
        )

    def _forward_cached_array(
        self,
        array: Any,
        *,
        name: str = "hidden_states",
        metadata: dict[str, Any] | None = None,
        cache_id: str,
        position_start: int,
    ) -> tuple[TensorPayload, QwenShardTrace]:
        import torch

        self._eval()
        cache = self._hf_caches.get(cache_id)
        if cache is None:
            raise KeyError(cache_id)
        hidden_states = torch.from_numpy(array).to(self.device)
        layer_range = self.manifest.layer_range
        if layer_range is None:
            raise ValueError("Qwen worker runtime requires a layer range")
        position_ids = torch.arange(position_start, position_start + hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        input_shape = list(hidden_states.shape)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.shard.forward(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        output_tensor = TensorPayload.from_numpy(
            output.detach().cpu().float().numpy(),
            name=name,
            metadata={
                **(metadata or {}),
                "qwen_shard_id": self.manifest.shard_id,
                "qwen_runtime_id": self.manifest.runtime_id,
                "qwen_layer_start": layer_range.start,
                "qwen_layer_end": layer_range.end,
                "kv_cache_id": cache_id,
                "kv_position_start": position_start,
                "kv_position_end": position_start + output.shape[1],
            },
        )
        trace = QwenShardTrace(
            shard_id=self.manifest.shard_id,
            runtime_id=self.manifest.runtime_id,
            layer_start=layer_range.start,
            layer_end=layer_range.end,
            input_shape=input_shape,
            output_shape=list(output.shape),
            input_bytes=int(getattr(array, "nbytes", 0)),
            output_bytes=output_tensor.byte_size(),
            elapsed_ms=elapsed_ms,
        )
        return output_tensor, trace

    def _kv_chunk(self, *, cache_id: str, sequence_id: str, position_start: int, position_end: int) -> KVCacheChunk:
        import numpy as np

        cache = self._hf_caches.get(cache_id)
        if cache is None:
            raise KeyError(cache_id)
        key_slices = []
        value_slices = []
        for layer_cache in cache.layers:
            if getattr(layer_cache, "keys", None) is None or getattr(layer_cache, "values", None) is None:
                continue
            key_slices.append(layer_cache.keys[:, :, position_start:position_end, :].detach().cpu().float().numpy())
            value_slices.append(layer_cache.values[:, :, position_start:position_end, :].detach().cpu().float().numpy())
        if not key_slices or not value_slices:
            raise ValueError(f"Qwen KV cache {cache_id} has no initialized key/value tensors")
        key = np.stack(key_slices, axis=0)
        value = np.stack(value_slices, axis=0)
        return KVCacheChunk(
            cache_id=cache_id,
            sequence_id=sequence_id,
            shard_id=self.manifest.shard_id,
            position_start=position_start,
            position_end=position_end,
            key=TensorPayload.from_numpy(key.astype("float32"), name="kv_key"),
            value=TensorPayload.from_numpy(value.astype("float32"), name="kv_value"),
        )

    def _eval(self) -> None:
        if hasattr(self.rotary_emb, "eval"):
            self.rotary_emb.eval()
        for layer in self.shard.layers:
            if hasattr(layer, "eval"):
                layer.eval()


class QwenLocalDistributedModel:
    """Local multi-shard Qwen runner for proving real layer-boundary execution before remote sharding."""

    def __init__(
        self,
        *,
        model_id: str,
        plan: QwenDistributedPlan,
        embed_tokens: Any,
        rotary_emb: Any,
        norm: Any,
        lm_head: Any,
        shards: Sequence[QwenLocalLayerShard],
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.plan = plan
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb
        self.norm = norm
        self.lm_head = lm_head
        self.shards = list(shards)
        self.device = device

    @classmethod
    def from_model(
        cls,
        *,
        model: Any,
        plan: QwenDistributedPlan,
        device: str = "cpu",
    ) -> "QwenLocalDistributedModel":
        backbone = _qwen_backbone(model)
        shards: list[QwenLocalLayerShard] = []
        for shard_manifest in plan.shards:
            if shard_manifest.layer_range is None:
                continue
            layers = list(backbone.layers[shard_manifest.layer_range.start : shard_manifest.layer_range.end])
            shards.append(QwenLocalLayerShard(manifest=shard_manifest, layers=layers))
        return cls(
            model_id=plan.model_id,
            plan=plan,
            embed_tokens=backbone.embed_tokens,
            rotary_emb=backbone.rotary_emb,
            norm=backbone.norm,
            lm_head=model.lm_head,
            shards=shards,
            device=device,
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = "Qwen/Qwen3-0.6B",
        resources: Sequence[ResourceProfile],
        precision: Precision = "fp32",
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> "QwenLocalDistributedModel":
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError("missing optional dependency: transformers")
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
        manifest = qwen_manifest_from_config(model_id=model_id, config=config, default_precision=precision)
        plan = build_qwen_distributed_plan(manifest, resources, precision=precision, device=device)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=_torch_dtype(torch, device=device, precision=precision),
            device_map=None,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model.to(device)
        model.eval()
        return cls.from_model(model=model, plan=plan, device=device)

    def forward_input_ids(self, input_ids: Any) -> QwenLocalForwardResult:
        import torch

        self._eval()
        resolved_input_ids = input_ids.to(self.device)
        with torch.no_grad():
            hidden_states = self.embed_tokens(resolved_input_ids)
            traces: list[QwenShardTrace] = []
            for shard in self.shards:
                position_ids = self._position_ids(hidden_states)
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                input_shape = list(hidden_states.shape)
                started = time.perf_counter()
                hidden_states = shard.forward(
                    hidden_states,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    use_cache=False,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                layer_range = shard.manifest.layer_range
                assert layer_range is not None
                traces.append(
                    QwenShardTrace(
                        shard_id=shard.manifest.shard_id,
                        runtime_id=shard.manifest.runtime_id,
                        layer_start=layer_range.start,
                        layer_end=layer_range.end,
                        input_shape=input_shape,
                        output_shape=list(hidden_states.shape),
                        input_bytes=int(hidden_states.numel() * hidden_states.element_size()),
                        output_bytes=int(hidden_states.numel() * hidden_states.element_size()),
                        elapsed_ms=elapsed_ms,
                    )
                )
            logits = self.lm_head(self.norm(hidden_states))
        return QwenLocalForwardResult(
            model_id=self.model_id,
            logits=TensorPayload.from_numpy(logits.detach().cpu().float().numpy(), name="logits"),
            traces=traces,
            metadata={
                "runner": "qwen-local-layer-shards-v0",
                "shard_count": len(self.shards),
                "edge_owner_policy": self.plan.edge_owner_policy,
                "real_qwen_modules": True,
            },
        )

    def generate_token_ids(self, input_ids: Any, *, max_new_tokens: int = 1) -> QwenLocalForwardResult:
        import torch

        generated = input_ids.to(self.device)
        generated_token_ids: list[int] = []
        last_result: QwenLocalForwardResult | None = None
        for _ in range(max_new_tokens):
            last_result = self.forward_input_ids(generated)
            logits = torch.from_numpy(last_result.logits.to_numpy()).to(self.device)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_token_ids.append(int(next_token[0, 0].item()))
            generated = torch.cat([generated, next_token], dim=-1)
        if last_result is None:
            last_result = self.forward_input_ids(generated)
        last_result.generated_token_ids = generated_token_ids
        last_result.metadata["generated_sequence_length"] = int(generated.shape[-1])
        return last_result

    def _position_ids(self, hidden_states: Any) -> Any:
        import torch

        sequence_length = int(hidden_states.shape[1])
        return torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)

    def _eval(self) -> None:
        for module in [self.embed_tokens, self.rotary_emb, self.norm, self.lm_head, *[layer for shard in self.shards for layer in shard.layers]]:
            if hasattr(module, "eval"):
                module.eval()


class QwenWorkerCacheCoordinator:
    """Coordinator-owned Qwen edges with worker-owned layer shards and KV caches."""

    def __init__(
        self,
        *,
        model_id: str,
        workers: Sequence[Any],
        embed_tokens: Any,
        norm: Any,
        lm_head: Any,
        device: str = "cpu",
    ) -> None:
        if not workers:
            raise ValueError("Qwen worker cache coordinator requires workers")
        self.model_id = model_id
        self.workers = list(workers)
        self.embed_tokens = embed_tokens
        self.norm = norm
        self.lm_head = lm_head
        self.device = device

    async def generate_token_ids(
        self,
        input_ids: Any,
        *,
        max_new_tokens: int,
        sequence_id: str,
        ttl_seconds: float | None = None,
        release_caches: bool = True,
        tensor_transport: str = "json",
        boundary_adapter_strategy: str = "identity",
        reference_token_ids: list[int] | None = None,
        sequence_metadata: dict[str, Any] | None = None,
        cache_update_callback: Callable[[str, list[KVCacheHandle], list[QwenShardTrace]], Awaitable[None]] | None = None,
    ) -> QwenWorkerCacheGenerationResult:
        import torch

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if tensor_transport not in {"json", "binary"}:
            raise ValueError("tensor_transport must be `json` or `binary`")
        resolved_boundary_adapter_strategy = _normalize_qwen_boundary_wire_strategy(
            boundary_adapter_strategy
        )
        if tensor_transport != "binary" and resolved_boundary_adapter_strategy != "identity":
            raise ValueError("boundary adapters require binary tensor transport")
        self._eval()
        resolved_input_ids = input_ids.to(self.device)
        prompt_length = int(resolved_input_ids.shape[-1])
        caches = [await worker.create_qwen_cache(sequence_id=sequence_id, ttl_seconds=ttl_seconds) for worker in self.workers]
        traces: list[QwenShardTrace] = []
        await self._emit_cache_update(cache_update_callback, "created", caches, traces)

        with torch.no_grad():
            prompt_hidden = self.embed_tokens(resolved_input_ids).detach().cpu().float().numpy()
        tensor = TensorPayload.from_numpy(prompt_hidden, name="hidden_states", metadata={"phase": "prefill", "model_id": self.model_id})
        for index, (worker, cache) in enumerate(zip(self.workers, caches, strict=True)):
            worker_sequence_metadata = _qwen_boundary_sequence_metadata(
                sequence_metadata,
                worker_index=index,
                cache=cache,
            )
            if tensor_transport == "binary":
                result, route_elapsed_ms = await worker.prefill_qwen_shard_binary(
                    tensor=tensor,
                    cache_id=cache.cache_id,
                    sequence_id=sequence_id,
                    position_start=0,
                    generation_step=0,
                    boundary_adapter_strategy=resolved_boundary_adapter_strategy,
                    sequence_metadata=worker_sequence_metadata,
                )
            else:
                result, route_elapsed_ms = await worker.prefill_qwen_shard(
                    tensor=tensor,
                    cache_id=cache.cache_id,
                    sequence_id=sequence_id,
                    position_start=0,
                )
            tensor = result.tensor
            caches[index] = result.cache
            traces.append(result.trace.model_copy(update={"route_elapsed_ms": route_elapsed_ms}))
            await self._emit_cache_update(cache_update_callback, "prefill", caches, traces)

        generated_token_ids: list[int] = []
        logit_summaries: list[dict[str, Any]] = []
        first_logit_summary = self._last_token_logit_summary(
            tensor,
            reference_token_id=_reference_token_id(reference_token_ids, 0),
        )
        first_logit_summary.update({"step": 0, "phase": "prefill"})
        logit_summaries.append(first_logit_summary)
        next_token_id = int(first_logit_summary["argmax_token_id"])
        generated_token_ids.append(next_token_id)

        for step in range(1, max_new_tokens):
            with torch.no_grad():
                token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
                decode_hidden = self.embed_tokens(token_tensor).detach().cpu().float().numpy()
            tensor = TensorPayload.from_numpy(decode_hidden, name="hidden_states", metadata={"phase": "decode", "model_id": self.model_id})
            next_caches: list[KVCacheHandle] = []
            for index, (worker, cache) in enumerate(zip(self.workers, caches, strict=True)):
                worker_sequence_metadata = _qwen_boundary_sequence_metadata(
                    sequence_metadata,
                    worker_index=index,
                    cache=cache,
                )
                if tensor_transport == "binary":
                    result, route_elapsed_ms = await worker.decode_qwen_shard_binary(
                        tensor=tensor,
                        cache_id=cache.cache_id,
                        sequence_id=sequence_id,
                        position_start=cache.current_position,
                        generation_step=step,
                        boundary_adapter_strategy=resolved_boundary_adapter_strategy,
                        sequence_metadata=worker_sequence_metadata,
                    )
                else:
                    result, route_elapsed_ms = await worker.decode_qwen_shard(
                        tensor=tensor,
                        cache_id=cache.cache_id,
                        sequence_id=sequence_id,
                        position_start=cache.current_position,
                    )
                tensor = result.tensor
                next_caches.append(result.cache)
                traces.append(result.trace.model_copy(update={"route_elapsed_ms": route_elapsed_ms}))
                await self._emit_cache_update(cache_update_callback, "decode", next_caches, traces)
            caches = next_caches
            logit_summary = self._last_token_logit_summary(
                tensor,
                reference_token_id=_reference_token_id(reference_token_ids, step),
            )
            logit_summary.update({"step": step, "phase": "decode"})
            logit_summaries.append(logit_summary)
            next_token_id = int(logit_summary["argmax_token_id"])
            generated_token_ids.append(next_token_id)

        if release_caches:
            caches = [
                await worker.release_qwen_cache(cache_id=cache.cache_id, sequence_id=sequence_id)
                for worker, cache in zip(self.workers, caches, strict=True)
            ]
            await self._emit_cache_update(cache_update_callback, "released", caches, traces)

        return QwenWorkerCacheGenerationResult(
            model_id=self.model_id,
            sequence_id=sequence_id,
            prompt_length=prompt_length,
            generated_token_ids=generated_token_ids,
            caches=caches,
            traces=traces,
            metadata={
                "runner": "qwen-worker-cache-coordinator-v0",
                "worker_count": len(self.workers),
                "decode_steps": max_new_tokens - 1,
                "released_caches": release_caches,
                "tensor_transport": tensor_transport,
                "boundary_adapter_strategy": resolved_boundary_adapter_strategy,
                "boundary_adapter_applied": resolved_boundary_adapter_strategy != "identity",
                "logit_topk_schema_version": "qwen-worker-cache-logit-topk-v0",
                "logit_topk": logit_summaries,
            },
        )

    def _argmax_last_token(self, tensor: TensorPayload) -> int:
        import torch

        with torch.no_grad():
            hidden = torch.from_numpy(tensor.to_numpy()).to(self.device)
            logits = self.lm_head(self.norm(hidden))
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
        return int(next_token[0].item())

    def _last_token_logit_summary(
        self,
        tensor: TensorPayload,
        *,
        top_k: int = 5,
        reference_token_id: int | None = None,
    ) -> dict[str, Any]:
        import torch

        with torch.no_grad():
            hidden = torch.from_numpy(tensor.to_numpy()).to(self.device)
            logits = self.lm_head(self.norm(hidden))[:, -1, :].float()
            k = min(top_k, int(logits.shape[-1]))
            values, indices = torch.topk(logits, k=k, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            argmax_logprob = float(log_probs[0, indices[0, 0]].detach().cpu().item()) if k else 0.0
            reference_logprob = (
                float(log_probs[0, reference_token_id].detach().cpu().item())
                if reference_token_id is not None and 0 <= reference_token_id < int(logits.shape[-1])
                else None
            )
        top_values = [float(value) for value in values[0].detach().cpu().tolist()]
        top_indices = [int(index) for index in indices[0].detach().cpu().tolist()]
        return {
            "schema_version": "qwen-worker-cache-logit-topk-step-v0",
            "top_k": k,
            "argmax_token_id": top_indices[0] if top_indices else -1,
            "top_k_token_ids": top_indices,
            "top_k_logits": top_values,
            "max_logit": top_values[0] if top_values else 0.0,
            "argmax_logprob": argmax_logprob,
            "reference_token_id": reference_token_id,
            "reference_token_logprob": reference_logprob,
            "reference_token_nll": -reference_logprob if reference_logprob is not None else None,
        }

    def _eval(self) -> None:
        for module in [self.embed_tokens, self.norm, self.lm_head]:
            if hasattr(module, "eval"):
                module.eval()

    async def _emit_cache_update(
        self,
        callback: Callable[[str, list[KVCacheHandle], list[QwenShardTrace]], Awaitable[None]] | None,
        phase: str,
        caches: list[KVCacheHandle],
        traces: list[QwenShardTrace],
    ) -> None:
        if callback is not None:
            await callback(phase, list(caches), list(traces))


    async def generate_token_ids_coalesced(
        self,
        input_id_batches: Sequence[Any],
        *,
        max_new_tokens: int,
        sequence_ids: Sequence[str],
        ttl_seconds: float | None = None,
        release_caches: bool = True,
        boundary_adapter_strategy: str = "identity",
        reference_token_ids_by_sequence: Sequence[list[int] | None] | None = None,
        sequence_metadata: dict[str, Any] | None = None,
        sequence_metadata_by_prompt: Sequence[dict[str, Any] | None] | None = None,
    ) -> list[QwenWorkerCacheGenerationResult]:
        import torch

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if not input_id_batches:
            raise ValueError("coalesced Qwen generation requires at least one prompt")
        if len(input_id_batches) != len(sequence_ids):
            raise ValueError("coalesced Qwen generation prompt/sequence counts must match")
        if (
            sequence_metadata_by_prompt is not None
            and len(sequence_metadata_by_prompt) != len(input_id_batches)
        ):
            raise ValueError(
                "coalesced Qwen generation prompt metadata count must match prompt count"
            )
        resolved_boundary_adapter_strategy = _normalize_qwen_boundary_wire_strategy(
            boundary_adapter_strategy
        )
        self._eval()
        prompt_count = len(input_id_batches)
        resolved_inputs = [input_ids.to(self.device) for input_ids in input_id_batches]
        prompt_lengths = [int(input_ids.shape[-1]) for input_ids in resolved_inputs]
        caches_by_prompt: list[list[KVCacheHandle]] = [
            [
                await worker.create_qwen_cache(
                    sequence_id=str(sequence_ids[prompt_index]),
                    ttl_seconds=ttl_seconds,
                )
                for worker in self.workers
            ]
            for prompt_index in range(prompt_count)
        ]
        traces_by_prompt: list[list[QwenShardTrace]] = [[] for _ in range(prompt_count)]
        logit_summaries_by_prompt: list[list[dict[str, Any]]] = [
            [] for _ in range(prompt_count)
        ]

        final_tensors: list[TensorPayload] = []
        for prompt_index, input_ids in enumerate(resolved_inputs):
            prompt_sequence_metadata = _metadata_for_prompt(
                sequence_metadata_by_prompt,
                prompt_index,
            )
            with torch.no_grad():
                prompt_hidden = self.embed_tokens(input_ids).detach().cpu().float().numpy()
            tensor = TensorPayload.from_numpy(
                prompt_hidden,
                name="hidden_states",
                metadata={
                    "phase": "prefill",
                    "model_id": self.model_id,
                    "coalesced_prompt_index": prompt_index,
                },
            )
            for worker_index, (worker, cache) in enumerate(
                zip(self.workers, caches_by_prompt[prompt_index], strict=True)
            ):
                worker_sequence_metadata = _qwen_boundary_sequence_metadata(
                    {**(sequence_metadata or {}), **prompt_sequence_metadata},
                    worker_index=worker_index,
                    cache=cache,
                )
                result, route_elapsed_ms = await worker.prefill_qwen_shard_binary(
                    tensor=tensor,
                    cache_id=cache.cache_id,
                    sequence_id=str(sequence_ids[prompt_index]),
                    position_start=0,
                    generation_step=0,
                    boundary_adapter_strategy=resolved_boundary_adapter_strategy,
                    sequence_metadata={
                        **worker_sequence_metadata,
                        "coalesced_prompt_index": prompt_index,
                        "coalesced_prompt_count": prompt_count,
                    },
                )
                tensor = result.tensor
                caches_by_prompt[prompt_index][worker_index] = result.cache
                traces_by_prompt[prompt_index].append(
                    result.trace.model_copy(update={"route_elapsed_ms": route_elapsed_ms})
                )
            final_tensors.append(tensor)

        generated_token_ids_by_prompt: list[list[int]] = []
        next_token_ids: list[int] = []
        for prompt_index, tensor in enumerate(final_tensors):
            summary = self._last_token_logit_summary(
                tensor,
                reference_token_id=_reference_token_id_for_prompt(
                    reference_token_ids_by_sequence,
                    prompt_index,
                    0,
                ),
            )
            summary.update({"step": 0, "phase": "prefill"})
            logit_summaries_by_prompt[prompt_index].append(summary)
            token_id = int(summary["argmax_token_id"])
            generated_token_ids_by_prompt.append([token_id])
            next_token_ids.append(token_id)

        coalesced_decode_route_count = 0
        coalesced_decode_member_count = 0
        for step in range(1, max_new_tokens):
            current_tensors: list[TensorPayload] = []
            for prompt_index, next_token_id in enumerate(next_token_ids):
                with torch.no_grad():
                    token_tensor = torch.tensor(
                        [[next_token_id]],
                        dtype=torch.long,
                        device=self.device,
                    )
                    decode_hidden = (
                        self.embed_tokens(token_tensor).detach().cpu().float().numpy()
                    )
                current_tensors.append(
                    TensorPayload.from_numpy(
                        decode_hidden,
                        name="hidden_states",
                        metadata={
                            "phase": "decode",
                            "model_id": self.model_id,
                            "coalesced_prompt_index": prompt_index,
                        },
                    )
                )

            for worker_index, worker in enumerate(self.workers):
                members: list[dict[str, Any]] = []
                for prompt_index, cache_set in enumerate(caches_by_prompt):
                    prompt_sequence_metadata = _metadata_for_prompt(
                        sequence_metadata_by_prompt,
                        prompt_index,
                    )
                    cache = cache_set[worker_index]
                    members.append(
                        {
                            "cache_id": cache.cache_id,
                            "sequence_id": str(sequence_ids[prompt_index]),
                            "position_start": cache.current_position,
                            "metadata": {
                                **prompt_sequence_metadata,
                                "coalesced_prompt_index": prompt_index,
                                "coalesced_prompt_count": prompt_count,
                            },
                        }
                    )
                results, route_elapsed_ms = await worker.decode_qwen_shard_binary_coalesced(
                    tensors=current_tensors,
                    members=members,
                    generation_step=step,
                    boundary_adapter_strategy=resolved_boundary_adapter_strategy,
                    sequence_metadata={
                        **(sequence_metadata or {}),
                        "coalesced_prompt_count": prompt_count,
                        "coalesced_worker_index": worker_index,
                    },
                )
                coalesced_decode_route_count += 1
                coalesced_decode_member_count += len(results)
                current_tensors = [result.tensor for result in results]
                for prompt_index, result in enumerate(results):
                    caches_by_prompt[prompt_index][worker_index] = result.cache
                    traces_by_prompt[prompt_index].append(
                        result.trace.model_copy(
                            update={"route_elapsed_ms": route_elapsed_ms}
                        )
                    )

            next_token_ids = []
            for prompt_index, tensor in enumerate(current_tensors):
                summary = self._last_token_logit_summary(
                    tensor,
                    reference_token_id=_reference_token_id_for_prompt(
                        reference_token_ids_by_sequence,
                        prompt_index,
                        step,
                    ),
                )
                summary.update({"step": step, "phase": "decode"})
                logit_summaries_by_prompt[prompt_index].append(summary)
                token_id = int(summary["argmax_token_id"])
                generated_token_ids_by_prompt[prompt_index].append(token_id)
                next_token_ids.append(token_id)

        if release_caches:
            for prompt_index, cache_set in enumerate(caches_by_prompt):
                caches_by_prompt[prompt_index] = [
                    await worker.release_qwen_cache(
                        cache_id=cache.cache_id,
                        sequence_id=str(sequence_ids[prompt_index]),
                    )
                    for worker, cache in zip(self.workers, cache_set, strict=True)
                ]

        results: list[QwenWorkerCacheGenerationResult] = []
        for prompt_index, sequence_id in enumerate(sequence_ids):
            results.append(
                QwenWorkerCacheGenerationResult(
                    model_id=self.model_id,
                    sequence_id=str(sequence_id),
                    prompt_length=prompt_lengths[prompt_index],
                    generated_token_ids=generated_token_ids_by_prompt[prompt_index],
                    caches=caches_by_prompt[prompt_index],
                    traces=traces_by_prompt[prompt_index],
                    metadata={
                        "runner": "qwen-worker-cache-coordinator-coalesced-v0",
                        "worker_count": len(self.workers),
                        "prompt_count": prompt_count,
                        "prompt_index": prompt_index,
                        "decode_steps": max_new_tokens - 1,
                        "released_caches": release_caches,
                        "tensor_transport": "binary-coalesced",
                        "boundary_adapter_strategy": resolved_boundary_adapter_strategy,
                        "boundary_adapter_applied": resolved_boundary_adapter_strategy
                        != "identity",
                        "coalesced_decode_route_count": coalesced_decode_route_count,
                        "coalesced_decode_member_count": coalesced_decode_member_count,
                        "coalesced_cache_isolation_preserved": True,
                        "coalesced_transformer_compute_claimed": False,
                        "logit_topk_schema_version": "qwen-worker-cache-logit-topk-v0",
                        "logit_topk": logit_summaries_by_prompt[prompt_index],
                    },
                )
            )
        return results


def _reference_token_id(reference_token_ids: list[int] | None, step: int) -> int | None:
    if reference_token_ids is None or step >= len(reference_token_ids):
        return None
    return int(reference_token_ids[step])


def _reference_token_id_for_prompt(
    reference_token_ids_by_sequence: Sequence[list[int] | None] | None,
    prompt_index: int,
    step: int,
) -> int | None:
    if (
        reference_token_ids_by_sequence is None
        or prompt_index >= len(reference_token_ids_by_sequence)
    ):
        return None
    return _reference_token_id(reference_token_ids_by_sequence[prompt_index], step)


def _metadata_for_prompt(
    sequence_metadata_by_prompt: Sequence[dict[str, Any] | None] | None,
    prompt_index: int,
) -> dict[str, Any]:
    if sequence_metadata_by_prompt is None:
        return {}
    if prompt_index >= len(sequence_metadata_by_prompt):
        return {}
    metadata = sequence_metadata_by_prompt[prompt_index]
    return dict(metadata) if isinstance(metadata, dict) else {}


def _normalize_qwen_boundary_wire_strategy(value: str | None) -> str:
    if value == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID:
        return TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID
    return normalize_boundary_adapter_strategy(str(value or "identity"))


def build_local_qwen_runner_from_model(
    *,
    model: Any,
    resources: Sequence[ResourceProfile],
    model_id: str = "Qwen/Local",
    precision: Precision = "fp32",
    device: str = "cpu",
) -> QwenLocalDistributedModel:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Qwen local runner requires a model with a Transformers config")
    manifest = qwen_manifest_from_config(model_id=model_id, config=config, default_precision=precision)
    plan = build_qwen_distributed_plan(manifest, resources, precision=precision, device=device)
    return QwenLocalDistributedModel.from_model(model=model, plan=plan, device=device)


def load_head_modules_shard(
    *,
    model_id: str,
    precision: "Precision" = "fp32",
    device: str = "cpu",
    local_files_only: bool = True,
) -> tuple[Any, Any, Any, int]:
    """Load ONLY the driver's head modules — embed / final-norm / lm_head.

    The split driver owns token embedding, final norm and the lm_head; loading
    the whole reference model just to reach them makes the driver machine
    unable to host a model too large for it. This meta-inits the architecture
    (zero memory) and materialises only those three modules' weights, so the
    driver's footprint is the head modules, not the model. The transformer
    layers stay on the meta device (never called). Returns
    (embed, final_norm, lm_head, hidden_size).
    """
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("missing optional dependency: transformers")
    import json as _json
    import os as _os

    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=local_files_only
    )
    hidden_size = int(
        getattr(config, "hidden_size", getattr(config, "n_embd", 0)) or 0
    )
    snapshot_dir = snapshot_download(
        model_id,
        local_files_only=local_files_only,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"],
    )
    dtype = _torch_dtype(torch, device=device, precision=precision)
    index_path = _os.path.join(snapshot_dir, "model.safetensors.index.json")
    if _os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as handle:
            weight_map = _json.load(handle)["weight_map"]
    else:
        single = _os.path.join(snapshot_dir, "model.safetensors")
        with safe_open(single, framework="pt", device="cpu") as handle:
            weight_map = {key: "model.safetensors" for key in handle.keys()}

    head_keys = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    needed = {key for key in head_keys if key in weight_map}
    # Tied lm_head: some models omit lm_head.weight and reuse embed_tokens.
    tied = "lm_head.weight" not in weight_map

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()
    by_file: dict[str, list[str]] = {}
    for key in needed:
        by_file.setdefault(weight_map[key], []).append(key)
    for file_name, keys in by_file.items():
        with safe_open(
            _os.path.join(snapshot_dir, file_name), framework="pt", device="cpu"
        ) as handle:
            for key in keys:
                set_module_tensor_to_device(
                    model, key, device, value=handle.get_tensor(key).to(dtype), dtype=dtype
                )
    if tied:
        set_module_tensor_to_device(
            model,
            "lm_head.weight",
            device,
            value=model.model.embed_tokens.weight.data,
            dtype=dtype,
        )
    backbone = _qwen_backbone(model)
    return backbone.embed_tokens, backbone.norm, model.get_output_embeddings(), hidden_size


def _qwen_backbone(model: Any) -> Any:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise ValueError("Qwen model does not expose a `.model` backbone")
    for name in ["embed_tokens", "layers", "norm", "rotary_emb"]:
        if not hasattr(backbone, name):
            raise ValueError(f"Qwen backbone does not expose `{name}`")
    if not hasattr(model, "lm_head"):
        raise ValueError("Qwen causal LM does not expose `lm_head`")
    return backbone


def _qwen_worker_loading_accounting(
    *,
    model: Any,
    manifest: QwenShardManifest,
    loading_mode: str,
) -> QwenWorkerLoadingAccounting:
    if manifest.layer_range is None:
        raise ValueError("Qwen worker runtime requires a layer range")
    backbone = _qwen_backbone(model)
    layers = list(backbone.layers)
    full_count, full_bytes = _module_parameter_stats(model)
    layer_stats = [_module_parameter_stats(layer) for layer in layers]
    decoder_layer_count = sum(count for count, _bytes in layer_stats)
    decoder_layer_bytes = sum(_bytes for _count, _bytes in layer_stats)
    assigned_layer_stats = layer_stats[manifest.layer_range.start : manifest.layer_range.end]
    assigned_count = sum(count for count, _bytes in assigned_layer_stats)
    assigned_bytes = sum(_bytes for _count, _bytes in assigned_layer_stats)
    loaded_count = full_count
    loaded_bytes = full_bytes
    return QwenWorkerLoadingAccounting(
        loading_mode=loading_mode,
        model_id=manifest.model_id,
        runtime_id=manifest.runtime_id,
        layer_start=manifest.layer_range.start,
        layer_end=manifest.layer_range.end,
        total_layers=len(layers),
        assigned_layer_count=manifest.layer_range.count,
        full_model_parameter_count=full_count,
        full_model_parameter_bytes=full_bytes,
        decoder_layer_parameter_count=decoder_layer_count,
        decoder_layer_parameter_bytes=decoder_layer_bytes,
        assigned_layer_parameter_count=assigned_count,
        assigned_layer_parameter_bytes=assigned_bytes,
        loaded_parameter_count=loaded_count,
        loaded_parameter_bytes=loaded_bytes,
        unassigned_loaded_parameter_count=max(loaded_count - assigned_count, 0),
        unassigned_loaded_parameter_bytes=max(loaded_bytes - assigned_bytes, 0),
        shard_only_weight_loading_claimed=False,
        per_machine_ram_reduction_claimed=False,
    )


def _module_parameter_stats(module: Any) -> tuple[int, int]:
    count = 0
    byte_count = 0
    for parameter in module.parameters():
        parameter_count = int(parameter.numel())
        count += parameter_count
        byte_count += parameter_count * int(parameter.element_size())
    return count, byte_count
