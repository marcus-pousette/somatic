from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, Field, model_validator

from somatic.sequence_model.interfaces import ModelManifest, Precision, ResourceProfile


EdgeOwnerPolicy = Literal["coordinator", "first_worker_last_worker"]
QwenShardRole = Literal["layer_range", "embedding", "final_norm", "lm_head"]


class QwenLayerRange(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "QwenLayerRange":
        if self.end <= self.start:
            raise ValueError("layer range end must be greater than start")
        return self

    @property
    def count(self) -> int:
        return self.end - self.start

    def contains(self, layer_index: int) -> bool:
        return self.start <= layer_index < self.end


class QwenShardManifest(BaseModel):
    shard_id: str
    runtime_id: str
    model_id: str
    weight_uri: str
    precision: Precision
    device: str = "cpu"
    layer_range: QwenLayerRange | None = None
    roles: list[QwenShardRole] = Field(default_factory=list)
    estimated_memory_gb: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ownership(self) -> "QwenShardManifest":
        if self.layer_range is None and not self.roles:
            raise ValueError("Qwen shard must own at least a layer range or edge role")
        if self.layer_range is not None and "layer_range" not in self.roles:
            self.roles.append("layer_range")
        return self

    def owns_layer(self, layer_index: int) -> bool:
        return self.layer_range is not None and self.layer_range.contains(layer_index)


class QwenDistributedPlan(BaseModel):
    model_id: str
    total_layers: int = Field(ge=1)
    precision: Precision
    tokenizer_owner: Literal["coordinator"] = "coordinator"
    embedding_owner: str
    final_norm_owner: str
    lm_head_owner: str
    shards: list[QwenShardManifest]
    edge_owner_policy: EdgeOwnerPolicy
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_layer_coverage(self) -> "QwenDistributedPlan":
        covered: dict[int, str] = {}
        for shard in self.shards:
            if shard.layer_range is None:
                continue
            for layer_index in range(shard.layer_range.start, shard.layer_range.end):
                if layer_index in covered:
                    raise ValueError(f"Qwen layer {layer_index} is assigned to multiple shards")
                covered[layer_index] = shard.shard_id
        missing = [index for index in range(self.total_layers) if index not in covered]
        if missing:
            raise ValueError(f"Qwen layer coverage is incomplete: {missing}")
        return self

    def shard_for_layer(self, layer_index: int) -> QwenShardManifest:
        for shard in self.shards:
            if shard.owns_layer(layer_index):
                return shard
        raise KeyError(layer_index)


def build_qwen_distributed_plan(
    manifest: ModelManifest,
    resources: Sequence[ResourceProfile],
    *,
    precision: Precision | None = None,
    edge_owner_policy: EdgeOwnerPolicy = "coordinator",
    device: str = "cpu",
) -> QwenDistributedPlan:
    if not resources:
        raise ValueError("cannot build Qwen distributed plan without resources")
    total_layers = infer_qwen_layer_count(manifest)
    ranges = split_qwen_layers(total_layers, len(resources))
    resolved_precision = precision or manifest.default_precision
    if resolved_precision not in manifest.supported_precisions:
        raise ValueError(f"manifest {manifest.model_id} does not support precision {resolved_precision}")

    shards: list[QwenShardManifest] = []
    hidden_size = _manifest_hidden_size(manifest)
    for index, (resource, layer_range) in enumerate(zip(resources, ranges, strict=True)):
        roles: list[QwenShardRole] = ["layer_range"]
        if edge_owner_policy == "first_worker_last_worker":
            if index == 0:
                roles.append("embedding")
            if index == len(resources) - 1:
                roles.extend(["final_norm", "lm_head"])
        shards.append(
            QwenShardManifest(
                shard_id=f"{manifest.model_id}:shard-{index}",
                runtime_id=resource.runtime_id,
                model_id=manifest.model_id,
                weight_uri=manifest.weight_uri or manifest.model_id,
                precision=resolved_precision,
                device=device,
                layer_range=layer_range,
                roles=roles,
                estimated_memory_gb=_estimate_shard_memory_gb(manifest, layer_range, roles),
                metadata={
                    "architecture_family": manifest.architecture_family,
                    "edge_owner_policy": edge_owner_policy,
                    "resource_memory_gb": resource.memory_gb + resource.gpu_memory_gb,
                    **({"qwen_hidden_size": hidden_size} if hidden_size is not None else {}),
                },
            )
        )

    first_runtime = resources[0].runtime_id
    last_runtime = resources[-1].runtime_id
    if edge_owner_policy == "coordinator":
        embedding_owner = "coordinator"
        final_norm_owner = "coordinator"
        lm_head_owner = "coordinator"
    else:
        embedding_owner = first_runtime
        final_norm_owner = last_runtime
        lm_head_owner = last_runtime

    return QwenDistributedPlan(
        model_id=manifest.model_id,
        total_layers=total_layers,
        precision=resolved_precision,
        embedding_owner=embedding_owner,
        final_norm_owner=final_norm_owner,
        lm_head_owner=lm_head_owner,
        shards=shards,
        edge_owner_policy=edge_owner_policy,
        metadata={
            "planner": "qwen-contiguous-layer-shards-v0",
            "generic_runtime_boundary": True,
            "qwen_specific_boundary": "adapter/manifest/loader/execution only",
        },
    )


def infer_qwen_layer_count(manifest: ModelManifest) -> int:
    layer_indices = {
        int(unit.metadata["layer"])
        for unit in manifest.graph.units
        if "layer" in unit.metadata and unit.kind in {"attention_block", "mlp", "deltanet_block", "moe"}
    }
    if not layer_indices:
        raise ValueError(f"manifest {manifest.model_id} does not expose layer metadata")
    return max(layer_indices) + 1


def split_qwen_layers(total_layers: int, shard_count: int) -> list[QwenLayerRange]:
    if total_layers <= 0:
        raise ValueError("total layers must be positive")
    if shard_count <= 0:
        raise ValueError("shard count must be positive")
    if shard_count > total_layers:
        raise ValueError("cannot create more Qwen layer shards than layers")

    base = total_layers // shard_count
    remainder = total_layers % shard_count
    ranges: list[QwenLayerRange] = []
    start = 0
    for index in range(shard_count):
        count = base + (1 if index < remainder else 0)
        end = start + count
        ranges.append(QwenLayerRange(start=start, end=end))
        start = end
    return ranges


def _estimate_shard_memory_gb(manifest: ModelManifest, layer_range: QwenLayerRange, roles: Sequence[QwenShardRole]) -> float:
    memory = 0.0
    for unit in manifest.graph.units:
        layer = unit.metadata.get("layer")
        if isinstance(layer, int) and layer_range.contains(layer):
            memory += unit.required_memory_gb
        elif unit.kind == "embedding" and "embedding" in roles:
            memory += unit.required_memory_gb
        elif unit.kind == "norm" and "final_norm" in roles:
            memory += unit.required_memory_gb
        elif unit.kind == "lm_head" and "lm_head" in roles:
            memory += unit.required_memory_gb
    return round(memory, 6)


def _manifest_hidden_size(manifest: ModelManifest) -> int | None:
    raw = manifest.metadata.get("hidden_size") or manifest.metadata.get("qwen_hidden_size")
    if raw is not None:
        return int(raw)
    for unit in manifest.graph.units:
        if unit.activation_bytes > 0:
            return int(unit.activation_bytes) // 2
    return None
