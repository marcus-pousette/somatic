from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, Field, model_validator

from soup.runtime.schemas import utc_now


UnitKind = Literal[
    "embedding",
    "attention_block",
    "ssm_block",
    "deltanet_block",
    "mlp",
    "moe",
    "norm",
    "adapter",
    "lm_head",
]
ArchitectureFamily = Literal["transformer", "ssm", "hybrid", "moe", "unknown"]
BackendKind = Literal["pytorch", "transformers", "llama_cpp", "mlx", "vllm", "phone", "numpy", "simulated"]
Precision = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4", "unknown"]
BoundaryAdapterKind = Literal["identity", "precision_cast", "quantized", "low_rank", "learned", "custom"]
BoundaryDecision = Literal["selected", "fallback_identity"]


class ModelUnit(BaseModel):
    unit_id: str
    kind: UnitKind
    ordinal: int = Field(ge=0)
    parameter_count: int = Field(default=0, ge=0)
    activation_bytes: int = Field(default=0, ge=0)
    required_memory_gb: float = Field(default=0.0, ge=0.0)
    required_capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelGraph(BaseModel):
    graph_id: str
    units: list[ModelUnit] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_unique_units(self) -> "ModelGraph":
        unit_ids = [unit.unit_id for unit in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("model graph unit_id values must be unique")
        return self

    def ordered_units(self) -> list[ModelUnit]:
        return sorted(self.units, key=lambda unit: (unit.ordinal, unit.unit_id))

    def unit_kinds(self) -> set[UnitKind]:
        return {unit.kind for unit in self.units}


class ModelManifest(BaseModel):
    model_id: str
    display_name: str
    architecture_family: ArchitectureFamily = "unknown"
    tokenizer_id: str | None = None
    weight_uri: str | None = None
    supported_precisions: list[Precision] = Field(default_factory=lambda: ["bf16", "fp16", "fp32"])
    default_precision: Precision = "bf16"
    adapter_compatibility: list[str] = Field(default_factory=list)
    graph: ModelGraph
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_default_precision(self) -> "ModelManifest":
        if self.default_precision not in self.supported_precisions:
            raise ValueError("default_precision must be included in supported_precisions")
        return self


def model_manifest_fingerprint(manifest: ModelManifest) -> str:
    """Return a stable content hash for a generic model manifest."""
    payload = manifest.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class ResourceProfile(BaseModel):
    runtime_id: str
    backend: BackendKind = "simulated"
    cpu_cores: int = Field(default=1, ge=0)
    memory_gb: float = Field(default=1.0, ge=0.0)
    gpu_kind: str | None = None
    gpu_memory_gb: float = Field(default=0.0, ge=0.0)
    network_latency_ms: float = Field(default=0.0, ge=0.0)
    bandwidth_mbps: float = Field(default=1000.0, gt=0.0)
    current_load: float = Field(default=0.0, ge=0.0)
    supported_unit_kinds: list[UnitKind] = Field(default_factory=list)
    supported_precisions: list[Precision] = Field(default_factory=lambda: ["bf16", "fp16", "fp32"])
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def memory_capacity_gb(self) -> float:
        advertised_capacity = self.memory_gb + self.gpu_memory_gb
        override = self.metadata.get("effective_memory_capacity_gb")
        if isinstance(override, (int, float)):
            return max(0.0, min(float(override), advertised_capacity))
        return advertised_capacity

    def supports(self, unit: ModelUnit, precision: Precision) -> bool:
        if precision not in self.supported_precisions:
            return False
        if self.supported_unit_kinds and unit.kind not in self.supported_unit_kinds:
            return False
        if unit.required_memory_gb > self.memory_capacity_gb():
            return False
        return all(capability in self.capabilities for capability in unit.required_capabilities)


class CalibrationProfile(BaseModel):
    runtime_id: str
    backend: BackendKind = "simulated"
    precision: Precision = "bf16"
    unit_latency_ms: dict[str, float] = Field(default_factory=dict)
    transfer_latency_ms_per_mb: float = Field(default=0.0, ge=0.0)
    memory_headroom_gb: float = Field(default=0.0, ge=0.0)
    measured_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def latency_for(self, unit: ModelUnit) -> float:
        if unit.unit_id in self.unit_latency_ms:
            return self.unit_latency_ms[unit.unit_id]
        if unit.kind in self.unit_latency_ms:
            return self.unit_latency_ms[unit.kind]
        return self.unit_latency_ms.get("default", 1.0)


class BoundaryAdapterSpec(BaseModel):
    adapter_id: str
    kind: BoundaryAdapterKind = "identity"
    display_name: str = ""
    supported_source_unit_kinds: list[UnitKind] = Field(default_factory=list)
    supported_target_unit_kinds: list[UnitKind] = Field(default_factory=list)
    estimated_raw_byte_ratio: float = Field(default=1.0, ge=0.0)
    estimated_frame_byte_ratio: float | None = Field(default=None, ge=0.0)
    estimated_mean_abs_error: float = Field(default=0.0, ge=0.0)
    estimated_max_abs_error: float = Field(default=0.0, ge=0.0)
    reversible: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def supports_boundary(self, source: ModelUnit, target: ModelUnit) -> bool:
        if self.supported_source_unit_kinds and source.kind not in self.supported_source_unit_kinds:
            return False
        if self.supported_target_unit_kinds and target.kind not in self.supported_target_unit_kinds:
            return False
        return True


class BoundaryAdapterConstraints(BaseModel):
    max_mean_abs_error: float = Field(default=0.0, ge=0.0)
    max_abs_error: float | None = Field(default=0.0, ge=0.0)
    min_raw_byte_savings_ratio: float = Field(default=0.0, ge=0.0)
    allowed_adapter_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BoundaryPlacement(BaseModel):
    boundary_id: str
    source_unit_id: str
    target_unit_id: str
    source_unit_kind: UnitKind
    target_unit_kind: UnitKind
    source_runtime_id: str
    target_runtime_id: str
    adapter_id: str
    adapter_kind: BoundaryAdapterKind
    decision: BoundaryDecision
    estimated_original_transfer_mb: float = Field(ge=0.0)
    estimated_adapted_transfer_mb: float = Field(ge=0.0)
    estimated_raw_byte_savings_ratio: float = Field(ge=0.0)
    estimated_mean_abs_error: float = Field(ge=0.0)
    estimated_max_abs_error: float = Field(ge=0.0)
    rejected_adapter_ids: list[str] = Field(default_factory=list)
    rejection_reasons: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionPlacement(BaseModel):
    unit_id: str
    runtime_id: str
    reason: str = ""


class ExecutionPlan(BaseModel):
    model_id: str
    precision: Precision
    placements: list[ExecutionPlacement]
    boundary_placements: list[BoundaryPlacement] = Field(default_factory=list)
    estimated_latency_ms: float = Field(ge=0.0)
    estimated_transfer_mb: float = Field(ge=0.0)
    planner_version: str = "sequence-planner-v0"
    generated_at: datetime = Field(default_factory=utc_now)
    assumptions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def runtime_ids(self) -> set[str]:
        return {placement.runtime_id for placement in self.placements}

    def placement_for(self, unit_id: str) -> ExecutionPlacement:
        for placement in self.placements:
            if placement.unit_id == unit_id:
                return placement
        raise KeyError(unit_id)


class RuntimeAdapter(Protocol):
    name: str
    backends: Sequence[BackendKind]

    def inspect_manifest(self, model_ref: str) -> ModelManifest:
        ...

    def calibrate(self, resource: ResourceProfile, manifest: ModelManifest) -> CalibrationProfile:
        ...

    def execute_unit(self, unit: ModelUnit, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class Planner:
    def __init__(
        self,
        *,
        default_precision: Precision | None = None,
        planner_version: str = "sequence-planner-v0",
        spread_penalty_ms: float = 0.0,
    ) -> None:
        self.default_precision = default_precision
        self.planner_version = planner_version
        self.spread_penalty_ms = spread_penalty_ms

    def plan(
        self,
        manifest: ModelManifest,
        resources: Sequence[ResourceProfile],
        calibrations: Sequence[CalibrationProfile] | None = None,
        *,
        boundary_adapter_specs: Sequence[BoundaryAdapterSpec] | None = None,
        boundary_constraints: BoundaryAdapterConstraints | None = None,
    ) -> ExecutionPlan:
        if not resources:
            raise ValueError("cannot plan without resources")

        precision = self.default_precision or manifest.default_precision
        if precision not in manifest.supported_precisions:
            raise ValueError(f"manifest {manifest.model_id} does not support precision {precision}")

        calibration_by_runtime = {profile.runtime_id: profile for profile in calibrations or []}
        used_memory: dict[str, float] = {resource.runtime_id: 0.0 for resource in resources}
        used_units: dict[str, int] = {resource.runtime_id: 0 for resource in resources}
        placements: list[ExecutionPlacement] = []
        boundary_placements: list[BoundaryPlacement] = []
        estimated_latency_ms = 0.0
        estimated_transfer_mb = 0.0
        previous_runtime: ResourceProfile | None = None
        previous_unit: ModelUnit | None = None
        resolved_boundary_specs = _with_identity_boundary_adapter(boundary_adapter_specs)
        resolved_boundary_constraints = boundary_constraints or BoundaryAdapterConstraints()

        for unit in manifest.graph.ordered_units():
            candidate = self._select_resource(unit, precision, resources, calibration_by_runtime, used_memory, used_units)
            calibration = calibration_by_runtime.get(candidate.runtime_id)
            unit_latency = calibration.latency_for(unit) if calibration is not None else self._fallback_latency(unit, candidate)

            if previous_runtime is not None and previous_runtime.runtime_id != candidate.runtime_id:
                boundary_adapter, boundary_decision, rejection_reasons = self._select_boundary_adapter(
                    source=previous_unit,
                    target=unit,
                    specs=resolved_boundary_specs,
                    constraints=resolved_boundary_constraints,
                )
                original_transfer_mb = unit.activation_bytes / 1_048_576
                adapted_transfer_bytes = int(round(unit.activation_bytes * boundary_adapter.estimated_raw_byte_ratio))
                transfer_mb = max(0, adapted_transfer_bytes) / 1_048_576
                raw_byte_savings_ratio = (
                    max(0.0, 1.0 - transfer_mb / original_transfer_mb)
                    if original_transfer_mb > 0
                    else 0.0
                )
                estimated_transfer_mb += transfer_mb
                estimated_latency_ms += self._transfer_latency_ms(transfer_mb, previous_runtime, candidate, calibration)
                boundary_placements.append(
                    BoundaryPlacement(
                        boundary_id=f"{previous_unit.unit_id if previous_unit is not None else 'unknown'}->{unit.unit_id}",
                        source_unit_id=previous_unit.unit_id if previous_unit is not None else "unknown",
                        target_unit_id=unit.unit_id,
                        source_unit_kind=previous_unit.kind if previous_unit is not None else unit.kind,
                        target_unit_kind=unit.kind,
                        source_runtime_id=previous_runtime.runtime_id,
                        target_runtime_id=candidate.runtime_id,
                        adapter_id=boundary_adapter.adapter_id,
                        adapter_kind=boundary_adapter.kind,
                        decision=boundary_decision,
                        estimated_original_transfer_mb=original_transfer_mb,
                        estimated_adapted_transfer_mb=transfer_mb,
                        estimated_raw_byte_savings_ratio=raw_byte_savings_ratio,
                        estimated_mean_abs_error=boundary_adapter.estimated_mean_abs_error,
                        estimated_max_abs_error=boundary_adapter.estimated_max_abs_error,
                        rejected_adapter_ids=sorted(rejection_reasons),
                        rejection_reasons=rejection_reasons,
                        metadata={
                            "adapter_display_name": boundary_adapter.display_name,
                            "architecture_neutral_boundary": True,
                        },
                    )
                )

            estimated_latency_ms += unit_latency
            used_memory[candidate.runtime_id] += unit.required_memory_gb
            used_units[candidate.runtime_id] += 1
            placements.append(
                ExecutionPlacement(
                    unit_id=unit.unit_id,
                    runtime_id=candidate.runtime_id,
                    reason=f"selected for {unit.kind} at {precision}",
                )
            )
            previous_runtime = candidate
            previous_unit = unit

        assumptions = [
            "planner uses architecture-neutral model units",
            "Qwen-specific behavior belongs in adapters/manifests, not placement logic",
        ]
        if self.spread_penalty_ms:
            assumptions.append("optional spread penalty can trade latency for distributed execution proof")

        return ExecutionPlan(
            model_id=manifest.model_id,
            precision=precision,
            placements=placements,
            boundary_placements=boundary_placements,
            estimated_latency_ms=estimated_latency_ms,
            estimated_transfer_mb=estimated_transfer_mb,
            planner_version=self.planner_version,
            assumptions=assumptions,
            metadata={
                "resource_count": len(resources),
                "unit_kinds": sorted(manifest.graph.unit_kinds()),
                "spread_penalty_ms": self.spread_penalty_ms,
                "boundary_adapter_candidates": [spec.adapter_id for spec in resolved_boundary_specs],
                "boundary_adapter_constraints": resolved_boundary_constraints.model_dump(mode="json"),
                "boundary_count": len(boundary_placements),
            },
        )

    def _select_resource(
        self,
        unit: ModelUnit,
        precision: Precision,
        resources: Sequence[ResourceProfile],
        calibration_by_runtime: dict[str, CalibrationProfile],
        used_memory: dict[str, float],
        used_units: dict[str, int],
    ) -> ResourceProfile:
        best_resource: ResourceProfile | None = None
        best_score = float("inf")
        rejection_reason = "no compatible resource"

        for resource in resources:
            if precision not in resource.supported_precisions:
                rejection_reason = f"{resource.runtime_id} does not support precision {precision}"
                continue
            if resource.supported_unit_kinds and unit.kind not in resource.supported_unit_kinds:
                rejection_reason = f"{resource.runtime_id} does not support {unit.kind}"
                continue
            missing_capabilities = [capability for capability in unit.required_capabilities if capability not in resource.capabilities]
            if missing_capabilities:
                rejection_reason = f"{resource.runtime_id} lacks capabilities for {unit.unit_id}: {','.join(missing_capabilities)}"
                continue
            capacity_gb = resource.memory_capacity_gb()
            if used_memory[resource.runtime_id] + unit.required_memory_gb > capacity_gb:
                rejection_reason = f"{resource.runtime_id} lacks memory for {unit.unit_id}"
                continue

            calibration = calibration_by_runtime.get(resource.runtime_id)
            latency = calibration.latency_for(unit) if calibration is not None else self._fallback_latency(unit, resource)
            score = latency + resource.network_latency_ms + resource.current_load * 100.0
            score += used_units[resource.runtime_id] * self.spread_penalty_ms
            if score < best_score:
                best_score = score
                best_resource = resource

        if best_resource is None:
            raise ValueError(f"cannot place {unit.unit_id}: {rejection_reason}")
        return best_resource

    def _fallback_latency(self, unit: ModelUnit, resource: ResourceProfile) -> float:
        unit_weight = {
            "embedding": 0.4,
            "attention_block": 2.0,
            "ssm_block": 1.2,
            "deltanet_block": 1.2,
            "mlp": 1.0,
            "moe": 1.8,
            "norm": 0.2,
            "adapter": 0.3,
            "lm_head": 0.6,
        }[unit.kind]
        compute_scale = max(resource.cpu_cores + resource.gpu_memory_gb, 1.0)
        return unit_weight / compute_scale * 10.0

    def _transfer_latency_ms(
        self,
        transfer_mb: float,
        previous: ResourceProfile,
        current: ResourceProfile,
        calibration: CalibrationProfile | None,
    ) -> float:
        if transfer_mb <= 0:
            return 0.0
        if calibration is not None and calibration.transfer_latency_ms_per_mb > 0:
            return transfer_mb * calibration.transfer_latency_ms_per_mb
        bottleneck_mbps = min(previous.bandwidth_mbps, current.bandwidth_mbps)
        return (transfer_mb * 8.0 / bottleneck_mbps) * 1000.0

    def _select_boundary_adapter(
        self,
        *,
        source: ModelUnit | None,
        target: ModelUnit,
        specs: Sequence[BoundaryAdapterSpec],
        constraints: BoundaryAdapterConstraints,
    ) -> tuple[BoundaryAdapterSpec, BoundaryDecision, dict[str, str]]:
        identity = _identity_boundary_adapter()
        if source is None:
            return identity, "fallback_identity", {}

        rejection_reasons: dict[str, str] = {}
        allowed_ids = set(constraints.allowed_adapter_ids)
        candidates: list[BoundaryAdapterSpec] = []
        for spec in specs:
            if allowed_ids and spec.adapter_id not in allowed_ids:
                rejection_reasons[spec.adapter_id] = "not in allowed adapter set"
                continue
            if not spec.supports_boundary(source, target):
                rejection_reasons[spec.adapter_id] = f"does not support {source.kind}->{target.kind}"
                continue
            if spec.estimated_mean_abs_error > constraints.max_mean_abs_error:
                rejection_reasons[spec.adapter_id] = "mean absolute error exceeds constraint"
                continue
            if constraints.max_abs_error is not None and spec.estimated_max_abs_error > constraints.max_abs_error:
                rejection_reasons[spec.adapter_id] = "max absolute error exceeds constraint"
                continue
            raw_savings_ratio = max(0.0, 1.0 - spec.estimated_raw_byte_ratio)
            if spec.kind != "identity" and raw_savings_ratio < constraints.min_raw_byte_savings_ratio:
                rejection_reasons[spec.adapter_id] = "raw byte savings below constraint"
                continue
            candidates.append(spec)

        non_identity = [spec for spec in candidates if spec.kind != "identity"]
        if non_identity:
            selected = min(
                non_identity,
                key=lambda spec: (
                    spec.estimated_raw_byte_ratio,
                    spec.estimated_mean_abs_error,
                    spec.adapter_id,
                ),
            )
            return selected, "selected", rejection_reasons
        return identity, "fallback_identity", rejection_reasons


def _identity_boundary_adapter() -> BoundaryAdapterSpec:
    return BoundaryAdapterSpec(
        adapter_id="identity",
        kind="identity",
        display_name="Full hidden-state transfer",
        estimated_raw_byte_ratio=1.0,
        estimated_frame_byte_ratio=1.0,
        estimated_mean_abs_error=0.0,
        estimated_max_abs_error=0.0,
        reversible=True,
        metadata={"baseline": True},
    )


def _with_identity_boundary_adapter(
    specs: Sequence[BoundaryAdapterSpec] | None,
) -> list[BoundaryAdapterSpec]:
    by_id = {_identity_boundary_adapter().adapter_id: _identity_boundary_adapter()}
    for spec in specs or []:
        by_id[spec.adapter_id] = spec
    return list(by_id.values())
