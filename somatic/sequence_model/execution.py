from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

from pydantic import BaseModel, Field

from somatic.sequence_model.interfaces import (
    BackendKind,
    CalibrationProfile,
    ExecutionPlan,
    ModelManifest,
    ModelUnit,
    Planner,
    ResourceProfile,
)


class UnitExecutionTrace(BaseModel):
    unit_id: str
    unit_kind: str
    runtime_id: str
    input_digest: str
    output_digest: str
    simulated_latency_ms: float = Field(ge=0.0)


class SequenceExecutionResult(BaseModel):
    model_id: str
    prompt: str
    plan: ExecutionPlan
    traces: list[UnitExecutionTrace]
    final_state: list[float]
    output_text: str
    architecture_neutral: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimulatedRuntimeAdapter:
    """Deterministic local adapter used to prove planning/execution plumbing."""

    name = "simulated-sequence-runtime"
    backends: Sequence[BackendKind] = ("simulated",)

    def inspect_manifest(self, model_ref: str) -> ModelManifest:
        raise NotImplementedError("simulated adapter receives explicit manifests")

    def calibrate(self, resource: ResourceProfile, manifest: ModelManifest) -> CalibrationProfile:
        throughput = max(resource.cpu_cores + resource.gpu_memory_gb * 2.0, 1.0)
        unit_latency_ms: dict[str, float] = {"default": 8.0 / throughput}
        for unit in manifest.graph.units:
            base = {
                "embedding": 2.0,
                "attention_block": 14.0,
                "ssm_block": 7.0,
                "deltanet_block": 7.5,
                "mlp": 6.0,
                "moe": 11.0,
                "norm": 1.0,
                "adapter": 1.5,
                "lm_head": 3.0,
            }[unit.kind]
            unit_latency_ms[unit.unit_id] = base / throughput * (1.0 + resource.current_load)
        transfer_latency_ms_per_mb = 8.0 / max(resource.bandwidth_mbps, 1.0) * 1000.0
        return CalibrationProfile(
            runtime_id=resource.runtime_id,
            backend=resource.backend,
            precision=manifest.default_precision,
            unit_latency_ms=unit_latency_ms,
            transfer_latency_ms_per_mb=transfer_latency_ms_per_mb,
            memory_headroom_gb=resource.memory_gb + resource.gpu_memory_gb,
        )

    def execute_unit(self, unit: ModelUnit, payload: dict[str, Any]) -> dict[str, Any]:
        state = [float(value) for value in payload.get("state", [])]
        prompt = str(payload.get("prompt", ""))
        if not state:
            state = _prompt_vector(prompt)

        salt = _stable_unit_scalar(unit.unit_id)
        if unit.kind == "embedding":
            next_state = [_clip(value + salt * 0.05) for value in state]
        elif unit.kind == "attention_block":
            mean = sum(state) / len(state)
            next_state = [_clip(value * 0.72 + mean * 0.24 + salt * 0.04) for value in state]
        elif unit.kind == "ssm_block":
            carry = 0.0
            next_state = []
            for value in state:
                carry = carry * 0.65 + value * 0.35 + salt * 0.02
                next_state.append(_clip(carry))
        elif unit.kind == "deltanet_block":
            next_state = []
            previous = state[-1]
            for value in state:
                next_state.append(_clip(value + (value - previous) * 0.18 + salt * 0.03))
                previous = value
        elif unit.kind == "mlp":
            next_state = [_clip(math.tanh(value * 1.4 + salt * 0.1)) for value in state]
        elif unit.kind == "moe":
            gate = int(abs(sum(state) * 10)) % 3
            multiplier = [0.85, 1.05, 1.2][gate]
            next_state = [_clip(value * multiplier + salt * 0.02) for value in state]
        elif unit.kind == "norm":
            magnitude = math.sqrt(sum(value * value for value in state)) or 1.0
            next_state = [_clip(value / magnitude) for value in state]
        elif unit.kind == "adapter":
            next_state = [_clip(value + salt * 0.01) for value in state]
        elif unit.kind == "lm_head":
            mean = sum(state) / len(state)
            next_state = [_clip(mean), _clip(max(state)), _clip(min(state)), _clip(salt)]
        else:
            next_state = state

        return {"prompt": prompt, "state": next_state}


class SequenceRuntimeCoordinator:
    def __init__(
        self,
        *,
        adapter: SimulatedRuntimeAdapter | None = None,
        planner: Planner | None = None,
    ) -> None:
        self.adapter = adapter or SimulatedRuntimeAdapter()
        self.planner = planner or Planner()

    def calibrate(self, manifest: ModelManifest, resources: Sequence[ResourceProfile]) -> list[CalibrationProfile]:
        return [self.adapter.calibrate(resource, manifest) for resource in resources]

    def execute(
        self,
        manifest: ModelManifest,
        resources: Sequence[ResourceProfile],
        *,
        prompt: str,
        calibrations: Sequence[CalibrationProfile] | None = None,
    ) -> SequenceExecutionResult:
        resolved_calibrations = list(calibrations or self.calibrate(manifest, resources))
        plan = self.planner.plan(manifest, resources, resolved_calibrations)
        calibration_by_runtime = {profile.runtime_id: profile for profile in resolved_calibrations}
        payload: dict[str, Any] = {"prompt": prompt, "state": []}
        traces: list[UnitExecutionTrace] = []

        for unit in manifest.graph.ordered_units():
            placement = plan.placement_for(unit.unit_id)
            before = payload_digest(payload)
            payload = self.adapter.execute_unit(unit, payload)
            after = payload_digest(payload)
            calibration = calibration_by_runtime.get(placement.runtime_id)
            latency = calibration.latency_for(unit) if calibration is not None else 0.0
            traces.append(
                UnitExecutionTrace(
                    unit_id=unit.unit_id,
                    unit_kind=unit.kind,
                    runtime_id=placement.runtime_id,
                    input_digest=before,
                    output_digest=after,
                    simulated_latency_ms=latency,
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
            output_text=f"simulated:{manifest.model_id}:{output_digest[:12]}",
            architecture_neutral=architecture_neutral,
            metadata={
                "adapter": self.adapter.name,
                "unit_count": len(traces),
                "unit_kinds": sorted(manifest.graph.unit_kinds()),
                "runtimes_used": sorted(plan.runtime_ids()),
            },
        )


def _prompt_vector(prompt: str, *, size: int = 8) -> list[float]:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(size)]


def _stable_unit_scalar(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return (digest[0] / 255.0) * 2.0 - 1.0


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def payload_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
