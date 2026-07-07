from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field

from soup.sequence_model.interfaces import (
    BackendKind,
    CalibrationProfile,
    ExecutionPlan,
    ModelManifest,
    ModelUnit,
    Planner,
    ResourceProfile,
)
from soup.sequence_model.tensors import TensorPayload


class TensorUnitExecutionTrace(BaseModel):
    unit_id: str
    unit_kind: str
    runtime_id: str
    input_digest: str
    output_digest: str
    observed_latency_ms: float = Field(ge=0.0)
    input_shape: list[int]
    output_shape: list[int]
    dtype: str


class TensorSequenceExecutionResult(BaseModel):
    model_id: str
    prompt: str
    plan: ExecutionPlan
    traces: list[TensorUnitExecutionTrace]
    final_tensor: TensorPayload
    output_text: str
    architecture_neutral: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class TensorRuntimeAdapter:
    """Deterministic tensor adapter for proving real payload movement and layer execution."""

    name = "numpy-tensor-sequence-runtime"
    backends: Sequence[BackendKind] = ("numpy",)

    def inspect_manifest(self, model_ref: str) -> ModelManifest:
        raise NotImplementedError("tensor adapter receives explicit manifests")

    def calibrate(self, resource: ResourceProfile, manifest: ModelManifest) -> CalibrationProfile:
        throughput = max(resource.cpu_cores + resource.gpu_memory_gb * 2.0, 1.0)
        unit_latency_ms: dict[str, float] = {"default": 4.0 / throughput}
        for unit in manifest.graph.units:
            base = {
                "embedding": 1.0,
                "attention_block": 8.0,
                "ssm_block": 4.0,
                "deltanet_block": 4.5,
                "mlp": 5.0,
                "moe": 6.5,
                "norm": 0.6,
                "adapter": 1.0,
                "lm_head": 1.5,
            }[unit.kind]
            unit_latency_ms[unit.unit_id] = base / throughput * (1.0 + resource.current_load)
        return CalibrationProfile(
            runtime_id=resource.runtime_id,
            backend=resource.backend,
            precision=manifest.default_precision,
            unit_latency_ms=unit_latency_ms,
            transfer_latency_ms_per_mb=8.0 / max(resource.bandwidth_mbps, 1.0) * 1000.0,
            memory_headroom_gb=resource.memory_gb + resource.gpu_memory_gb,
            metadata={"calibration_mode": "typed-tensor"},
        )

    def execute_unit(self, unit: ModelUnit, tensor: TensorPayload) -> TensorPayload:
        return self.execute_array_unit(unit, tensor.to_numpy(), name=tensor.name, metadata=tensor.metadata)

    def execute_array_unit(
        self,
        unit: ModelUnit,
        array: np.ndarray,
        *,
        name: str = "activation",
        metadata: dict[str, Any] | None = None,
    ) -> TensorPayload:
        original_array = np.ascontiguousarray(array)
        array = original_array.astype(np.float32, copy=False)
        output = _execute_array_unit(unit, array)
        output = _canonical_float32(output)
        output = output.astype(original_array.dtype, copy=False)
        output_metadata = dict(metadata or {})
        output_metadata["last_unit_id"] = unit.unit_id
        output_metadata["last_unit_kind"] = unit.kind
        output_metadata["execution_adapter"] = self.name
        prior_history = output_metadata.get("unit_history", [])
        if not isinstance(prior_history, list):
            prior_history = []
        output_metadata["unit_history"] = [*prior_history, unit.unit_id]
        return TensorPayload.from_numpy(output, name=name, metadata=output_metadata)


class TensorSequenceRuntimeCoordinator:
    def __init__(
        self,
        *,
        adapter: TensorRuntimeAdapter | None = None,
        planner: Planner | None = None,
    ) -> None:
        self.adapter = adapter or TensorRuntimeAdapter()
        self.planner = planner or Planner()

    def calibrate(self, manifest: ModelManifest, resources: Sequence[ResourceProfile]) -> list[CalibrationProfile]:
        return [self.adapter.calibrate(resource, manifest) for resource in resources]

    def execute(
        self,
        manifest: ModelManifest,
        resources: Sequence[ResourceProfile],
        *,
        prompt: str,
        initial_tensor: TensorPayload | None = None,
        hidden_size: int = 16,
        calibrations: Sequence[CalibrationProfile] | None = None,
    ) -> TensorSequenceExecutionResult:
        resolved_calibrations = list(calibrations or self.calibrate(manifest, resources))
        plan = self.planner.plan(manifest, resources, resolved_calibrations)
        tensor = initial_tensor or prompt_to_tensor(prompt, hidden_size=hidden_size)
        traces: list[TensorUnitExecutionTrace] = []

        for unit in manifest.graph.ordered_units():
            placement = plan.placement_for(unit.unit_id)
            before = tensor.digest()
            input_shape = list(tensor.shape)
            started = time.perf_counter()
            tensor = self.adapter.execute_unit(unit, tensor)
            observed_latency_ms = (time.perf_counter() - started) * 1000.0
            after = tensor.digest()
            traces.append(
                TensorUnitExecutionTrace(
                    unit_id=unit.unit_id,
                    unit_kind=unit.kind,
                    runtime_id=placement.runtime_id,
                    input_digest=before,
                    output_digest=after,
                    observed_latency_ms=observed_latency_ms,
                    input_shape=input_shape,
                    output_shape=list(tensor.shape),
                    dtype=tensor.dtype,
                )
            )

        output_digest = tensor.digest()
        architecture_neutral = bool(manifest.graph.unit_kinds() - {"embedding", "attention_block", "mlp", "norm", "lm_head"})
        return TensorSequenceExecutionResult(
            model_id=manifest.model_id,
            prompt=prompt,
            plan=plan,
            traces=traces,
            final_tensor=tensor,
            output_text=f"tensor-pipeline:{manifest.model_id}:{output_digest[:12]}",
            architecture_neutral=architecture_neutral,
            metadata={
                "adapter": self.adapter.name,
                "transport": "in-process",
                "unit_count": len(traces),
                "unit_kinds": sorted(manifest.graph.unit_kinds()),
                "runtimes_used": sorted(plan.runtime_ids()),
                "payload": "typed-tensor-base64-json",
                "weight_source": "deterministic-prototype-weights",
            },
        )


def prompt_to_tensor(prompt: str, *, hidden_size: int = 16, max_tokens: int = 8) -> TensorPayload:
    words = prompt.split() or [prompt or "<empty>"]
    sequence_length = max(1, min(len(words), max_tokens))
    rows: list[np.ndarray] = []
    for index in range(sequence_length):
        rows.append(_stable_signed_array(f"{prompt}\0{index}", (hidden_size,), scale=0.35))
    array = np.stack(rows, axis=0)
    return TensorPayload.from_numpy(
        array,
        name="activation",
        metadata={
            "source": "prompt",
            "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "hidden_size": hidden_size,
        },
    )


def _execute_array_unit(unit: ModelUnit, array: np.ndarray) -> np.ndarray:
    x = _as_2d(array)
    if unit.kind == "embedding":
        result = np.tanh(x + _bias(unit, x.shape[-1], scale=0.08))
    elif unit.kind == "attention_block":
        scores = _stable_matmul(x, x.T) / math.sqrt(max(x.shape[-1], 1))
        weights = _softmax(scores)
        context = _stable_matmul(weights, x)
        result = _layer_norm(x * 0.65 + context * 0.30 + _linear(unit, x, scale=0.05))
    elif unit.kind == "ssm_block":
        carry = np.zeros((x.shape[-1],), dtype=np.float32)
        rows = []
        bias = _bias(unit, x.shape[-1], scale=0.03)
        for row in x:
            carry = carry * 0.70 + row * 0.30 + bias
            rows.append(carry.copy())
        result = _layer_norm(np.stack(rows, axis=0))
    elif unit.kind == "deltanet_block":
        previous = np.concatenate([np.zeros_like(x[:1]), x[:-1]], axis=0)
        delta = x - previous
        result = _layer_norm(x + delta * 0.25 + _linear(unit, x, scale=0.04))
    elif unit.kind == "mlp":
        hidden = np.tanh(_linear(unit, x, scale=0.20) + _bias(unit, x.shape[-1], scale=0.05))
        result = _layer_norm(x + hidden * 0.35)
    elif unit.kind == "moe":
        gates = (np.abs(np.floor(np.sum(x, axis=-1) * 10)).astype(int) % 3).reshape(-1, 1)
        multipliers = np.take(np.array([0.82, 1.05, 1.22], dtype=np.float32), gates)
        result = _layer_norm(x * multipliers + _linear(unit, x, scale=0.03))
    elif unit.kind == "norm":
        result = _layer_norm(x)
    elif unit.kind == "adapter":
        result = _layer_norm(x + _linear(unit, x, scale=0.02))
    elif unit.kind == "lm_head":
        result = _layer_norm(np.mean(x, axis=0, keepdims=True))
    else:
        result = x
    return result.reshape(array.shape) if array.ndim == 1 and result.shape[0] == 1 else result


def _as_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        return array.reshape(-1, array.shape[-1])
    return array


def _bias(unit: ModelUnit, width: int, *, scale: float) -> np.ndarray:
    return _stable_signed_array(f"{unit.unit_id}:bias", (width,), scale=scale)


def _linear(unit: ModelUnit, array: np.ndarray, *, scale: float) -> np.ndarray:
    matrix = _stable_signed_array(
        f"{unit.unit_id}:matrix:{array.shape[-1]}",
        (array.shape[-1], array.shape[-1]),
        scale=scale,
    )
    return _stable_matmul(array, matrix)


def _layer_norm(array: np.ndarray) -> np.ndarray:
    x = _as_2d(np.asarray(array, dtype=np.float64))
    output = np.empty_like(x, dtype=np.float64)
    for row_index, row in enumerate(x):
        mean = sum(float(value) for value in row) / max(len(row), 1)
        variance = sum((float(value) - mean) ** 2 for value in row) / max(len(row), 1)
        denominator = math.sqrt(variance + 1e-6)
        output[row_index] = [(float(value) - mean) / denominator for value in row]
    return output.astype(np.float32)


def _softmax(array: np.ndarray) -> np.ndarray:
    x = _as_2d(np.asarray(array, dtype=np.float64))
    output = np.empty_like(x, dtype=np.float64)
    for row_index, row in enumerate(x):
        max_value = max(float(value) for value in row)
        exp_values = [math.exp(float(value) - max_value) for value in row]
        total = sum(exp_values)
        output[row_index] = [value / total for value in exp_values]
    return output.astype(np.float32)


def _stable_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs = _as_2d(np.asarray(left, dtype=np.float64))
    rhs = _as_2d(np.asarray(right, dtype=np.float64))
    if lhs.shape[-1] != rhs.shape[0]:
        raise ValueError(f"matmul shape mismatch: {lhs.shape} x {rhs.shape}")
    output = np.empty((lhs.shape[0], rhs.shape[1]), dtype=np.float64)
    for row_index in range(lhs.shape[0]):
        for column_index in range(rhs.shape[1]):
            total = 0.0
            for shared_index in range(lhs.shape[1]):
                total += float(lhs[row_index, shared_index]) * float(rhs[shared_index, column_index])
            output[row_index, column_index] = total
    return output.astype(np.float32)


def _canonical_float32(array: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(array, dtype=np.float64), decimals=6)
    rounded[np.abs(rounded) < 0.0000005] = 0.0
    return rounded.astype(np.float32)


def _stable_signed_array(seed: str, shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    element_count = math.prod(shape)
    byte_count = element_count * 4
    chunks: list[bytes] = []
    counter = 0
    seed_bytes = seed.encode("utf-8")
    while sum(len(chunk) for chunk in chunks) < byte_count:
        chunks.append(hashlib.sha256(seed_bytes + counter.to_bytes(8, "big")).digest())
        counter += 1
    raw = b"".join(chunks)[:byte_count]
    words = np.frombuffer(raw, dtype=">u4").astype(np.float64)
    values = ((words / 4294967295.0) * 2.0 - 1.0) * scale
    return values.astype(np.float32).reshape(shape)
