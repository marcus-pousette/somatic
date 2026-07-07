from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


BOUNDARY_CODEC_BACKEND_CONTRACT_SCHEMA_VERSION = "boundary-codec-backend-contract-v0"
BOUNDARY_CODEC_BACKEND_RUNTIME_DECISION_SCHEMA_VERSION = "boundary-codec-backend-runtime-decision-v0"
BOUNDARY_CODEC_BACKEND_CONTRACT_METADATA_KEY = "boundary_codec_backend_contract"
BOUNDARY_CODEC_BACKEND_REQUEST_METADATA_KEY = "requested_boundary_codec_backend_id"
BOUNDARY_CODEC_BACKEND_RUNTIME_METADATA_KEY = "boundary_codec_backend"
BOUNDARY_CODEC_RAW_FRAME_RECEIVE_ENABLED_METADATA_KEY = "boundary_codec_raw_frame_receive_enabled"
BOUNDARY_CODEC_RAW_FRAME_RECEIVE_METADATA_KEY = "boundary_codec_raw_frame_receive"
BoundaryCodecBackendId = Literal["python_reference", "rust_native"]
BoundaryCodecBackendSelectionStatus = Literal["selected", "fallback"]
BoundaryCodecBackendOperation = Literal["encode", "decode"]


class BoundaryCodecBackendContract(BaseModel):
    schema_version: str = BOUNDARY_CODEC_BACKEND_CONTRACT_SCHEMA_VERSION
    status: Literal["rust_native_optional", "python_reference_only", "blocked"]
    default_backend_id: BoundaryCodecBackendId = "python_reference"
    optional_backend_ids: list[BoundaryCodecBackendId] = Field(default_factory=lambda: ["python_reference"])
    recommended_backend_id: BoundaryCodecBackendId = "python_reference"
    adapter_ids: list[str] = Field(default_factory=list)
    planner_adapter_semantics_unchanged: bool = True
    runtime_default_changed: bool = False
    python_reference_fallback_required: bool = True
    rust_native_runtime_execution_allowed: bool = False
    rust_native_production_default_allowed: bool = False
    source_artifacts: dict[str, str] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    non_claims: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BoundaryCodecBackendSelection(BaseModel):
    schema_version: str = "boundary-codec-backend-selection-v0"
    requested_backend_id: BoundaryCodecBackendId | None = None
    selected_backend_id: BoundaryCodecBackendId
    status: BoundaryCodecBackendSelectionStatus
    reason_codes: list[str] = Field(default_factory=list)
    shape_aware_policy_matched: bool = False
    planner_adapter_semantics_unchanged: bool = True
    runtime_default_changed: bool = False


class BoundaryCodecBackendRuntimeDecision(BaseModel):
    schema_version: str = BOUNDARY_CODEC_BACKEND_RUNTIME_DECISION_SCHEMA_VERSION
    operation: BoundaryCodecBackendOperation
    adapter_id: str
    requested_backend_id: BoundaryCodecBackendId | None = None
    selected_backend_id: BoundaryCodecBackendId
    executed_backend_id: BoundaryCodecBackendId
    selection_status: BoundaryCodecBackendSelectionStatus
    status: Literal["executed", "fallback"]
    reason_codes: list[str] = Field(default_factory=list)
    rust_native_runtime_binding_available: bool = False
    planner_adapter_semantics_unchanged: bool = True
    runtime_default_changed: bool = False


def build_boundary_codec_backend_contract(
    *,
    adapter_ids: list[str],
    parity_passed: bool,
    benchmark_passed: bool,
    rust_benchmark_available: bool,
    benchmark_result_path: Path | str | None = None,
    parity_result_path: Path | str | None = None,
    rust_benchmark_report_path: Path | str | None = None,
    recommended_backend_id: BoundaryCodecBackendId | None = None,
    metadata: dict[str, Any] | None = None,
) -> BoundaryCodecBackendContract:
    reason_codes: list[str] = []
    optional_backend_ids: list[BoundaryCodecBackendId] = ["python_reference"]
    rust_allowed = parity_passed and benchmark_passed and rust_benchmark_available
    if rust_allowed:
        optional_backend_ids.append("rust_native")
        status: Literal["rust_native_optional", "python_reference_only", "blocked"] = "rust_native_optional"
    else:
        status = "python_reference_only"
        if not parity_passed:
            reason_codes.append("rust_native_parity_not_proven")
        if not benchmark_passed:
            reason_codes.append("rust_native_benchmark_not_passed")
        if not rust_benchmark_available:
            reason_codes.append("rust_native_benchmark_unavailable")
    resolved_recommended = recommended_backend_id or ("rust_native" if rust_allowed else "python_reference")
    if resolved_recommended == "rust_native" and "rust_native" not in optional_backend_ids:
        resolved_recommended = "python_reference"
        reason_codes.append("requested_rust_native_recommendation_fell_back_to_python_reference")
    source_artifacts = {}
    if benchmark_result_path is not None:
        source_artifacts["benchmark_result_path"] = str(benchmark_result_path)
    if parity_result_path is not None:
        source_artifacts["parity_result_path"] = str(parity_result_path)
    if rust_benchmark_report_path is not None:
        source_artifacts["rust_benchmark_report_path"] = str(rust_benchmark_report_path)
    return BoundaryCodecBackendContract(
        status=status,
        optional_backend_ids=optional_backend_ids,
        recommended_backend_id=resolved_recommended,
        adapter_ids=sorted(set(adapter_ids)),
        rust_native_runtime_execution_allowed=rust_allowed,
        source_artifacts=source_artifacts,
        reason_codes=reason_codes,
        non_claims=[
            "Codec backend selection does not change planner-selectable adapter ids.",
            "Python/NumPy remains the default and fallback codec backend.",
            "Rust native codec support is optional local evidence, not a production transport default.",
            "This contract does not claim full distributed training, pretrained-Qwen fine-tuning, public-worker security, crypto settlement, token launch readiness, payouts, custody, or slashing.",
        ],
        metadata=metadata or {},
    )


def coerce_boundary_codec_backend_contract(value: Any) -> BoundaryCodecBackendContract | None:
    if isinstance(value, BoundaryCodecBackendContract):
        return value
    if isinstance(value, dict):
        try:
            return BoundaryCodecBackendContract.model_validate(value)
        except ValueError:
            return None
    return None


def select_boundary_codec_backend(
    contract: BoundaryCodecBackendContract,
    *,
    requested_backend_id: BoundaryCodecBackendId | None = None,
    adapter_id: str | None = None,
    tensor_shape: list[int] | None = None,
    operation: BoundaryCodecBackendOperation | None = None,
    allow_shape_aware_policy: bool = False,
) -> BoundaryCodecBackendSelection:
    reason_codes: list[str] = []
    shape_aware_policy_matched = False
    requested = requested_backend_id
    if requested is None and allow_shape_aware_policy:
        candidate = _shape_aware_backend_policy_candidate(
            contract=contract,
            adapter_id=adapter_id,
            tensor_shape=tensor_shape,
            operation=operation,
        )
        if candidate == "rust_native":
            requested = "rust_native"
            shape_aware_policy_matched = True
            reason_codes.append("shape_aware_backend_candidate_region_matched")
    if requested is None:
        requested = contract.recommended_backend_id
    if requested in contract.optional_backend_ids:
        return BoundaryCodecBackendSelection(
            requested_backend_id=requested_backend_id,
            selected_backend_id=requested,
            status="selected",
            reason_codes=reason_codes,
            shape_aware_policy_matched=shape_aware_policy_matched,
        )
    return BoundaryCodecBackendSelection(
        requested_backend_id=requested_backend_id,
        selected_backend_id=contract.default_backend_id,
        status="fallback",
        reason_codes=[*reason_codes, "requested_backend_not_allowed_by_contract"],
        shape_aware_policy_matched=shape_aware_policy_matched,
    )


def resolve_boundary_codec_backend_runtime_decision(
    *,
    contract: BoundaryCodecBackendContract | None,
    requested_backend_id: BoundaryCodecBackendId | None = None,
    adapter_id: str,
    operation: BoundaryCodecBackendOperation,
    rust_native_runtime_binding_available: bool = False,
    tensor_shape: list[int] | None = None,
    allow_shape_aware_policy: bool = False,
) -> BoundaryCodecBackendRuntimeDecision:
    if contract is None:
        reason_codes = ["boundary_codec_backend_contract_missing"]
        selection_status: BoundaryCodecBackendSelectionStatus = "selected"
        if requested_backend_id == "rust_native":
            selection_status = "fallback"
            reason_codes.append("requested_rust_native_requires_contract")
        return BoundaryCodecBackendRuntimeDecision(
            operation=operation,
            adapter_id=adapter_id,
            requested_backend_id=requested_backend_id,
            selected_backend_id="python_reference",
            executed_backend_id="python_reference",
            selection_status=selection_status,
            status="fallback" if selection_status == "fallback" else "executed",
            reason_codes=reason_codes,
            rust_native_runtime_binding_available=rust_native_runtime_binding_available,
        )

    selection = select_boundary_codec_backend(
        contract,
        requested_backend_id=requested_backend_id,
        adapter_id=adapter_id,
        tensor_shape=tensor_shape,
        operation=operation,
        allow_shape_aware_policy=allow_shape_aware_policy,
    )
    reason_codes = list(selection.reason_codes)
    if selection.selected_backend_id == "rust_native" and not rust_native_runtime_binding_available:
        reason_codes.append("rust_native_runtime_binding_unavailable")
        reason_codes.append("python_reference_fallback_executed")
        return BoundaryCodecBackendRuntimeDecision(
            operation=operation,
            adapter_id=adapter_id,
            requested_backend_id=requested_backend_id,
            selected_backend_id=selection.selected_backend_id,
            executed_backend_id="python_reference",
            selection_status=selection.status,
            status="fallback",
            reason_codes=reason_codes,
            rust_native_runtime_binding_available=False,
            planner_adapter_semantics_unchanged=contract.planner_adapter_semantics_unchanged,
            runtime_default_changed=contract.runtime_default_changed,
        )
    return BoundaryCodecBackendRuntimeDecision(
        operation=operation,
        adapter_id=adapter_id,
        requested_backend_id=requested_backend_id,
        selected_backend_id=selection.selected_backend_id,
        executed_backend_id=selection.selected_backend_id,
        selection_status=selection.status,
        status="fallback" if selection.status == "fallback" else "executed",
        reason_codes=reason_codes,
        rust_native_runtime_binding_available=rust_native_runtime_binding_available,
        planner_adapter_semantics_unchanged=contract.planner_adapter_semantics_unchanged,
        runtime_default_changed=contract.runtime_default_changed,
    )


def _shape_aware_backend_policy_candidate(
    *,
    contract: BoundaryCodecBackendContract,
    adapter_id: str | None,
    tensor_shape: list[int] | None,
    operation: BoundaryCodecBackendOperation | None = None,
) -> BoundaryCodecBackendId | None:
    if adapter_id is None or tensor_shape is None:
        return None
    policy = contract.metadata.get("shape_aware_backend_policy") or {}
    if not isinstance(policy, dict) or policy.get("enabled") is not True:
        return None
    if policy.get("default_backend_id", "python_reference") != "python_reference":
        return None
    regions = policy.get("candidate_regions") or []
    if not isinstance(regions, list):
        return None
    normalized_shape = [int(dim) for dim in tensor_shape]
    for region in regions:
        if not isinstance(region, dict):
            continue
        if str(region.get("adapter_id") or "") != adapter_id:
            continue
        region_shape = region.get("input_shape") or region.get("shape")
        if not isinstance(region_shape, list):
            continue
        if [int(dim) for dim in region_shape] != normalized_shape:
            continue
        region_operation = region.get("operation") or region.get("backend_operation") or "encode_decode"
        if operation is not None and region_operation not in {operation, "encode_decode", "both", "any"}:
            continue
        backend_id = region.get("backend_id") or "rust_native"
        if backend_id == "rust_native":
            return "rust_native"
    return None
