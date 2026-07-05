from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from somatic.sequence_model.execution import SimulatedRuntimeAdapter
from somatic.sequence_model.interfaces import CalibrationProfile, ModelManifest, ModelUnit, Precision, ResourceProfile, UnitKind
from somatic.sequence_model.boundary_adapters import (
    BOUNDARY_RAW_RESPONSE_ENCODE_TIMING_METADATA_KEY,
    BOUNDARY_POSITION_SCOPED_TRANSPORT_METADATA_KEY,
    boundary_adapter_summary_from_metadata,
    compact_boundary_adapter_metrics,
    decode_boundary_frame_parts_to_numpy,
    decode_payload_from_boundary,
    encode_payload_for_boundary_raw_frame,
    normalize_boundary_adapter_strategy,
)
from somatic.sequence_model.boundary_codec_backend import (
    BOUNDARY_CODEC_BACKEND_CONTRACT_METADATA_KEY,
    BOUNDARY_CODEC_BACKEND_REQUEST_METADATA_KEY,
    BOUNDARY_CODEC_RAW_FRAME_RECEIVE_ENABLED_METADATA_KEY,
)
from somatic.sequence_model.boundary_compression.wire_codec_runtime import (
    LEARNED_INT8_BOUNDARY_ADAPTER_ID,
    LEARNED_INT8_WIRE_PAYLOAD_KIND,
    TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
    TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND,
    LearnedInt8WireCodec,
    LearnedInt8WireCodecArtifact,
    decode_boundary_wire_payload,
    encode_boundary_wire_payload,
    load_learned_int8_wire_codec_artifact,
    load_learned_int8_wire_codec_from_artifact,
)
from somatic.sequence_model.boundary_compression.trainable_autoencoder_wire import (
    load_trainable_autoencoder_basis_artifact,
)
from somatic.sequence_model.kv_cache import KVCacheHandle
from somatic.sequence_model.qwen_real import QwenCacheForwardResult, QwenShardTrace, QwenWorkerLayerRuntime
from somatic.sequence_model.tensor_execution import TensorRuntimeAdapter
from somatic.sequence_model.telemetry import SequenceWorkerMetrics, collect_sequence_worker_metrics
from somatic.sequence_model.tensors import (
    TENSOR_FRAME_MEDIA_TYPE,
    TensorPayload,
    decode_tensor_frame_to_numpy,
    decode_tensor_frame_to_raw,
    encode_tensor_frame,
    encode_tensor_frame_from_raw,
)
from somatic.sequence_model.transformers_qwen import QwenGenerationResult, TransformersQwenAdapter
from somatic.servers.provider_probe import (
    ProviderProbeResponder,
    mount_provider_probe_endpoint,
    mount_provider_startup_observation_endpoint,
)
from somatic.security.public_worker import (
    PeerClass,
    ProductionWorkerSecurityVerification,
    SecurityAdmissionDecision,
    public_worker_security_decision_metadata,
    validate_public_worker_security_decision_metadata,
)


class SequenceProfileResponse(BaseModel):
    resource_profile: ResourceProfile


class SequenceMetricsResponse(BaseModel):
    metrics: SequenceWorkerMetrics


class SequenceCalibrationRequest(BaseModel):
    manifest: ModelManifest


class SequenceCalibrationResponse(BaseModel):
    calibration: CalibrationProfile


class SequenceExecuteUnitRequest(BaseModel):
    unit: ModelUnit
    payload: dict[str, Any] = Field(default_factory=dict)
    precision: Precision = "bf16"
    public_worker_security_decision: SecurityAdmissionDecision | None = None
    production_worker_security_verification: ProductionWorkerSecurityVerification | None = None


class SequenceExecuteUnitResponse(BaseModel):
    payload: dict[str, Any]


class SequenceExecuteTensorUnitRequest(BaseModel):
    unit: ModelUnit
    tensor: TensorPayload
    precision: Precision = "bf16"
    public_worker_security_decision: SecurityAdmissionDecision | None = None
    production_worker_security_verification: ProductionWorkerSecurityVerification | None = None


class SequenceExecuteTensorUnitResponse(BaseModel):
    tensor: TensorPayload


class SequenceGenerateRequest(BaseModel):
    adapter: str = "transformers-qwen"
    model_id: str = "Qwen/Qwen3-0.6B"
    prompt: str
    max_new_tokens: int = Field(default=24, ge=1, le=256)
    device: str = "cpu"
    local_files_only: bool = False
    precision: Precision = "bf16"
    public_worker_security_decision: SecurityAdmissionDecision | None = None
    production_worker_security_verification: ProductionWorkerSecurityVerification | None = None


class SequenceGenerateResponse(BaseModel):
    result: QwenGenerationResult


class SequenceQwenShardForwardRequest(BaseModel):
    tensor: TensorPayload
    public_worker_security_decision: SecurityAdmissionDecision | None = None
    production_worker_security_verification: ProductionWorkerSecurityVerification | None = None


class SequenceQwenShardForwardResponse(BaseModel):
    tensor: TensorPayload
    trace: QwenShardTrace


class SequenceQwenCacheCreateRequest(BaseModel):
    sequence_id: str
    ttl_seconds: float | None = Field(default=None, gt=0.0)
    public_worker_security_decision: SecurityAdmissionDecision | None = None
    production_worker_security_verification: ProductionWorkerSecurityVerification | None = None


class SequenceQwenCacheResponse(BaseModel):
    cache: KVCacheHandle


class SequenceQwenCacheForwardRequest(BaseModel):
    tensor: TensorPayload
    cache_id: str
    sequence_id: str
    position_start: int = Field(ge=0)
    public_worker_security_decision: SecurityAdmissionDecision | None = None
    production_worker_security_verification: ProductionWorkerSecurityVerification | None = None


class SequenceQwenCacheForwardResponse(BaseModel):
    result: QwenCacheForwardResult


class SequenceQwenCacheReleaseRequest(BaseModel):
    cache_id: str
    sequence_id: str
    public_worker_security_decision: SecurityAdmissionDecision | None = None
    production_worker_security_verification: ProductionWorkerSecurityVerification | None = None


class SequenceQwenCacheTruncateRequest(BaseModel):
    cache_id: str
    sequence_id: str
    length: int
    public_worker_security_decision: SecurityAdmissionDecision | None = None
    production_worker_security_verification: ProductionWorkerSecurityVerification | None = None


def create_app(
    *,
    resource_profile: ResourceProfile | None = None,
    qwen_adapter: TransformersQwenAdapter | None = None,
    qwen_shard_runtime: QwenWorkerLayerRuntime | None = None,
    auth_token: str | None = None,
    max_request_bytes: int = 16 * 1024 * 1024,
    request_timeout_seconds: float = 600.0,
    audit_log_path: str | Path | None = None,
    require_public_worker_security: bool | None = None,
    public_worker_security_production_mode: bool = False,
    trusted_production_worker_security_verifier_ids: set[str] | None = None,
    trusted_production_worker_security_verifier_keys: dict[str, str] | None = None,
    enable_provider_probe_endpoint: bool = False,
    provider_probe_responder: ProviderProbeResponder | None = None,
    provider_probe_auth_token: str | None = None,
    provider_runtime_startup_observation: dict[str, Any] | None = None,
    learned_wire_codec_artifact_path: str | Path | None = None,
    trainable_autoencoder_basis_artifact_path: str | Path | None = None,
    trainable_autoencoder_source_strategy: str = "int8_symmetric",
    trainable_autoencoder_allow_source_only_fallback: bool = False,
) -> FastAPI:
    resolved_profile = resource_profile or default_resource_profile()
    learned_wire_codec: LearnedInt8WireCodec | None = None
    learned_wire_codec_artifact: LearnedInt8WireCodecArtifact | None = None
    if learned_wire_codec_artifact_path is not None:
        learned_wire_codec_artifact = load_learned_int8_wire_codec_artifact(learned_wire_codec_artifact_path)
        learned_wire_codec = load_learned_int8_wire_codec_from_artifact(learned_wire_codec_artifact_path)
    trainable_autoencoder_basis_artifact: dict[str, Any] | None = None
    if trainable_autoencoder_basis_artifact_path is not None:
        trainable_autoencoder_basis_artifact = load_trainable_autoencoder_basis_artifact(
            str(trainable_autoencoder_basis_artifact_path)
        )
    resolved_profile = _profile_with_stable_split_metadata(
        resolved_profile,
        qwen_shard_runtime=qwen_shard_runtime,
        learned_wire_codec_artifact=learned_wire_codec_artifact,
        trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
    )
    adapter = SimulatedRuntimeAdapter()
    tensor_adapter = TensorRuntimeAdapter()
    resolved_qwen_adapter = qwen_adapter or TransformersQwenAdapter()
    app = FastAPI(title=f"Somatic sequence worker {resolved_profile.runtime_id}")
    resolved_audit_log_path = Path(audit_log_path) if audit_log_path is not None else None
    shard_id = qwen_shard_runtime.shard.manifest.shard_id if qwen_shard_runtime is not None else None
    resolved_peer_class = _peer_class_for_profile(resolved_profile)
    public_worker_security_required = (
        resolved_peer_class in {"public", "rented"}
        if require_public_worker_security is None
        else require_public_worker_security
    )
    resolved_trusted_production_verifier_ids = (
        set(trusted_production_worker_security_verifier_ids or [])
        if public_worker_security_production_mode
        else None
    )
    resolved_trusted_production_verifier_keys = (
        dict(trusted_production_worker_security_verifier_keys or {})
        if public_worker_security_production_mode
        else None
    )
    if enable_provider_probe_endpoint or provider_probe_responder is not None:
        mount_provider_probe_endpoint(
            app,
            runtime_id=resolved_profile.runtime_id,
            supported_job_kinds={"inference"},
            responder=provider_probe_responder,
            auth_token=provider_probe_auth_token if provider_probe_auth_token is not None else auth_token,
        )
    startup_observation = _sequence_runtime_startup_observation(
        provider_runtime_startup_observation,
        qwen_shard_runtime=qwen_shard_runtime,
    )
    if startup_observation is not None:
        mount_provider_startup_observation_endpoint(
            app,
            observation=startup_observation,
            auth_token=provider_probe_auth_token if provider_probe_auth_token is not None else auth_token,
        )

    @app.middleware("http")
    async def sequence_security_middleware(request: Request, call_next: Any) -> Response:
        if not request.url.path.startswith("/sequence/"):
            return await call_next(request)

        request_id = request.headers.get("x-request-id") or str(uuid4())
        started = time.perf_counter()
        request_bytes = 0
        response: Response | None = None

        try:
            body = await request.body()
            request_bytes = len(body)
            if request_bytes > max_request_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": f"sequence request body exceeds {max_request_bytes} bytes"},
                )
            else:
                replayed = False

                async def receive() -> dict[str, Any]:
                    nonlocal replayed
                    if replayed:
                        return {"type": "http.request", "body": b"", "more_body": False}
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}

                replay_request = Request(request.scope, receive)
                if request_timeout_seconds > 0:
                    response = await asyncio.wait_for(call_next(replay_request), timeout=request_timeout_seconds)
                else:
                    response = await call_next(replay_request)
        except asyncio.TimeoutError:
            response = JSONResponse(status_code=504, content={"detail": "sequence request timed out"})
        except Exception as exc:
            _write_audit_event(
                path=resolved_audit_log_path,
                event={
                    "request_id": request_id,
                    "runtime_id": resolved_profile.runtime_id,
                    "shard_id": shard_id,
                    "method": request.method,
                    "route": request.url.path,
                    "status_code": 500,
                    "request_bytes": request_bytes,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "auth_configured": auth_token is not None,
                    "error_type": type(exc).__name__,
                },
            )
            raise

        response.headers["x-request-id"] = request_id
        _write_audit_event(
            path=resolved_audit_log_path,
            event={
                "request_id": request_id,
                "runtime_id": resolved_profile.runtime_id,
                "shard_id": shard_id,
                "method": request.method,
                "route": request.url.path,
                "status_code": response.status_code,
                "request_bytes": request_bytes,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "auth_configured": auth_token is not None,
            },
        )
        return response

    def require_sequence_auth(authorization: str | None = Header(default=None)) -> None:
        if auth_token is None:
            return
        if authorization != f"Bearer {auth_token}":
            raise HTTPException(status_code=401, detail="sequence worker authorization failed")

    def require_inference_security(
        decision: SecurityAdmissionDecision | None,
        production_verification: ProductionWorkerSecurityVerification | None = None,
    ) -> None:
        if not public_worker_security_required and decision is None:
            return
        if decision is None:
            raise HTTPException(status_code=403, detail="public_worker_security_decision_required")
        metadata = public_worker_security_decision_metadata(decision, production_verification)
        reason_codes = validate_public_worker_security_decision_metadata(
            metadata,
            runtime_id=resolved_profile.runtime_id,
            peer_class=resolved_peer_class if resolved_peer_class in {"owner", "friend", "rented", "public"} else None,
            job_kind="inference",
            production_mode=public_worker_security_production_mode,
            trusted_production_verifier_ids=resolved_trusted_production_verifier_ids,
            trusted_production_verifier_keys=resolved_trusted_production_verifier_keys,
        )
        if reason_codes:
            raise HTTPException(
                status_code=403,
                detail=f"public worker security admission rejected: {reason_codes[0]}",
            )

    def worker_telemetry_headers() -> dict[str, str]:
        qwen_metrics_runtime = (
            qwen_shard_runtime if hasattr(qwen_shard_runtime, "manifest") else None
        )
        metrics = collect_sequence_worker_metrics(
            runtime_id=resolved_profile.runtime_id,
            qwen_shard_runtime=qwen_metrics_runtime,
        )
        return _worker_telemetry_headers(
            process_cpu_seconds=metrics.process_cpu_seconds,
            rss_bytes=metrics.rss_bytes or metrics.max_rss_bytes,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "runtime_id": resolved_profile.runtime_id}

    @app.get("/sequence/profile", response_model=SequenceProfileResponse)
    def profile(_: None = Depends(require_sequence_auth)) -> SequenceProfileResponse:
        return SequenceProfileResponse(resource_profile=resolved_profile)

    @app.get("/sequence/metrics", response_model=SequenceMetricsResponse)
    def metrics(_: None = Depends(require_sequence_auth)) -> SequenceMetricsResponse:
        return SequenceMetricsResponse(
            metrics=collect_sequence_worker_metrics(
                runtime_id=resolved_profile.runtime_id,
                qwen_shard_runtime=qwen_shard_runtime,
            )
        )

    @app.post("/sequence/calibrate", response_model=SequenceCalibrationResponse)
    def calibrate(request: SequenceCalibrationRequest, _: None = Depends(require_sequence_auth)) -> SequenceCalibrationResponse:
        return SequenceCalibrationResponse(calibration=adapter.calibrate(resolved_profile, request.manifest))

    @app.post("/sequence/execute-unit", response_model=SequenceExecuteUnitResponse)
    def execute_unit(request: SequenceExecuteUnitRequest, _: None = Depends(require_sequence_auth)) -> SequenceExecuteUnitResponse:
        require_inference_security(
            request.public_worker_security_decision,
            request.production_worker_security_verification,
        )
        if not resolved_profile.supports(request.unit, request.precision):
            raise HTTPException(status_code=422, detail=f"{resolved_profile.runtime_id} cannot execute {request.unit.kind} at {request.precision}")
        return SequenceExecuteUnitResponse(payload=adapter.execute_unit(request.unit, request.payload))

    @app.post("/sequence/execute-tensor-unit", response_model=SequenceExecuteTensorUnitResponse)
    def execute_tensor_unit(
        request: SequenceExecuteTensorUnitRequest,
        _: None = Depends(require_sequence_auth),
    ) -> SequenceExecuteTensorUnitResponse:
        require_inference_security(
            request.public_worker_security_decision,
            request.production_worker_security_verification,
        )
        if not resolved_profile.supports(request.unit, request.precision):
            raise HTTPException(status_code=422, detail=f"{resolved_profile.runtime_id} cannot execute {request.unit.kind} at {request.precision}")
        return SequenceExecuteTensorUnitResponse(tensor=tensor_adapter.execute_unit(request.unit, request.tensor))

    @app.post("/sequence/execute-tensor-unit-binary")
    async def execute_tensor_unit_binary(
        request: Request,
        _: None = Depends(require_sequence_auth),
    ) -> Response:
        try:
            array, tensor_header, header = decode_tensor_frame_to_numpy(await request.body())
            unit = ModelUnit.model_validate(header.get("unit"))
            precision = header.get("precision", "bf16")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid tensor frame request: {exc}") from exc
        require_inference_security(
            _public_worker_security_decision_from_frame_header(header),
            _production_worker_security_verification_from_frame_header(header),
        )
        if not resolved_profile.supports(unit, precision):
            raise HTTPException(status_code=422, detail=f"{resolved_profile.runtime_id} cannot execute {unit.kind} at {precision}")
        output = tensor_adapter.execute_array_unit(
            unit,
            array,
            name=str(tensor_header.get("name") or "activation"),
            metadata=tensor_header.get("metadata") or {},
        )
        return Response(
            content=encode_tensor_frame(output, extra_header={"precision": precision, "transport": "binary-tensor-frame-v1"}),
            media_type=TENSOR_FRAME_MEDIA_TYPE,
            headers=worker_telemetry_headers(),
        )

    @app.post("/sequence/generate", response_model=SequenceGenerateResponse)
    def generate(request: SequenceGenerateRequest, _: None = Depends(require_sequence_auth)) -> SequenceGenerateResponse:
        require_inference_security(
            request.public_worker_security_decision,
            request.production_worker_security_verification,
        )
        if request.adapter != "transformers-qwen":
            raise HTTPException(status_code=422, detail=f"unsupported generation adapter {request.adapter}")
        result = resolved_qwen_adapter.generate(
            model_id=request.model_id,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            device=request.device,
            local_files_only=request.local_files_only,
            default_precision=request.precision,
        )
        result.metadata["worker_runtime_id"] = resolved_profile.runtime_id
        result.metadata["transport"] = "sequence-worker-http"
        return SequenceGenerateResponse(result=result)

    @app.post("/sequence/qwen/forward-shard", response_model=SequenceQwenShardForwardResponse)
    def qwen_forward_shard(
        request: SequenceQwenShardForwardRequest,
        _: None = Depends(require_sequence_auth),
    ) -> SequenceQwenShardForwardResponse:
        if qwen_shard_runtime is None:
            raise HTTPException(status_code=422, detail="Qwen shard runtime is not configured on this worker")
        require_inference_security(
            request.public_worker_security_decision,
            request.production_worker_security_verification,
        )
        tensor, trace = qwen_shard_runtime.forward_tensor(request.tensor)
        return SequenceQwenShardForwardResponse(tensor=tensor, trace=trace)

    @app.post("/sequence/qwen/forward-shard-binary")
    async def qwen_forward_shard_binary(
        request: Request,
        _: None = Depends(require_sequence_auth),
    ) -> Response:
        if qwen_shard_runtime is None:
            raise HTTPException(status_code=422, detail="Qwen shard runtime is not configured on this worker")
        try:
            array, tensor_header, header, request_boundary_adapter, request_frame_bytes = await _decode_qwen_boundary_frame_request(
                request,
                learned_wire_codec=learned_wire_codec,
                learned_wire_codec_artifact=learned_wire_codec_artifact,
                trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
            )
            require_inference_security(
                _public_worker_security_decision_from_frame_header(header),
                _production_worker_security_verification_from_frame_header(header),
            )
            tensor, trace = qwen_shard_runtime.forward_array(
                array,
                name=str(tensor_header.get("name") or "hidden_states"),
                metadata=tensor_header.get("metadata") or {},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid Qwen tensor frame request: {exc}") from exc
        return _qwen_binary_response(
            tensor=tensor,
            trace=trace,
            transport="qwen-binary-tensor-frame-v1",
            boundary_adapter_strategy=_response_boundary_adapter_strategy_from_header(
                header,
                request_boundary_adapter,
            ),
            request_boundary_adapter=request_boundary_adapter,
            request_frame_bytes=request_frame_bytes,
            compact_response_trace=_compact_qwen_response_trace_requested(header),
            response_position_scoped_transport=_response_position_scoped_transport_from_header(header),
            trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
            trainable_autoencoder_source_strategy=trainable_autoencoder_source_strategy,
            trainable_autoencoder_allow_source_only_fallback=(
                trainable_autoencoder_allow_source_only_fallback
            ),
            trainable_autoencoder_context=_trainable_autoencoder_response_context(
                header,
                path="/sequence/qwen/forward-shard-binary",
                tensor=tensor,
            ),
            headers=worker_telemetry_headers(),
        )

    @app.post("/sequence/qwen/cache/create", response_model=SequenceQwenCacheResponse)
    def qwen_cache_create(
        request: SequenceQwenCacheCreateRequest,
        _: None = Depends(require_sequence_auth),
    ) -> SequenceQwenCacheResponse:
        if qwen_shard_runtime is None:
            raise HTTPException(status_code=422, detail="Qwen shard runtime is not configured on this worker")
        require_inference_security(
            request.public_worker_security_decision,
            request.production_worker_security_verification,
        )
        try:
            return SequenceQwenCacheResponse(cache=qwen_shard_runtime.create_cache(sequence_id=request.sequence_id, ttl_seconds=request.ttl_seconds))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/sequence/qwen/cache/prefill", response_model=SequenceQwenCacheForwardResponse)
    def qwen_cache_prefill(
        request: SequenceQwenCacheForwardRequest,
        _: None = Depends(require_sequence_auth),
    ) -> SequenceQwenCacheForwardResponse:
        if qwen_shard_runtime is None:
            raise HTTPException(status_code=422, detail="Qwen shard runtime is not configured on this worker")
        require_inference_security(
            request.public_worker_security_decision,
            request.production_worker_security_verification,
        )
        try:
            result = qwen_shard_runtime.prefill_tensor(
                tensor=request.tensor,
                cache_id=request.cache_id,
                sequence_id=request.sequence_id,
                position_start=request.position_start,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown Qwen KV cache {request.cache_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SequenceQwenCacheForwardResponse(result=result)

    @app.post("/sequence/qwen/cache/prefill-binary")
    async def qwen_cache_prefill_binary(
        request: Request,
        _: None = Depends(require_sequence_auth),
    ) -> Response:
        if qwen_shard_runtime is None:
            raise HTTPException(status_code=422, detail="Qwen shard runtime is not configured on this worker")
        try:
            array, tensor_header, header, request_boundary_adapter, request_frame_bytes = await _decode_qwen_boundary_frame_request(
                request,
                learned_wire_codec=learned_wire_codec,
                learned_wire_codec_artifact=learned_wire_codec_artifact,
                trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
            )
            require_inference_security(
                _public_worker_security_decision_from_frame_header(header),
                _production_worker_security_verification_from_frame_header(header),
            )
            result = qwen_shard_runtime.prefill_array(
                array,
                name=str(tensor_header.get("name") or "hidden_states"),
                metadata=tensor_header.get("metadata") or {},
                cache_id=_required_frame_string(header, "cache_id"),
                sequence_id=_required_frame_string(header, "sequence_id"),
                position_start=_optional_frame_int(header, "position_start", default=0),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown Qwen KV cache {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _qwen_cache_binary_response(
            result,
            boundary_adapter_strategy=_response_boundary_adapter_strategy_from_header(
                header,
                request_boundary_adapter,
            ),
            request_boundary_adapter=request_boundary_adapter,
            request_frame_bytes=request_frame_bytes,
            compact_response_trace=_compact_qwen_response_trace_requested(header),
            response_position_scoped_transport=_response_position_scoped_transport_from_header(header),
            trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
            trainable_autoencoder_source_strategy=trainable_autoencoder_source_strategy,
            trainable_autoencoder_allow_source_only_fallback=(
                trainable_autoencoder_allow_source_only_fallback
            ),
            trainable_autoencoder_context=_trainable_autoencoder_response_context(
                header,
                path="/sequence/qwen/cache/prefill-binary",
                tensor=result.tensor,
            ),
            headers=worker_telemetry_headers(),
        )

    @app.post("/sequence/qwen/cache/decode", response_model=SequenceQwenCacheForwardResponse)
    def qwen_cache_decode(
        request: SequenceQwenCacheForwardRequest,
        _: None = Depends(require_sequence_auth),
    ) -> SequenceQwenCacheForwardResponse:
        if qwen_shard_runtime is None:
            raise HTTPException(status_code=422, detail="Qwen shard runtime is not configured on this worker")
        require_inference_security(
            request.public_worker_security_decision,
            request.production_worker_security_verification,
        )
        try:
            result = qwen_shard_runtime.decode_tensor(
                tensor=request.tensor,
                cache_id=request.cache_id,
                sequence_id=request.sequence_id,
                position_start=request.position_start,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown Qwen KV cache {request.cache_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SequenceQwenCacheForwardResponse(result=result)

    @app.post("/sequence/qwen/cache/decode-binary")
    async def qwen_cache_decode_binary(
        request: Request,
        _: None = Depends(require_sequence_auth),
    ) -> Response:
        if qwen_shard_runtime is None:
            raise HTTPException(status_code=422, detail="Qwen shard runtime is not configured on this worker")
        try:
            array, tensor_header, header, request_boundary_adapter, request_frame_bytes = await _decode_qwen_boundary_frame_request(
                request,
                learned_wire_codec=learned_wire_codec,
                learned_wire_codec_artifact=learned_wire_codec_artifact,
                trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
            )
            require_inference_security(
                _public_worker_security_decision_from_frame_header(header),
                _production_worker_security_verification_from_frame_header(header),
            )
            result = qwen_shard_runtime.decode_array(
                array,
                name=str(tensor_header.get("name") or "hidden_states"),
                metadata=tensor_header.get("metadata") or {},
                cache_id=_required_frame_string(header, "cache_id"),
                sequence_id=_required_frame_string(header, "sequence_id"),
                position_start=_required_frame_int(header, "position_start"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown Qwen KV cache {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _qwen_cache_binary_response(
            result,
            boundary_adapter_strategy=_response_boundary_adapter_strategy_from_header(
                header,
                request_boundary_adapter,
            ),
            request_boundary_adapter=request_boundary_adapter,
            request_frame_bytes=request_frame_bytes,
            compact_response_trace=_compact_qwen_response_trace_requested(header),
            trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
            trainable_autoencoder_source_strategy=trainable_autoencoder_source_strategy,
            trainable_autoencoder_allow_source_only_fallback=(
                trainable_autoencoder_allow_source_only_fallback
            ),
            trainable_autoencoder_context=_trainable_autoencoder_response_context(
                header,
                path="/sequence/qwen/cache/decode-binary",
                tensor=result.tensor,
            ),
            response_position_scoped_transport=_response_position_scoped_transport_from_header(header),
            headers=worker_telemetry_headers(),
        )

    @app.post("/sequence/qwen/cache/decode-binary-coalesced")
    async def qwen_cache_decode_binary_coalesced(
        request: Request,
        _: None = Depends(require_sequence_auth),
    ) -> Response:
        if qwen_shard_runtime is None:
            raise HTTPException(status_code=422, detail="Qwen shard runtime is not configured on this worker")
        try:
            array, tensor_header, header, request_boundary_adapter, request_frame_bytes = await _decode_qwen_boundary_frame_request(
                request,
                learned_wire_codec=learned_wire_codec,
                learned_wire_codec_artifact=learned_wire_codec_artifact,
                trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
            )
            require_inference_security(
                _public_worker_security_decision_from_frame_header(header),
                _production_worker_security_verification_from_frame_header(header),
            )
            members = _coalesced_qwen_decode_members_from_header(header)
            tensor, traces, caches = qwen_shard_runtime.decode_arrays_coalesced(
                array,
                members=members,
                name=str(tensor_header.get("name") or "hidden_states"),
                metadata=tensor_header.get("metadata") or {},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown Qwen KV cache {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _qwen_cache_coalesced_binary_response(
            tensor=tensor,
            traces=traces,
            caches=caches,
            boundary_adapter_strategy=_response_boundary_adapter_strategy_from_header(
                header,
                request_boundary_adapter,
            ),
            request_boundary_adapter=request_boundary_adapter,
            request_frame_bytes=request_frame_bytes,
            compact_response_trace=_compact_qwen_response_trace_requested(header),
            response_position_scoped_transport=_response_position_scoped_transport_from_header(header),
            trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
            trainable_autoencoder_source_strategy=trainable_autoencoder_source_strategy,
            trainable_autoencoder_allow_source_only_fallback=(
                trainable_autoencoder_allow_source_only_fallback
            ),
            trainable_autoencoder_context=_trainable_autoencoder_response_context(
                header,
                path="/sequence/qwen/cache/decode-binary-coalesced",
                tensor=tensor,
            ),
            headers=worker_telemetry_headers(),
        )

    @app.post("/sequence/qwen/cache/release", response_model=SequenceQwenCacheResponse)
    def qwen_cache_release(
        request: SequenceQwenCacheReleaseRequest,
        _: None = Depends(require_sequence_auth),
    ) -> SequenceQwenCacheResponse:
        return _qwen_cache_release_impl(request)

    @app.post("/sequence/qwen/cache/truncate", response_model=SequenceQwenCacheResponse)
    def qwen_cache_truncate(
        request: SequenceQwenCacheTruncateRequest,
        _: None = Depends(require_sequence_auth),
    ) -> SequenceQwenCacheResponse:
        if qwen_shard_runtime is None:
            raise HTTPException(status_code=422, detail="Qwen shard runtime is not configured on this worker")
        require_inference_security(
            request.public_worker_security_decision,
            request.production_worker_security_verification,
        )
        try:
            return SequenceQwenCacheResponse(
                cache=qwen_shard_runtime.truncate_cache(
                    cache_id=request.cache_id,
                    sequence_id=request.sequence_id,
                    length=request.length,
                )
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown Qwen KV cache {request.cache_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def _qwen_cache_release_impl(
        request: SequenceQwenCacheReleaseRequest,
    ) -> SequenceQwenCacheResponse:
        if qwen_shard_runtime is None:
            raise HTTPException(status_code=422, detail="Qwen shard runtime is not configured on this worker")
        require_inference_security(
            request.public_worker_security_decision,
            request.production_worker_security_verification,
        )
        try:
            return SequenceQwenCacheResponse(cache=qwen_shard_runtime.release_cache(cache_id=request.cache_id, sequence_id=request.sequence_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown Qwen KV cache {request.cache_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


async def _decode_qwen_boundary_frame_request(
    request: Request,
    *,
    learned_wire_codec: LearnedInt8WireCodec | None = None,
    learned_wire_codec_artifact: LearnedInt8WireCodecArtifact | None = None,
    trainable_autoencoder_basis_artifact: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], int]:
    body = await request.body()
    parts = decode_tensor_frame_to_raw(body)
    if _learned_wire_codec_frame_requested(parts.extra, parts.metadata):
        return _decode_qwen_learned_wire_frame_request(
            parts,
            body_length=len(body),
            learned_wire_codec=learned_wire_codec,
            learned_wire_codec_artifact=learned_wire_codec_artifact,
        )
    if _trainable_autoencoder_wire_frame_requested(parts.extra, parts.metadata):
        return _decode_qwen_trainable_autoencoder_wire_frame_request(
            parts,
            body_length=len(body),
            trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
        )
    if _raw_frame_receive_enabled(parts.extra, parts.metadata):
        return _decode_qwen_boundary_frame_request_raw(parts, body_length=len(body))
    header = parts.extra
    tensor = TensorPayload.from_raw_bytes(
        parts.raw,
        dtype=parts.dtype,
        shape=parts.shape,
        name=parts.name,
        metadata=parts.metadata,
    )
    tensor_metadata = dict(tensor.metadata)
    if BOUNDARY_CODEC_BACKEND_CONTRACT_METADATA_KEY in header:
        tensor_metadata.setdefault(
            BOUNDARY_CODEC_BACKEND_CONTRACT_METADATA_KEY,
            header[BOUNDARY_CODEC_BACKEND_CONTRACT_METADATA_KEY],
        )
    if BOUNDARY_CODEC_BACKEND_REQUEST_METADATA_KEY in header:
        tensor_metadata.setdefault(
            BOUNDARY_CODEC_BACKEND_REQUEST_METADATA_KEY,
            header[BOUNDARY_CODEC_BACKEND_REQUEST_METADATA_KEY],
        )
    if tensor_metadata != tensor.metadata:
        tensor = tensor.model_copy(update={"metadata": tensor_metadata})
    decoded_tensor, request_boundary_adapter = decode_payload_from_boundary(tensor)
    return (
        decoded_tensor.to_numpy(),
        {
            "name": decoded_tensor.name,
            "metadata": decoded_tensor.metadata,
            "dtype": decoded_tensor.dtype,
            "shape": decoded_tensor.shape,
        },
        header,
        request_boundary_adapter,
        len(body),
    )


def _learned_wire_codec_frame_requested(header: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return (
        header.get("boundary_adapter_strategy") == LEARNED_INT8_BOUNDARY_ADAPTER_ID
        or metadata.get("boundary_adapter_strategy") == LEARNED_INT8_BOUNDARY_ADAPTER_ID
        or metadata.get("encoded_payload_kind") == LEARNED_INT8_WIRE_PAYLOAD_KIND
    )


def _trainable_autoencoder_wire_frame_requested(
    header: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    return (
        header.get("boundary_adapter_strategy")
        == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID
        or metadata.get("boundary_adapter_strategy")
        == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID
        or metadata.get("encoded_payload_kind")
        == TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND
    )


def _decode_qwen_learned_wire_frame_request(
    parts: Any,
    *,
    body_length: int,
    learned_wire_codec: LearnedInt8WireCodec | None,
    learned_wire_codec_artifact: LearnedInt8WireCodecArtifact | None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], int]:
    original_shape = [
        int(dim)
        for dim in (
            parts.extra.get("original_shape")
            or parts.extra.get("os")
            or parts.extra.get("shape")
            or parts.shape
        )
    ]
    header = {
        **parts.extra,
        "shape": list(parts.extra.get("shape") or parts.shape),
        "original_shape": original_shape,
        "dtype": parts.extra.get("dtype") or parts.dtype,
        "tensor_name": parts.extra.get("tensor_name") or parts.name,
        "tensor_metadata": {
            **dict(parts.metadata),
            **dict(parts.extra.get("tensor_metadata") or {}),
        },
    }
    decoded_array = decode_boundary_wire_payload(
        header,
        parts.raw,
        learned_int8_codec=learned_wire_codec,
    )
    artifact_summary = (
        _learned_wire_codec_artifact_profile_metadata(learned_wire_codec_artifact)
        if learned_wire_codec_artifact is not None
        else None
    )
    original_raw_bytes = int(decoded_array.nbytes)
    encoded_raw_bytes = int(len(parts.raw))
    raw_byte_ratio = encoded_raw_bytes / original_raw_bytes if original_raw_bytes else 1.0
    boundary_adapter = {
        **dict(header.get("boundary_adapter") or {}),
        "adapter_id": LEARNED_INT8_BOUNDARY_ADAPTER_ID,
        "adapter_kind": "learned_int8_wire_codec",
        "original_raw_bytes": original_raw_bytes,
        "encoded_raw_bytes": encoded_raw_bytes,
        "raw_byte_ratio": raw_byte_ratio,
        "raw_byte_savings_ratio": max(0.0, 1.0 - raw_byte_ratio),
        "decoded_on_receive": True,
        "encoded_payload_kind": LEARNED_INT8_WIRE_PAYLOAD_KIND,
        "stable_sequence_worker_decode": True,
        "compact_stable_frame": "boundary_adapter" not in header,
        "artifact_hash_validated": learned_wire_codec_artifact is not None,
        "artifact_id": artifact_summary.get("artifact_id") if artifact_summary else None,
        "learned_payload_family": (
            artifact_summary.get("learned_payload_family")
            if artifact_summary
            else None
        ),
        "boundary_compression_strategy": (
            artifact_summary.get("boundary_compression_strategy")
            if artifact_summary
            else None
        ),
        "production_runtime_claimed": False,
        "planner_selectable_claimed": False,
    }
    decoded_metadata = {
        **dict(parts.metadata),
        "boundary_adapter": boundary_adapter,
        "learned_wire_codec_artifact": artifact_summary,
    }
    decoded_metadata = {key: value for key, value in decoded_metadata.items() if value is not None}
    return (
        decoded_array,
        {
            "name": str(header.get("tensor_name") or parts.name),
            "metadata": decoded_metadata,
            "dtype": str(decoded_array.dtype),
            "shape": list(decoded_array.shape),
        },
        header,
        boundary_adapter,
        body_length,
    )


def _decode_qwen_trainable_autoencoder_wire_frame_request(
    parts: Any,
    *,
    body_length: int,
    trainable_autoencoder_basis_artifact: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], int]:
    header = {
        **parts.extra,
        "shape": list(parts.extra.get("shape") or parts.shape),
        "original_shape": list(parts.extra.get("original_shape") or parts.shape),
        "dtype": parts.extra.get("dtype") or parts.dtype,
        "tensor_name": parts.extra.get("tensor_name") or parts.name,
        "tensor_metadata": {
            **dict(parts.metadata),
            **dict(parts.extra.get("tensor_metadata") or {}),
        },
    }
    decoded_array = decode_boundary_wire_payload(
        header,
        parts.raw,
        trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
    )
    original_raw_bytes = int(decoded_array.nbytes)
    encoded_raw_bytes = int(len(parts.raw))
    raw_byte_ratio = encoded_raw_bytes / original_raw_bytes if original_raw_bytes else 1.0
    basis_hash = (
        trainable_autoencoder_basis_artifact.get("basis_hash")
        if isinstance(trainable_autoencoder_basis_artifact, dict)
        else None
    )
    boundary_adapter = {
        **dict(header.get("boundary_adapter") or {}),
        "adapter_id": TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
        "adapter_kind": "learned",
        "original_raw_bytes": original_raw_bytes,
        "encoded_raw_bytes": encoded_raw_bytes,
        "raw_byte_ratio": raw_byte_ratio,
        "raw_byte_savings_ratio": max(0.0, 1.0 - raw_byte_ratio),
        "decoded_on_receive": True,
        "encoded_payload_kind": TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND,
        "basis_hash": basis_hash,
        "artifact_hash_validated": trainable_autoencoder_basis_artifact is not None,
        "runtime_transport_implemented_claimed": False,
        "worker_route_transport_exercised": True,
        "planner_selectable_claimed": False,
        "production_runtime_claimed": False,
    }
    encoded_metadata = dict(parts.metadata)
    source_metadata = (
        encoded_metadata.get("source_metadata")
        if isinstance(encoded_metadata.get("source_metadata"), dict)
        else {}
    )
    decoded_metadata = {
        **dict(source_metadata),
        "boundary_adapter": boundary_adapter,
        "trainable_autoencoder_basis_artifact": {
            "basis_hash": basis_hash,
            "basis_kind": (
                trainable_autoencoder_basis_artifact.get("basis", {}).get("basis_kind")
                if isinstance(trainable_autoencoder_basis_artifact, dict)
                and isinstance(trainable_autoencoder_basis_artifact.get("basis"), dict)
                else None
            ),
            "runtime_transport_implemented_claimed": False,
            "worker_route_transport_exercised": True,
            "production_runtime_claimed": False,
        },
        "trainable_autoencoder_source_latent_request": {
            "encoded_payload_kind": encoded_metadata.get("encoded_payload_kind"),
            "basis_hash": encoded_metadata.get("basis_hash"),
            "basis_selection_key": encoded_metadata.get("basis_selection_key"),
            "basis_selection_kind": encoded_metadata.get("basis_selection_kind"),
            "source_strategy": encoded_metadata.get("source_strategy"),
            "source_only_fallback": encoded_metadata.get("source_only_fallback"),
            "source_only_fallback_reason": encoded_metadata.get(
                "source_only_fallback_reason"
            ),
            "basis_correction_applied": encoded_metadata.get(
                "basis_correction_applied"
            ),
            "source_raw_byte_count": encoded_metadata.get("source_raw_byte_count"),
            "latent_byte_count": encoded_metadata.get("latent_byte_count"),
            "decoded_on_receive": True,
            "worker_route_transport_exercised": True,
            "production_runtime_claimed": False,
        },
    }
    decoded_metadata = {
        key: value for key, value in decoded_metadata.items() if value is not None
    }
    return (
        decoded_array,
        {
            "name": str(header.get("tensor_name") or parts.name),
            "metadata": decoded_metadata,
            "dtype": str(decoded_array.dtype),
            "shape": list(decoded_array.shape),
        },
        header,
        boundary_adapter,
        body_length,
    )


def _raw_frame_receive_enabled(header: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return _optional_frame_bool(
        header,
        BOUNDARY_CODEC_RAW_FRAME_RECEIVE_ENABLED_METADATA_KEY,
        default=False,
    ) or _optional_frame_bool(
        metadata,
        BOUNDARY_CODEC_RAW_FRAME_RECEIVE_ENABLED_METADATA_KEY,
        default=False,
    )


def _decode_qwen_boundary_frame_request_raw(
    parts: Any,
    *,
    body_length: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], int]:
    header = parts.extra
    tensor_metadata = dict(parts.metadata)
    if BOUNDARY_CODEC_BACKEND_CONTRACT_METADATA_KEY in header:
        tensor_metadata.setdefault(
            BOUNDARY_CODEC_BACKEND_CONTRACT_METADATA_KEY,
            header[BOUNDARY_CODEC_BACKEND_CONTRACT_METADATA_KEY],
        )
    if BOUNDARY_CODEC_BACKEND_REQUEST_METADATA_KEY in header:
        tensor_metadata.setdefault(
            BOUNDARY_CODEC_BACKEND_REQUEST_METADATA_KEY,
            header[BOUNDARY_CODEC_BACKEND_REQUEST_METADATA_KEY],
        )
    if tensor_metadata != parts.metadata:
        parts = replace(parts, metadata=tensor_metadata)
    requested_backend_id = header.get(BOUNDARY_CODEC_BACKEND_REQUEST_METADATA_KEY)
    if requested_backend_id not in {"python_reference", "rust_native"}:
        requested_backend_id = None
    decoded_array, decoded_metadata, request_boundary_adapter = decode_boundary_frame_parts_to_numpy(
        parts,
        backend_contract=header.get(BOUNDARY_CODEC_BACKEND_CONTRACT_METADATA_KEY),
        requested_backend_id=requested_backend_id,  # type: ignore[arg-type]
    )
    return (
        decoded_array,
        {
            "name": parts.name,
            "metadata": decoded_metadata,
            "dtype": str(decoded_array.dtype),
            "shape": list(decoded_array.shape),
        },
        header,
        request_boundary_adapter,
        body_length,
    )


def _required_frame_string(header: dict[str, Any], name: str) -> str:
    value = header.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"tensor frame header requires non-empty `{name}`")
    return value


def _coalesced_qwen_decode_members_from_header(header: dict[str, Any]) -> list[dict[str, Any]]:
    members = header.get("coalesced_members")
    if not isinstance(members, list) or not members:
        raise ValueError("tensor frame header requires non-empty `coalesced_members`")
    expected_count = header.get("coalesced_member_count")
    if expected_count is not None and int(expected_count) != len(members):
        raise ValueError(
            "tensor frame header `coalesced_member_count` does not match `coalesced_members`"
        )
    normalized: list[dict[str, Any]] = []
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise ValueError("tensor frame header `coalesced_members` entries must be objects")
        cache_id = member.get("cache_id")
        sequence_id = member.get("sequence_id")
        if not isinstance(cache_id, str) or not cache_id:
            raise ValueError("coalesced member requires non-empty `cache_id`")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("coalesced member requires non-empty `sequence_id`")
        raw_metadata = member.get("metadata")
        normalized.append(
            {
                "member_index": int(member.get("member_index", index)),
                "cache_id": cache_id,
                "sequence_id": sequence_id,
                "position_start": _frame_int(
                    member.get("position_start"),
                    "coalesced_members.position_start",
                ),
                "metadata": dict(raw_metadata) if isinstance(raw_metadata, dict) else {},
            }
        )
    return normalized


def _required_frame_int(header: dict[str, Any], name: str) -> int:
    if name not in header:
        raise ValueError(f"tensor frame header requires `{name}`")
    return _frame_int(header[name], name)


def _optional_frame_int(header: dict[str, Any], name: str, *, default: int) -> int:
    if name not in header:
        return default
    return _frame_int(header[name], name)


def _optional_frame_bool(header: dict[str, Any], name: str, *, default: bool) -> bool:
    if name not in header:
        return default
    value = header[name]
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"tensor frame header `{name}` must be a boolean")


def _frame_int(value: Any, name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tensor frame header `{name}` must be an integer") from exc
    if resolved < 0:
        raise ValueError(f"tensor frame header `{name}` must be non-negative")
    return resolved


def _boundary_adapter_strategy_from_summary(summary: dict[str, Any]) -> str:
    return str(summary.get("adapter_id") or "identity")


def _response_boundary_adapter_strategy_from_header(
    header: dict[str, Any],
    request_boundary_adapter: dict[str, Any],
) -> str:
    requested = header.get("response_boundary_adapter_strategy")
    if requested is not None:
        return _normalize_qwen_boundary_wire_strategy(str(requested))
    request_strategy = _boundary_adapter_strategy_from_summary(request_boundary_adapter)
    if request_strategy in {
        LEARNED_INT8_BOUNDARY_ADAPTER_ID,
        TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
    }:
        return "identity"
    return _normalize_qwen_boundary_wire_strategy(request_strategy)


def _normalize_qwen_boundary_wire_strategy(value: str | None) -> str:
    if value == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID:
        return TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID
    return normalize_boundary_adapter_strategy(str(value or "identity"))


def _trainable_autoencoder_response_context(
    header: dict[str, Any],
    *,
    path: str,
    tensor: TensorPayload,
) -> dict[str, Any]:
    return {
        "schema_version": "sequence-worker-trainable-autoencoder-response-context-v0",
        "path": path,
        "role": "response",
        "shape": list(tensor.shape),
        "position_start": header.get("position_start"),
        "generation_step": header.get("generation_step"),
        "generation_phase": header.get("generation_phase"),
        "extra_header": dict(header),
    }


def _response_position_scoped_transport_from_header(
    header: dict[str, Any],
) -> dict[str, Any] | None:
    raw = header.get("response_boundary_position_scoped_transport")
    if not isinstance(raw, dict):
        return None
    return dict(raw)


def _tensor_with_response_position_scoped_transport(
    tensor: TensorPayload,
    response_position_scoped_transport: dict[str, Any] | None,
) -> TensorPayload:
    if not isinstance(response_position_scoped_transport, dict):
        return tensor
    metadata = dict(tensor.metadata)
    metadata[BOUNDARY_POSITION_SCOPED_TRANSPORT_METADATA_KEY] = dict(
        response_position_scoped_transport
    )
    return tensor.model_copy(update={"metadata": metadata})


def _compact_qwen_response_trace_requested(header: dict[str, Any]) -> bool:
    return _optional_frame_bool(header, "compact_qwen_response_trace", default=False) or _optional_frame_bool(
        header,
        "cqtr",
        default=False,
    )


def _compact_qwen_trace(trace: QwenShardTrace) -> dict[str, Any]:
    return {
        "sid": trace.shard_id,
        "rid": trace.runtime_id,
        "ls": trace.layer_start,
        "le": trace.layer_end,
        "is": trace.input_shape,
        "os": trace.output_shape,
        "ib": trace.input_bytes,
        "ob": trace.output_bytes,
        "iob": trace.input_original_bytes,
        "oob": trace.output_original_bytes,
        "ieb": trace.input_encoded_bytes,
        "oeb": trace.output_encoded_bytes,
        "rfb": trace.request_frame_bytes,
        "sfb": trace.response_frame_bytes,
        "ba": trace.boundary_adapter_id,
        "baa": trace.boundary_adapter_applied,
        "em": trace.elapsed_ms,
        "rem": trace.route_elapsed_ms,
    }


def _qwen_cache_coalesced_binary_response(
    *,
    tensor: TensorPayload,
    traces: list[QwenShardTrace],
    caches: list[KVCacheHandle],
    boundary_adapter_strategy: str = "identity",
    request_boundary_adapter: dict[str, Any] | None = None,
    request_frame_bytes: int | None = None,
    compact_response_trace: bool = False,
    response_position_scoped_transport: dict[str, Any] | None = None,
    trainable_autoencoder_basis_artifact: dict[str, Any] | None = None,
    trainable_autoencoder_source_strategy: str = "int8_symmetric",
    trainable_autoencoder_allow_source_only_fallback: bool = False,
    trainable_autoencoder_context: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Response:
    if not traces or not caches or len(traces) != len(caches):
        raise ValueError("coalesced Qwen response requires matching traces and caches")
    first_trace = traces[0]
    combined_trace = first_trace.model_copy(
        update={
            "input_shape": [
                len(traces),
                *list(first_trace.input_shape[1:]),
            ],
            "output_shape": list(tensor.shape),
            "input_bytes": sum(int(trace.input_bytes) for trace in traces),
            "output_bytes": tensor.byte_size(),
            "elapsed_ms": sum(float(trace.elapsed_ms) for trace in traces),
        }
    )
    coalesced_results = [
        {
            "member_index": index,
            "trace": trace.model_dump(mode="json", exclude_defaults=True, exclude_none=True),
            "cache": cache.model_dump(mode="json"),
        }
        for index, (trace, cache) in enumerate(zip(traces, caches, strict=True))
    ]
    return _qwen_binary_response(
        tensor=tensor,
        trace=combined_trace,
        transport="qwen-kv-cache-binary-coalesced-tensor-frame-v1",
        boundary_adapter_strategy=boundary_adapter_strategy,
        request_boundary_adapter=request_boundary_adapter,
        request_frame_bytes=request_frame_bytes,
        compact_response_trace=compact_response_trace,
        response_position_scoped_transport=response_position_scoped_transport,
        trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
        trainable_autoencoder_source_strategy=trainable_autoencoder_source_strategy,
        trainable_autoencoder_allow_source_only_fallback=(
            trainable_autoencoder_allow_source_only_fallback
        ),
        trainable_autoencoder_context=trainable_autoencoder_context,
        headers=headers,
        extra_header={
            "coalesced_member_count": len(coalesced_results),
            "coalesced_cache_isolation_preserved": True,
            "coalesced_transformer_compute_claimed": False,
            "coalesced_results": coalesced_results,
        },
    )


def _qwen_cache_binary_response(
    result: QwenCacheForwardResult,
    *,
    boundary_adapter_strategy: str = "identity",
    request_boundary_adapter: dict[str, Any] | None = None,
    request_frame_bytes: int | None = None,
    compact_response_trace: bool = False,
    response_position_scoped_transport: dict[str, Any] | None = None,
    trainable_autoencoder_basis_artifact: dict[str, Any] | None = None,
    trainable_autoencoder_source_strategy: str = "int8_symmetric",
    trainable_autoencoder_allow_source_only_fallback: bool = False,
    trainable_autoencoder_context: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Response:
    return _qwen_binary_response(
        tensor=result.tensor,
        trace=result.trace,
        transport="qwen-kv-cache-binary-tensor-frame-v1",
        boundary_adapter_strategy=boundary_adapter_strategy,
        request_boundary_adapter=request_boundary_adapter,
        request_frame_bytes=request_frame_bytes,
        compact_response_trace=compact_response_trace,
        response_position_scoped_transport=response_position_scoped_transport,
        trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
        trainable_autoencoder_source_strategy=trainable_autoencoder_source_strategy,
        trainable_autoencoder_allow_source_only_fallback=(
            trainable_autoencoder_allow_source_only_fallback
        ),
        trainable_autoencoder_context=trainable_autoencoder_context,
        headers=headers,
        extra_header={"cache": result.cache.model_dump(mode="json")},
    )


def _qwen_binary_response(
    *,
    tensor: TensorPayload,
    trace: QwenShardTrace,
    transport: str,
    boundary_adapter_strategy: str = "identity",
    request_boundary_adapter: dict[str, Any] | None = None,
    request_frame_bytes: int | None = None,
    compact_response_trace: bool = False,
    response_position_scoped_transport: dict[str, Any] | None = None,
    trainable_autoencoder_basis_artifact: dict[str, Any] | None = None,
    trainable_autoencoder_source_strategy: str = "int8_symmetric",
    trainable_autoencoder_allow_source_only_fallback: bool = False,
    trainable_autoencoder_context: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    extra_header: dict[str, Any] | None = None,
) -> Response:
    strategy = _normalize_qwen_boundary_wire_strategy(boundary_adapter_strategy)
    tensor = _tensor_with_response_position_scoped_transport(
        tensor,
        response_position_scoped_transport,
    )
    response_boundary_encode_started = time.perf_counter()
    if strategy == TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID:
        if trainable_autoencoder_basis_artifact is None:
            raise ValueError("trainable autoencoder response transport requires a basis artifact")
        encoded = encode_boundary_wire_payload(
            tensor.to_numpy(),
            boundary_adapter_strategy=TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
            tensor_name=tensor.name,
            tensor_metadata=tensor.metadata,
            trainable_autoencoder_basis_artifact=trainable_autoencoder_basis_artifact,
            trainable_autoencoder_source_strategy=trainable_autoencoder_source_strategy,
            trainable_autoencoder_context=trainable_autoencoder_context,
            trainable_autoencoder_allow_source_only_fallback=(
                trainable_autoencoder_allow_source_only_fallback
            ),
        )
        encoded_raw = encoded.payload
        encoded_dtype = encoded.header_fields["dtype"]
        encoded_shape = [int(dim) for dim in encoded.header_fields["shape"]]
        encoded_name = str(encoded.header_fields.get("tensor_name") or tensor.name)
        encoded_metadata = dict(encoded.header_fields.get("tensor_metadata") or {})
        response_boundary_adapter = dict(encoded.header_fields.get("boundary_adapter") or {})
        raw_encode_path = "trainable_autoencoder_source_latent_wire_payload"
        tensor_payload_constructed_for_frame = True
    else:
        encoded_frame, response_boundary_adapter = encode_payload_for_boundary_raw_frame(tensor, strategy=strategy)
        encoded_raw = encoded_frame.raw
        encoded_dtype = encoded_frame.dtype
        encoded_shape = encoded_frame.shape
        encoded_name = encoded_frame.name
        encoded_metadata = encoded_frame.metadata
        raw_encode_path = encoded_frame.raw_encode_path
        tensor_payload_constructed_for_frame = encoded_frame.tensor_payload_constructed_for_frame
    response_boundary_encode_ms = (time.perf_counter() - response_boundary_encode_started) * 1000.0
    request_summary = boundary_adapter_summary_from_metadata(request_boundary_adapter)
    response_summary = boundary_adapter_summary_from_metadata(response_boundary_adapter)
    updated_trace = trace.model_copy(
        update={
            "input_bytes": int(request_summary.get("encoded_raw_bytes") or trace.input_bytes),
            "output_bytes": int(response_summary.get("encoded_raw_bytes") or trace.output_bytes),
            "input_original_bytes": request_summary.get("original_raw_bytes") or trace.input_original_bytes,
            "output_original_bytes": response_summary.get("original_raw_bytes") or trace.output_original_bytes,
            "input_encoded_bytes": request_summary.get("encoded_raw_bytes") or trace.input_encoded_bytes,
            "output_encoded_bytes": response_summary.get("encoded_raw_bytes") or trace.output_encoded_bytes,
            "request_frame_bytes": request_frame_bytes or trace.request_frame_bytes,
            "boundary_adapter_id": strategy,
            "boundary_adapter_applied": strategy != "identity",
            "boundary_adapter_input": request_summary,
            "boundary_adapter_output": response_summary,
        }
    )
    extra = {**(extra_header or {}), "transport": transport}
    if compact_response_trace:
        extra["qt"] = _compact_qwen_trace(updated_trace)
        extra["re"] = {
            "bem": response_boundary_encode_ms,
            "erb": len(encoded_raw),
            "ba": strategy,
            "rep": raw_encode_path,
            "tpc": tensor_payload_constructed_for_frame,
            "qpm": response_boundary_adapter.get("quality_probe_mode"),
        }
    else:
        extra["trace"] = updated_trace.model_dump(
            mode="json",
            exclude_defaults=True,
            exclude_none=True,
        )
        extra["response_encode"] = {
            "schema_version": "qwen-binary-response-encode-v0",
            "boundary_encode_ms": response_boundary_encode_ms,
            "encoded_tensor_raw_bytes": len(encoded_raw),
            "boundary_adapter_id": strategy,
            "raw_encode_path": raw_encode_path,
            "tensor_payload_constructed_for_frame": tensor_payload_constructed_for_frame,
            "quality_probe_mode": response_boundary_adapter.get("quality_probe_mode"),
            BOUNDARY_RAW_RESPONSE_ENCODE_TIMING_METADATA_KEY: response_boundary_adapter.get(
                BOUNDARY_RAW_RESPONSE_ENCODE_TIMING_METADATA_KEY
            ),
        }
    if strategy != "identity":
        extra["bo"] = compact_boundary_adapter_metrics(response_boundary_adapter)
    response_frame_encode_started = time.perf_counter()
    content = encode_tensor_frame_from_raw(
        encoded_raw,
        dtype=encoded_dtype,
        shape=encoded_shape,
        name=encoded_name,
        metadata=encoded_metadata,
        extra_header=extra,
    )
    response_frame_encode_ms = (time.perf_counter() - response_frame_encode_started) * 1000.0
    response_headers = dict(headers or {})
    response_headers["x-somatic-response-frame-encode-ms"] = f"{response_frame_encode_ms:.9f}"
    return Response(
        content=content,
        media_type=TENSOR_FRAME_MEDIA_TYPE,
        headers=response_headers,
    )


def _worker_telemetry_headers(
    *,
    process_cpu_seconds: float,
    rss_bytes: int | None,
) -> dict[str, str]:
    headers = {"x-somatic-worker-cpu-seconds": f"{process_cpu_seconds:.9f}"}
    if rss_bytes is not None:
        headers["x-somatic-worker-rss-bytes"] = str(int(rss_bytes))
    return headers


def _sequence_runtime_startup_observation(
    observation: dict[str, Any] | None,
    *,
    qwen_shard_runtime: Any | None,
) -> dict[str, Any] | None:
    if observation is None:
        return None
    resolved = dict(observation)
    qwen_payloads = _qwen_shard_binary_route_payloads(qwen_shard_runtime)
    if qwen_payloads:
        existing_payloads = (
            resolved.get("binary_route_payloads")
            if isinstance(resolved.get("binary_route_payloads"), dict)
            else {}
        )
        resolved["binary_route_payloads"] = {
            **existing_payloads,
            **qwen_payloads,
        }
    return resolved


def _qwen_shard_binary_route_payloads(qwen_shard_runtime: Any | None) -> dict[str, dict[str, Any]]:
    if qwen_shard_runtime is None:
        return {}
    manifest = getattr(qwen_shard_runtime, "manifest", None)
    if manifest is None:
        shard = getattr(qwen_shard_runtime, "shard", None)
        manifest = getattr(shard, "manifest", None)
    hidden_size = _qwen_manifest_hidden_size(manifest)
    if hidden_size is None:
        return {}
    payload = {
        "tensor_name": "hidden_states",
        "tensor_dtype": _qwen_tensor_dtype_from_precision(
            str(getattr(manifest, "precision", "") or "")
        ),
        "tensor_shape": [1, 1, hidden_size],
    }
    return {
        "/sequence/qwen/forward-shard-binary": dict(payload),
        "/sequence/qwen/cache/prefill-binary": dict(payload),
        "/sequence/qwen/cache/decode-binary": dict(payload),
        "/sequence/qwen/cache/decode-binary-coalesced": {
            **dict(payload),
            "tensor_shape": ["N", 1, hidden_size],
            "coalesced_member_axis": 0,
        },
    }


def _qwen_manifest_hidden_size(manifest: Any | None) -> int | None:
    metadata = getattr(manifest, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("hidden_size", "qwen_hidden_size"):
            hidden_size = _positive_int_or_none(metadata.get(key))
            if hidden_size is not None:
                return hidden_size
    activation_bytes = _positive_int_or_none(getattr(manifest, "activation_bytes", None))
    if activation_bytes is not None and activation_bytes % 2 == 0:
        return activation_bytes // 2
    return None


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _qwen_tensor_dtype_from_precision(precision: str) -> str:
    if precision == "fp16":
        return "float16"
    return "float32"


def _public_worker_security_decision_from_frame_header(header: dict[str, Any]) -> SecurityAdmissionDecision | None:
    payload = header.get("public_worker_security_decision")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("tensor frame header public_worker_security_decision must be an object")
    return SecurityAdmissionDecision.model_validate(payload)


def _production_worker_security_verification_from_frame_header(
    header: dict[str, Any],
) -> ProductionWorkerSecurityVerification | None:
    payload = header.get("production_worker_security_verification")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("tensor frame header production_worker_security_verification must be an object")
    return ProductionWorkerSecurityVerification.model_validate(payload)


def _peer_class_for_profile(profile: ResourceProfile) -> PeerClass | None:
    peer_class = (
        profile.metadata.get("peer_class")
        or profile.metadata.get("admission_class")
        or profile.metadata.get("trust_tier")
    )
    if peer_class in {"owner", "friend", "rented", "public"}:
        return peer_class
    return None


def _profile_with_stable_split_metadata(
    profile: ResourceProfile,
    *,
    qwen_shard_runtime: Any | None,
    learned_wire_codec_artifact: LearnedInt8WireCodecArtifact | None,
    trainable_autoencoder_basis_artifact: dict[str, Any] | None,
) -> ResourceProfile:
    qwen_routes = sorted(_qwen_shard_binary_route_payloads(qwen_shard_runtime))
    qwen_loading_accounting = _qwen_loading_accounting_profile_metadata(qwen_shard_runtime)
    artifact_metadata = (
        _learned_wire_codec_artifact_profile_metadata(learned_wire_codec_artifact)
        if learned_wire_codec_artifact is not None
        else None
    )
    trainable_autoencoder_metadata = (
        _trainable_autoencoder_basis_profile_metadata(trainable_autoencoder_basis_artifact)
        if trainable_autoencoder_basis_artifact is not None
        else None
    )
    supported_boundary_adapters = ["identity", "fp16", "int8_symmetric"]
    if artifact_metadata is not None:
        supported_boundary_adapters.append(LEARNED_INT8_BOUNDARY_ADAPTER_ID)
    if trainable_autoencoder_metadata is not None:
        supported_boundary_adapters.append(TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID)
    boundary_wire_codecs = [
        item
        for item in [artifact_metadata, trainable_autoencoder_metadata]
        if item is not None
    ]
    split_capability = {
        "schema_version": "stable-sequence-worker-split-capability-v0",
        "qwen_shard_runtime_configured": qwen_shard_runtime is not None,
        "qwen_binary_routes": qwen_routes,
        "qwen_kv_cache_binary_routes": [
            route for route in qwen_routes if route.startswith("/sequence/qwen/cache/")
        ],
        "qwen_loading_accounting": qwen_loading_accounting,
        "qwen_shard_only_weight_loading_claimed": (
            qwen_loading_accounting.get("shard_only_weight_loading_claimed") is True
            if qwen_loading_accounting is not None
            else False
        ),
        "qwen_per_machine_ram_reduction_claimed": (
            qwen_loading_accounting.get("per_machine_ram_reduction_claimed") is True
            if qwen_loading_accounting is not None
            else False
        ),
        "boundary_wire_codecs": boundary_wire_codecs,
        "supported_boundary_adapter_ids": supported_boundary_adapters,
        "stable_runtime_learned_wire_codec_decode": artifact_metadata is not None,
        "stable_runtime_trainable_autoencoder_wire_codec_decode": (
            trainable_autoencoder_metadata is not None
        ),
        "research_live_split_feature_equivalent_claimed": False,
        "production_runtime_claimed": False,
        "public_worker_security_claimed": False,
        "distributed_training_claimed": False,
        "token_launch_claimed": False,
    }
    metadata = {
        **profile.metadata,
        "stable_sequence_worker_split_capability": split_capability,
        "stable_sequence_worker_split_capable": qwen_shard_runtime is not None,
        "qwen_loading_accounting": qwen_loading_accounting,
        "learned_wire_codec_artifact_loaded": artifact_metadata is not None,
        "trainable_autoencoder_basis_artifact_loaded": (
            trainable_autoencoder_metadata is not None
        ),
    }
    if artifact_metadata is not None:
        metadata.update(
            {
                "learned_wire_codec_artifact_id": artifact_metadata["artifact_id"],
                "learned_wire_codec_adapter_id": artifact_metadata["adapter_id"],
                "learned_wire_codec_width": artifact_metadata["width"],
                "learned_wire_codec_hash_validated": True,
                "learned_wire_codec_planner_selectable_claimed": False,
                "learned_wire_codec_production_runtime_claimed": False,
            }
        )
    if trainable_autoencoder_metadata is not None:
        metadata.update(
            {
                "trainable_autoencoder_basis_hash": trainable_autoencoder_metadata[
                    "basis_hash"
                ],
                "trainable_autoencoder_basis_kind": trainable_autoencoder_metadata[
                    "basis_kind"
                ],
                "trainable_autoencoder_parameter_byte_count": (
                    trainable_autoencoder_metadata["parameter_byte_count"]
                ),
                "trainable_autoencoder_production_runtime_claimed": False,
            }
        )
    return profile.model_copy(update={"metadata": metadata})


def _qwen_loading_accounting_profile_metadata(qwen_shard_runtime: Any | None) -> dict[str, Any] | None:
    loading_accounting = getattr(qwen_shard_runtime, "loading_accounting", None)
    if loading_accounting is None:
        return None
    return loading_accounting.summary()


def _learned_wire_codec_artifact_profile_metadata(
    artifact: LearnedInt8WireCodecArtifact | None,
) -> dict[str, Any] | None:
    if artifact is None:
        return None
    return {
        "schema_version": "stable-sequence-worker-learned-wire-codec-artifact-v0",
        "artifact_id": artifact.artifact_id,
        "adapter_id": artifact.adapter_id,
        "region_id": artifact.region_id,
        "width": artifact.width,
        "payload_kind": artifact.payload_kind,
        "fallback_adapter_id": artifact.fallback_adapter_id,
        "parameter_hash": artifact.parameter_hash,
        "parameter_count": artifact.parameter_count,
        "low_rank_rank": artifact.low_rank_rank,
        "learned_payload_family": artifact.learned_payload_family,
        "boundary_compression_strategy": artifact.boundary_compression_strategy,
        "hash_validated": True,
        "architecture_neutral_boundary_codec": artifact.claims.get("architecture_neutral_boundary_codec") is True,
        "planner_selectable_claimed": False,
        "production_runtime_claimed": False,
        "public_worker_security_claimed": False,
        "distributed_training_claimed": False,
        "token_launch_claimed": False,
    }


def _trainable_autoencoder_basis_profile_metadata(
    basis_artifact: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if basis_artifact is None:
        return None
    basis = (
        basis_artifact.get("basis")
        if isinstance(basis_artifact.get("basis"), dict)
        else {}
    )
    return {
        "schema_version": "stable-sequence-worker-trainable-autoencoder-basis-v0",
        "artifact_id": basis_artifact.get("candidate_id"),
        "adapter_id": TRAINABLE_AUTOENCODER_SOURCE_LATENT_ADAPTER_ID,
        "basis_hash": basis_artifact.get("basis_hash"),
        "basis_kind": basis.get("basis_kind"),
        "payload_kind": TRAINABLE_AUTOENCODER_SOURCE_LATENT_PAYLOAD_KIND,
        "parameter_byte_count": basis_artifact.get("parameter_byte_count"),
        "bottleneck_rank": basis_artifact.get("bottleneck_rank"),
        "source_strategy": "int8_symmetric",
        "hash_validated": True,
        "architecture_neutral_boundary_codec": True,
        "planner_selectable_claimed": False,
        "production_runtime_claimed": False,
        "public_worker_security_claimed": False,
        "distributed_training_claimed": False,
        "token_launch_claimed": False,
    }


def default_resource_profile(
    *,
    runtime_id: str = "sequence-worker-0",
    cpu_cores: int = 4,
    memory_gb: float = 1.0,
    current_load: float = 0.0,
    supported_unit_kinds: list[UnitKind] | None = None,
) -> ResourceProfile:
    return ResourceProfile(
        runtime_id=runtime_id,
        backend="simulated",
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        current_load=current_load,
        supported_unit_kinds=supported_unit_kinds
        or ["embedding", "attention_block", "ssm_block", "deltanet_block", "mlp", "moe", "norm", "adapter", "lm_head"],
        supported_precisions=["bf16", "fp16", "fp32", "int8", "int4"],
        metadata={
            "protocol": "sequence-worker-v0",
            "tensor_protocol": "typed-tensor-base64-json-v0,tensor-frame-v1",
            "metrics_protocol": "sequence-worker-metrics-v0",
        },
    )


def serve(
    *,
    runtime_id: str = "sequence-worker-0",
    port: int = 8201,
    host: str = "0.0.0.0",
    cpu_cores: int = 4,
    memory_gb: float = 1.0,
    current_load: float = 0.0,
    supported_unit_kinds: list[UnitKind] | None = None,
    auth_token: str | None = None,
    max_request_bytes: int = 16 * 1024 * 1024,
    request_timeout_seconds: float = 600.0,
    audit_log_path: str | Path | None = None,
    qwen_shard_model_id: str | None = None,
    qwen_layer_start: int | None = None,
    qwen_layer_end: int | None = None,
    qwen_precision: Precision = "fp32",
    qwen_device: str = "cpu",
    qwen_local_files_only: bool = True,
    learned_wire_codec_artifact_path: str | Path | None = None,
    trainable_autoencoder_basis_artifact_path: str | Path | None = None,
    admission_class: PeerClass | None = None,
    require_public_worker_security: bool | None = None,
    public_worker_security_production_mode: bool = False,
    trusted_production_worker_security_verifier_ids: set[str] | None = None,
    trusted_production_worker_security_verifier_keys: dict[str, str] | None = None,
    enable_provider_probe_endpoint: bool = False,
    provider_probe_responder: ProviderProbeResponder | None = None,
    provider_probe_auth_token: str | None = None,
    provider_runtime_startup_observation: dict[str, Any] | None = None,
) -> None:
    import uvicorn

    profile = default_resource_profile(
        runtime_id=runtime_id,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        current_load=current_load,
        supported_unit_kinds=supported_unit_kinds,
    )
    if admission_class is not None:
        profile = profile.model_copy(update={"metadata": {**profile.metadata, "admission_class": admission_class}})
    qwen_shard_runtime = None
    if qwen_shard_model_id is not None:
        if qwen_layer_start is None or qwen_layer_end is None:
            raise ValueError("Qwen shard startup requires both qwen_layer_start and qwen_layer_end")
        qwen_shard_runtime = QwenWorkerLayerRuntime.from_pretrained(
            model_id=qwen_shard_model_id,
            runtime_id=runtime_id,
            layer_start=qwen_layer_start,
            layer_end=qwen_layer_end,
            precision=qwen_precision,
            device=qwen_device,
            local_files_only=qwen_local_files_only,
        )
        profile = profile.model_copy(
            update={
                "backend": "transformers",
                "metadata": {
                    **profile.metadata,
                    "qwen_shard_model_id": qwen_shard_model_id,
                    "qwen_layer_start": qwen_layer_start,
                    "qwen_layer_end": qwen_layer_end,
                    "qwen_precision": qwen_precision,
                    "qwen_device": qwen_device,
                    "qwen_local_files_only": qwen_local_files_only,
                },
            }
        )
    uvicorn.run(
        create_app(
            resource_profile=profile,
            qwen_shard_runtime=qwen_shard_runtime,
            learned_wire_codec_artifact_path=learned_wire_codec_artifact_path,
            trainable_autoencoder_basis_artifact_path=trainable_autoencoder_basis_artifact_path,
            auth_token=auth_token,
            max_request_bytes=max_request_bytes,
            request_timeout_seconds=request_timeout_seconds,
            audit_log_path=audit_log_path,
            require_public_worker_security=require_public_worker_security,
            public_worker_security_production_mode=public_worker_security_production_mode,
            trusted_production_worker_security_verifier_ids=trusted_production_worker_security_verifier_ids,
            trusted_production_worker_security_verifier_keys=trusted_production_worker_security_verifier_keys,
            enable_provider_probe_endpoint=enable_provider_probe_endpoint,
            provider_probe_responder=provider_probe_responder,
            provider_probe_auth_token=provider_probe_auth_token,
            provider_runtime_startup_observation=provider_runtime_startup_observation,
        ),
        host=host,
        port=port,
    )


def _write_audit_event(*, path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
