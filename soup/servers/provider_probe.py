from __future__ import annotations

import inspect
import hashlib
import json
from datetime import datetime
from typing import Any, Callable, Literal, Protocol

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from soup.security.public_worker import (
    OSSandboxVerifier,
    PeerClass,
    ProductionRuntimeAttestationEvidence,
    ProductionSandboxEvidence,
    ProductionVerifierKind,
    ProviderRuntimeAttestationVerifier,
    SecurityAdmissionDecision,
    WorkKind,
    public_worker_security_decision_metadata,
    sign_production_worker_security_verification,
    verify_production_worker_security_evidence,
)
from soup.runtime.schemas import utc_now


PROVIDER_PROBE_ENDPOINT_PATH = "/.well-known/soup/public-worker-security/probe"
PROVIDER_STARTUP_OBSERVATION_ENDPOINT_PATH = (
    "/.well-known/soup/public-worker-security/startup-observation"
)
PROVIDER_PROBE_REQUEST_SCHEMA_VERSION = "public-worker-security-provider-load-probe-request-v0"
PROVIDER_PROBE_RESPONSE_SCHEMA_VERSION = "public-worker-security-provider-load-probe-response-v0"
PROVIDER_RUNTIME_STARTUP_OBSERVATION_SCHEMA_VERSION = (
    "public-worker-security-provider-runtime-startup-observation-v0"
)

_UNCLAIMED_REPORTING_CLAIMS = {
    "production_public_worker_security_claimed",
    "production_provider_tee_attestation_claimed",
    "production_sandbox_enforcement_claimed",
    "provider_accounting_claimed",
    "crypto_settlement_claimed",
    "token_launch_claimed",
    "payouts_claimed",
    "custody_claimed",
    "slashing_claimed",
}


class ProviderProbeAssignmentBinding(BaseModel):
    assignment_id: str
    assignment_nonce: str
    runtime_id: str
    peer_class: PeerClass
    job_kind: WorkKind


class ProviderProbeRequest(BaseModel):
    schema_version: str = PROVIDER_PROBE_REQUEST_SCHEMA_VERSION
    probe_id: str
    endpoint_set_fingerprint: str
    probe_package_context_fingerprint: str | None = None
    runtime_startup_receipts_fingerprint: str | None = None
    probe_request_context_fingerprint: str
    route: str
    runtime_id: str
    job_kind: WorkKind
    verification_mode: Literal["challenge", "duplicate"]
    assignment_binding: ProviderProbeAssignmentBinding
    required_production_verifier_id: str | None = None
    result_schema_version: str | None = None


class ProviderProbeResponse(BaseModel):
    schema_version: str = PROVIDER_PROBE_RESPONSE_SCHEMA_VERSION
    probe_id: str
    probe_request_context_fingerprint: str | None = None
    status: Literal["passed", "failed", "blocked"] = "failed"
    reason_codes: list[str] = Field(default_factory=list)
    public_worker_security_metadata: dict[str, Any] | None = None
    runtime_attestation: ProductionRuntimeAttestationEvidence | None = None
    sandbox_evidence: ProductionSandboxEvidence | None = None
    claims: dict[str, bool] = Field(default_factory=lambda: {claim: False for claim in _UNCLAIMED_REPORTING_CLAIMS})


class BinaryRoutePayloadConfig(BaseModel):
    tensor_name: str | None = None
    tensor_dtype: Literal["float32", "float16"] | None = None
    tensor_shape: list[int] | None = None
    cache_sequence_id: str | None = None
    cache_ttl_seconds: float | None = None

    @model_validator(mode="after")
    def _validate_shape_and_cache(self) -> "BinaryRoutePayloadConfig":
        if self.tensor_name is not None and not self.tensor_name:
            raise ValueError("tensor_name must be non-empty when supplied")
        if self.tensor_shape is not None and (
            len(self.tensor_shape) != 3 or any(dim <= 0 for dim in self.tensor_shape)
        ):
            raise ValueError("tensor_shape must contain three positive dimensions when supplied")
        if self.cache_sequence_id is not None and not self.cache_sequence_id:
            raise ValueError("cache_sequence_id must be non-empty when supplied")
        if self.cache_ttl_seconds is not None and self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive when supplied")
        return self


class ProviderProbeEnvelope(BaseModel):
    probe_response: ProviderProbeResponse


class ProviderRuntimeStartupObservation(BaseModel):
    schema_version: str = PROVIDER_RUNTIME_STARTUP_OBSERVATION_SCHEMA_VERSION
    runtime_id: str
    admission_class: PeerClass | None = None
    supported_job_kinds: list[WorkKind] = Field(default_factory=list)
    provider_probe_endpoint_path: str = PROVIDER_PROBE_ENDPOINT_PATH
    provider_probe_endpoint_enabled: bool = False
    provider_probe_adapter_backed_responder_configured: bool = False
    provider_probe_auth_token_env: str | None = None
    provider_probe_runtime_attestation_factory: str | None = None
    provider_probe_os_sandbox_factory: str | None = None
    provider_probe_required_production_verifier_id: str | None = None
    provider_probe_signing_key_id: str | None = None
    provider_probe_signing_secret_env: str | None = None
    public_worker_security_production_mode: bool = False
    trusted_production_verifier_ids: list[str] = Field(default_factory=list)
    trusted_production_verifier_key_envs: list[str] = Field(default_factory=list)
    observed_provider_probe_flags: list[str] = Field(default_factory=list)
    observed_production_admission_flags: list[str] = Field(default_factory=list)
    observed_required_env_names: list[str] = Field(default_factory=list)
    binary_route_payloads: dict[str, BinaryRoutePayloadConfig] = Field(default_factory=dict)
    runtime_process_observed_alive: bool = True
    secret_values_included: bool = False
    provider_probe_endpoint_contacted_by_observation: bool = False
    claims: dict[str, bool] = Field(default_factory=lambda: {claim: False for claim in _UNCLAIMED_REPORTING_CLAIMS})


class ProviderRuntimeStartupObservationEnvelope(BaseModel):
    runtime_startup_observation: ProviderRuntimeStartupObservation


def provider_probe_request_context_fingerprint(
    request: ProviderProbeRequest | dict[str, Any],
) -> str | None:
    if isinstance(request, ProviderProbeRequest):
        request_body = request.model_dump(mode="json")
    elif isinstance(request, dict):
        request_body = request
    else:
        return None
    canonical = {
        key: value
        for key, value in request_body.items()
        if key != "probe_request_context_fingerprint"
        and value is not None
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ProviderProbeResponder(Protocol):
    def respond_provider_probe(self, request: ProviderProbeRequest) -> ProviderProbeResponse | dict[str, Any]:
        ...


class AdapterBackedProviderProbeResponder:
    def __init__(
        self,
        *,
        runtime_attestation_verifier: ProviderRuntimeAttestationVerifier,
        sandbox_verifier: OSSandboxVerifier,
        verifier_id: str,
        signing_key_id: str,
        signing_secret: str,
        verifier_kind: ProductionVerifierKind = "provider-and-os-sandbox",
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self.runtime_attestation_verifier = runtime_attestation_verifier
        self.sandbox_verifier = sandbox_verifier
        self.verifier_id = verifier_id
        self.signing_key_id = signing_key_id
        self.signing_secret = signing_secret
        self.verifier_kind = verifier_kind
        self.now = now

    def respond_provider_probe(self, request: ProviderProbeRequest) -> ProviderProbeResponse:
        if request.required_production_verifier_id and request.required_production_verifier_id != self.verifier_id:
            return ProviderProbeResponse(
                probe_id=request.probe_id,
                status="failed",
                reason_codes=["provider_probe_required_production_verifier_mismatch"],
            )
        decision = security_decision_from_provider_probe_request(request)
        now = self.now()
        try:
            runtime_attestation = self.runtime_attestation_verifier.verify_runtime_attestation(
                decision=decision,
                verifier_id=self.verifier_id,
                verifier_kind=self.verifier_kind,
                current_time=now,
            )
            sandbox = self.sandbox_verifier.verify_sandbox(
                decision=decision,
                verifier_id=self.verifier_id,
                verifier_kind=self.verifier_kind,
                current_time=now,
            )
            verification = sign_production_worker_security_verification(
                verify_production_worker_security_evidence(
                    decision=decision,
                    runtime_attestation=runtime_attestation,
                    sandbox=sandbox,
                    verifier_id=self.verifier_id,
                    verifier_kind=self.verifier_kind,
                    current_time=now,
                ),
                key_id=self.signing_key_id,
                signing_secret=self.signing_secret,
            )
        except Exception as exc:
            return ProviderProbeResponse(
                probe_id=request.probe_id,
                status="failed",
                reason_codes=[f"provider_probe_adapter_error:{type(exc).__name__}"],
            )
        return ProviderProbeResponse(
            probe_id=request.probe_id,
            probe_request_context_fingerprint=request.probe_request_context_fingerprint,
            status="passed" if verification.status == "verified" else "failed",
            reason_codes=list(verification.reason_codes),
            public_worker_security_metadata=public_worker_security_decision_metadata(decision, verification),
            runtime_attestation=runtime_attestation,
            sandbox_evidence=sandbox,
        )


def mount_provider_probe_endpoint(
    app: FastAPI,
    *,
    runtime_id: str,
    supported_job_kinds: set[WorkKind],
    responder: ProviderProbeResponder | None = None,
    auth_token: str | None = None,
    endpoint_path: str = PROVIDER_PROBE_ENDPOINT_PATH,
) -> None:
    @app.post(endpoint_path, response_model=ProviderProbeEnvelope)
    async def provider_probe(
        request: ProviderProbeRequest,
        authorization: str | None = Header(default=None),
    ) -> ProviderProbeEnvelope:
        if auth_token is not None and authorization != f"Bearer {auth_token}":
            raise HTTPException(status_code=401, detail="provider_probe_authorization_failed")
        request_reason_codes = validate_provider_probe_request(
            request,
            runtime_id=runtime_id,
            supported_job_kinds=supported_job_kinds,
        )
        if request_reason_codes:
            raise HTTPException(
                status_code=403,
                detail=f"provider probe request rejected: {request_reason_codes[0]}",
            )
        if responder is None:
            raise HTTPException(status_code=503, detail="provider_probe_responder_unavailable")
        raw_response = responder.respond_provider_probe(request)
        if inspect.isawaitable(raw_response):
            raw_response = await raw_response
        response = ProviderProbeResponse.model_validate(raw_response)
        response_reason_codes = validate_provider_probe_response(response, request)
        if response_reason_codes:
            raise HTTPException(
                status_code=502,
                detail=f"provider probe response rejected: {response_reason_codes[0]}",
            )
        return ProviderProbeEnvelope(probe_response=response)


def mount_provider_startup_observation_endpoint(
    app: FastAPI,
    *,
    observation: ProviderRuntimeStartupObservation | dict[str, Any],
    auth_token: str | None = None,
    endpoint_path: str = PROVIDER_STARTUP_OBSERVATION_ENDPOINT_PATH,
) -> None:
    startup_observation = ProviderRuntimeStartupObservation.model_validate(observation)

    @app.get(endpoint_path, response_model=ProviderRuntimeStartupObservationEnvelope)
    async def provider_runtime_startup_observation(
        authorization: str | None = Header(default=None),
    ) -> ProviderRuntimeStartupObservationEnvelope:
        if auth_token is not None and authorization != f"Bearer {auth_token}":
            raise HTTPException(status_code=401, detail="provider_startup_observation_authorization_failed")
        reason_codes = validate_provider_runtime_startup_observation(startup_observation)
        if reason_codes:
            raise HTTPException(
                status_code=503,
                detail=f"provider startup observation rejected: {reason_codes[0]}",
            )
        return ProviderRuntimeStartupObservationEnvelope(
            runtime_startup_observation=startup_observation
        )


def validate_provider_runtime_startup_observation(
    observation: ProviderRuntimeStartupObservation,
) -> list[str]:
    reason_codes: list[str] = []
    if observation.schema_version != PROVIDER_RUNTIME_STARTUP_OBSERVATION_SCHEMA_VERSION:
        reason_codes.append("provider_startup_observation_schema_version_unsupported")
    if not observation.runtime_id:
        reason_codes.append("provider_startup_observation_runtime_id_missing")
    if observation.provider_probe_endpoint_enabled and not observation.provider_probe_endpoint_path.startswith("/"):
        reason_codes.append("provider_startup_observation_probe_endpoint_path_invalid")
    for route in observation.binary_route_payloads:
        if not route.startswith("/"):
            reason_codes.append("provider_startup_observation_binary_route_payload_route_invalid")
    if observation.secret_values_included:
        reason_codes.append("provider_startup_observation_secret_values_included")
    if observation.provider_probe_endpoint_contacted_by_observation:
        reason_codes.append("provider_startup_observation_contacted_probe_endpoint")
    reason_codes.extend(_forbidden_claim_reason_codes(observation.claims))
    return list(dict.fromkeys(reason_codes))


def validate_provider_probe_request(
    request: ProviderProbeRequest,
    *,
    runtime_id: str,
    supported_job_kinds: set[WorkKind],
) -> list[str]:
    reason_codes: list[str] = []
    if request.schema_version != PROVIDER_PROBE_REQUEST_SCHEMA_VERSION:
        reason_codes.append("provider_probe_request_schema_version_unsupported")
    if request.runtime_id != runtime_id:
        reason_codes.append("provider_probe_runtime_mismatch")
    if request.job_kind not in supported_job_kinds:
        reason_codes.append("provider_probe_job_kind_unsupported")
    binding = request.assignment_binding
    if not binding.assignment_id:
        reason_codes.append("provider_probe_assignment_id_missing")
    if not binding.assignment_nonce:
        reason_codes.append("provider_probe_assignment_nonce_missing")
    if binding.runtime_id != request.runtime_id:
        reason_codes.append("provider_probe_assignment_runtime_mismatch")
    if binding.job_kind != request.job_kind:
        reason_codes.append("provider_probe_assignment_job_kind_mismatch")
    if not request.endpoint_set_fingerprint:
        reason_codes.append("provider_probe_endpoint_set_fingerprint_missing")
    if not request.probe_request_context_fingerprint:
        reason_codes.append("provider_probe_request_context_fingerprint_missing")
    elif request.probe_request_context_fingerprint != provider_probe_request_context_fingerprint(
        request
    ):
        reason_codes.append("provider_probe_request_context_fingerprint_mismatch")
    if not request.route:
        reason_codes.append("provider_probe_route_missing")
    return list(dict.fromkeys(reason_codes))


def security_decision_from_provider_probe_request(request: ProviderProbeRequest) -> SecurityAdmissionDecision:
    binding = request.assignment_binding
    return SecurityAdmissionDecision(
        decision_id=f"worker-security-decision-{request.probe_id}",
        assignment_id=binding.assignment_id,
        assignment_nonce=binding.assignment_nonce,
        runtime_id=request.runtime_id,
        peer_class=binding.peer_class,
        job_kind=request.job_kind,
        status="accepted",
        admitted=True,
        challenge_required=True,
        challenge_verified=True,
        replay_decision="fresh",
        claims={
            "production_public_worker_security_claimed": False,
            "provider_accounting_claimed": False,
            "token_launch_claimed": False,
        },
    )


def validate_provider_probe_response(
    response: ProviderProbeResponse,
    request: ProviderProbeRequest,
) -> list[str]:
    reason_codes: list[str] = []
    if response.schema_version != PROVIDER_PROBE_RESPONSE_SCHEMA_VERSION:
        reason_codes.append("provider_probe_response_schema_version_unsupported")
    if response.probe_id != request.probe_id:
        reason_codes.append("provider_probe_response_probe_id_mismatch")
    if (
        response.status == "passed"
        and response.probe_request_context_fingerprint
        != request.probe_request_context_fingerprint
    ):
        reason_codes.append("provider_probe_response_request_context_fingerprint_mismatch")
    reason_codes.extend(_forbidden_claim_reason_codes(response.claims))
    if response.status != "passed":
        return list(dict.fromkeys(reason_codes))
    if response.public_worker_security_metadata is None:
        reason_codes.append("provider_probe_public_worker_security_metadata_missing")
    else:
        reason_codes.extend(_metadata_binding_reason_codes(response.public_worker_security_metadata, request))
    if response.runtime_attestation is None:
        reason_codes.append("provider_probe_runtime_attestation_missing")
    else:
        reason_codes.extend(_runtime_attestation_binding_reason_codes(response.runtime_attestation, request))
    if response.sandbox_evidence is None:
        reason_codes.append("provider_probe_sandbox_evidence_missing")
    else:
        reason_codes.extend(_sandbox_binding_reason_codes(response.sandbox_evidence, request))
    return list(dict.fromkeys(reason_codes))


def _forbidden_claim_reason_codes(claims: dict[str, bool]) -> list[str]:
    return [
        f"provider_probe_response_claim_forbidden:{claim}"
        for claim in sorted(_UNCLAIMED_REPORTING_CLAIMS)
        if claims.get(claim) is True
    ]


def _metadata_binding_reason_codes(metadata: dict[str, Any], request: ProviderProbeRequest) -> list[str]:
    binding = request.assignment_binding
    reason_codes: list[str] = []
    expected_values = {
        "public_worker_security_runtime_id": request.runtime_id,
        "public_worker_security_assignment_id": binding.assignment_id,
        "public_worker_security_assignment_nonce": binding.assignment_nonce,
        "public_worker_security_peer_class": binding.peer_class,
        "public_worker_security_job_kind": request.job_kind,
        "public_worker_security_production_runtime_id": request.runtime_id,
        "public_worker_security_production_assignment_id": binding.assignment_id,
        "public_worker_security_production_assignment_nonce": binding.assignment_nonce,
        "public_worker_security_production_peer_class": binding.peer_class,
        "public_worker_security_production_job_kind": request.job_kind,
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            reason_codes.append(f"provider_probe_metadata_{key}_mismatch")
    if request.required_production_verifier_id and (
        metadata.get("public_worker_security_production_verifier_id") != request.required_production_verifier_id
    ):
        reason_codes.append("provider_probe_metadata_production_verifier_mismatch")
    if metadata.get("public_worker_security_provider_accounting_claimed") is True:
        reason_codes.append("provider_probe_metadata_provider_accounting_claim_forbidden")
    if metadata.get("public_worker_security_token_launch_claimed") is True:
        reason_codes.append("provider_probe_metadata_token_launch_claim_forbidden")
    return reason_codes


def _runtime_attestation_binding_reason_codes(
    evidence: ProductionRuntimeAttestationEvidence,
    request: ProviderProbeRequest,
) -> list[str]:
    binding = request.assignment_binding
    reason_codes: list[str] = []
    if evidence.runtime_id != request.runtime_id:
        reason_codes.append("provider_probe_runtime_attestation_runtime_mismatch")
    if evidence.assignment_id != binding.assignment_id:
        reason_codes.append("provider_probe_runtime_attestation_assignment_mismatch")
    if evidence.assignment_nonce != binding.assignment_nonce:
        reason_codes.append("provider_probe_runtime_attestation_nonce_mismatch")
    if evidence.job_kind != request.job_kind:
        reason_codes.append("provider_probe_runtime_attestation_job_kind_mismatch")
    if request.required_production_verifier_id and evidence.verifier_id != request.required_production_verifier_id:
        reason_codes.append("provider_probe_runtime_attestation_verifier_mismatch")
    return reason_codes


def _sandbox_binding_reason_codes(
    evidence: ProductionSandboxEvidence,
    request: ProviderProbeRequest,
) -> list[str]:
    binding = request.assignment_binding
    reason_codes: list[str] = []
    if evidence.runtime_id != request.runtime_id:
        reason_codes.append("provider_probe_sandbox_runtime_mismatch")
    if evidence.assignment_id != binding.assignment_id:
        reason_codes.append("provider_probe_sandbox_assignment_mismatch")
    if evidence.assignment_nonce != binding.assignment_nonce:
        reason_codes.append("provider_probe_sandbox_nonce_mismatch")
    if evidence.job_kind != request.job_kind:
        reason_codes.append("provider_probe_sandbox_job_kind_mismatch")
    if request.required_production_verifier_id and evidence.verifier_id != request.required_production_verifier_id:
        reason_codes.append("provider_probe_sandbox_verifier_mismatch")
    return reason_codes
