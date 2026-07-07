from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from soup.runtime.schemas import utc_now


PUBLIC_WORKER_SECURITY_EVIDENCE_SCHEMA_VERSION = "public-worker-security-evidence-v0"
PUBLIC_WORKER_SECURITY_POLICY_SCHEMA_VERSION = "public-worker-security-policy-v0"
PUBLIC_WORKER_SECURITY_DECISION_SCHEMA_VERSION = "public-worker-security-decision-v0"
PUBLIC_WORKER_SECURITY_QUARANTINE_SCHEMA_VERSION = "public-worker-security-quarantine-v0"
PUBLIC_WORKER_RUNTIME_ATTESTATION_SCHEMA_VERSION = "public-worker-runtime-attestation-v0"
PUBLIC_WORKER_SANDBOX_PROFILE_SCHEMA_VERSION = "public-worker-sandbox-profile-v0"
PUBLIC_WORKER_ASSIGNMENT_SCHEMA_VERSION = "public-worker-assignment-v0"
PUBLIC_WORKER_CHALLENGE_SCHEMA_VERSION = "public-worker-verification-challenge-v0"
PUBLIC_WORKER_LOCAL_SIGNATURE_SCHEMA_VERSION = "public-worker-local-signature-v0"
PUBLIC_WORKER_EVIDENCE_VERIFICATION_SCHEMA_VERSION = "public-worker-evidence-verification-v0"
PUBLIC_WORKER_PRODUCTION_RUNTIME_ATTESTATION_SCHEMA_VERSION = (
    "public-worker-production-runtime-attestation-v0"
)
PUBLIC_WORKER_PRODUCTION_SANDBOX_EVIDENCE_SCHEMA_VERSION = "public-worker-production-sandbox-evidence-v0"
PUBLIC_WORKER_PRODUCTION_VERIFICATION_SCHEMA_VERSION = "public-worker-production-verification-v0"
PUBLIC_WORKER_PRODUCTION_VERIFICATION_SIGNATURE_SCHEMA_VERSION = (
    "public-worker-production-verification-signature-v0"
)
PRODUCTION_VERIFICATION_CLOCK_SKEW_TOLERANCE = timedelta(seconds=5)

PeerClass = Literal["owner", "friend", "rented", "public"]
WorkKind = Literal["inference", "training"]
DataScope = Literal["public", "friends", "private"]
CacheScope = Literal["none", "public", "ephemeral_public", "friends", "private"]
AdmissionStatus = Literal["accepted", "rejected"]
ProductionVerificationStatus = Literal["verified", "failed"]
ProductionEvidenceSource = Literal["live-provider", "provider-api-fixture", "local-contract-fixture"]
ProductionVerifierKind = Literal[
    "provider-runtime-attestation",
    "os-sandbox",
    "provider-and-os-sandbox",
    "local-contract-fixture",
    "unavailable",
]

_PRODUCTION_WORKER_SECURITY_VERIFICATION_METADATA_KEYS = [
    "public_worker_security_production_verification_schema_version",
    "public_worker_security_production_verification_id",
    "public_worker_security_production_worker_security_decision_id",
    "public_worker_security_production_assignment_id",
    "public_worker_security_production_assignment_nonce",
    "public_worker_security_production_runtime_id",
    "public_worker_security_production_peer_class",
    "public_worker_security_production_job_kind",
    "public_worker_security_production_verifier_id",
    "public_worker_security_production_verifier_kind",
    "public_worker_security_production_status",
    "public_worker_security_production_reason_codes",
    "public_worker_security_production_runtime_attestation_evidence_id",
    "public_worker_security_production_sandbox_evidence_id",
    "public_worker_security_production_runtime_attestation_verified",
    "public_worker_security_production_sandbox_verified",
    "public_worker_security_production_live_provider_evidence",
    "public_worker_security_production_evidence_source",
    "public_worker_security_production_runtime_attestation_evidence_source",
    "public_worker_security_production_sandbox_evidence_source",
    "public_worker_security_production_evidence_digest",
    "public_worker_security_production_issued_at",
    "public_worker_security_production_expires_at",
    "public_worker_security_production_public_worker_security_claimed",
    "public_worker_security_production_provider_tee_attestation_claimed",
    "public_worker_security_production_sandbox_enforcement_claimed",
]


class RuntimeAttestation(BaseModel):
    schema_version: str = PUBLIC_WORKER_RUNTIME_ATTESTATION_SCHEMA_VERSION
    runtime_id: str
    runtime_kind: str = "unknown"
    build_fingerprint: str
    measurement_digest: str
    nonce: str
    verifier_kind: str = "local-fixture-verifier"
    local_attestation_verified: bool = False
    production_attestation_verified: bool = False


class SandboxProfile(BaseModel):
    schema_version: str = PUBLIC_WORKER_SANDBOX_PROFILE_SCHEMA_VERSION
    sandbox_id: str
    runtime_id: str
    network_policy: Literal["deny-all", "restricted", "open"] = "restricted"
    filesystem_policy: Literal["read-only", "scoped-write", "open"] = "scoped-write"
    process_isolation: bool = True
    local_sandbox_verified: bool = False
    production_sandbox_verified: bool = False


class ProductionRuntimeAttestationEvidence(BaseModel):
    schema_version: str = PUBLIC_WORKER_PRODUCTION_RUNTIME_ATTESTATION_SCHEMA_VERSION
    evidence_id: str = Field(default_factory=lambda: f"production-runtime-attestation-{uuid4().hex}")
    adapter_id: str | None = None
    runtime_id: str
    assignment_id: str
    assignment_nonce: str
    job_kind: WorkKind
    worker_security_decision_id: str
    verifier_id: str
    verifier_kind: ProductionVerifierKind = "unavailable"
    provider_id: str | None = None
    runtime_kind: str = "unknown"
    build_fingerprint: str
    measurement_digest: str
    nonce: str
    issued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    attestation_verified: bool = False
    live_provider_evidence: bool = False
    evidence_source: ProductionEvidenceSource = "local-contract-fixture"
    provider_tee_attestation_claimed: bool = False


class ProductionSandboxEvidence(BaseModel):
    schema_version: str = PUBLIC_WORKER_PRODUCTION_SANDBOX_EVIDENCE_SCHEMA_VERSION
    evidence_id: str = Field(default_factory=lambda: f"production-sandbox-evidence-{uuid4().hex}")
    adapter_id: str | None = None
    sandbox_id: str
    runtime_id: str
    assignment_id: str
    assignment_nonce: str
    job_kind: WorkKind
    worker_security_decision_id: str
    verifier_id: str
    verifier_kind: ProductionVerifierKind = "unavailable"
    network_policy: Literal["deny-all", "restricted", "open"] = "restricted"
    filesystem_policy: Literal["read-only", "scoped-write", "open"] = "scoped-write"
    process_isolation: bool = True
    sandbox_policy_digest: str
    issued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    sandbox_verified: bool = False
    live_provider_evidence: bool = False
    evidence_source: ProductionEvidenceSource = "local-contract-fixture"
    production_sandbox_enforcement_claimed: bool = False


class LocalEvidenceSignature(BaseModel):
    schema_version: str = PUBLIC_WORKER_LOCAL_SIGNATURE_SCHEMA_VERSION
    key_id: str
    payload_digest: str
    signature_algorithm: Literal["hmac-sha256"] = "hmac-sha256"
    signature: str


class ProductionVerificationSignature(BaseModel):
    schema_version: str = PUBLIC_WORKER_PRODUCTION_VERIFICATION_SIGNATURE_SCHEMA_VERSION
    key_id: str
    payload_digest: str
    signature_algorithm: Literal["hmac-sha256"] = "hmac-sha256"
    signature: str


class WorkerSecurityEvidence(BaseModel):
    schema_version: str = PUBLIC_WORKER_SECURITY_EVIDENCE_SCHEMA_VERSION
    evidence_id: str = Field(default_factory=lambda: f"worker-evidence-{uuid4().hex}")
    worker_id: str
    runtime_id: str
    peer_class: PeerClass
    issued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    endpoint_auth_required: bool = False
    audit_enabled: bool = False
    allowed_job_kinds: list[WorkKind] = Field(default_factory=list)
    allowed_model_scopes: list[str] = Field(default_factory=list)
    allowed_dataset_scopes: list[DataScope] = Field(default_factory=lambda: ["public"])
    allowed_cache_scopes: list[CacheScope] = Field(default_factory=lambda: ["none", "public"])
    runtime_attestation: RuntimeAttestation
    sandbox_profile: SandboxProfile
    signature: LocalEvidenceSignature | None = None


class WorkAssignment(BaseModel):
    schema_version: str = PUBLIC_WORKER_ASSIGNMENT_SCHEMA_VERSION
    assignment_id: str = Field(default_factory=lambda: f"assignment-{uuid4().hex}")
    runtime_id: str
    peer_class: PeerClass
    job_kind: WorkKind
    model_scope: str
    dataset_scope: DataScope = "public"
    cache_scope: CacheScope = "none"
    unit_ids: list[str] = Field(default_factory=list)
    nonce: str
    issued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    requires_challenge: bool = True
    quarantine_on_reject: bool = True


class VerificationChallenge(BaseModel):
    schema_version: str = PUBLIC_WORKER_CHALLENGE_SCHEMA_VERSION
    challenge_id: str = Field(default_factory=lambda: f"challenge-{uuid4().hex}")
    assignment_id: str
    runtime_id: str
    nonce: str
    verification_mode: Literal["challenge", "duplicate"] = "challenge"
    input_digest: str
    expected_output_digest: str
    issued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime


class EvidenceVerificationResult(BaseModel):
    schema_version: str = PUBLIC_WORKER_EVIDENCE_VERIFICATION_SCHEMA_VERSION
    evidence_id: str
    runtime_id: str
    status: Literal["verified", "failed"]
    reason_codes: list[str] = Field(default_factory=list)
    signature_verified: bool = False
    payload_digest_matches: bool = False
    payload_digest: str | None = None
    production_attestation_claimed: bool = False


class ProductionWorkerSecurityVerification(BaseModel):
    schema_version: str = PUBLIC_WORKER_PRODUCTION_VERIFICATION_SCHEMA_VERSION
    verification_id: str = Field(default_factory=lambda: f"production-worker-security-{uuid4().hex}")
    runtime_id: str
    assignment_id: str
    assignment_nonce: str
    peer_class: PeerClass
    job_kind: WorkKind
    worker_security_decision_id: str
    verifier_id: str
    verifier_kind: ProductionVerifierKind = "unavailable"
    status: ProductionVerificationStatus
    reason_codes: list[str] = Field(default_factory=list)
    runtime_attestation_evidence_id: str | None = None
    sandbox_evidence_id: str | None = None
    runtime_attestation_verified: bool = False
    sandbox_verified: bool = False
    live_provider_evidence: bool = False
    evidence_source: ProductionEvidenceSource = "local-contract-fixture"
    runtime_attestation_evidence_source: ProductionEvidenceSource | None = None
    sandbox_evidence_source: ProductionEvidenceSource | None = None
    evidence_digest: str | None = None
    issued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    claims: dict[str, bool] = Field(default_factory=lambda: _production_verification_claims(False))
    signature: ProductionVerificationSignature | None = None


class ProductionWorkerSecurityVerifier(Protocol):
    verifier_id: str
    verifier_kind: ProductionVerifierKind

    def verify(
        self,
        *,
        decision: "SecurityAdmissionDecision",
        runtime_attestation: ProductionRuntimeAttestationEvidence | None,
        sandbox: ProductionSandboxEvidence | None,
        current_time: datetime | None = None,
    ) -> ProductionWorkerSecurityVerification:
        ...


class ProviderRuntimeAttestationVerifier(Protocol):
    adapter_id: str

    def verify_runtime_attestation(
        self,
        *,
        decision: "SecurityAdmissionDecision",
        verifier_id: str,
        verifier_kind: ProductionVerifierKind,
        current_time: datetime | None = None,
    ) -> ProductionRuntimeAttestationEvidence:
        ...


class OSSandboxVerifier(Protocol):
    adapter_id: str

    def verify_sandbox(
        self,
        *,
        decision: "SecurityAdmissionDecision",
        verifier_id: str,
        verifier_kind: ProductionVerifierKind,
        current_time: datetime | None = None,
    ) -> ProductionSandboxEvidence:
        ...


class QuarantineRecord(BaseModel):
    schema_version: str = PUBLIC_WORKER_SECURITY_QUARANTINE_SCHEMA_VERSION
    quarantine_id: str = Field(default_factory=lambda: f"quarantine-{uuid4().hex}")
    assignment_id: str
    runtime_id: str
    worker_id: str | None = None
    peer_class: PeerClass
    job_kind: WorkKind
    evidence_id: str | None = None
    challenge_id: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    payload_digest: str | None = None
    action: str = "reject_output_before_training_or_inference_use"
    created_at: datetime = Field(default_factory=utc_now)


class SecurityAdmissionDecision(BaseModel):
    schema_version: str = PUBLIC_WORKER_SECURITY_DECISION_SCHEMA_VERSION
    decision_id: str = Field(default_factory=lambda: f"worker-security-decision-{uuid4().hex}")
    assignment_id: str
    assignment_nonce: str | None = None
    runtime_id: str
    peer_class: PeerClass
    job_kind: WorkKind
    status: AdmissionStatus
    admitted: bool
    reason_codes: list[str] = Field(default_factory=list)
    evidence_verification: EvidenceVerificationResult | None = None
    challenge_required: bool = False
    challenge_verified: bool = False
    replay_decision: Literal["fresh", "reused", "missing"] = "missing"
    quarantine_record: QuarantineRecord | None = None
    app_visible_admission_state: dict[str, Any] = Field(default_factory=dict)
    claims: dict[str, bool] = Field(default_factory=dict)


class PublicWorkerSecurityPolicy(BaseModel):
    schema_version: str = PUBLIC_WORKER_SECURITY_POLICY_SCHEMA_VERSION
    less_trusted_peer_classes: list[PeerClass] = Field(default_factory=lambda: ["rented", "public"])
    require_signature_for_less_trusted: bool = True
    require_local_attestation_for_less_trusted: bool = True
    require_local_sandbox_for_less_trusted: bool = True
    require_endpoint_auth_for_less_trusted: bool = True
    require_audit_for_less_trusted: bool = True
    require_challenge_for_less_trusted: bool = True
    allowed_dataset_scopes_for_less_trusted: list[DataScope] = Field(default_factory=lambda: ["public"])
    allowed_cache_scopes_for_less_trusted: list[CacheScope] = Field(
        default_factory=lambda: ["none", "public", "ephemeral_public"]
    )
    record_nonce_on_accept: bool = True
    quarantine_rejected_less_trusted_outputs: bool = True


class AssignmentReplayGuard:
    def __init__(self, used_nonces: set[str] | None = None) -> None:
        self._used_nonces = set(used_nonces or set())

    @property
    def used_nonces(self) -> set[str]:
        return set(self._used_nonces)

    def check(self, assignment: WorkAssignment) -> Literal["fresh", "reused", "missing"]:
        if not assignment.nonce:
            return "missing"
        nonce_key = self._nonce_key(assignment)
        if nonce_key in self._used_nonces:
            return "reused"
        return "fresh"

    def record(self, assignment: WorkAssignment) -> None:
        if assignment.nonce:
            self._used_nonces.add(self._nonce_key(assignment))

    @staticmethod
    def _nonce_key(assignment: WorkAssignment) -> str:
        return f"{assignment.runtime_id}:{assignment.assignment_id}:{assignment.nonce}"


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def challenge_output_digest(payload: Any) -> str:
    return stable_digest({"challenge_output": payload})


def sign_worker_security_evidence(
    evidence: WorkerSecurityEvidence,
    *,
    key_id: str,
    signing_secret: str,
) -> WorkerSecurityEvidence:
    payload_digest = _worker_evidence_payload_digest(evidence)
    signature = _hmac_signature(payload_digest, signing_secret)
    return evidence.model_copy(
        update={
            "signature": LocalEvidenceSignature(
                key_id=key_id,
                payload_digest=payload_digest,
                signature=signature,
            )
        },
        deep=True,
    )


def verify_worker_security_evidence(
    evidence: WorkerSecurityEvidence,
    *,
    trusted_keys: dict[str, str],
    current_time: datetime | None = None,
) -> EvidenceVerificationResult:
    now = _as_utc(current_time or utc_now())
    reason_codes: list[str] = []
    if evidence.schema_version != PUBLIC_WORKER_SECURITY_EVIDENCE_SCHEMA_VERSION:
        reason_codes.append("worker_security_evidence_schema_version_unsupported")
    if evidence.runtime_attestation.runtime_id != evidence.runtime_id:
        reason_codes.append("runtime_attestation_runtime_mismatch")
    if evidence.sandbox_profile.runtime_id != evidence.runtime_id:
        reason_codes.append("sandbox_profile_runtime_mismatch")
    if _as_utc(evidence.issued_at) > now:
        reason_codes.append("worker_security_evidence_not_yet_valid")
    if _as_utc(evidence.expires_at) <= now:
        reason_codes.append("worker_security_evidence_expired")

    payload_digest = _worker_evidence_payload_digest(evidence)
    payload_digest_matches = False
    signature_verified = False
    if evidence.signature is None:
        reason_codes.append("worker_security_evidence_signature_missing")
    else:
        payload_digest_matches = hmac.compare_digest(evidence.signature.payload_digest, payload_digest)
        if not payload_digest_matches:
            reason_codes.append("worker_security_evidence_payload_digest_mismatch")
        trusted_secret = trusted_keys.get(evidence.signature.key_id)
        if trusted_secret is None:
            reason_codes.append("worker_security_evidence_signature_key_unknown")
        else:
            expected_signature = _hmac_signature(evidence.signature.payload_digest, trusted_secret)
            signature_verified = payload_digest_matches and hmac.compare_digest(
                evidence.signature.signature,
                expected_signature,
            )
            if not signature_verified:
                reason_codes.append("worker_security_evidence_signature_invalid")

    return EvidenceVerificationResult(
        evidence_id=evidence.evidence_id,
        runtime_id=evidence.runtime_id,
        status="verified" if not reason_codes else "failed",
        reason_codes=sorted(set(reason_codes)),
        signature_verified=signature_verified,
        payload_digest_matches=payload_digest_matches,
        payload_digest=payload_digest,
        production_attestation_claimed=bool(evidence.runtime_attestation.production_attestation_verified),
    )


def verify_production_worker_security_evidence(
    *,
    decision: SecurityAdmissionDecision,
    runtime_attestation: ProductionRuntimeAttestationEvidence | None,
    sandbox: ProductionSandboxEvidence | None,
    verifier_id: str,
    verifier_kind: ProductionVerifierKind,
    current_time: datetime | None = None,
    require_live_provider_evidence: bool = True,
) -> ProductionWorkerSecurityVerification:
    now = _as_utc(current_time or utc_now())
    reason_codes: list[str] = []
    decision_nonce = decision.assignment_nonce or ""
    runtime_verified = False
    sandbox_verified = False
    live_provider_evidence = False
    runtime_evidence_source: ProductionEvidenceSource | None = None
    sandbox_evidence_source: ProductionEvidenceSource | None = None

    if decision.status != "accepted" or not decision.admitted:
        reason_codes.append("production_worker_security_decision_not_accepted")
    if not decision_nonce:
        reason_codes.append("production_worker_security_assignment_nonce_missing")
    if verifier_kind in {"unavailable", "local-contract-fixture"}:
        reason_codes.append("production_worker_security_verifier_not_live")

    if runtime_attestation is None:
        reason_codes.append("production_runtime_attestation_evidence_missing")
    else:
        reason_codes.extend(
            _production_evidence_binding_reason_codes(
                schema_version=runtime_attestation.schema_version,
                expected_schema_version=PUBLIC_WORKER_PRODUCTION_RUNTIME_ATTESTATION_SCHEMA_VERSION,
                runtime_id=runtime_attestation.runtime_id,
                assignment_id=runtime_attestation.assignment_id,
                assignment_nonce=runtime_attestation.assignment_nonce,
                job_kind=runtime_attestation.job_kind,
                worker_security_decision_id=runtime_attestation.worker_security_decision_id,
                verifier_id=runtime_attestation.verifier_id,
                verifier_kind=runtime_attestation.verifier_kind,
                expected_runtime_id=decision.runtime_id,
                expected_assignment_id=decision.assignment_id,
                expected_assignment_nonce=decision_nonce,
                expected_job_kind=decision.job_kind,
                expected_worker_security_decision_id=decision.decision_id,
                expected_verifier_id=verifier_id,
                expected_verifier_kind=verifier_kind,
                evidence_label="production_runtime_attestation",
            )
        )
        if _production_time_is_after_now(runtime_attestation.issued_at, now):
            reason_codes.append("production_runtime_attestation_not_yet_valid")
        if _as_utc(runtime_attestation.expires_at) <= now:
            reason_codes.append("production_runtime_attestation_expired")
        if not runtime_attestation.attestation_verified:
            reason_codes.append("production_runtime_attestation_unverified")
        runtime_verified = runtime_attestation.attestation_verified
        runtime_evidence_source = runtime_attestation.evidence_source
        runtime_live_provider_evidence = (
            runtime_attestation.attestation_verified
            and runtime_attestation.live_provider_evidence
            and runtime_attestation.evidence_source == "live-provider"
        )
        if runtime_attestation.live_provider_evidence and runtime_attestation.evidence_source != "live-provider":
            reason_codes.append("production_runtime_attestation_live_source_mismatch")
        live_provider_evidence = runtime_live_provider_evidence

    if sandbox is None:
        reason_codes.append("production_sandbox_evidence_missing")
    else:
        reason_codes.extend(
            _production_evidence_binding_reason_codes(
                schema_version=sandbox.schema_version,
                expected_schema_version=PUBLIC_WORKER_PRODUCTION_SANDBOX_EVIDENCE_SCHEMA_VERSION,
                runtime_id=sandbox.runtime_id,
                assignment_id=sandbox.assignment_id,
                assignment_nonce=sandbox.assignment_nonce,
                job_kind=sandbox.job_kind,
                worker_security_decision_id=sandbox.worker_security_decision_id,
                verifier_id=sandbox.verifier_id,
                verifier_kind=sandbox.verifier_kind,
                expected_runtime_id=decision.runtime_id,
                expected_assignment_id=decision.assignment_id,
                expected_assignment_nonce=decision_nonce,
                expected_job_kind=decision.job_kind,
                expected_worker_security_decision_id=decision.decision_id,
                expected_verifier_id=verifier_id,
                expected_verifier_kind=verifier_kind,
                evidence_label="production_sandbox",
            )
        )
        if _production_time_is_after_now(sandbox.issued_at, now):
            reason_codes.append("production_sandbox_not_yet_valid")
        if _as_utc(sandbox.expires_at) <= now:
            reason_codes.append("production_sandbox_expired")
        if not sandbox.sandbox_verified:
            reason_codes.append("production_sandbox_unverified")
        if sandbox.network_policy == "open":
            reason_codes.append("production_sandbox_network_policy_open")
        if sandbox.filesystem_policy == "open":
            reason_codes.append("production_sandbox_filesystem_policy_open")
        if not sandbox.process_isolation:
            reason_codes.append("production_sandbox_process_isolation_missing")
        sandbox_verified = sandbox.sandbox_verified
        sandbox_evidence_source = sandbox.evidence_source
        sandbox_live_provider_evidence = (
            sandbox.sandbox_verified
            and sandbox.live_provider_evidence
            and sandbox.evidence_source == "live-provider"
        )
        if sandbox.live_provider_evidence and sandbox.evidence_source != "live-provider":
            reason_codes.append("production_sandbox_live_source_mismatch")
        live_provider_evidence = live_provider_evidence and sandbox_live_provider_evidence

    if require_live_provider_evidence and not live_provider_evidence:
        reason_codes.append("production_worker_security_live_provider_evidence_missing")

    unique_reason_codes = list(dict.fromkeys(reason_codes))
    evidence_digest = stable_digest(
        {
            "schema_version": PUBLIC_WORKER_PRODUCTION_VERIFICATION_SCHEMA_VERSION,
            "runtime_attestation": runtime_attestation.model_dump(mode="json")
            if runtime_attestation is not None
            else None,
            "sandbox": sandbox.model_dump(mode="json") if sandbox is not None else None,
            "decision_binding": {
                "decision_id": decision.decision_id,
                "runtime_id": decision.runtime_id,
                "assignment_id": decision.assignment_id,
                "assignment_nonce": decision_nonce,
                "peer_class": decision.peer_class,
                "job_kind": decision.job_kind,
            },
            "verifier_id": verifier_id,
            "verifier_kind": verifier_kind,
        }
    )
    status: ProductionVerificationStatus = "verified" if not unique_reason_codes else "failed"
    verification_suffix_payload = {
        "decision_id": decision.decision_id,
        "evidence_digest": evidence_digest,
        "status": status,
    }
    return ProductionWorkerSecurityVerification(
        verification_id=f"production-worker-security-{_digest_suffix(verification_suffix_payload)}",
        runtime_id=decision.runtime_id,
        assignment_id=decision.assignment_id,
        assignment_nonce=decision_nonce,
        peer_class=decision.peer_class,
        job_kind=decision.job_kind,
        worker_security_decision_id=decision.decision_id,
        verifier_id=verifier_id,
        verifier_kind=verifier_kind,
        status=status,
        reason_codes=unique_reason_codes,
        runtime_attestation_evidence_id=runtime_attestation.evidence_id
        if runtime_attestation is not None
        else None,
        sandbox_evidence_id=sandbox.evidence_id if sandbox is not None else None,
        runtime_attestation_verified=runtime_verified,
        sandbox_verified=sandbox_verified,
        live_provider_evidence=live_provider_evidence,
        evidence_source="live-provider" if live_provider_evidence else "local-contract-fixture",
        runtime_attestation_evidence_source=runtime_evidence_source,
        sandbox_evidence_source=sandbox_evidence_source,
        evidence_digest=evidence_digest,
        issued_at=now,
        expires_at=now + timedelta(minutes=5),
        claims=_production_verification_claims(status == "verified" and live_provider_evidence),
    )


def sign_production_worker_security_verification(
    verification: ProductionWorkerSecurityVerification,
    *,
    key_id: str,
    signing_secret: str,
) -> ProductionWorkerSecurityVerification:
    unsigned = verification.model_copy(update={"signature": None}, deep=True)
    payload_digest = _production_worker_security_verification_metadata_payload_digest(unsigned)
    return unsigned.model_copy(
        update={
            "signature": ProductionVerificationSignature(
                key_id=key_id,
                payload_digest=payload_digest,
                signature=_hmac_signature(payload_digest, signing_secret),
            )
        },
        deep=True,
    )


def build_signed_production_worker_security_verification(
    *,
    decision: SecurityAdmissionDecision,
    runtime_attestation_verifier: ProviderRuntimeAttestationVerifier,
    sandbox_verifier: OSSandboxVerifier,
    verifier_id: str,
    signing_key_id: str,
    signing_secret: str,
    verifier_kind: ProductionVerifierKind = "provider-and-os-sandbox",
    current_time: datetime | None = None,
    require_live_provider_evidence: bool = True,
) -> ProductionWorkerSecurityVerification:
    runtime_attestation = runtime_attestation_verifier.verify_runtime_attestation(
        decision=decision,
        verifier_id=verifier_id,
        verifier_kind=verifier_kind,
        current_time=current_time,
    )
    sandbox = sandbox_verifier.verify_sandbox(
        decision=decision,
        verifier_id=verifier_id,
        verifier_kind=verifier_kind,
        current_time=current_time,
    )
    verification = verify_production_worker_security_evidence(
        decision=decision,
        runtime_attestation=runtime_attestation,
        sandbox=sandbox,
        verifier_id=verifier_id,
        verifier_kind=verifier_kind,
        current_time=current_time,
        require_live_provider_evidence=require_live_provider_evidence,
    )
    return sign_production_worker_security_verification(
        verification,
        key_id=signing_key_id,
        signing_secret=signing_secret,
    )


class StaticProviderRuntimeAttestationVerifier:
    def __init__(
        self,
        *,
        adapter_id: str = "static-provider-runtime-attestation",
        provider_id: str | None = None,
        runtime_kind: str = "unknown",
        build_fingerprint: str = "sha256:static-runtime-build",
        measurement_digest: str = "sha256:static-runtime-measurement",
        nonce: str = "static-runtime-attestation-nonce",
        attestation_verified: bool = False,
        live_provider_evidence: bool = False,
        evidence_source: ProductionEvidenceSource = "local-contract-fixture",
        ttl_seconds: float = 300.0,
    ) -> None:
        self.adapter_id = adapter_id
        self.provider_id = provider_id
        self.runtime_kind = runtime_kind
        self.build_fingerprint = build_fingerprint
        self.measurement_digest = measurement_digest
        self.nonce = nonce
        self.attestation_verified = attestation_verified
        self.live_provider_evidence = live_provider_evidence
        self.evidence_source = evidence_source
        self.ttl_seconds = ttl_seconds

    def verify_runtime_attestation(
        self,
        *,
        decision: SecurityAdmissionDecision,
        verifier_id: str,
        verifier_kind: ProductionVerifierKind,
        current_time: datetime | None = None,
    ) -> ProductionRuntimeAttestationEvidence:
        now = _as_utc(current_time or utc_now())
        evidence_suffix_payload = {
            "adapter_id": self.adapter_id,
            "decision_id": decision.decision_id,
            "runtime_id": decision.runtime_id,
        }
        return ProductionRuntimeAttestationEvidence(
            evidence_id=f"production-runtime-attestation-{_digest_suffix(evidence_suffix_payload)}",
            adapter_id=self.adapter_id,
            runtime_id=decision.runtime_id,
            assignment_id=decision.assignment_id,
            assignment_nonce=decision.assignment_nonce or "",
            job_kind=decision.job_kind,
            worker_security_decision_id=decision.decision_id,
            verifier_id=verifier_id,
            verifier_kind=verifier_kind,
            provider_id=self.provider_id,
            runtime_kind=self.runtime_kind,
            build_fingerprint=self.build_fingerprint,
            measurement_digest=self.measurement_digest,
            nonce=self.nonce,
            issued_at=now,
            expires_at=now + timedelta(seconds=self.ttl_seconds),
            attestation_verified=self.attestation_verified,
            live_provider_evidence=self.live_provider_evidence,
            evidence_source=self.evidence_source,
            provider_tee_attestation_claimed=False,
        )


class StaticOSSandboxVerifier:
    def __init__(
        self,
        *,
        adapter_id: str = "static-os-sandbox",
        sandbox_id_prefix: str = "static-sandbox",
        network_policy: Literal["deny-all", "restricted", "open"] = "restricted",
        filesystem_policy: Literal["read-only", "scoped-write", "open"] = "scoped-write",
        process_isolation: bool = True,
        sandbox_policy_digest: str = "sha256:static-sandbox-policy",
        sandbox_verified: bool = False,
        live_provider_evidence: bool = False,
        evidence_source: ProductionEvidenceSource = "local-contract-fixture",
        ttl_seconds: float = 300.0,
    ) -> None:
        self.adapter_id = adapter_id
        self.sandbox_id_prefix = sandbox_id_prefix
        self.network_policy = network_policy
        self.filesystem_policy = filesystem_policy
        self.process_isolation = process_isolation
        self.sandbox_policy_digest = sandbox_policy_digest
        self.sandbox_verified = sandbox_verified
        self.live_provider_evidence = live_provider_evidence
        self.evidence_source = evidence_source
        self.ttl_seconds = ttl_seconds

    def verify_sandbox(
        self,
        *,
        decision: SecurityAdmissionDecision,
        verifier_id: str,
        verifier_kind: ProductionVerifierKind,
        current_time: datetime | None = None,
    ) -> ProductionSandboxEvidence:
        now = _as_utc(current_time or utc_now())
        evidence_suffix_payload = {
            "adapter_id": self.adapter_id,
            "decision_id": decision.decision_id,
            "runtime_id": decision.runtime_id,
        }
        return ProductionSandboxEvidence(
            evidence_id=f"production-sandbox-{_digest_suffix(evidence_suffix_payload)}",
            adapter_id=self.adapter_id,
            sandbox_id=f"{self.sandbox_id_prefix}-{decision.runtime_id}",
            runtime_id=decision.runtime_id,
            assignment_id=decision.assignment_id,
            assignment_nonce=decision.assignment_nonce or "",
            job_kind=decision.job_kind,
            worker_security_decision_id=decision.decision_id,
            verifier_id=verifier_id,
            verifier_kind=verifier_kind,
            network_policy=self.network_policy,
            filesystem_policy=self.filesystem_policy,
            process_isolation=self.process_isolation,
            sandbox_policy_digest=self.sandbox_policy_digest,
            issued_at=now,
            expires_at=now + timedelta(seconds=self.ttl_seconds),
            sandbox_verified=self.sandbox_verified,
            live_provider_evidence=self.live_provider_evidence,
            evidence_source=self.evidence_source,
            production_sandbox_enforcement_claimed=False,
        )


class LinuxProcessRuntimeAttestationVerifier:
    def __init__(
        self,
        *,
        adapter_id: str = "linux-process-runtime-attestation-v0",
        provider_id: str | None = None,
        runtime_kind: str = "linux-process",
        platform_system: str | None = None,
        machine: str | None = None,
        python_executable: str | None = None,
        pid: int | None = None,
        ttl_seconds: float = 300.0,
    ) -> None:
        self.adapter_id = adapter_id
        self.provider_id = provider_id
        self.runtime_kind = runtime_kind
        self.platform_system = platform_system
        self.machine = machine
        self.python_executable = python_executable
        self.pid = pid
        self.ttl_seconds = ttl_seconds

    def verify_runtime_attestation(
        self,
        *,
        decision: SecurityAdmissionDecision,
        verifier_id: str,
        verifier_kind: ProductionVerifierKind,
        current_time: datetime | None = None,
    ) -> ProductionRuntimeAttestationEvidence:
        now = _as_utc(current_time or utc_now())
        system = self.platform_system or platform.system()
        machine = self.machine or platform.machine()
        executable = self.python_executable or sys.executable
        pid = self.pid if self.pid is not None else os.getpid()
        linux_posture_verified = system.lower() == "linux" and bool(decision.assignment_nonce)
        runtime_descriptor = {
            "adapter_id": self.adapter_id,
            "runtime_id": decision.runtime_id,
            "system": system,
            "machine": machine,
            "python_version": sys.version.split()[0],
            "python_executable_name": Path(executable).name,
            "pid_present": bool(pid),
            "provider_id": self.provider_id,
        }
        measurement_descriptor = {
            **runtime_descriptor,
            "assignment_id": decision.assignment_id,
            "assignment_nonce_digest": stable_digest(decision.assignment_nonce or ""),
            "worker_security_decision_id": decision.decision_id,
        }
        evidence_suffix_payload = {
            "adapter_id": self.adapter_id,
            "decision_id": decision.decision_id,
            "runtime_id": decision.runtime_id,
        }
        return ProductionRuntimeAttestationEvidence(
            evidence_id=f"production-runtime-attestation-{_digest_suffix(evidence_suffix_payload)}",
            adapter_id=self.adapter_id,
            runtime_id=decision.runtime_id,
            assignment_id=decision.assignment_id,
            assignment_nonce=decision.assignment_nonce or "",
            job_kind=decision.job_kind,
            worker_security_decision_id=decision.decision_id,
            verifier_id=verifier_id,
            verifier_kind=verifier_kind,
            provider_id=self.provider_id,
            runtime_kind=self.runtime_kind,
            build_fingerprint=stable_digest(runtime_descriptor),
            measurement_digest=stable_digest(measurement_descriptor),
            nonce=stable_digest(
                {
                    "runtime_id": decision.runtime_id,
                    "assignment_id": decision.assignment_id,
                    "assignment_nonce": decision.assignment_nonce or "",
                    "adapter_id": self.adapter_id,
                }
            ),
            issued_at=now,
            expires_at=now + timedelta(seconds=self.ttl_seconds),
            attestation_verified=linux_posture_verified,
            live_provider_evidence=linux_posture_verified,
            evidence_source="live-provider" if linux_posture_verified else "local-contract-fixture",
            provider_tee_attestation_claimed=False,
        )


class LinuxProcSandboxVerifier:
    def __init__(
        self,
        *,
        adapter_id: str = "linux-proc-sandbox-verifier-v0",
        sandbox_id_prefix: str = "linux-proc-sandbox",
        proc_root: Path | str = Path("/proc"),
        platform_system: str | None = None,
        status_text: str | None = None,
        namespace_inodes: dict[str, str] | None = None,
        require_seccomp: bool = True,
        network_policy: Literal["deny-all", "restricted", "open"] = "restricted",
        filesystem_policy: Literal["read-only", "scoped-write", "open"] = "scoped-write",
        ttl_seconds: float = 300.0,
    ) -> None:
        self.adapter_id = adapter_id
        self.sandbox_id_prefix = sandbox_id_prefix
        self.proc_root = Path(proc_root)
        self.platform_system = platform_system
        self.status_text = status_text
        self.namespace_inodes = namespace_inodes
        self.require_seccomp = require_seccomp
        self.network_policy = network_policy
        self.filesystem_policy = filesystem_policy
        self.ttl_seconds = ttl_seconds

    def verify_sandbox(
        self,
        *,
        decision: SecurityAdmissionDecision,
        verifier_id: str,
        verifier_kind: ProductionVerifierKind,
        current_time: datetime | None = None,
    ) -> ProductionSandboxEvidence:
        now = _as_utc(current_time or utc_now())
        system = self.platform_system or platform.system()
        status_text = self.status_text if self.status_text is not None else _read_proc_status(self.proc_root)
        namespace_inodes = (
            dict(self.namespace_inodes)
            if self.namespace_inodes is not None
            else _read_proc_namespace_inodes(self.proc_root)
        )
        seccomp_mode = _linux_proc_status_int(status_text, "Seccomp")
        no_new_privs = _linux_proc_status_int(status_text, "NoNewPrivs")
        namespace_count = len(namespace_inodes)
        seccomp_ok = not self.require_seccomp or (seccomp_mode is not None and seccomp_mode > 0)
        process_isolation = namespace_count >= 2
        sandbox_verified = (
            system.lower() == "linux"
            and seccomp_ok
            and process_isolation
            and self.network_policy != "open"
            and self.filesystem_policy != "open"
        )
        policy_descriptor = {
            "adapter_id": self.adapter_id,
            "runtime_id": decision.runtime_id,
            "system": system,
            "network_policy": self.network_policy,
            "filesystem_policy": self.filesystem_policy,
            "seccomp_mode": seccomp_mode,
            "no_new_privs": no_new_privs,
            "namespace_inodes": namespace_inodes,
            "process_isolation": process_isolation,
        }
        evidence_suffix_payload = {
            "adapter_id": self.adapter_id,
            "decision_id": decision.decision_id,
            "runtime_id": decision.runtime_id,
        }
        return ProductionSandboxEvidence(
            evidence_id=f"production-sandbox-{_digest_suffix(evidence_suffix_payload)}",
            adapter_id=self.adapter_id,
            sandbox_id=f"{self.sandbox_id_prefix}-{decision.runtime_id}",
            runtime_id=decision.runtime_id,
            assignment_id=decision.assignment_id,
            assignment_nonce=decision.assignment_nonce or "",
            job_kind=decision.job_kind,
            worker_security_decision_id=decision.decision_id,
            verifier_id=verifier_id,
            verifier_kind=verifier_kind,
            network_policy=self.network_policy,
            filesystem_policy=self.filesystem_policy,
            process_isolation=process_isolation,
            sandbox_policy_digest=stable_digest(policy_descriptor),
            issued_at=now,
            expires_at=now + timedelta(seconds=self.ttl_seconds),
            sandbox_verified=sandbox_verified,
            live_provider_evidence=sandbox_verified,
            evidence_source="live-provider" if sandbox_verified else "local-contract-fixture",
            production_sandbox_enforcement_claimed=False,
        )


class ProductionEvidenceBundleRuntimeAttestationVerifier:
    def __init__(
        self,
        runtime_attestations: list[ProductionRuntimeAttestationEvidence | dict[str, Any]],
        *,
        adapter_id: str = "production-evidence-bundle-runtime-attestation",
    ) -> None:
        self.adapter_id = adapter_id
        self._runtime_attestations = [
            item
            if isinstance(item, ProductionRuntimeAttestationEvidence)
            else ProductionRuntimeAttestationEvidence.model_validate(item)
            for item in runtime_attestations
        ]

    def verify_runtime_attestation(
        self,
        *,
        decision: SecurityAdmissionDecision,
        verifier_id: str,
        verifier_kind: ProductionVerifierKind,
        current_time: datetime | None = None,
    ) -> ProductionRuntimeAttestationEvidence:
        key = _production_bundle_lookup_key(decision, verifier_id, verifier_kind)
        for evidence in self._runtime_attestations:
            if _production_runtime_attestation_lookup_key(evidence) == key:
                return evidence.model_copy(
                    update={"adapter_id": evidence.adapter_id or self.adapter_id},
                    deep=True,
                )
        now = _as_utc(current_time or utc_now())
        evidence_suffix_payload = {
            "adapter_id": self.adapter_id,
            "decision_id": decision.decision_id,
            "runtime_id": decision.runtime_id,
        }
        return ProductionRuntimeAttestationEvidence(
            evidence_id=f"production-runtime-attestation-missing-{_digest_suffix(evidence_suffix_payload)}",
            adapter_id=self.adapter_id,
            runtime_id=decision.runtime_id,
            assignment_id=decision.assignment_id,
            assignment_nonce=decision.assignment_nonce or "",
            job_kind=decision.job_kind,
            worker_security_decision_id=decision.decision_id,
            verifier_id=verifier_id,
            verifier_kind=verifier_kind,
            runtime_kind="unknown",
            build_fingerprint="sha256:missing-runtime-attestation",
            measurement_digest="sha256:missing-runtime-attestation",
            nonce="missing-runtime-attestation",
            issued_at=now,
            expires_at=now + timedelta(seconds=1),
            attestation_verified=False,
            live_provider_evidence=False,
            evidence_source="local-contract-fixture",
            provider_tee_attestation_claimed=False,
        )


class ProductionEvidenceBundleOSSandboxVerifier:
    def __init__(
        self,
        sandbox_evidence: list[ProductionSandboxEvidence | dict[str, Any]],
        *,
        adapter_id: str = "production-evidence-bundle-os-sandbox",
    ) -> None:
        self.adapter_id = adapter_id
        self._sandbox_evidence = [
            item
            if isinstance(item, ProductionSandboxEvidence)
            else ProductionSandboxEvidence.model_validate(item)
            for item in sandbox_evidence
        ]

    def verify_sandbox(
        self,
        *,
        decision: SecurityAdmissionDecision,
        verifier_id: str,
        verifier_kind: ProductionVerifierKind,
        current_time: datetime | None = None,
    ) -> ProductionSandboxEvidence:
        key = _production_bundle_lookup_key(decision, verifier_id, verifier_kind)
        for evidence in self._sandbox_evidence:
            if _production_sandbox_lookup_key(evidence) == key:
                return evidence.model_copy(
                    update={"adapter_id": evidence.adapter_id or self.adapter_id},
                    deep=True,
                )
        now = _as_utc(current_time or utc_now())
        evidence_suffix_payload = {
            "adapter_id": self.adapter_id,
            "decision_id": decision.decision_id,
            "runtime_id": decision.runtime_id,
        }
        return ProductionSandboxEvidence(
            evidence_id=f"production-sandbox-missing-{_digest_suffix(evidence_suffix_payload)}",
            adapter_id=self.adapter_id,
            sandbox_id=f"missing-sandbox-{decision.runtime_id}",
            runtime_id=decision.runtime_id,
            assignment_id=decision.assignment_id,
            assignment_nonce=decision.assignment_nonce or "",
            job_kind=decision.job_kind,
            worker_security_decision_id=decision.decision_id,
            verifier_id=verifier_id,
            verifier_kind=verifier_kind,
            sandbox_policy_digest="sha256:missing-sandbox-evidence",
            issued_at=now,
            expires_at=now + timedelta(seconds=1),
            sandbox_verified=False,
            live_provider_evidence=False,
            evidence_source="local-contract-fixture",
            production_sandbox_enforcement_claimed=False,
        )


class UnavailableProviderRuntimeAttestationVerifier(StaticProviderRuntimeAttestationVerifier):
    def __init__(self) -> None:
        super().__init__(
            adapter_id="provider-runtime-attestation-unavailable",
            attestation_verified=False,
            live_provider_evidence=False,
            evidence_source="local-contract-fixture",
        )


class UnavailableOSSandboxVerifier(StaticOSSandboxVerifier):
    def __init__(self) -> None:
        super().__init__(
            adapter_id="os-sandbox-unavailable",
            sandbox_verified=False,
            live_provider_evidence=False,
            evidence_source="local-contract-fixture",
        )


class UnavailableProductionWorkerSecurityVerifier:
    verifier_id = "production-worker-security-verifier-unavailable"
    verifier_kind: ProductionVerifierKind = "unavailable"

    def verify(
        self,
        *,
        decision: SecurityAdmissionDecision,
        runtime_attestation: ProductionRuntimeAttestationEvidence | None,
        sandbox: ProductionSandboxEvidence | None,
        current_time: datetime | None = None,
    ) -> ProductionWorkerSecurityVerification:
        verification = verify_production_worker_security_evidence(
            decision=decision,
            runtime_attestation=runtime_attestation,
            sandbox=sandbox,
            verifier_id=self.verifier_id,
            verifier_kind=self.verifier_kind,
            current_time=current_time,
        )
        if "production_worker_security_verifier_unavailable" in verification.reason_codes:
            return verification
        return verification.model_copy(
            update={
                "status": "failed",
                "reason_codes": list(
                    dict.fromkeys(
                        [
                            "production_worker_security_verifier_unavailable",
                            *verification.reason_codes,
                        ]
                    )
                ),
                "claims": _production_verification_claims(False),
            },
            deep=True,
        )


def admit_public_worker_assignment(
    *,
    assignment: WorkAssignment,
    evidence: WorkerSecurityEvidence | None,
    policy: PublicWorkerSecurityPolicy | None = None,
    trusted_keys: dict[str, str] | None = None,
    replay_guard: AssignmentReplayGuard | None = None,
    challenge: VerificationChallenge | None = None,
    observed_output_digest: str | None = None,
    current_time: datetime | None = None,
) -> SecurityAdmissionDecision:
    resolved_policy = policy or PublicWorkerSecurityPolicy()
    keyring = trusted_keys or {}
    guard = replay_guard or AssignmentReplayGuard()
    now = _as_utc(current_time or utc_now())
    reason_codes: list[str] = []
    evidence_verification: EvidenceVerificationResult | None = None
    worker_id: str | None = None
    evidence_id: str | None = None
    less_trusted = assignment.peer_class in set(resolved_policy.less_trusted_peer_classes)

    replay_decision = guard.check(assignment)
    if replay_decision == "missing":
        reason_codes.append("assignment_nonce_missing")
    elif replay_decision == "reused":
        reason_codes.append("assignment_nonce_reused")

    if _as_utc(assignment.issued_at) > now:
        reason_codes.append("assignment_not_yet_valid")
    if _as_utc(assignment.expires_at) <= now:
        reason_codes.append("assignment_expired")

    if evidence is None:
        reason_codes.append("worker_security_evidence_missing")
    else:
        worker_id = evidence.worker_id
        evidence_id = evidence.evidence_id
        evidence_verification = verify_worker_security_evidence(
            evidence,
            trusted_keys=keyring,
            current_time=now,
        )
        reason_codes.extend(evidence_verification.reason_codes)
        if evidence.runtime_id != assignment.runtime_id:
            reason_codes.append("assignment_runtime_mismatch")
        if evidence.peer_class != assignment.peer_class:
            reason_codes.append("assignment_peer_class_mismatch")
        if assignment.job_kind not in evidence.allowed_job_kinds:
            reason_codes.append("assignment_job_kind_unsupported")
        if assignment.model_scope not in evidence.allowed_model_scopes:
            reason_codes.append("assignment_model_scope_unsupported")
        if assignment.dataset_scope not in evidence.allowed_dataset_scopes:
            reason_codes.append("assignment_dataset_scope_unsupported")
        if assignment.cache_scope not in evidence.allowed_cache_scopes:
            reason_codes.append("assignment_cache_scope_unsupported")
        if less_trusted:
            if resolved_policy.require_signature_for_less_trusted and not evidence_verification.signature_verified:
                reason_codes.append("less_trusted_evidence_signature_unverified")
            if (
                resolved_policy.require_local_attestation_for_less_trusted
                and not evidence.runtime_attestation.local_attestation_verified
            ):
                reason_codes.append("less_trusted_runtime_attestation_unverified")
            if (
                resolved_policy.require_local_sandbox_for_less_trusted
                and not evidence.sandbox_profile.local_sandbox_verified
            ):
                reason_codes.append("less_trusted_sandbox_unverified")
            if resolved_policy.require_endpoint_auth_for_less_trusted and not evidence.endpoint_auth_required:
                reason_codes.append("less_trusted_endpoint_auth_missing")
            if resolved_policy.require_audit_for_less_trusted and not evidence.audit_enabled:
                reason_codes.append("less_trusted_audit_missing")

    if less_trusted and assignment.dataset_scope not in resolved_policy.allowed_dataset_scopes_for_less_trusted:
        reason_codes.append("less_trusted_dataset_scope_not_allowed")
    if less_trusted and assignment.cache_scope not in resolved_policy.allowed_cache_scopes_for_less_trusted:
        reason_codes.append("less_trusted_cache_scope_not_allowed")

    challenge_required = assignment.requires_challenge or (
        less_trusted and resolved_policy.require_challenge_for_less_trusted
    )
    challenge_verified = False
    if challenge_required:
        if challenge is None:
            reason_codes.append("verification_challenge_missing")
        else:
            if _as_utc(challenge.issued_at) > now:
                reason_codes.append("verification_challenge_not_yet_valid")
            if _as_utc(challenge.expires_at) <= now:
                reason_codes.append("verification_challenge_expired")
            if challenge.assignment_id != assignment.assignment_id:
                reason_codes.append("verification_challenge_assignment_mismatch")
            if challenge.runtime_id != assignment.runtime_id:
                reason_codes.append("verification_challenge_runtime_mismatch")
            if challenge.nonce != assignment.nonce:
                reason_codes.append("verification_challenge_nonce_mismatch")
            if observed_output_digest is None:
                reason_codes.append("verification_challenge_output_missing")
            elif not hmac.compare_digest(challenge.expected_output_digest, observed_output_digest):
                reason_codes.append("verification_challenge_output_digest_mismatch")
            else:
                challenge_verified = True

    unique_reason_codes = sorted(set(reason_codes))
    admitted = not unique_reason_codes
    quarantine_record: QuarantineRecord | None = None
    if admitted and resolved_policy.record_nonce_on_accept:
        guard.record(assignment)
    elif _should_quarantine(assignment, less_trusted, resolved_policy):
        quarantine_suffix_payload = {
            "assignment_id": assignment.assignment_id,
            "runtime_id": assignment.runtime_id,
            "reason_codes": unique_reason_codes,
        }
        quarantine_record = QuarantineRecord(
            quarantine_id=f"quarantine-{_digest_suffix(quarantine_suffix_payload)}",
            assignment_id=assignment.assignment_id,
            runtime_id=assignment.runtime_id,
            worker_id=worker_id,
            peer_class=assignment.peer_class,
            job_kind=assignment.job_kind,
            evidence_id=evidence_id,
            challenge_id=challenge.challenge_id if challenge is not None else None,
            reason_codes=unique_reason_codes,
            payload_digest=observed_output_digest,
            created_at=now,
        )

    claims = _decision_claims()
    decision_suffix_payload = {
        "assignment_id": assignment.assignment_id,
        "runtime_id": assignment.runtime_id,
        "status": "accepted" if admitted else "rejected",
        "reason_codes": unique_reason_codes,
    }
    return SecurityAdmissionDecision(
        decision_id=f"worker-security-decision-{_digest_suffix(decision_suffix_payload)}",
        assignment_id=assignment.assignment_id,
        assignment_nonce=assignment.nonce,
        runtime_id=assignment.runtime_id,
        peer_class=assignment.peer_class,
        job_kind=assignment.job_kind,
        status="accepted" if admitted else "rejected",
        admitted=admitted,
        reason_codes=unique_reason_codes,
        evidence_verification=evidence_verification,
        challenge_required=challenge_required,
        challenge_verified=challenge_verified,
        replay_decision=replay_decision,
        quarantine_record=quarantine_record,
        app_visible_admission_state=_app_visible_state(
            assignment=assignment,
            admitted=admitted,
            reason_codes=unique_reason_codes,
            challenge_required=challenge_required,
            challenge_verified=challenge_verified,
            replay_decision=replay_decision,
            quarantine_record=quarantine_record,
        ),
        claims=claims,
    )


def summarize_security_decisions(decisions: list[SecurityAdmissionDecision]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for decision in decisions:
        reason_counts.update(decision.reason_codes)
    return {
        "schema_version": "public-worker-security-decision-summary-v0",
        "decision_count": len(decisions),
        "accepted_count": sum(1 for decision in decisions if decision.admitted),
        "rejected_count": sum(1 for decision in decisions if not decision.admitted),
        "quarantine_count": sum(1 for decision in decisions if decision.quarantine_record is not None),
        "accepted_assignment_ids": [
            decision.assignment_id for decision in decisions if decision.admitted
        ],
        "rejected_assignment_ids": [
            decision.assignment_id for decision in decisions if not decision.admitted
        ],
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "production_public_worker_security_claimed": False,
        "production_attestation_claimed": False,
        "provider_accounting_claimed": False,
        "token_launch_claimed": False,
    }


def public_worker_security_decision_metadata(
    decision: SecurityAdmissionDecision,
    production_verification: ProductionWorkerSecurityVerification | None = None,
) -> dict[str, Any]:
    metadata = {
        "public_worker_security_schema_version": decision.schema_version,
        "public_worker_security_decision_id": decision.decision_id,
        "public_worker_security_assignment_id": decision.assignment_id,
        "public_worker_security_assignment_nonce": decision.assignment_nonce,
        "public_worker_security_runtime_id": decision.runtime_id,
        "public_worker_security_peer_class": decision.peer_class,
        "public_worker_security_job_kind": decision.job_kind,
        "public_worker_security_status": decision.status,
        "public_worker_security_admitted": decision.admitted,
        "public_worker_security_reason_codes": list(decision.reason_codes),
        "public_worker_security_challenge_required": decision.challenge_required,
        "public_worker_security_challenge_verified": decision.challenge_verified,
        "public_worker_security_replay_decision": decision.replay_decision,
        "public_worker_security_quarantined": decision.quarantine_record is not None,
        "public_worker_security_production_public_worker_security_claimed": bool(
            decision.claims.get("production_public_worker_security_claimed")
        ),
        "public_worker_security_provider_accounting_claimed": bool(
            decision.claims.get("provider_accounting_claimed")
        ),
        "public_worker_security_token_launch_claimed": bool(decision.claims.get("token_launch_claimed")),
    }
    if production_verification is not None:
        metadata.update(production_worker_security_verification_metadata(production_verification))
    return metadata


def production_worker_security_verification_metadata(
    verification: ProductionWorkerSecurityVerification,
) -> dict[str, Any]:
    metadata = _production_worker_security_verification_metadata_payload(verification)
    if verification.signature is not None:
        metadata.update(
            {
                "public_worker_security_production_signature_schema_version": (
                    verification.signature.schema_version
                ),
                "public_worker_security_production_signature_key_id": verification.signature.key_id,
                "public_worker_security_production_signature_algorithm": (
                    verification.signature.signature_algorithm
                ),
                "public_worker_security_production_signature_payload_digest": (
                    verification.signature.payload_digest
                ),
                "public_worker_security_production_signature": verification.signature.signature,
            }
        )
    return metadata


def _production_worker_security_verification_metadata_payload(
    verification: ProductionWorkerSecurityVerification,
) -> dict[str, Any]:
    return {
        "public_worker_security_production_verification_schema_version": verification.schema_version,
        "public_worker_security_production_verification_id": verification.verification_id,
        "public_worker_security_production_worker_security_decision_id": (
            verification.worker_security_decision_id
        ),
        "public_worker_security_production_assignment_id": verification.assignment_id,
        "public_worker_security_production_assignment_nonce": verification.assignment_nonce,
        "public_worker_security_production_runtime_id": verification.runtime_id,
        "public_worker_security_production_peer_class": verification.peer_class,
        "public_worker_security_production_job_kind": verification.job_kind,
        "public_worker_security_production_verifier_id": verification.verifier_id,
        "public_worker_security_production_verifier_kind": verification.verifier_kind,
        "public_worker_security_production_status": verification.status,
        "public_worker_security_production_reason_codes": list(verification.reason_codes),
        "public_worker_security_production_runtime_attestation_evidence_id": (
            verification.runtime_attestation_evidence_id
        ),
        "public_worker_security_production_sandbox_evidence_id": verification.sandbox_evidence_id,
        "public_worker_security_production_runtime_attestation_verified": (
            verification.runtime_attestation_verified
        ),
        "public_worker_security_production_sandbox_verified": verification.sandbox_verified,
        "public_worker_security_production_live_provider_evidence": verification.live_provider_evidence,
        "public_worker_security_production_evidence_source": verification.evidence_source,
        "public_worker_security_production_runtime_attestation_evidence_source": (
            verification.runtime_attestation_evidence_source
        ),
        "public_worker_security_production_sandbox_evidence_source": (
            verification.sandbox_evidence_source
        ),
        "public_worker_security_production_evidence_digest": verification.evidence_digest,
        "public_worker_security_production_issued_at": verification.issued_at.isoformat(),
        "public_worker_security_production_expires_at": verification.expires_at.isoformat(),
        "public_worker_security_production_public_worker_security_claimed": bool(
            verification.claims.get("production_public_worker_security_claimed")
        ),
        "public_worker_security_production_provider_tee_attestation_claimed": bool(
            verification.claims.get("production_provider_tee_attestation_claimed")
        ),
        "public_worker_security_production_sandbox_enforcement_claimed": bool(
            verification.claims.get("production_sandbox_enforcement_claimed")
        ),
    }


def validate_public_worker_security_decision_metadata(
    metadata: dict[str, Any],
    *,
    required_decision_id: str | None = None,
    runtime_id: str | None = None,
    peer_class: PeerClass | None = None,
    job_kind: WorkKind | None = None,
    require_challenge_verified: bool = True,
    production_mode: bool = False,
    trusted_production_verifier_ids: set[str] | None = None,
    trusted_production_verifier_keys: dict[str, str] | None = None,
    current_time: datetime | None = None,
) -> list[str]:
    reason_codes: list[str] = []
    decision_id = metadata.get("public_worker_security_decision_id")
    if not isinstance(decision_id, str) or not decision_id:
        reason_codes.append("public_worker_security_decision_missing")
    elif required_decision_id is not None and decision_id != required_decision_id:
        reason_codes.append("public_worker_security_decision_mismatch")

    if metadata.get("public_worker_security_schema_version") not in {
        None,
        PUBLIC_WORKER_SECURITY_DECISION_SCHEMA_VERSION,
    }:
        reason_codes.append("public_worker_security_decision_schema_version_unsupported")

    status = metadata.get("public_worker_security_status")
    admitted = metadata.get("public_worker_security_admitted")
    if status != "accepted" or admitted is not True:
        reason_codes.append("public_worker_security_decision_not_accepted")

    if runtime_id is not None and metadata.get("public_worker_security_runtime_id") != runtime_id:
        reason_codes.append("public_worker_security_runtime_mismatch")
    if peer_class is not None and metadata.get("public_worker_security_peer_class") != peer_class:
        reason_codes.append("public_worker_security_peer_class_mismatch")
    if job_kind is not None and metadata.get("public_worker_security_job_kind") != job_kind:
        reason_codes.append("public_worker_security_job_kind_mismatch")

    challenge_required = metadata.get("public_worker_security_challenge_required")
    challenge_verified = metadata.get("public_worker_security_challenge_verified")
    if require_challenge_verified and challenge_required is True and challenge_verified is not True:
        reason_codes.append("public_worker_security_challenge_not_verified")

    if metadata.get("public_worker_security_quarantined") is True:
        reason_codes.append("public_worker_security_decision_quarantined")
    if metadata.get("public_worker_security_provider_accounting_claimed") is True:
        reason_codes.append("public_worker_security_provider_accounting_claim_forbidden")
    if metadata.get("public_worker_security_token_launch_claimed") is True:
        reason_codes.append("public_worker_security_token_launch_claim_forbidden")
    if production_mode:
        reason_codes.extend(
            validate_production_worker_security_verification_metadata(
                metadata,
                required_decision_id=decision_id if isinstance(decision_id, str) else required_decision_id,
                runtime_id=runtime_id,
                assignment_id=_optional_metadata_str(metadata.get("public_worker_security_assignment_id")),
                assignment_nonce=_optional_metadata_str(metadata.get("public_worker_security_assignment_nonce")),
                peer_class=peer_class,
                job_kind=job_kind,
                trusted_verifier_ids=trusted_production_verifier_ids,
                trusted_verifier_keys=trusted_production_verifier_keys,
                current_time=current_time,
            )
        )

    return list(dict.fromkeys(reason_codes))


def validate_production_worker_security_verification_metadata(
    metadata: dict[str, Any],
    *,
    required_decision_id: str | None = None,
    runtime_id: str | None = None,
    assignment_id: str | None = None,
    assignment_nonce: str | None = None,
    peer_class: PeerClass | None = None,
    job_kind: WorkKind | None = None,
    trusted_verifier_ids: set[str] | None = None,
    trusted_verifier_keys: dict[str, str] | None = None,
    current_time: datetime | None = None,
) -> list[str]:
    if metadata.get("public_worker_security_production_verification_schema_version") is None:
        return ["production_worker_security_verification_missing"]

    reason_codes: list[str] = []
    if (
        metadata.get("public_worker_security_production_verification_schema_version")
        != PUBLIC_WORKER_PRODUCTION_VERIFICATION_SCHEMA_VERSION
    ):
        reason_codes.append("production_worker_security_verification_schema_version_unsupported")
    if metadata.get("public_worker_security_production_status") != "verified":
        reason_codes.append("production_worker_security_verification_not_verified")

    verification_reasons = metadata.get("public_worker_security_production_reason_codes")
    if verification_reasons:
        reason_codes.append("production_worker_security_verification_has_reason_codes")

    if (
        required_decision_id is not None
        and metadata.get("public_worker_security_production_worker_security_decision_id") != required_decision_id
    ):
        reason_codes.append("production_worker_security_decision_binding_mismatch")
    if (
        runtime_id is not None
        and metadata.get("public_worker_security_production_runtime_id") != runtime_id
    ):
        reason_codes.append("production_worker_security_runtime_binding_mismatch")
    if (
        assignment_id is not None
        and metadata.get("public_worker_security_production_assignment_id") != assignment_id
    ):
        reason_codes.append("production_worker_security_assignment_binding_mismatch")
    if not assignment_nonce:
        reason_codes.append("production_worker_security_assignment_nonce_missing")
    elif metadata.get("public_worker_security_production_assignment_nonce") != assignment_nonce:
        reason_codes.append("production_worker_security_assignment_nonce_mismatch")
    if (
        peer_class is not None
        and metadata.get("public_worker_security_production_peer_class") != peer_class
    ):
        reason_codes.append("production_worker_security_peer_class_binding_mismatch")
    if job_kind is not None and metadata.get("public_worker_security_production_job_kind") != job_kind:
        reason_codes.append("production_worker_security_job_kind_binding_mismatch")

    if metadata.get("public_worker_security_production_runtime_attestation_verified") is not True:
        reason_codes.append("production_runtime_attestation_unverified")
    if metadata.get("public_worker_security_production_sandbox_verified") is not True:
        reason_codes.append("production_sandbox_unverified")
    if metadata.get("public_worker_security_production_live_provider_evidence") is not True:
        reason_codes.append("production_worker_security_live_provider_evidence_missing")
    if metadata.get("public_worker_security_production_evidence_source") != "live-provider":
        reason_codes.append("production_worker_security_live_provider_evidence_source_missing")
    if (
        metadata.get("public_worker_security_production_runtime_attestation_evidence_source")
        != "live-provider"
    ):
        reason_codes.append("production_runtime_attestation_live_source_missing")
    if metadata.get("public_worker_security_production_sandbox_evidence_source") != "live-provider":
        reason_codes.append("production_sandbox_live_source_missing")

    verifier_kind = metadata.get("public_worker_security_production_verifier_kind")
    if verifier_kind in {None, "unavailable", "local-contract-fixture"}:
        reason_codes.append("production_worker_security_verifier_not_live")
    verifier_id = metadata.get("public_worker_security_production_verifier_id")
    if trusted_verifier_ids is not None:
        if not trusted_verifier_ids:
            reason_codes.append("production_worker_security_trusted_verifier_missing")
        elif verifier_id not in trusted_verifier_ids:
            reason_codes.append("production_worker_security_verifier_untrusted")
    if not isinstance(metadata.get("public_worker_security_production_evidence_digest"), str):
        reason_codes.append("production_worker_security_evidence_digest_missing")
    reason_codes.extend(
        _validate_production_worker_security_verification_signature_metadata(
            metadata,
            trusted_verifier_keys=trusted_verifier_keys,
        )
    )
    now = _as_utc(current_time or utc_now())
    issued_at = _parse_metadata_datetime(metadata.get("public_worker_security_production_issued_at"))
    expires_at = _parse_metadata_datetime(metadata.get("public_worker_security_production_expires_at"))
    if issued_at is None:
        reason_codes.append("production_worker_security_issued_at_missing")
    elif _production_time_is_after_now(issued_at, now):
        reason_codes.append("production_worker_security_not_yet_valid")
    if expires_at is None:
        reason_codes.append("production_worker_security_expires_at_missing")
    elif expires_at <= now:
        reason_codes.append("production_worker_security_expired")
    return list(dict.fromkeys(reason_codes))


def _worker_evidence_payload_digest(evidence: WorkerSecurityEvidence) -> str:
    return stable_digest(evidence.model_dump(mode="json", exclude={"signature"}))


def _production_time_is_after_now(value: datetime, now: datetime) -> bool:
    return _as_utc(value) > _as_utc(now) + PRODUCTION_VERIFICATION_CLOCK_SKEW_TOLERANCE


def _production_worker_security_verification_metadata_payload_digest(
    verification: ProductionWorkerSecurityVerification,
) -> str:
    return stable_digest(_production_worker_security_verification_metadata_payload(verification))


def _production_worker_security_metadata_payload_digest(metadata: dict[str, Any]) -> str:
    return stable_digest(
        {
            key: metadata.get(key)
            for key in _PRODUCTION_WORKER_SECURITY_VERIFICATION_METADATA_KEYS
        }
    )


def _validate_production_worker_security_verification_signature_metadata(
    metadata: dict[str, Any],
    *,
    trusted_verifier_keys: dict[str, str] | None,
) -> list[str]:
    if trusted_verifier_keys is None:
        return []
    reason_codes: list[str] = []
    key_id = metadata.get("public_worker_security_production_signature_key_id")
    payload_digest = metadata.get("public_worker_security_production_signature_payload_digest")
    signature = metadata.get("public_worker_security_production_signature")
    if (
        metadata.get("public_worker_security_production_signature_schema_version")
        != PUBLIC_WORKER_PRODUCTION_VERIFICATION_SIGNATURE_SCHEMA_VERSION
    ):
        reason_codes.append("production_worker_security_signature_schema_version_unsupported")
    if metadata.get("public_worker_security_production_signature_algorithm") != "hmac-sha256":
        reason_codes.append("production_worker_security_signature_algorithm_unsupported")
    if not isinstance(key_id, str) or not key_id:
        reason_codes.append("production_worker_security_signature_key_missing")
    if not isinstance(payload_digest, str) or not payload_digest:
        reason_codes.append("production_worker_security_signature_payload_digest_missing")
    if not isinstance(signature, str) or not signature:
        reason_codes.append("production_worker_security_signature_missing")
    if reason_codes:
        return reason_codes

    expected_payload_digest = _production_worker_security_metadata_payload_digest(metadata)
    if not hmac.compare_digest(payload_digest, expected_payload_digest):
        reason_codes.append("production_worker_security_signature_payload_digest_mismatch")
    signing_secret = trusted_verifier_keys.get(key_id)
    if signing_secret is None:
        reason_codes.append("production_worker_security_signature_key_untrusted")
    else:
        expected_signature = _hmac_signature(payload_digest, signing_secret)
        if not hmac.compare_digest(signature, expected_signature):
            reason_codes.append("production_worker_security_signature_invalid")
    return reason_codes


def _hmac_signature(payload_digest: str, signing_secret: str) -> str:
    signature = hmac.new(
        signing_secret.encode("utf-8"),
        payload_digest.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"hmac-sha256:{signature}"


def _read_proc_status(proc_root: Path) -> str:
    try:
        return (proc_root / "self" / "status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_proc_namespace_inodes(proc_root: Path) -> dict[str, str]:
    namespace_dir = proc_root / "self" / "ns"
    namespace_inodes: dict[str, str] = {}
    for namespace_name in ("pid", "mnt", "net", "uts", "ipc", "user", "cgroup"):
        namespace_path = namespace_dir / namespace_name
        try:
            namespace_inodes[namespace_name] = os.readlink(namespace_path)
        except OSError:
            continue
    return namespace_inodes


def _linux_proc_status_int(status_text: str, field_name: str) -> int | None:
    prefix = f"{field_name}:"
    for line in status_text.splitlines():
        if not line.startswith(prefix):
            continue
        raw_value = line.split(":", 1)[1].strip().split(maxsplit=1)[0]
        try:
            return int(raw_value)
        except ValueError:
            return None
    return None


def _digest_suffix(payload: Any) -> str:
    return stable_digest(payload).split(":", 1)[1][:16]


def _production_bundle_lookup_key(
    decision: SecurityAdmissionDecision,
    verifier_id: str,
    verifier_kind: ProductionVerifierKind,
) -> tuple[str, str, str, str, str, str, str]:
    return (
        decision.runtime_id,
        decision.assignment_id,
        decision.assignment_nonce or "",
        decision.job_kind,
        decision.decision_id,
        verifier_id,
        verifier_kind,
    )


def _production_runtime_attestation_lookup_key(
    evidence: ProductionRuntimeAttestationEvidence,
) -> tuple[str, str, str, str, str, str, str]:
    return (
        evidence.runtime_id,
        evidence.assignment_id,
        evidence.assignment_nonce,
        evidence.job_kind,
        evidence.worker_security_decision_id,
        evidence.verifier_id,
        evidence.verifier_kind,
    )


def _production_sandbox_lookup_key(
    evidence: ProductionSandboxEvidence,
) -> tuple[str, str, str, str, str, str, str]:
    return (
        evidence.runtime_id,
        evidence.assignment_id,
        evidence.assignment_nonce,
        evidence.job_kind,
        evidence.worker_security_decision_id,
        evidence.verifier_id,
        evidence.verifier_kind,
    )


def _optional_metadata_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_metadata_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _as_utc(parsed)


def _should_quarantine(
    assignment: WorkAssignment,
    less_trusted: bool,
    policy: PublicWorkerSecurityPolicy,
) -> bool:
    if not assignment.quarantine_on_reject:
        return False
    return less_trusted and policy.quarantine_rejected_less_trusted_outputs


def _production_evidence_binding_reason_codes(
    *,
    schema_version: str,
    expected_schema_version: str,
    runtime_id: str,
    assignment_id: str,
    assignment_nonce: str,
    job_kind: WorkKind,
    worker_security_decision_id: str,
    verifier_id: str,
    verifier_kind: ProductionVerifierKind,
    expected_runtime_id: str,
    expected_assignment_id: str,
    expected_assignment_nonce: str,
    expected_job_kind: WorkKind,
    expected_worker_security_decision_id: str,
    expected_verifier_id: str,
    expected_verifier_kind: ProductionVerifierKind,
    evidence_label: str,
) -> list[str]:
    reason_codes: list[str] = []
    if schema_version != expected_schema_version:
        reason_codes.append(f"{evidence_label}_schema_version_unsupported")
    if runtime_id != expected_runtime_id:
        reason_codes.append(f"{evidence_label}_runtime_binding_mismatch")
    if assignment_id != expected_assignment_id:
        reason_codes.append(f"{evidence_label}_assignment_binding_mismatch")
    if assignment_nonce != expected_assignment_nonce:
        reason_codes.append(f"{evidence_label}_assignment_nonce_mismatch")
    if job_kind != expected_job_kind:
        reason_codes.append(f"{evidence_label}_job_kind_mismatch")
    if worker_security_decision_id != expected_worker_security_decision_id:
        reason_codes.append(f"{evidence_label}_decision_binding_mismatch")
    if verifier_id != expected_verifier_id:
        reason_codes.append(f"{evidence_label}_verifier_id_mismatch")
    if verifier_kind != expected_verifier_kind:
        reason_codes.append(f"{evidence_label}_verifier_kind_mismatch")
    return reason_codes


def _decision_claims() -> dict[str, bool]:
    return {
        "local_public_worker_enforcement_evaluated": True,
        "training_inference_shared_contract": True,
        "production_public_worker_security_claimed": False,
        "production_provider_tee_attestation_claimed": False,
        "production_sandbox_enforcement_claimed": False,
        "provider_accounting_claimed": False,
        "crypto_settlement_claimed": False,
        "token_launch_claimed": False,
        "payouts_claimed": False,
        "custody_claimed": False,
        "slashing_claimed": False,
    }


def _production_verification_claims(live_provider_evidence_passed: bool) -> dict[str, bool]:
    return {
        "production_public_worker_security_claimed": bool(live_provider_evidence_passed),
        "production_provider_tee_attestation_claimed": False,
        "production_sandbox_enforcement_claimed": bool(live_provider_evidence_passed),
        "provider_accounting_claimed": False,
        "crypto_settlement_claimed": False,
        "token_launch_claimed": False,
        "payouts_claimed": False,
        "custody_claimed": False,
        "slashing_claimed": False,
    }


def _app_visible_state(
    *,
    assignment: WorkAssignment,
    admitted: bool,
    reason_codes: list[str],
    challenge_required: bool,
    challenge_verified: bool,
    replay_decision: str,
    quarantine_record: QuarantineRecord | None,
) -> dict[str, Any]:
    return {
        "schema_version": "public-worker-security-app-admission-state-v0",
        "assignment_id": assignment.assignment_id,
        "runtime_id": assignment.runtime_id,
        "peer_class": assignment.peer_class,
        "job_kind": assignment.job_kind,
        "admission_status": "accepted" if admitted else "rejected",
        "reason_codes": reason_codes,
        "challenge_required": challenge_required,
        "challenge_verified": challenge_verified,
        "replay_decision": replay_decision,
        "quarantined": quarantine_record is not None,
        "quarantine_id": quarantine_record.quarantine_id if quarantine_record is not None else None,
        "production_public_worker_security_claimed": False,
    }
