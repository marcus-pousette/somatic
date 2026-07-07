from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


TrustTier = Literal["owner", "friend", "verified", "unknown"]
RuntimeStatus = Literal["idle", "busy", "offline", "unhealthy"]
RuntimeType = Literal["local", "docker", "native_macos", "linux_gpu", "simulated"]
PrivacyLevel = Literal["public", "friends", "private"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]
ReceiptStatus = Literal["success", "failure"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Provider(BaseModel):
    provider_id: str
    display_name: str
    trust_tier: TrustTier = "unknown"
    credit_balance: float = 0.0


class RuntimeResources(BaseModel):
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_kind: str | None = None
    speed_score: float = Field(default=1.0, ge=0.0)


class RuntimeTrust(BaseModel):
    tier: TrustTier = "unknown"
    privacy_allowed: list[PrivacyLevel] = Field(default_factory=lambda: ["public"])


class RuntimeNetwork(BaseModel):
    latency_ms_to_core: float = Field(default=10.0, ge=0.0)
    bandwidth_mbps_to_core: float = Field(default=100.0, ge=0.0)
    reliability: float = Field(default=1.0, ge=0.0, le=1.0)


class RuntimeNode(BaseModel):
    runtime_id: str
    provider_id: str
    status: RuntimeStatus = "idle"
    runtime_type: RuntimeType = "simulated"
    capabilities: list[str] = Field(default_factory=list)
    resources: RuntimeResources = Field(default_factory=RuntimeResources)
    trust: RuntimeTrust = Field(default_factory=RuntimeTrust)
    network: RuntimeNetwork = Field(default_factory=RuntimeNetwork)
    current_load: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Capability(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: f"job_{uuid4().hex}")
    requester_id: str
    capability: str
    privacy: PrivacyLevel = "public"
    input_payload: dict[str, Any] = Field(default_factory=dict)
    requirements: dict[str, Any] = Field(default_factory=dict)
    max_credits: float = Field(default=1.0, ge=0.0)
    status: JobStatus = "queued"


class Verification(BaseModel):
    method: str = "none"
    passed: bool = True


class Receipt(BaseModel):
    receipt_id: str = Field(default_factory=lambda: f"receipt_{uuid4().hex}")
    job_id: str
    runtime_id: str
    provider_id: str
    status: ReceiptStatus
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    output_hash: str | None = None
    credits_charged: float = 0.0
    verification: Verification = Field(default_factory=Verification)
    signature: str | None = None


class BodyState(BaseModel):
    runtimes: list[RuntimeNode] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    jobs: list[Job] = Field(default_factory=list)
    receipts: list[Receipt] = Field(default_factory=list)
    credits: dict[str, float] = Field(default_factory=dict)
    recent_failures: list[dict[str, Any]] = Field(default_factory=list)
