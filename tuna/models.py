"""Data models for the hybrid GPU inference orchestrator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from tuna.scaling import ScalingPolicy, default_scaling_policy


@dataclass
class DeployRequest:
    """What the user asks for."""

    model_name: str
    gpu: str
    gpu_count: int = 1
    tp_size: int = 1
    max_model_len: int = 4096
    serverless_provider: str = "modal"  # "modal", "runpod", "cloudrun"
    spots_cloud: str = "aws"
    region: Optional[str] = None
    cold_start_mode: str = "fast_boot"  # "fast_boot" or "no_fast_boot"
    scaling: ScalingPolicy = field(default_factory=default_scaling_policy)
    service_name: Optional[str] = None  # auto-generated if None
    public: bool = False  # If True, make endpoints publicly accessible (no auth)
    vllm_version: str = "0.15.1"  # Set by orchestrator to match serverless provider
    serverless_only: bool = False  # skip spot + router

    def __post_init__(self):
        from tuna.catalog import normalize_gpu_name
        try:
            self.gpu = normalize_gpu_name(self.gpu)
        except KeyError:
            pass  # Let provider-level validation handle unknown GPUs
        if self.service_name is None:
            short_id = uuid.uuid4().hex[:8]
            self.service_name = f"tuna-{short_id}"


@dataclass
class ProviderPlan:
    """Rendered deployment artifact, ready to execute."""

    provider: str  # "modal", "skyserve", "router"
    rendered_script: str  # File contents (Python, YAML, etc.)
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Outcome of a single backend deployment."""

    provider: str
    endpoint_url: Optional[str] = None
    health_url: Optional[str] = None
    error: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class PreflightCheck:
    """Result of a single preflight validation step."""

    name: str                       # e.g. "gcloud_installed"
    passed: bool
    message: str                    # Human-readable status
    fix_command: str | None = None  # Exact shell command to fix
    auto_fixed: bool = False        # True if we fixed it automatically


@dataclass
class PreflightResult:
    """Aggregated result of all preflight checks for a provider."""

    provider: str
    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[PreflightCheck]:
        return [c for c in self.checks if not c.passed]


@dataclass
class HybridDeployment:
    """Combined result returned to the user."""

    serverless: Optional[DeploymentResult] = None
    spot: Optional[DeploymentResult] = None
    router: Optional[DeploymentResult] = None
    router_url: Optional[str] = None
