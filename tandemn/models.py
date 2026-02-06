"""Data models for the hybrid GPU inference orchestrator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeployRequest:
    """What the user asks for."""

    model_name: str
    gpu: str
    gpu_count: int = 1
    tp_size: int = 1
    max_model_len: int = 4096
    serverless_provider: str = "modal"  # "modal", later "cloudrun", "runpod"
    spots_cloud: str = "aws"
    region: Optional[str] = None
    concurrency: int = 32
    cold_start_mode: str = "fast_boot"  # "fast_boot" or "no_fast_boot"
    scale_to_zero: bool = True
    service_name: Optional[str] = None  # auto-generated if None

    def __post_init__(self):
        if self.service_name is None:
            short_id = uuid.uuid4().hex[:8]
            self.service_name = f"tandemn-{short_id}"


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
class HybridDeployment:
    """Combined result returned to the user."""

    serverless: Optional[DeploymentResult] = None
    spot: Optional[DeploymentResult] = None
    router: Optional[DeploymentResult] = None
    router_url: Optional[str] = None
