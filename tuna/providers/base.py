"""Abstract base class for inference providers (serverless and spot)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tuna.models import DeployRequest, DeploymentResult, PreflightResult, ProviderPlan


class InferenceProvider(ABC):
    """Base class for inference providers (serverless and spot).

    Each provider implements: plan -> deploy -> destroy.
    The router never touches this — it only knows about URLs.
    """

    @abstractmethod
    def name(self) -> str:
        """Provider identifier: 'modal', 'skyserve', 'cloudrun', 'runpod'."""
        ...

    @abstractmethod
    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        """Render the deployment artifact from the shared vllm_cmd and request.

        Does NOT deploy anything. Pure function.
        """
        ...

    @abstractmethod
    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        """Execute the plan. Returns endpoint URL on success, error on failure."""
        ...

    @abstractmethod
    def destroy(self, result: DeploymentResult) -> None:
        """Tear down the deployment."""
        ...

    def preflight(self, request: DeployRequest) -> PreflightResult:
        """Validate environment before plan/deploy. Override per provider."""
        return PreflightResult(provider=self.name())

    def status(self, service_name: str) -> dict:
        """Check deployment status. Override for provider-native status APIs."""
        return {"provider": self.name(), "status": "unknown"}

    def vllm_version(self) -> str:
        """Return the vLLM version this provider uses. Override per provider."""
        return "0.15.1"

    def auth_token(self) -> str:
        """Return the auth token the router should use when proxying to this backend.

        Empty string means no auth needed. Override per provider.
        """
        return ""

    def health_check(self, result: DeploymentResult) -> bool:
        """HTTP GET to health_url. Override for provider-native status APIs."""
        try:
            import requests

            resp = requests.get(result.health_url, timeout=5)
            return 200 <= resp.status_code < 300
        except Exception:
            return False
