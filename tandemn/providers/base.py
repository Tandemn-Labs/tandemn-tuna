"""Abstract base class for serverless GPU providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tandemn.models import DeployRequest, DeploymentResult, ProviderPlan


class ServerlessProvider(ABC):
    """Base class for serverless GPU providers.

    Each provider implements: plan -> deploy -> destroy.
    The router never touches this — it only knows about URLs.
    """

    @abstractmethod
    def name(self) -> str:
        """Provider identifier: 'modal', 'cloudrun', 'runpod'."""
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

    def health_check(self, result: DeploymentResult) -> bool:
        """HTTP GET to health_url. Override for provider-native status APIs."""
        try:
            import requests

            resp = requests.get(result.health_url, timeout=5)
            return 200 <= resp.status_code < 300
        except Exception:
            return False
