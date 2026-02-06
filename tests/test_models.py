"""Tests for tandemn.models."""

from tandemn.models import DeployRequest, DeploymentResult, HybridDeployment, ProviderPlan
from tandemn.scaling import ScalingPolicy, SpotScaling, ServerlessScaling


class TestDeployRequest:
    def test_auto_generates_service_name(self):
        req = DeployRequest(model_name="Qwen/Qwen3-0.6B", gpu="L40S")
        assert req.service_name is not None
        assert req.service_name.startswith("tandemn-")
        assert len(req.service_name) == len("tandemn-") + 8

    def test_preserves_explicit_service_name(self):
        req = DeployRequest(
            model_name="Qwen/Qwen3-0.6B", gpu="L40S", service_name="my-svc"
        )
        assert req.service_name == "my-svc"

    def test_unique_service_names(self):
        names = set()
        for _ in range(100):
            req = DeployRequest(model_name="m", gpu="g")
            names.add(req.service_name)
        assert len(names) == 100

    def test_defaults(self):
        req = DeployRequest(model_name="m", gpu="g")
        assert req.gpu_count == 1
        assert req.tp_size == 1
        assert req.max_model_len == 4096
        assert req.serverless_provider == "modal"
        assert req.spots_cloud == "aws"
        assert req.cold_start_mode == "fast_boot"
        # Scaling policy defaults
        assert req.scaling.serverless.concurrency == 32
        assert req.scaling.spot.min_replicas == 0
        assert req.scaling.spot.max_replicas == 5


class TestProviderPlan:
    def test_defaults(self):
        plan = ProviderPlan(provider="modal", rendered_script="# script")
        assert plan.env == {}
        assert plan.metadata == {}


class TestDeploymentResult:
    def test_error_result(self):
        r = DeploymentResult(provider="modal", error="boom")
        assert r.endpoint_url is None
        assert r.error == "boom"

    def test_success_result(self):
        r = DeploymentResult(
            provider="modal",
            endpoint_url="https://x.modal.run",
            health_url="https://x.modal.run/health",
        )
        assert r.error is None


class TestHybridDeployment:
    def test_all_none(self):
        h = HybridDeployment()
        assert h.serverless is None
        assert h.spot is None
        assert h.router is None
        assert h.router_url is None
