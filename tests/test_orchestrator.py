"""Tests for tandemn.orchestrator — unit tests with mocked subprocess/HTTP."""

from unittest.mock import MagicMock, patch

from tandemn.models import DeployRequest, DeploymentResult
from tandemn.orchestrator import build_vllm_cmd, destroy_hybrid, push_url_to_router, status_hybrid


class TestBuildVllmCmd:
    def test_contains_model(self):
        req = DeployRequest(model_name="Qwen/Qwen3-0.6B", gpu="L40S")
        cmd = build_vllm_cmd(req)
        assert "Qwen/Qwen3-0.6B" in cmd

    def test_default_port_8001(self):
        req = DeployRequest(model_name="m", gpu="g")
        cmd = build_vllm_cmd(req)
        assert "--port 8001" in cmd

    def test_custom_port(self):
        req = DeployRequest(model_name="m", gpu="g")
        cmd = build_vllm_cmd(req, port="9000")
        assert "--port 9000" in cmd

    def test_fast_boot_uses_enforce_eager(self):
        req = DeployRequest(model_name="m", gpu="g", cold_start_mode="fast_boot")
        cmd = build_vllm_cmd(req)
        assert "--enforce-eager" in cmd

    def test_no_fast_boot_no_enforce_eager(self):
        req = DeployRequest(model_name="m", gpu="g", cold_start_mode="no_fast_boot")
        cmd = build_vllm_cmd(req)
        assert "--enforce-eager" not in cmd

    def test_tp_size(self):
        req = DeployRequest(model_name="m", gpu="g", tp_size=4)
        cmd = build_vllm_cmd(req)
        assert "--tensor-parallel-size 4" in cmd

    def test_max_model_len(self):
        req = DeployRequest(model_name="m", gpu="g", max_model_len=8192)
        cmd = build_vllm_cmd(req)
        assert "--max-model-len 8192" in cmd


class TestPushUrlToRouter:
    @patch("tandemn.orchestrator.requests")
    def test_pushes_serverless_url(self, mock_requests):
        mock_requests.post.return_value = MagicMock(status_code=200)
        result = push_url_to_router(
            "http://router:8080", serverless_url="https://modal.run"
        )
        assert result is True
        mock_requests.post.assert_called_once()
        call_kwargs = mock_requests.post.call_args
        assert call_kwargs[1]["json"]["serverless_url"] == "https://modal.run"

    @patch("tandemn.orchestrator.requests")
    def test_pushes_spot_url(self, mock_requests):
        mock_requests.post.return_value = MagicMock(status_code=200)
        result = push_url_to_router(
            "http://router:8080", spot_url="http://spot:30001"
        )
        assert result is True
        call_kwargs = mock_requests.post.call_args
        assert call_kwargs[1]["json"]["spot_url"] == "http://spot:30001"

    @patch("tandemn.orchestrator.requests")
    def test_empty_payload_returns_true(self, mock_requests):
        result = push_url_to_router("http://router:8080")
        assert result is True
        mock_requests.post.assert_not_called()

    @patch("tandemn.orchestrator.requests")
    def test_handles_connection_error(self, mock_requests):
        import requests
        mock_requests.post.side_effect = requests.ConnectionError("nope")
        result = push_url_to_router(
            "http://router:8080", serverless_url="https://modal.run"
        )
        assert result is False


class TestDestroyHybrid:
    @patch("tandemn.orchestrator.subprocess")
    @patch("tandemn.orchestrator.get_provider")
    def test_calls_provider_destroy_for_spot(self, mock_get_provider, mock_subprocess):
        mock_spot = MagicMock()
        mock_spot.name.return_value = "skyserve"
        mock_modal = MagicMock()
        mock_modal.name.return_value = "modal"
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        destroy_hybrid("my-svc")

        mock_spot.destroy.assert_called_once()
        result_arg = mock_spot.destroy.call_args[0][0]
        assert result_arg.metadata["service_name"] == "my-svc-spot"

    @patch("tandemn.orchestrator.subprocess")
    @patch("tandemn.orchestrator.get_provider")
    def test_calls_provider_destroy_for_serverless(self, mock_get_provider, mock_subprocess):
        mock_spot = MagicMock()
        mock_spot.name.return_value = "skyserve"
        mock_modal = MagicMock()
        mock_modal.name.return_value = "modal"
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        destroy_hybrid("my-svc")

        mock_modal.destroy.assert_called_once()
        result_arg = mock_modal.destroy.call_args[0][0]
        assert result_arg.metadata["app_name"] == "my-svc-serverless"

    @patch("tandemn.orchestrator.subprocess")
    @patch("tandemn.orchestrator.get_provider")
    def test_router_teardown_uses_sky_down(self, mock_get_provider, mock_subprocess):
        mock_spot = MagicMock()
        mock_spot.name.return_value = "skyserve"
        mock_modal = MagicMock()
        mock_modal.name.return_value = "modal"
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        destroy_hybrid("my-svc")

        # Router teardown should still use subprocess (infrastructure, not provider)
        mock_subprocess.run.assert_called()
        first_call = mock_subprocess.run.call_args_list[0]
        assert "sky" in first_call[0][0]
        assert "down" in first_call[0][0]
        assert "my-svc-router" in first_call[0][0]


class TestStatusHybrid:
    @patch("tandemn.orchestrator._get_cluster_ip", return_value=None)
    @patch("tandemn.orchestrator.get_provider")
    def test_calls_provider_status_for_spot(self, mock_get_provider, mock_get_ip):
        mock_spot = MagicMock()
        mock_spot.status.return_value = {"provider": "skyserve", "status": "running"}
        mock_modal = MagicMock()
        mock_modal.status.return_value = {"provider": "modal", "status": "running"}
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        result = status_hybrid("my-svc")

        mock_spot.status.assert_called_once_with("my-svc")
        assert result["spot"]["provider"] == "skyserve"

    @patch("tandemn.orchestrator._get_cluster_ip", return_value=None)
    @patch("tandemn.orchestrator.get_provider")
    def test_calls_provider_status_for_serverless(self, mock_get_provider, mock_get_ip):
        mock_spot = MagicMock()
        mock_spot.status.return_value = {"provider": "skyserve", "status": "running"}
        mock_modal = MagicMock()
        mock_modal.status.return_value = {"provider": "modal", "status": "running"}
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        result = status_hybrid("my-svc")

        mock_modal.status.assert_called_once_with("my-svc")
        assert result["serverless"]["provider"] == "modal"

    @patch("tandemn.orchestrator._get_cluster_ip", return_value=None)
    @patch("tandemn.orchestrator.get_provider")
    def test_serverless_status_included(self, mock_get_provider, mock_get_ip):
        mock_spot = MagicMock()
        mock_spot.status.return_value = {"provider": "skyserve", "raw": "UP"}
        mock_modal = MagicMock()
        mock_modal.status.return_value = {"provider": "modal", "app_name": "my-svc-serverless", "status": "running"}
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        result = status_hybrid("my-svc")

        # Serverless status was missing before — now it's included
        assert result["serverless"] is not None
        assert result["serverless"]["status"] == "running"
