"""Tests for tandemn.orchestrator — unit tests with mocked subprocess/HTTP."""

from unittest.mock import MagicMock, patch

from tandemn.models import DeployRequest, DeploymentResult
from tandemn.orchestrator import build_vllm_cmd, destroy_hybrid, push_url_to_router, status_hybrid
from tandemn.state import DeploymentRecord


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
    def _make_record(self, **kwargs):
        defaults = dict(
            service_name="my-svc",
            serverless_provider_name="modal",
            serverless_metadata={"app_name": "my-svc-serverless"},
            spot_provider_name="skyserve",
            spot_metadata={"service_name": "my-svc-spot"},
        )
        defaults.update(kwargs)
        return DeploymentRecord(**defaults)

    @patch("tandemn.orchestrator.subprocess")
    @patch("tandemn.orchestrator.get_provider")
    def test_calls_provider_destroy_for_spot(self, mock_get_provider, mock_subprocess):
        mock_spot = MagicMock()
        mock_spot.name.return_value = "skyserve"
        mock_modal = MagicMock()
        mock_modal.name.return_value = "modal"
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        record = self._make_record()
        destroy_hybrid("my-svc", record=record)

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

        record = self._make_record()
        destroy_hybrid("my-svc", record=record)

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

        record = self._make_record()
        destroy_hybrid("my-svc", record=record)

        # Router teardown should still use subprocess (infrastructure, not provider)
        mock_subprocess.run.assert_called()
        first_call = mock_subprocess.run.call_args_list[0]
        assert "sky" in first_call[0][0]
        assert "down" in first_call[0][0]
        assert "my-svc-router" in first_call[0][0]

    @patch("tandemn.orchestrator.subprocess")
    @patch("tandemn.orchestrator.get_provider")
    def test_record_provider_names_used(self, mock_get_provider, mock_subprocess):
        """Verify that provider names come from the record, not hardcoded."""
        mock_custom_spot = MagicMock()
        mock_custom_spot.name.return_value = "custom_spot"
        mock_custom_sl = MagicMock()
        mock_custom_sl.name.return_value = "custom_sl"
        mock_get_provider.side_effect = lambda name: {
            "custom_spot": mock_custom_spot,
            "custom_sl": mock_custom_sl,
        }[name]

        record = self._make_record(
            spot_provider_name="custom_spot",
            spot_metadata={"service_name": "my-svc-spot"},
            serverless_provider_name="custom_sl",
            serverless_metadata={"app_name": "my-svc-serverless"},
        )
        destroy_hybrid("my-svc", record=record)

        mock_get_provider.assert_any_call("custom_spot")
        mock_get_provider.assert_any_call("custom_sl")

    @patch("tandemn.orchestrator.subprocess")
    @patch("tandemn.orchestrator.get_provider")
    def test_fallback_without_record(self, mock_get_provider, mock_subprocess):
        """Without a record, falls back to hardcoded provider names."""
        mock_spot = MagicMock()
        mock_spot.name.return_value = "skyserve"
        mock_modal = MagicMock()
        mock_modal.name.return_value = "modal"
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        destroy_hybrid("my-svc")

        mock_get_provider.assert_any_call("skyserve")
        mock_get_provider.assert_any_call("modal")


class TestStatusHybrid:
    def _make_record(self, **kwargs):
        defaults = dict(
            service_name="my-svc",
            serverless_provider_name="modal",
            spot_provider_name="skyserve",
        )
        defaults.update(kwargs)
        return DeploymentRecord(**defaults)

    @patch("tandemn.orchestrator._get_cluster_ip", return_value=None)
    @patch("tandemn.orchestrator.get_provider")
    def test_calls_provider_status_for_spot(self, mock_get_provider, mock_get_ip):
        mock_spot = MagicMock()
        mock_spot.status.return_value = {"provider": "skyserve", "status": "running"}
        mock_modal = MagicMock()
        mock_modal.status.return_value = {"provider": "modal", "status": "running"}
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        record = self._make_record()
        result = status_hybrid("my-svc", record=record)

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

        record = self._make_record()
        result = status_hybrid("my-svc", record=record)

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

        record = self._make_record()
        result = status_hybrid("my-svc", record=record)

        assert result["serverless"] is not None
        assert result["serverless"]["status"] == "running"

    @patch("tandemn.orchestrator._get_cluster_ip", return_value=None)
    @patch("tandemn.orchestrator.get_provider")
    def test_record_provider_names_used(self, mock_get_provider, mock_get_ip):
        """Verify that provider names come from the record, not hardcoded."""
        mock_custom_spot = MagicMock()
        mock_custom_spot.status.return_value = {"provider": "custom_spot", "status": "up"}
        mock_custom_sl = MagicMock()
        mock_custom_sl.status.return_value = {"provider": "custom_sl", "status": "up"}
        mock_get_provider.side_effect = lambda name: {
            "custom_spot": mock_custom_spot,
            "custom_sl": mock_custom_sl,
        }[name]

        record = self._make_record(
            spot_provider_name="custom_spot",
            serverless_provider_name="custom_sl",
        )
        result = status_hybrid("my-svc", record=record)

        mock_get_provider.assert_any_call("custom_spot")
        mock_get_provider.assert_any_call("custom_sl")

    @patch("tandemn.orchestrator._get_cluster_ip", return_value=None)
    @patch("tandemn.orchestrator.get_provider")
    def test_fallback_without_record(self, mock_get_provider, mock_get_ip):
        """Without a record, falls back to hardcoded provider names."""
        mock_spot = MagicMock()
        mock_spot.status.return_value = {"provider": "skyserve", "status": "running"}
        mock_modal = MagicMock()
        mock_modal.status.return_value = {"provider": "modal", "status": "running"}
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        result = status_hybrid("my-svc")

        mock_get_provider.assert_any_call("skyserve")
        mock_get_provider.assert_any_call("modal")
