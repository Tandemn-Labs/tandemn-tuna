"""Tests for tandemn.orchestrator — unit tests with mocked subprocess/HTTP."""

from unittest.mock import MagicMock, patch

from tandemn.models import DeployRequest, DeploymentResult
from tandemn.orchestrator import build_vllm_cmd, push_url_to_router


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
