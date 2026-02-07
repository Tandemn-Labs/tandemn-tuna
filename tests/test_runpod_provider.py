"""Tests for tandemn.providers.runpod_provider."""

from unittest.mock import MagicMock, patch

import pytest

from tandemn.models import DeployRequest, DeploymentResult
from tandemn.providers.runpod_provider import GPU_MAP, RunPodProvider


class TestRunPodGpuMap:
    def test_all_short_names_resolve(self):
        for short, full in GPU_MAP.items():
            assert isinstance(full, str)
            assert len(full) > 0

    def test_unknown_gpu_raises(self):
        provider = RunPodProvider()
        request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="UNKNOWN_GPU",
            service_name="test-svc",
        )
        with pytest.raises(ValueError, match="Unknown GPU type for RunPod"):
            provider.plan(request, "vllm serve Qwen/Qwen3-0.6B")


class TestRunPodPlan:
    def setup_method(self):
        self.provider = RunPodProvider()
        self.request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="L40S",
            gpu_count=1,
            tp_size=2,
            max_model_len=8192,
            service_name="test-svc",
        )
        self.vllm_cmd = "vllm serve Qwen/Qwen3-0.6B --port 8001"

    def test_env_vars_from_request(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.env["MODEL_NAME"] == "Qwen/Qwen3-0.6B"
        assert plan.env["MAX_MODEL_LEN"] == "8192"
        assert plan.env["TENSOR_PARALLEL_SIZE"] == "2"

    def test_fast_boot_sets_enforce_eager(self):
        self.request.cold_start_mode = "fast_boot"
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.env["ENFORCE_EAGER"] == "true"

    def test_no_fast_boot_no_enforce_eager(self):
        self.request.cold_start_mode = "no_fast_boot"
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert "ENFORCE_EAGER" not in plan.env

    def test_gpu_mapped_in_metadata(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["gpu_type_id"] == "NVIDIA L40S"

    def test_scaling_params_in_metadata(self):
        self.request.scaling.serverless.scaledown_window = 120
        self.request.scaling.serverless.timeout = 300
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["idle_timeout"] == "120"
        assert plan.metadata["execution_timeout_ms"] == "300000"
        assert plan.metadata["workers_min"] == "0"
        assert plan.metadata["workers_max"] == "1"

    def test_provider_name(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.provider == "runpod"

    def test_rendered_script_empty(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.rendered_script == ""

    def test_hf_token_included_when_set(self):
        with patch.dict("os.environ", {"HF_TOKEN": "hf_test123"}):
            plan = self.provider.plan(self.request, self.vllm_cmd)
            assert plan.env["HF_TOKEN"] == "hf_test123"

    def test_hf_token_excluded_when_empty(self):
        with patch.dict("os.environ", {"HF_TOKEN": ""}, clear=False):
            plan = self.provider.plan(self.request, self.vllm_cmd)
            assert "HF_TOKEN" not in plan.env

    def test_concurrency_from_scaling(self):
        self.request.scaling.serverless.concurrency = 64
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.env["MAX_CONCURRENCY"] == "64"

    def test_image_name_in_metadata(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["image_name"] == "runpod/worker-v1-vllm:v2.11.3"


class TestRunPodDeploy:
    def setup_method(self):
        self.provider = RunPodProvider()
        self.request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="L40S",
            service_name="test-svc",
        )
        self.vllm_cmd = "vllm serve Qwen/Qwen3-0.6B"

    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    @patch("tandemn.providers.runpod_provider.requests.post")
    def test_successful_deploy(self, mock_post):
        plan = self.provider.plan(self.request, self.vllm_cmd)

        # Mock template creation response
        template_resp = MagicMock()
        template_resp.json.return_value = {"id": "tpl_abc123"}
        template_resp.raise_for_status = MagicMock()

        # Mock endpoint creation response
        endpoint_resp = MagicMock()
        endpoint_resp.json.return_value = {"id": "ep_xyz789"}
        endpoint_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [template_resp, endpoint_resp]

        result = self.provider.deploy(plan)

        assert result.error is None
        assert result.endpoint_url == "https://api.runpod.ai/v2/ep_xyz789/openai/v1"
        assert result.health_url == "https://api.runpod.ai/v2/ep_xyz789/health"
        assert result.metadata["endpoint_id"] == "ep_xyz789"
        assert result.metadata["template_id"] == "tpl_abc123"
        assert mock_post.call_count == 2

    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    @patch("tandemn.providers.runpod_provider.requests.post")
    def test_template_env_is_flat_dict(self, mock_post):
        """RunPod REST API expects env as a flat dict {KEY: value}."""
        plan = self.provider.plan(self.request, self.vllm_cmd)

        template_resp = MagicMock()
        template_resp.json.return_value = {"id": "tpl_abc123"}
        template_resp.raise_for_status = MagicMock()

        endpoint_resp = MagicMock()
        endpoint_resp.json.return_value = {"id": "ep_xyz789"}
        endpoint_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [template_resp, endpoint_resp]
        self.provider.deploy(plan)

        # First call is template creation
        template_call_kwargs = mock_post.call_args_list[0]
        payload = template_call_kwargs.kwargs["json"]
        env = payload["env"]
        assert isinstance(env, dict)
        assert env["MODEL_NAME"] == "Qwen/Qwen3-0.6B"

    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    @patch("tandemn.providers.runpod_provider.requests.post")
    def test_template_creation_fails(self, mock_post):
        plan = self.provider.plan(self.request, self.vllm_cmd)

        mock_post.side_effect = Exception("API error: 403 Forbidden")

        result = self.provider.deploy(plan)

        assert result.error is not None
        assert "Template creation failed" in result.error
        assert mock_post.call_count == 1

    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    @patch("tandemn.providers.runpod_provider.requests.delete")
    @patch("tandemn.providers.runpod_provider.requests.post")
    def test_endpoint_creation_fails(self, mock_post, mock_delete):
        plan = self.provider.plan(self.request, self.vllm_cmd)

        # Template succeeds
        template_resp = MagicMock()
        template_resp.json.return_value = {"id": "tpl_abc123"}
        template_resp.raise_for_status = MagicMock()

        # Endpoint fails
        mock_post.side_effect = [template_resp, Exception("Endpoint error")]
        mock_delete.return_value = MagicMock()

        result = self.provider.deploy(plan)

        assert result.error is not None
        assert "Endpoint creation failed" in result.error
        assert result.metadata["template_id"] == "tpl_abc123"
        # Verify cleanup was attempted
        mock_delete.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        result = self.provider.deploy(plan)

        assert result.error is not None
        assert "RUNPOD_API_KEY" in result.error


class TestRunPodDestroy:
    def setup_method(self):
        self.provider = RunPodProvider()

    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    @patch("tandemn.providers.runpod_provider.requests.delete")
    def test_destroy_deletes_endpoint_and_template(self, mock_delete):
        mock_delete.return_value = MagicMock()

        result = DeploymentResult(
            provider="runpod",
            endpoint_url="https://api.runpod.ai/v2/ep_xyz/openai/v1",
            metadata={
                "endpoint_id": "ep_xyz",
                "template_id": "tpl_abc",
                "endpoint_name": "test-svc-serverless",
            },
        )

        self.provider.destroy(result)

        assert mock_delete.call_count == 2
        calls = [c.args[0] for c in mock_delete.call_args_list]
        assert "endpoints/ep_xyz" in calls[0]
        assert "templates/tpl_abc" in calls[1]

    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    @patch("tandemn.providers.runpod_provider.requests.delete")
    def test_destroy_missing_metadata(self, mock_delete):
        result = DeploymentResult(
            provider="runpod",
            metadata={},
        )

        # Should not crash
        self.provider.destroy(result)
        mock_delete.assert_not_called()


class TestRunPodStatus:
    def setup_method(self):
        self.provider = RunPodProvider()

    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    @patch("tandemn.providers.runpod_provider.requests.get")
    def test_health_check(self, mock_get):
        # Mock list endpoints response
        list_resp = MagicMock()
        list_resp.json.return_value = [
            {"name": "test-svc-serverless", "id": "ep_xyz"},
        ]
        list_resp.raise_for_status = MagicMock()

        # Mock detailed endpoint response
        detail_resp = MagicMock()
        detail_resp.json.return_value = {
            "id": "ep_xyz",
            "name": "test-svc-serverless",
            "workers": {"running": 2, "idle": 1, "ready": 3},
        }
        detail_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [list_resp, detail_resp]

        status = self.provider.status("test-svc")

        assert status["status"] == "running"
        assert status["endpoint_id"] == "ep_xyz"
        assert status["workers"] == {"running": 2, "idle": 1, "ready": 3}

    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    @patch("tandemn.providers.runpod_provider.requests.get")
    def test_status_endpoint_not_found(self, mock_get):
        list_resp = MagicMock()
        list_resp.json.return_value = []
        list_resp.raise_for_status = MagicMock()

        mock_get.return_value = list_resp

        status = self.provider.status("test-svc")

        assert status["status"] == "not found"
