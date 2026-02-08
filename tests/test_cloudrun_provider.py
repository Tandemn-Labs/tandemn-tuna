"""Tests for tandemn.providers.cloudrun_provider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tandemn.models import DeployRequest, DeploymentResult
from tandemn.providers.cloudrun_provider import (
    GPU_MAP,
    CloudRunProvider,
    _get_project_id,
)


class TestCloudRunGpuMap:
    def test_all_short_names_resolve(self):
        for short, full in GPU_MAP.items():
            assert isinstance(full, str)
            assert len(full) > 0

    def test_unknown_gpu_raises(self):
        provider = CloudRunProvider()
        request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="UNKNOWN_GPU",
            service_name="test-svc",
        )
        with pytest.raises(ValueError, match="Unknown GPU type for Cloud Run"):
            provider.plan(request, "vllm serve Qwen/Qwen3-0.6B")


class TestGetProjectId:
    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "my-project"})
    def test_from_env_var(self):
        assert _get_project_id() == "my-project"

    @patch.dict("os.environ", {}, clear=True)
    @patch("tandemn.providers.cloudrun_provider.subprocess.run")
    def test_from_gcloud_fallback(self, mock_run):
        mock_run.return_value = MagicMock(stdout="gcloud-project\n", returncode=0)
        assert _get_project_id() == "gcloud-project"

    @patch.dict("os.environ", {}, clear=True)
    @patch("tandemn.providers.cloudrun_provider.subprocess.run")
    def test_raises_when_not_found(self, mock_run):
        mock_run.return_value = MagicMock(stdout="(unset)\n", returncode=0)
        with pytest.raises(RuntimeError, match="Cannot determine Google Cloud project"):
            _get_project_id()


class TestCloudRunPlan:
    def setup_method(self):
        self.provider = CloudRunProvider()
        self.request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="L4",
            gpu_count=1,
            tp_size=1,
            max_model_len=8192,
            service_name="test-svc",
        )
        self.vllm_cmd = "vllm serve Qwen/Qwen3-0.6B --port 8000"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_provider_name(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.provider == "cloudrun"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_rendered_script_empty(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.rendered_script == ""

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_env_vars_from_request(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.env["MODEL_NAME"] == "Qwen/Qwen3-0.6B"
        assert plan.env["MAX_MODEL_LEN"] == "8192"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_fast_boot_sets_enforce_eager(self):
        self.request.cold_start_mode = "fast_boot"
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.env["ENFORCE_EAGER"] == "true"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_no_fast_boot_no_enforce_eager(self):
        self.request.cold_start_mode = "no_fast_boot"
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert "ENFORCE_EAGER" not in plan.env

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_gpu_mapped_in_metadata(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["gpu_accelerator"] == "nvidia-l4"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_service_name_in_metadata(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["service_name"] == "test-svc-serverless"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_image_in_metadata(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["image"] == "vllm/vllm-openai:v0.15.1"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_default_region(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["region"] == "us-central1"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_custom_region_from_request(self):
        self.request.region = "europe-west4"
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["region"] == "europe-west4"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_container_args_contain_model(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        args = json.loads(plan.metadata["container_args"])
        assert "--model" in args
        idx = args.index("--model")
        assert args[idx + 1] == "Qwen/Qwen3-0.6B"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project", "HF_TOKEN": "hf_test123"})
    def test_hf_token_included_when_set(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.env["HF_TOKEN"] == "hf_test123"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project", "HF_TOKEN": ""}, clear=False)
    def test_hf_token_excluded_when_empty(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert "HF_TOKEN" not in plan.env

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_scaling_params_in_metadata(self):
        self.request.scaling.serverless.workers_min = 0
        self.request.scaling.serverless.workers_max = 3
        self.request.scaling.serverless.concurrency = 64
        self.request.scaling.serverless.timeout = 300
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["min_instance_count"] == "0"
        assert plan.metadata["max_instance_count"] == "3"
        assert plan.metadata["max_concurrency"] == "64"
        assert plan.metadata["timeout_seconds"] == "300"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_tp_size_greater_than_1_raises(self):
        self.request.tp_size = 2
        with pytest.raises(ValueError, match="Cloud Run supports only 1 GPU per instance"):
            self.provider.plan(self.request, self.vllm_cmd)

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_gpu_count_greater_than_1_raises(self):
        self.request.gpu_count = 2
        with pytest.raises(ValueError, match="Cloud Run supports only 1 GPU per instance"):
            self.provider.plan(self.request, self.vllm_cmd)

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_tp_size_always_1_in_container_args(self):
        # Even if someone sneaks tp_size=1 through, verify it's hardcoded to 1
        plan = self.provider.plan(self.request, self.vllm_cmd)
        args = json.loads(plan.metadata["container_args"])
        idx = args.index("--tensor-parallel-size")
        assert args[idx + 1] == "1"


class TestCloudRunDeploy:
    def setup_method(self):
        self.provider = CloudRunProvider()
        self.request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="L4",
            service_name="test-svc",
        )
        self.vllm_cmd = "vllm serve Qwen/Qwen3-0.6B"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    @patch("tandemn.providers.cloudrun_provider.CloudRunProvider._set_public_access")
    def test_successful_deploy(self, mock_set_public):
        plan = self.provider.plan(self.request, self.vllm_cmd)

        mock_service = MagicMock()
        mock_service.uri = "https://test-svc-serverless-abc123.a.run.app"
        mock_service.name = "projects/test-project/locations/us-central1/services/test-svc-serverless"

        mock_operation = MagicMock()
        mock_operation.result.return_value = mock_service

        mock_client = MagicMock()
        mock_client.create_service.return_value = mock_operation

        with patch("google.cloud.run_v2.ServicesClient", return_value=mock_client):
            result = self.provider.deploy(plan)

        assert result.error is None
        assert result.endpoint_url == "https://test-svc-serverless-abc123.a.run.app"
        assert result.health_url == "https://test-svc-serverless-abc123.a.run.app/health"
        mock_client.create_service.assert_called_once()
        mock_set_public.assert_called_once()

    def test_missing_sdk_returns_error(self):
        plan = MagicMock()
        plan.metadata = {
            "service_name": "test-svc-serverless",
            "project_id": "test-project",
            "region": "us-central1",
            "image": "vllm/vllm-openai:v0.15.1",
            "container_args": "[]",
            "container_port": "8000",
            "gpu_accelerator": "nvidia-l4",
            "cpu": "8",
            "memory": "32Gi",
            "min_instance_count": "0",
            "max_instance_count": "1",
            "max_concurrency": "32",
            "timeout_seconds": "600",
        }
        plan.env = {}

        # Temporarily remove the real module so the lazy import fails
        import sys as _sys
        real_mod = _sys.modules.pop("google.cloud.run_v2", None)
        real_cloud = _sys.modules.pop("google.cloud", None)
        real_google = _sys.modules.pop("google", None)
        try:
            _sys.modules["google.cloud.run_v2"] = None  # block import
            _sys.modules["google.cloud"] = None
            _sys.modules["google"] = None
            provider = CloudRunProvider()
            result = provider.deploy(plan)
        finally:
            # Restore
            _sys.modules.pop("google.cloud.run_v2", None)
            _sys.modules.pop("google.cloud", None)
            _sys.modules.pop("google", None)
            if real_mod is not None:
                _sys.modules["google.cloud.run_v2"] = real_mod
            if real_cloud is not None:
                _sys.modules["google.cloud"] = real_cloud
            if real_google is not None:
                _sys.modules["google"] = real_google

        assert result.error is not None
        assert "google-cloud-run SDK not installed" in result.error

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_create_service_failure(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)

        mock_client = MagicMock()
        mock_client.create_service.side_effect = Exception("Permission denied")

        with patch("google.cloud.run_v2.ServicesClient", return_value=mock_client):
            result = self.provider.deploy(plan)

        assert result.error is not None
        assert "Service creation failed" in result.error

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    @patch("tandemn.providers.cloudrun_provider.CloudRunProvider._set_public_access")
    def test_already_exists_triggers_update(self, mock_set_public):
        plan = self.provider.plan(self.request, self.vllm_cmd)

        mock_service = MagicMock()
        mock_service.uri = "https://test-svc-serverless-abc123.a.run.app"
        mock_service.name = "projects/test-project/locations/us-central1/services/test-svc-serverless"

        mock_update_op = MagicMock()
        mock_update_op.result.return_value = mock_service

        mock_client = MagicMock()
        mock_client.create_service.side_effect = Exception("409 AlreadyExists")
        mock_client.update_service.return_value = mock_update_op

        with patch("google.cloud.run_v2.ServicesClient", return_value=mock_client):
            result = self.provider.deploy(plan)

        assert result.error is None
        assert result.endpoint_url == "https://test-svc-serverless-abc123.a.run.app"
        mock_client.update_service.assert_called_once()


class TestCloudRunDestroy:
    def setup_method(self):
        self.provider = CloudRunProvider()

    def test_destroy_deletes_service(self):
        result = DeploymentResult(
            provider="cloudrun",
            endpoint_url="https://test-svc-serverless-abc123.a.run.app",
            metadata={
                "service_name": "test-svc-serverless",
                "project_id": "test-project",
                "region": "us-central1",
            },
        )

        mock_client = MagicMock()

        with patch("google.cloud.run_v2.ServicesClient", return_value=mock_client):
            self.provider.destroy(result)

        mock_client.delete_service.assert_called_once_with(
            name="projects/test-project/locations/us-central1/services/test-svc-serverless"
        )

    def test_destroy_missing_metadata(self):
        result = DeploymentResult(
            provider="cloudrun",
            metadata={},
        )

        # Should not crash
        self.provider.destroy(result)


class TestCloudRunStatus:
    def setup_method(self):
        self.provider = CloudRunProvider()

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_status_running(self):
        mock_svc = MagicMock()
        mock_svc.uri = "https://test-svc-serverless-abc123.a.run.app"
        mock_svc.conditions = []

        mock_client = MagicMock()
        mock_client.get_service.return_value = mock_svc

        with patch("google.cloud.run_v2.ServicesClient", return_value=mock_client):
            status = self.provider.status("test-svc")

        assert status["status"] == "running"
        assert status["uri"] == "https://test-svc-serverless-abc123.a.run.app"

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"})
    def test_status_not_found(self):
        mock_client = MagicMock()
        mock_client.get_service.side_effect = Exception("404 NotFound")

        with patch("google.cloud.run_v2.ServicesClient", return_value=mock_client):
            status = self.provider.status("test-svc")

        assert status["status"] == "not found"
