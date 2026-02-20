"""Tests for tuna.providers.cerebrium_provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tuna.catalog import provider_gpu_map
from tuna.models import DeployRequest, DeploymentResult, ProviderPlan
from tuna.providers.cerebrium_provider import (
    CerebriumProvider,
    _get_project_id,
    _GPU_RESOURCES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    return CerebriumProvider()


@pytest.fixture
def request_t4():
    return DeployRequest(
        model_name="Qwen/Qwen3-0.6B",
        gpu="T4",
        service_name="test-svc",
        serverless_provider="cerebrium",
    )


@pytest.fixture
def request_h100():
    return DeployRequest(
        model_name="meta-llama/Llama-3-8B",
        gpu="H100",
        service_name="test-svc",
        serverless_provider="cerebrium",
    )


@pytest.fixture
def vllm_cmd():
    return (
        "vllm serve Qwen/Qwen3-0.6B "
        "--host 0.0.0.0 --port 8001 --max-model-len 4096 "
        "--served-model-name Qwen/Qwen3-0.6B --tensor-parallel-size 1 "
        "--disable-log-requests --uvicorn-log-level info --enforce-eager"
    )


# ---------------------------------------------------------------------------
# Provider basics
# ---------------------------------------------------------------------------

class TestCerebriumProviderBasics:
    def test_name(self, provider):
        assert provider.name() == "cerebrium"

    def test_vllm_version(self, provider):
        assert provider.vllm_version() == "0.15.1"

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test123"})
    def test_auth_token(self, provider):
        assert provider.auth_token() == "ck-test123"

    @patch.dict("os.environ", {}, clear=True)
    def test_auth_token_empty_when_not_set(self, provider):
        assert provider.auth_token() == ""


# ---------------------------------------------------------------------------
# GPU map
# ---------------------------------------------------------------------------

class TestCerebriumGpuMap:
    def test_all_short_names_resolve(self):
        gpu_map = provider_gpu_map("cerebrium")
        for short, full in gpu_map.items():
            assert isinstance(full, str)
            assert len(full) > 0

    def test_t4_maps(self):
        gpu_map = provider_gpu_map("cerebrium")
        assert gpu_map["T4"] == "TURING_T4"

    def test_l4_maps(self):
        gpu_map = provider_gpu_map("cerebrium")
        assert gpu_map["L4"] == "ADA_L4"

    def test_a10_maps(self):
        gpu_map = provider_gpu_map("cerebrium")
        assert gpu_map["A10"] == "AMPERE_A10"

    def test_l40s_maps(self):
        gpu_map = provider_gpu_map("cerebrium")
        assert gpu_map["L40S"] == "ADA_L40"

    def test_a100_40gb_maps(self):
        gpu_map = provider_gpu_map("cerebrium")
        assert gpu_map["A100_40GB"] == "AMPERE_A100_40GB"

    def test_a100_80gb_maps(self):
        gpu_map = provider_gpu_map("cerebrium")
        assert gpu_map["A100_80GB"] == "AMPERE_A100_80GB"

    def test_h100_maps(self):
        gpu_map = provider_gpu_map("cerebrium")
        assert gpu_map["H100"] == "HOPPER_H100"


# ---------------------------------------------------------------------------
# _get_project_id()
# ---------------------------------------------------------------------------

class TestGetProjectId:
    def test_returns_none_when_no_config(self):
        with patch("tuna.providers.cerebrium_provider.Path.home") as mock_home:
            mock_home.return_value = MagicMock()
            mock_home.return_value.__truediv__ = MagicMock(return_value=MagicMock())
            config = mock_home.return_value / ".cerebrium" / "config.yaml"
            config.exists.return_value = False
            # _get_project_id checks Path.home() / ".cerebrium" / "config.yaml"
            # Simplest approach: mock at a higher level
        with patch("pathlib.Path.exists", return_value=False):
            assert _get_project_id() is None


# ---------------------------------------------------------------------------
# plan() tests
# ---------------------------------------------------------------------------

class TestCerebriumPlan:
    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_provider_name(self, mock_pid, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.provider == "cerebrium"

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_service_name(self, mock_pid, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["service_name"] == "test-svc-serverless"

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_default_region(self, mock_pid, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["region"] == "us-east-1"

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_custom_region(self, mock_pid, provider, vllm_cmd):
        req = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="T4",
            service_name="test-svc",
            region="eu-west-1",
        )
        plan = provider.plan(req, vllm_cmd)
        assert plan.metadata["region"] == "eu-west-1"

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_project_id(self, mock_pid, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["project_id"] == "proj-123"

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_gpu_compute_t4(self, mock_pid, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["gpu_compute"] == "TURING_T4"

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_gpu_compute_h100(self, mock_pid, provider, request_h100, vllm_cmd):
        plan = provider.plan(request_h100, vllm_cmd)
        assert plan.metadata["gpu_compute"] == "HOPPER_H100"

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_rendered_toml_contains_service_name(self, mock_pid, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert 'name = "test-svc-serverless"' in plan.rendered_script

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_rendered_toml_contains_gpu(self, mock_pid, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert 'compute = "TURING_T4"' in plan.rendered_script

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_rendered_toml_contains_model(self, mock_pid, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert "Qwen/Qwen3-0.6B" in plan.rendered_script

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_rendered_toml_contains_vllm_version(self, mock_pid, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert "0.15.1" in plan.rendered_script

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_fast_boot_includes_enforce_eager(self, mock_pid, provider, request_t4, vllm_cmd):
        request_t4.cold_start_mode = "fast_boot"
        plan = provider.plan(request_t4, vllm_cmd)
        assert '"--enforce-eager"' in plan.rendered_script

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_no_fast_boot_excludes_enforce_eager(self, mock_pid, provider, request_t4, vllm_cmd):
        request_t4.cold_start_mode = "no_fast_boot"
        plan = provider.plan(request_t4, vllm_cmd)
        assert "--enforce-eager" not in plan.rendered_script

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_scaling_params(self, mock_pid, provider, request_t4, vllm_cmd):
        request_t4.scaling.serverless.workers_min = 0
        request_t4.scaling.serverless.workers_max = 5
        request_t4.scaling.serverless.scaledown_window = 120
        plan = provider.plan(request_t4, vllm_cmd)
        assert "min_replicas = 0" in plan.rendered_script
        assert "max_replicas = 5" in plan.rendered_script
        assert "cooldown = 120" in plan.rendered_script

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_unknown_gpu_raises(self, mock_pid, provider, vllm_cmd):
        req = DeployRequest(model_name="m", gpu="UNKNOWN_GPU", service_name="s")
        with pytest.raises(ValueError, match="Unknown GPU type for Cerebrium"):
            provider.plan(req, vllm_cmd)

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_h100_resources(self, mock_pid, provider, request_h100, vllm_cmd):
        plan = provider.plan(request_h100, vllm_cmd)
        assert "cpu = 12" in plan.rendered_script
        assert "memory = 64" in plan.rendered_script

    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    def test_plan_t4_resources(self, mock_pid, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert "cpu = 4" in plan.rendered_script
        assert "memory = 16" in plan.rendered_script


# ---------------------------------------------------------------------------
# preflight() tests
# ---------------------------------------------------------------------------

class TestCerebriumPreflight:
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key(self, provider, request_t4):
        result = provider.preflight(request_t4)
        assert not result.ok
        assert result.checks[0].name == "api_key"
        assert not result.checks[0].passed

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("shutil.which", return_value=None)
    def test_cli_not_installed(self, mock_which, provider, request_t4):
        result = provider.preflight(request_t4)
        assert not result.ok
        cli_check = [c for c in result.checks if c.name == "cli_installed"][0]
        assert not cli_check.passed
        assert "pip install cerebrium" in cli_check.fix_command

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("shutil.which", return_value="/usr/bin/cerebrium")
    @patch("tuna.providers.cerebrium_provider.subprocess.run")
    def test_not_authenticated(self, mock_run, mock_which, provider, request_t4):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not logged in")
        result = provider.preflight(request_t4)
        assert not result.ok
        auth_check = [c for c in result.checks if c.name == "cli_authenticated"][0]
        assert not auth_check.passed
        assert "cerebrium login" in auth_check.fix_command

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("shutil.which", return_value="/usr/bin/cerebrium")
    @patch("tuna.providers.cerebrium_provider.subprocess.run")
    def test_all_checks_pass(self, mock_run, mock_which, provider, request_t4):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        result = provider.preflight(request_t4)
        assert result.ok
        assert len(result.failed) == 0

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("shutil.which", return_value="/usr/bin/cerebrium")
    @patch("tuna.providers.cerebrium_provider.subprocess.run")
    def test_gpu_not_supported(self, mock_run, mock_which, provider):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        req = DeployRequest(
            model_name="m",
            gpu="UNKNOWN_GPU",
            service_name="s",
            serverless_provider="cerebrium",
        )
        result = provider.preflight(req)
        gpu_check = [c for c in result.checks if c.name == "gpu_supported"][0]
        assert not gpu_check.passed


# ---------------------------------------------------------------------------
# deploy() tests
# ---------------------------------------------------------------------------

class TestCerebriumDeploy:
    def _make_plan(self, **overrides) -> ProviderPlan:
        metadata = {
            "service_name": "test-svc-serverless",
            "region": "us-east-1",
            "project_id": "proj-123",
            "gpu_compute": "TURING_T4",
        }
        metadata.update(overrides)
        return ProviderPlan(
            provider="cerebrium",
            rendered_script='[cerebrium.deployment]\nname = "test-svc-serverless"\n',
            env={},
            metadata=metadata,
        )

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    @patch("tuna.providers.cerebrium_provider.subprocess.run")
    def test_deploy_success(self, mock_run, mock_pid, provider):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Deployed successfully",
            stderr="",
        )
        plan = self._make_plan()
        result = provider.deploy(plan)

        assert result.error is None
        assert result.endpoint_url is not None
        assert "proj-123" in result.endpoint_url
        assert "test-svc-serverless" in result.endpoint_url
        assert result.endpoint_url.endswith("/v1")
        assert result.health_url is not None
        assert result.health_url.endswith("/health")

    @patch.dict("os.environ", {}, clear=True)
    def test_deploy_no_api_key(self, provider):
        plan = self._make_plan()
        result = provider.deploy(plan)
        assert result.error is not None
        assert "CEREBRIUM_API_KEY" in result.error

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    @patch("tuna.providers.cerebrium_provider.subprocess.run")
    def test_deploy_cli_failure(self, mock_run, mock_pid, provider):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: deployment failed",
        )
        plan = self._make_plan()
        result = provider.deploy(plan)
        assert result.error is not None
        assert "cerebrium deploy failed" in result.error

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    @patch("tuna.providers.cerebrium_provider.subprocess.run")
    def test_deploy_timeout(self, mock_run, mock_pid, provider):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="cerebrium deploy", timeout=600)
        plan = self._make_plan()
        result = provider.deploy(plan)
        assert result.error is not None
        assert "timed out" in result.error

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value=None)
    @patch("tuna.providers.cerebrium_provider.subprocess.run")
    def test_deploy_no_project_id(self, mock_run, mock_pid, provider):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        plan = self._make_plan(project_id="")
        result = provider.deploy(plan)
        # Should succeed but endpoint_url will be None
        assert result.endpoint_url is None


# ---------------------------------------------------------------------------
# destroy() tests
# ---------------------------------------------------------------------------

class TestCerebriumDestroy:
    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("requests.delete")
    def test_destroy_rest_api(self, mock_delete, provider):
        mock_delete.return_value = MagicMock(status_code=200)
        result = DeploymentResult(
            provider="cerebrium",
            endpoint_url="https://api.aws.us-east-1.cerebrium.ai/v4/proj-123/test-svc-serverless/v1",
            metadata={
                "service_name": "test-svc-serverless",
                "project_id": "proj-123",
            },
        )
        provider.destroy(result)
        mock_delete.assert_called_once()
        call_url = mock_delete.call_args[0][0]
        assert "proj-123" in call_url
        assert "test-svc-serverless" in call_url

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("requests.delete", side_effect=Exception("connection error"))
    @patch("tuna.providers.cerebrium_provider.subprocess.run")
    def test_destroy_fallback_to_cli(self, mock_run, mock_delete, provider):
        mock_run.return_value = MagicMock(returncode=0)
        result = DeploymentResult(
            provider="cerebrium",
            metadata={
                "service_name": "test-svc-serverless",
                "project_id": "proj-123",
            },
        )
        provider.destroy(result)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "cerebrium" in cmd
        assert "test-svc-serverless" in cmd

    def test_destroy_missing_metadata(self, provider):
        result = DeploymentResult(provider="cerebrium", metadata={})
        # Should not crash
        provider.destroy(result)


# ---------------------------------------------------------------------------
# status() tests
# ---------------------------------------------------------------------------

class TestCerebriumStatus:
    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    @patch("requests.get")
    def test_status_running(self, mock_get, mock_pid, provider):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "running"}),
        )
        status = provider.status("test-svc")
        assert status["status"] == "running"

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    @patch("requests.get")
    def test_status_not_found(self, mock_get, mock_pid, provider):
        mock_get.return_value = MagicMock(status_code=404)
        status = provider.status("test-svc")
        assert status["status"] == "not found"

    @patch.dict("os.environ", {"CEREBRIUM_API_KEY": "ck-test"})
    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value="proj-123")
    @patch("requests.get", side_effect=Exception("timeout"))
    def test_status_error(self, mock_get, mock_pid, provider):
        status = provider.status("test-svc")
        assert status["status"] == "unknown"
        assert "timeout" in status["error"]

    @patch.dict("os.environ", {}, clear=True)
    @patch("tuna.providers.cerebrium_provider._get_project_id", return_value=None)
    def test_status_no_credentials(self, mock_pid, provider):
        status = provider.status("test-svc")
        assert status["status"] == "unknown"


# ---------------------------------------------------------------------------
# GPU resources dict
# ---------------------------------------------------------------------------

class TestGpuResources:
    def test_all_catalog_gpus_have_resources(self):
        gpu_map = provider_gpu_map("cerebrium")
        for short_name, cerebrium_id in gpu_map.items():
            assert cerebrium_id in _GPU_RESOURCES, (
                f"Missing _GPU_RESOURCES entry for {cerebrium_id} ({short_name})"
            )

    def test_resources_have_cpu_and_memory(self):
        for gpu_id, resources in _GPU_RESOURCES.items():
            assert "cpu" in resources
            assert "memory" in resources
            assert resources["cpu"] > 0
            assert resources["memory"] > 0
