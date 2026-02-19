"""Tests for tuna.providers.azure_provider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from tuna.catalog import provider_gpu_map
from tuna.models import DeployRequest, DeploymentResult, ProviderPlan
from tuna.providers.azure_provider import (
    AzureProvider,
    _resolve_subscription_id,
    _resolve_resource_group,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    return AzureProvider()


@pytest.fixture
def request_t4():
    return DeployRequest(
        model_name="Qwen/Qwen3-0.6B",
        gpu="T4",
        service_name="test-svc",
        serverless_provider="azure",
    )


@pytest.fixture
def request_a100():
    return DeployRequest(
        model_name="meta-llama/Llama-3-8B",
        gpu="A100_80GB",
        service_name="test-svc",
        serverless_provider="azure",
    )


@pytest.fixture
def vllm_cmd():
    return "vllm serve Qwen/Qwen3-0.6B --port 8000"


# ---------------------------------------------------------------------------
# Provider basics
# ---------------------------------------------------------------------------

class TestAzureProviderBasics:
    def test_name(self, provider):
        assert provider.name() == "azure"

    def test_vllm_version(self, provider):
        assert provider.vllm_version() == "0.15.1"

    def test_auth_token_is_empty(self, provider):
        assert provider.auth_token() == ""


# ---------------------------------------------------------------------------
# GPU map
# ---------------------------------------------------------------------------

class TestAzureGpuMap:
    def test_all_short_names_resolve(self):
        gpu_map = provider_gpu_map("azure")
        for short, full in gpu_map.items():
            assert isinstance(full, str)
            assert len(full) > 0

    def test_t4_maps_to_profile(self):
        gpu_map = provider_gpu_map("azure")
        assert gpu_map["T4"] == "Consumption-GPU-NC8as-T4"

    def test_a100_maps_to_profile(self):
        gpu_map = provider_gpu_map("azure")
        assert gpu_map["A100_80GB"] == "Consumption-GPU-NC24-A100"


# ---------------------------------------------------------------------------
# _resolve_subscription_id() / _resolve_resource_group()
# ---------------------------------------------------------------------------

class TestResolveSubscriptionId:
    @patch.dict("os.environ", {"AZURE_SUBSCRIPTION_ID": "sub-from-env"})
    def test_from_env_var(self):
        assert _resolve_subscription_id() == "sub-from-env"

    @patch.dict("os.environ", {}, clear=True)
    @patch("tuna.providers.azure_provider.subprocess.run")
    def test_from_az_cli_fallback(self, mock_run):
        mock_run.return_value = MagicMock(stdout="sub-from-cli\n", returncode=0)
        assert _resolve_subscription_id() == "sub-from-cli"

    @patch.dict("os.environ", {}, clear=True)
    @patch("tuna.providers.azure_provider.subprocess.run")
    def test_returns_none_when_not_found(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=1)
        assert _resolve_subscription_id() is None


class TestResolveResourceGroup:
    @patch.dict("os.environ", {"AZURE_RESOURCE_GROUP": "rg-from-env"})
    def test_from_env_var(self):
        assert _resolve_resource_group() == "rg-from-env"

    @patch.dict("os.environ", {}, clear=True)
    @patch("tuna.providers.azure_provider.subprocess.run")
    def test_from_az_cli_fallback(self, mock_run):
        mock_run.return_value = MagicMock(stdout="rg-from-cli\n", returncode=0)
        assert _resolve_resource_group() == "rg-from-cli"

    @patch.dict("os.environ", {}, clear=True)
    @patch("tuna.providers.azure_provider.subprocess.run")
    def test_returns_none_when_not_found(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=1)
        assert _resolve_resource_group() is None


# ---------------------------------------------------------------------------
# plan() tests
# ---------------------------------------------------------------------------

class TestAzurePlan:
    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_provider_name(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.provider == "azure"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_service_name(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["service_name"] == "test-svc-serverless"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_env_name(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["env_name"] == "test-svc-env"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_rendered_script_empty(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.rendered_script == ""

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_gpu_profile_t4(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["gpu_profile"] == "Consumption-GPU-NC8as-T4"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_gpu_profile_a100(self, provider, request_a100, vllm_cmd):
        plan = provider.plan(request_a100, vllm_cmd)
        assert plan.metadata["gpu_profile"] == "Consumption-GPU-NC24-A100"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_default_region(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["region"] == "eastus"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
        "AZURE_REGION": "westus2",
    })
    def test_plan_custom_region(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["region"] == "westus2"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_env_vars(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.env["MODEL_NAME"] == "Qwen/Qwen3-0.6B"
        assert plan.env["MAX_MODEL_LEN"] == "4096"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_fast_boot_sets_enforce_eager(self, provider, request_t4, vllm_cmd):
        request_t4.cold_start_mode = "fast_boot"
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.env["ENFORCE_EAGER"] == "true"
        args = json.loads(plan.metadata["container_args"])
        assert "--enforce-eager" in args

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_no_fast_boot_no_enforce_eager(self, provider, request_t4, vllm_cmd):
        request_t4.cold_start_mode = "no_fast_boot"
        plan = provider.plan(request_t4, vllm_cmd)
        assert "ENFORCE_EAGER" not in plan.env

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
        "HF_TOKEN": "hf_test123",
    })
    def test_plan_hf_token_included(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.env["HF_TOKEN"] == "hf_test123"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
        "HF_TOKEN": "",
    }, clear=False)
    def test_plan_hf_token_excluded_when_empty(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert "HF_TOKEN" not in plan.env

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_container_args_contain_model(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        args = json.loads(plan.metadata["container_args"])
        assert "--model" in args
        idx = args.index("--model")
        assert args[idx + 1] == "Qwen/Qwen3-0.6B"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_scaling_params(self, provider, request_t4, vllm_cmd):
        request_t4.scaling.serverless.workers_min = 0
        request_t4.scaling.serverless.workers_max = 3
        request_t4.scaling.serverless.concurrency = 64
        request_t4.scaling.serverless.timeout = 300
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["min_replicas"] == "0"
        assert plan.metadata["max_replicas"] == "3"
        assert plan.metadata["concurrency"] == "64"
        assert plan.metadata["timeout"] == "300"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_unknown_gpu_raises(self, provider, vllm_cmd):
        req = DeployRequest(model_name="m", gpu="UNKNOWN_GPU", service_name="s")
        with pytest.raises(ValueError, match="Unknown GPU type for Azure"):
            provider.plan(req, vllm_cmd)

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_tp_size_greater_than_1_raises(self, provider, request_t4, vllm_cmd):
        request_t4.tp_size = 2
        with pytest.raises(ValueError, match="Azure Container Apps supports only 1 GPU"):
            provider.plan(request_t4, vllm_cmd)

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_gpu_count_greater_than_1_raises(self, provider, request_t4, vllm_cmd):
        request_t4.gpu_count = 2
        with pytest.raises(ValueError, match="Azure Container Apps supports only 1 GPU"):
            provider.plan(request_t4, vllm_cmd)

    @patch.dict("os.environ", {}, clear=True)
    @patch("tuna.providers.azure_provider.subprocess.run")
    def test_plan_no_subscription_raises(self, mock_run, provider, request_t4, vllm_cmd):
        mock_run.return_value = MagicMock(stdout="", returncode=1)
        with pytest.raises(RuntimeError, match="Cannot determine Azure subscription"):
            provider.plan(request_t4, vllm_cmd)

    @patch.dict("os.environ", {"AZURE_SUBSCRIPTION_ID": "test-sub"}, clear=True)
    @patch("tuna.providers.azure_provider.subprocess.run")
    def test_plan_no_resource_group_raises(self, mock_run, provider, request_t4, vllm_cmd):
        mock_run.return_value = MagicMock(stdout="", returncode=1)
        with pytest.raises(RuntimeError, match="Cannot determine Azure resource group"):
            provider.plan(request_t4, vllm_cmd)

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_t4_cpu_memory(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["cpu"] == "8"
        assert plan.metadata["memory"] == "56Gi"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_a100_cpu_memory(self, provider, request_a100, vllm_cmd):
        plan = provider.plan(request_a100, vllm_cmd)
        assert plan.metadata["cpu"] == "24"
        assert plan.metadata["memory"] == "220Gi"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_public_flag(self, provider, request_t4, vllm_cmd):
        request_t4.public = True
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["public_access"] == "true"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_plan_private_by_default(self, provider, request_t4, vllm_cmd):
        plan = provider.plan(request_t4, vllm_cmd)
        assert plan.metadata["public_access"] == "false"


# ---------------------------------------------------------------------------
# _find_existing_environment() tests
# ---------------------------------------------------------------------------

class TestFindExistingEnvironment:
    @patch.dict("os.environ", {"AZURE_ENVIRONMENT": "my-env"})
    def test_explicit_env_var_returns_immediately(self, provider):
        result = provider._find_existing_environment("sub", "rg", "Consumption-GPU-NC8as-T4")
        assert result == "my-env"

    @patch.dict("os.environ", {}, clear=True)
    def test_finds_matching_gpu_profile(self, provider):
        mock_wp = MagicMock()
        mock_wp.workload_profile_type = "Consumption-GPU-NC8as-T4"

        mock_env = MagicMock()
        mock_env.name = "found-env"
        mock_env.workload_profiles = [mock_wp]

        mock_client = MagicMock()
        mock_client.managed_environments.list_by_resource_group.return_value = [mock_env]

        with patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk:
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            result = provider._find_existing_environment("sub", "rg", "Consumption-GPU-NC8as-T4")

        assert result == "found-env"

    @patch.dict("os.environ", {}, clear=True)
    def test_returns_none_when_no_match(self, provider):
        mock_wp = MagicMock()
        mock_wp.workload_profile_type = "Consumption"  # Not a GPU profile

        mock_env = MagicMock()
        mock_env.name = "cpu-env"
        mock_env.workload_profiles = [mock_wp]

        mock_client = MagicMock()
        mock_client.managed_environments.list_by_resource_group.return_value = [mock_env]

        with patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk:
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            result = provider._find_existing_environment("sub", "rg", "Consumption-GPU-NC8as-T4")

        assert result is None

    @patch.dict("os.environ", {}, clear=True)
    def test_returns_none_when_empty_list(self, provider):
        mock_client = MagicMock()
        mock_client.managed_environments.list_by_resource_group.return_value = []

        with patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk:
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            result = provider._find_existing_environment("sub", "rg", "Consumption-GPU-NC8as-T4")

        assert result is None

    @patch.dict("os.environ", {}, clear=True)
    def test_returns_none_on_sdk_import_error(self, provider):
        with patch("tuna.providers.azure_provider._require_azure_sdk", side_effect=ImportError):
            result = provider._find_existing_environment("sub", "rg", "Consumption-GPU-NC8as-T4")
        assert result is None

    @patch.dict("os.environ", {}, clear=True)
    def test_returns_none_on_api_error(self, provider):
        mock_client = MagicMock()
        mock_client.managed_environments.list_by_resource_group.side_effect = Exception("API error")

        with patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk:
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            result = provider._find_existing_environment("sub", "rg", "Consumption-GPU-NC8as-T4")

        assert result is None


# ---------------------------------------------------------------------------
# deploy() tests
# ---------------------------------------------------------------------------

class TestAzureDeploy:
    def _make_plan(self, **overrides) -> ProviderPlan:
        metadata = {
            "service_name": "test-svc-serverless",
            "env_name": "test-svc-env",
            "subscription_id": "test-sub",
            "resource_group": "test-rg",
            "region": "eastus",
            "image": "vllm/vllm-openai:v0.15.1",
            "gpu_profile": "Consumption-GPU-NC8as-T4",
            "container_port": "8000",
            "container_args": json.dumps(["--model", "Qwen/Qwen3-0.6B"]),
            "min_replicas": "0",
            "max_replicas": "1",
            "concurrency": "32",
            "timeout": "600",
            "cpu": "8",
            "memory": "56Gi",
            "public_access": "false",
        }
        metadata.update(overrides)
        return ProviderPlan(
            provider="azure",
            rendered_script="",
            env={"MODEL_NAME": "Qwen/Qwen3-0.6B"},
            metadata=metadata,
        )

    def test_deploy_reuses_existing_environment(self, provider):
        plan = self._make_plan()

        mock_app = MagicMock()
        mock_app.configuration.ingress.fqdn = "test-svc-serverless.eastus.azurecontainerapps.io"

        mock_poller = MagicMock()
        mock_poller.result.return_value = mock_app

        mock_client = MagicMock()
        mock_client.container_apps.begin_create_or_update.return_value = mock_poller

        with (
            patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk,
            patch.object(provider, "_find_existing_environment", return_value="existing-env"),
        ):
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            result = provider.deploy(plan)

        assert result.error is None
        assert result.endpoint_url == "https://test-svc-serverless.eastus.azurecontainerapps.io"
        assert result.health_url == "https://test-svc-serverless.eastus.azurecontainerapps.io/health"
        assert result.metadata["env_name"] == "existing-env"
        # Should NOT have called begin_create_or_update on environments
        mock_client.managed_environments.begin_create_or_update.assert_not_called()

    def test_deploy_creates_environment_when_none_exists(self, provider):
        plan = self._make_plan()

        mock_app = MagicMock()
        mock_app.configuration.ingress.fqdn = "test-svc-serverless.eastus.azurecontainerapps.io"

        mock_env_poller = MagicMock()
        mock_env_poller.result.return_value = MagicMock()

        mock_app_poller = MagicMock()
        mock_app_poller.result.return_value = mock_app

        mock_client = MagicMock()
        mock_client.managed_environments.begin_create_or_update.return_value = mock_env_poller
        mock_client.container_apps.begin_create_or_update.return_value = mock_app_poller

        with (
            patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk,
            patch.object(provider, "_find_existing_environment", return_value=None),
        ):
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            result = provider.deploy(plan)

        assert result.error is None
        assert result.endpoint_url == "https://test-svc-serverless.eastus.azurecontainerapps.io"
        # Should have created the environment
        mock_client.managed_environments.begin_create_or_update.assert_called_once()

    def test_deploy_environment_creation_failure(self, provider):
        plan = self._make_plan()

        mock_poller = MagicMock()
        mock_poller.result.side_effect = Exception("Quota exceeded")

        mock_client = MagicMock()
        mock_client.managed_environments.begin_create_or_update.return_value = mock_poller

        with (
            patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk,
            patch.object(provider, "_find_existing_environment", return_value=None),
        ):
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            result = provider.deploy(plan)

        assert result.error is not None
        assert "Environment creation failed" in result.error

    def test_deploy_container_app_creation_failure(self, provider):
        plan = self._make_plan()

        mock_app_poller = MagicMock()
        mock_app_poller.result.side_effect = Exception("Permission denied")

        mock_client = MagicMock()
        mock_client.container_apps.begin_create_or_update.return_value = mock_app_poller

        with (
            patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk,
            patch.object(provider, "_find_existing_environment", return_value="existing-env"),
        ):
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            result = provider.deploy(plan)

        assert result.error is not None
        assert "Container app creation failed" in result.error

    def test_deploy_missing_sdk_returns_error(self, provider):
        plan = self._make_plan()

        with patch("tuna.providers.azure_provider._require_azure_sdk", side_effect=ImportError):
            result = provider.deploy(plan)

        assert result.error is not None
        assert "Azure SDK not installed" in result.error


# ---------------------------------------------------------------------------
# destroy() tests
# ---------------------------------------------------------------------------

class TestAzureDestroy:
    def test_destroy_only_deletes_container_app(self, provider):
        result = DeploymentResult(
            provider="azure",
            endpoint_url="https://test-svc-serverless.eastus.azurecontainerapps.io",
            metadata={
                "service_name": "test-svc-serverless",
                "resource_group": "test-rg",
                "subscription_id": "test-sub",
                "env_name": "test-svc-env",
            },
        )

        mock_poller = MagicMock()
        mock_poller.result.return_value = None

        mock_client = MagicMock()
        mock_client.container_apps.begin_delete.return_value = mock_poller

        with patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk:
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            provider.destroy(result)

        # Container App deleted
        mock_client.container_apps.begin_delete.assert_called_once_with("test-rg", "test-svc-serverless")
        # Environment NOT deleted (preserved for reuse)
        mock_client.managed_environments.begin_delete.assert_not_called()

    def test_destroy_missing_metadata(self, provider):
        result = DeploymentResult(provider="azure", metadata={})
        # Should not crash
        provider.destroy(result)

    def test_destroy_missing_sdk(self, provider):
        result = DeploymentResult(
            provider="azure",
            metadata={
                "service_name": "test-svc-serverless",
                "resource_group": "test-rg",
                "subscription_id": "test-sub",
            },
        )

        with patch("tuna.providers.azure_provider._require_azure_sdk", side_effect=ImportError):
            # Should not crash
            provider.destroy(result)


# ---------------------------------------------------------------------------
# destroy_environment() tests
# ---------------------------------------------------------------------------

class TestAzureDestroyEnvironment:
    def test_destroy_environment_deletes_env(self, provider):
        result = DeploymentResult(
            provider="azure",
            metadata={
                "service_name": "test-svc-serverless",
                "resource_group": "test-rg",
                "subscription_id": "test-sub",
                "env_name": "test-svc-env",
            },
        )

        mock_poller = MagicMock()
        mock_poller.result.return_value = None

        mock_client = MagicMock()
        mock_client.managed_environments.begin_delete.return_value = mock_poller

        with patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk:
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            provider.destroy_environment(result)

        mock_client.managed_environments.begin_delete.assert_called_once_with("test-rg", "test-svc-env")

    def test_destroy_environment_missing_metadata(self, provider):
        result = DeploymentResult(provider="azure", metadata={})
        # Should not crash
        provider.destroy_environment(result)


# ---------------------------------------------------------------------------
# preflight() tests
# ---------------------------------------------------------------------------

class TestAzurePreflight:
    def _mock_az_success(self, mock_run, responses=None):
        """Configure mock_run to return success for common az commands."""
        if responses is None:
            responses = {}

        def side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            if "az version" in cmd_str:
                return MagicMock(returncode=0, stdout="2.66.0\tcore\n")
            if "account show" in cmd_str and "user.name" in cmd_str:
                return MagicMock(returncode=0, stdout="user@test.com\n")
            if "account show" in cmd_str and "id" in cmd_str:
                return MagicMock(returncode=0, stdout="test-sub-id\n")
            if "config get defaults.group" in cmd_str:
                return MagicMock(returncode=0, stdout="test-rg\n")
            if "provider show" in cmd_str and "Microsoft.App" in cmd_str:
                return MagicMock(returncode=0, stdout="Registered\n")
            if "provider show" in cmd_str and "Microsoft.OperationalInsights" in cmd_str:
                return MagicMock(returncode=0, stdout="Registered\n")
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect

    @patch("tuna.providers.azure_provider.subprocess.run")
    def test_az_cli_not_installed(self, mock_run, provider, request_t4):
        mock_run.side_effect = FileNotFoundError
        result = provider.preflight(request_t4)
        assert not result.ok
        assert result.checks[0].name == "az_installed"
        assert not result.checks[0].passed

    @patch("tuna.providers.azure_provider.subprocess.run")
    def test_not_logged_in(self, mock_run, provider, request_t4):
        def side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            if "az version" in cmd_str:
                return MagicMock(returncode=0, stdout="2.66.0\n")
            return MagicMock(returncode=1, stdout="")

        mock_run.side_effect = side_effect
        result = provider.preflight(request_t4)
        assert not result.ok
        login_check = [c for c in result.checks if c.name == "logged_in"][0]
        assert not login_check.passed
        assert "az login" in login_check.fix_command

    @patch("tuna.providers.azure_provider.subprocess.run")
    @patch.dict("os.environ", {"AZURE_SUBSCRIPTION_ID": "test-sub", "AZURE_RESOURCE_GROUP": "test-rg"})
    def test_all_checks_pass(self, mock_run, provider, request_t4):
        self._mock_az_success(mock_run)

        with (
            patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk,
            patch.object(provider, "_find_existing_environment", return_value="my-env"),
        ):
            mock_sdk.return_value = (MagicMock(), MagicMock())
            result = provider.preflight(request_t4)

        assert result.ok
        assert len(result.failed) == 0

    @patch("tuna.providers.azure_provider.subprocess.run")
    @patch.dict("os.environ", {}, clear=True)
    def test_no_subscription(self, mock_run, provider, request_t4):
        def side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            if "az version" in cmd_str:
                return MagicMock(returncode=0, stdout="2.66.0\n")
            if "user.name" in cmd_str:
                return MagicMock(returncode=0, stdout="user@test.com\n")
            return MagicMock(returncode=1, stdout="")

        mock_run.side_effect = side_effect
        result = provider.preflight(request_t4)
        assert not result.ok
        sub_check = [c for c in result.checks if c.name == "subscription"][0]
        assert not sub_check.passed

    @patch("tuna.providers.azure_provider.subprocess.run")
    @patch.dict("os.environ", {"AZURE_SUBSCRIPTION_ID": "test-sub"}, clear=True)
    def test_no_resource_group(self, mock_run, provider, request_t4):
        def side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            if "az version" in cmd_str:
                return MagicMock(returncode=0, stdout="2.66.0\n")
            if "user.name" in cmd_str:
                return MagicMock(returncode=0, stdout="user@test.com\n")
            return MagicMock(returncode=1, stdout="")

        mock_run.side_effect = side_effect
        result = provider.preflight(request_t4)
        assert not result.ok
        rg_check = [c for c in result.checks if c.name == "resource_group"][0]
        assert not rg_check.passed

    @patch("tuna.providers.azure_provider.subprocess.run")
    @patch.dict("os.environ", {"AZURE_SUBSCRIPTION_ID": "test-sub", "AZURE_RESOURCE_GROUP": "test-rg"})
    def test_resource_provider_not_registered(self, mock_run, provider, request_t4):
        def side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            if "az version" in cmd_str:
                return MagicMock(returncode=0, stdout="2.66.0\n")
            if "user.name" in cmd_str:
                return MagicMock(returncode=0, stdout="user@test.com\n")
            if "provider show" in cmd_str:
                return MagicMock(returncode=0, stdout="NotRegistered\n")
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect
        result = provider.preflight(request_t4)
        assert not result.ok
        rp_check = [c for c in result.checks if c.name == "resource_provider_app"][0]
        assert not rp_check.passed
        assert "az provider register" in rp_check.fix_command

    @patch("tuna.providers.azure_provider.subprocess.run")
    @patch.dict("os.environ", {"AZURE_SUBSCRIPTION_ID": "test-sub", "AZURE_RESOURCE_GROUP": "test-rg"})
    def test_sdk_not_installed(self, mock_run, provider, request_t4):
        self._mock_az_success(mock_run)

        with patch("tuna.providers.azure_provider._require_azure_sdk", side_effect=ImportError):
            result = provider.preflight(request_t4)

        sdk_check = [c for c in result.checks if c.name == "azure_sdk"][0]
        assert not sdk_check.passed
        assert "pip install" in sdk_check.fix_command

    @patch("tuna.providers.azure_provider.subprocess.run")
    @patch.dict("os.environ", {"AZURE_SUBSCRIPTION_ID": "test-sub", "AZURE_RESOURCE_GROUP": "test-rg"})
    def test_preflight_environment_found(self, mock_run, provider, request_t4):
        self._mock_az_success(mock_run)

        with (
            patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk,
            patch.object(provider, "_find_existing_environment", return_value="my-env"),
        ):
            mock_sdk.return_value = (MagicMock(), MagicMock())
            result = provider.preflight(request_t4)

        env_check = [c for c in result.checks if c.name == "environment"][0]
        assert env_check.passed
        assert "my-env" in env_check.message

    @patch("tuna.providers.azure_provider.subprocess.run")
    @patch.dict("os.environ", {"AZURE_SUBSCRIPTION_ID": "test-sub", "AZURE_RESOURCE_GROUP": "test-rg"})
    def test_preflight_environment_not_found(self, mock_run, provider, request_t4):
        self._mock_az_success(mock_run)

        with (
            patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk,
            patch.object(provider, "_find_existing_environment", return_value=None),
        ):
            mock_sdk.return_value = (MagicMock(), MagicMock())
            result = provider.preflight(request_t4)

        env_check = [c for c in result.checks if c.name == "environment"][0]
        assert env_check.passed  # Still passes, just a warning
        assert "30+ min" in env_check.message


# ---------------------------------------------------------------------------
# status() tests
# ---------------------------------------------------------------------------

class TestAzureStatus:
    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_status_running(self, provider):
        mock_app = MagicMock()
        mock_app.provisioning_state = "Succeeded"
        mock_app.configuration.ingress.fqdn = "test-svc-serverless.eastus.azurecontainerapps.io"

        mock_client = MagicMock()
        mock_client.container_apps.get.return_value = mock_app

        with patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk:
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            status = provider.status("test-svc")

        assert status["status"] == "Succeeded"
        assert status["uri"] == "https://test-svc-serverless.eastus.azurecontainerapps.io"

    @patch.dict("os.environ", {
        "AZURE_SUBSCRIPTION_ID": "test-sub",
        "AZURE_RESOURCE_GROUP": "test-rg",
    })
    def test_status_not_found(self, provider):
        mock_client = MagicMock()
        mock_client.container_apps.get.side_effect = Exception("ResourceNotFound")

        with patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk:
            mock_sdk.return_value = (MagicMock(), MagicMock(return_value=mock_client))
            status = provider.status("test-svc")

        assert status["status"] == "not found"

    def test_status_missing_sdk(self, provider):
        with patch("tuna.providers.azure_provider._require_azure_sdk", side_effect=ImportError):
            status = provider.status("test-svc")
        assert status["status"] == "unknown"
        assert "Azure SDK not installed" in status["error"]

    @patch.dict("os.environ", {}, clear=True)
    @patch("tuna.providers.azure_provider.subprocess.run")
    def test_status_no_subscription(self, mock_run, provider):
        mock_run.return_value = MagicMock(stdout="", returncode=1)
        with patch("tuna.providers.azure_provider._require_azure_sdk") as mock_sdk:
            mock_sdk.return_value = (MagicMock(), MagicMock())
            status = provider.status("test-svc")
        assert status["status"] == "unknown"
        assert "No subscription" in status["error"]


# ---------------------------------------------------------------------------
# GPU region check
# ---------------------------------------------------------------------------

class TestGpuRegionCheck:
    def test_valid_region(self, provider):
        check = provider._check_gpu_region(
            "Consumption-GPU-NC8as-T4", "eastus",
            valid_regions=("eastus", "westus2"),
        )
        assert check.passed

    def test_invalid_region(self, provider):
        check = provider._check_gpu_region(
            "Consumption-GPU-NC8as-T4", "antarctica",
            valid_regions=("eastus", "westus2"),
        )
        assert not check.passed
        assert "antarctica" in check.message

    def test_no_valid_regions_skips(self, provider):
        check = provider._check_gpu_region(
            "Consumption-GPU-NC8as-T4", "eastus",
            valid_regions=(),
        )
        assert check.passed
        assert "skipped" in check.message
