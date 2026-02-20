"""Tests for tuna.providers.baseten_provider — plan(), preflight(), and deploy()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml

from tuna.models import DeployRequest, ProviderPlan
from tuna.providers.baseten_provider import BasetenProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    return BasetenProvider()


@pytest.fixture
def request_l4():
    return DeployRequest(
        model_name="Qwen/Qwen3-0.6B",
        gpu="L4",
        service_name="test-svc",
        serverless_provider="baseten",
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
# plan() tests
# ---------------------------------------------------------------------------

class TestBasetenPlan:
    def test_plan_provider_name(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        assert plan.provider == "baseten"

    def test_plan_metadata(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        assert plan.metadata["service_name"] == "test-svc-serverless"
        assert plan.metadata["model_name"] == "Qwen/Qwen3-0.6B"

    def test_plan_rendered_contains_model(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        assert "Qwen/Qwen3-0.6B" in plan.rendered_script

    def test_plan_rendered_contains_gpu(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["resources"]["accelerator"] == "L4"

    def test_plan_rendered_is_valid_yaml(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed is not None
        assert "docker_server" in parsed
        assert "resources" in parsed

    def test_plan_fast_boot_sets_enforce_eager(self, provider, request_l4, vllm_cmd):
        request_l4.cold_start_mode = "fast_boot"
        plan = provider.plan(request_l4, vllm_cmd)
        assert "--enforce-eager" in plan.rendered_script

    def test_plan_no_fast_boot_omits_enforce_eager(self, provider, request_l4, vllm_cmd):
        request_l4.cold_start_mode = "no_fast_boot"
        plan = provider.plan(request_l4, vllm_cmd)
        assert "--enforce-eager" not in plan.rendered_script

    def test_plan_concurrency(self, provider, request_l4, vllm_cmd):
        request_l4.scaling.serverless.concurrency = 64
        plan = provider.plan(request_l4, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["runtime"]["predict_concurrency"] == 64

    def test_plan_gpu_mapping_t4(self, provider, vllm_cmd):
        req = DeployRequest(model_name="m", gpu="T4", service_name="s", serverless_provider="baseten")
        plan = provider.plan(req, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["resources"]["accelerator"] == "T4"

    def test_plan_gpu_mapping_a10g(self, provider, vllm_cmd):
        req = DeployRequest(model_name="m", gpu="A10G", service_name="s", serverless_provider="baseten")
        plan = provider.plan(req, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["resources"]["accelerator"] == "A10G"

    def test_plan_gpu_mapping_a100(self, provider, vllm_cmd):
        req = DeployRequest(model_name="m", gpu="A100_80GB", service_name="s", serverless_provider="baseten")
        plan = provider.plan(req, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["resources"]["accelerator"] == "A100"

    def test_plan_gpu_mapping_h100_mig(self, provider, vllm_cmd):
        req = DeployRequest(model_name="m", gpu="H100_MIG", service_name="s", serverless_provider="baseten")
        plan = provider.plan(req, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["resources"]["accelerator"] == "H100MIG"

    def test_plan_gpu_mapping_h100(self, provider, vllm_cmd):
        req = DeployRequest(model_name="m", gpu="H100", service_name="s", serverless_provider="baseten")
        plan = provider.plan(req, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["resources"]["accelerator"] == "H100"

    def test_plan_gpu_mapping_b200(self, provider, vllm_cmd):
        req = DeployRequest(model_name="m", gpu="B200", service_name="s", serverless_provider="baseten")
        plan = provider.plan(req, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["resources"]["accelerator"] == "B200"

    def test_plan_unknown_gpu_raises(self, provider, vllm_cmd):
        req = DeployRequest(model_name="m", gpu="FUTURE_GPU_9000", service_name="s", serverless_provider="baseten")
        with pytest.raises(ValueError, match="Unknown GPU type for Baseten"):
            provider.plan(req, vllm_cmd)

    def test_plan_docker_server_port(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["docker_server"]["server_port"] == 8000

    def test_plan_health_endpoints(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["docker_server"]["readiness_endpoint"] == "/health"
        assert parsed["docker_server"]["liveness_endpoint"] == "/health"

    def test_plan_env_is_empty(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        assert plan.env == {}

    def test_plan_model_name_in_yaml(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["model_name"] == "test-svc-serverless"


# ---------------------------------------------------------------------------
# _parse_model_id() tests
# ---------------------------------------------------------------------------

class TestParseModelId:
    def test_parse_dashboard_url(self, provider):
        stdout = (
            "\n"
            "Deploying truss using T4x4x16 instance type.\n"
            "✨ Model baseten-test-serverless was successfully pushed ✨\n"
            "🪵  View logs for your deployment at "
            "https://app.baseten.co/models/31d5m413/logs/31dgo51\n"
        )
        assert provider._parse_model_id(stdout) == "31d5m413"

    def test_parse_endpoint_url(self, provider):
        stdout = "Endpoint: https://model-abc123.api.baseten.co/environments/production/sync/v1\n"
        assert provider._parse_model_id(stdout) == "abc123"

    def test_parse_model_id_key_value(self, provider):
        stdout = "model_id: xyz789\n"
        assert provider._parse_model_id(stdout) == "xyz789"

    def test_parse_empty_output(self, provider):
        assert provider._parse_model_id("") is None

    def test_parse_no_match(self, provider):
        assert provider._parse_model_id("some random output\nno ids here\n") is None


# ---------------------------------------------------------------------------
# preflight() tests
# ---------------------------------------------------------------------------

class TestBasetenPreflight:
    @patch.dict("os.environ", {}, clear=True)
    def test_api_key_missing(self, provider, request_l4):
        result = provider.preflight(request_l4)
        assert result.ok is False
        assert len(result.checks) == 1
        assert result.checks[0].name == "api_key"
        assert result.checks[0].passed is False
        assert "BASETEN_API_KEY" in result.checks[0].fix_command

    @patch.dict("os.environ", {"BASETEN_API_KEY": "test-key"}, clear=True)
    def test_api_key_invalid(self, provider, request_l4):
        mock_resp = MagicMock(status_code=401)
        with patch("tuna.providers.baseten_provider.requests.get", return_value=mock_resp):
            result = provider.preflight(request_l4)
        assert result.ok is False
        assert result.checks[0].name == "api_key"
        assert result.checks[0].passed is True  # key is set
        assert result.checks[1].name == "api_key_valid"
        assert result.checks[1].passed is False

    @patch.dict("os.environ", {"BASETEN_API_KEY": "valid-key"}, clear=True)
    def test_all_checks_pass(self, provider, request_l4):
        mock_api_resp = MagicMock(status_code=200)
        mock_truss_proc = MagicMock(returncode=0, stdout="user@baseten.co\n")

        with (
            patch("tuna.providers.baseten_provider.requests.get", return_value=mock_api_resp),
            patch("tuna.providers.baseten_provider.shutil.which", return_value="/usr/bin/truss"),
            patch("tuna.providers.baseten_provider.subprocess.run", return_value=mock_truss_proc),
        ):
            result = provider.preflight(request_l4)

        assert result.ok is True
        assert len(result.failed) == 0
        # api_key, api_key_valid, truss_installed, truss_authenticated, gpu_supported
        assert len(result.checks) == 5

    @patch.dict("os.environ", {"BASETEN_API_KEY": "valid-key"}, clear=True)
    def test_truss_not_installed(self, provider, request_l4):
        mock_api_resp = MagicMock(status_code=200)

        with (
            patch("tuna.providers.baseten_provider.requests.get", return_value=mock_api_resp),
            patch("tuna.providers.baseten_provider.shutil.which", return_value=None),
        ):
            result = provider.preflight(request_l4)

        assert result.ok is False
        truss_check = [c for c in result.checks if c.name == "truss_installed"][0]
        assert truss_check.passed is False
        assert "pip install" in truss_check.fix_command

    @patch.dict("os.environ", {"BASETEN_API_KEY": "valid-key"}, clear=True)
    def test_truss_not_authenticated(self, provider, request_l4):
        mock_api_resp = MagicMock(status_code=200)
        mock_truss_proc = MagicMock(returncode=1, stdout="")

        with (
            patch("tuna.providers.baseten_provider.requests.get", return_value=mock_api_resp),
            patch("tuna.providers.baseten_provider.shutil.which", return_value="/usr/bin/truss"),
            patch("tuna.providers.baseten_provider.subprocess.run", return_value=mock_truss_proc),
        ):
            result = provider.preflight(request_l4)

        assert result.ok is False
        auth_check = [c for c in result.checks if c.name == "truss_authenticated"][0]
        assert auth_check.passed is False
        assert "truss login" in auth_check.fix_command


# ---------------------------------------------------------------------------
# Provider basics
# ---------------------------------------------------------------------------

class TestBasetenProviderBasics:
    def test_name(self, provider):
        assert provider.name() == "baseten"

    def test_vllm_version(self, provider):
        assert provider.vllm_version() == "0.15.1"

    @patch.dict("os.environ", {"BASETEN_API_KEY": "my-key"}, clear=True)
    def test_auth_token(self, provider):
        assert provider.auth_token() == "my-key"

    @patch.dict("os.environ", {}, clear=True)
    def test_auth_token_empty_when_unset(self, provider):
        assert provider.auth_token() == ""


# ---------------------------------------------------------------------------
# model_cache + truss-transfer-cli template tests
# ---------------------------------------------------------------------------

class TestBasetenModelCache:
    def test_template_has_model_cache_block(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert "model_cache" in parsed
        assert len(parsed["model_cache"]) == 1
        assert parsed["model_cache"][0]["repo_id"] == "Qwen/Qwen3-0.6B"
        assert parsed["model_cache"][0]["revision"] == "main"
        assert parsed["model_cache"][0]["use_volume"] is True
        assert parsed["model_cache"][0]["volume_folder"] == "Qwen--Qwen3-0.6B"

    def test_template_has_truss_transfer_cli(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        assert "truss-transfer-cli &&" in plan.rendered_script

    def test_start_command_begins_with_bash_truss_transfer(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        start_cmd = parsed["docker_server"]["start_command"]
        assert start_cmd.strip().startswith("bash -c")
        assert "truss-transfer-cli &&" in start_cmd

    def test_model_cache_repo_id_matches_model(self, provider, vllm_cmd):
        req = DeployRequest(
            model_name="meta-llama/Llama-3-8B",
            gpu="H100",
            service_name="llama-test",
            serverless_provider="baseten",
        )
        plan = provider.plan(req, vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["model_cache"][0]["repo_id"] == "meta-llama/Llama-3-8B"


# ---------------------------------------------------------------------------
# plan() metadata for autoscaling
# ---------------------------------------------------------------------------

class TestBasetenPlanMetadata:
    def test_plan_metadata_has_concurrency_target(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        assert "concurrency_target" in plan.metadata
        assert plan.metadata["concurrency_target"] == "32"  # default

    def test_plan_metadata_has_scale_down_delay(self, provider, request_l4, vllm_cmd):
        plan = provider.plan(request_l4, vllm_cmd)
        assert "scale_down_delay" in plan.metadata
        assert plan.metadata["scale_down_delay"] == "60"  # default

    def test_plan_metadata_custom_concurrency(self, provider, request_l4, vllm_cmd):
        request_l4.scaling.serverless.concurrency = 64
        plan = provider.plan(request_l4, vllm_cmd)
        assert plan.metadata["concurrency_target"] == "64"

    def test_plan_metadata_custom_scaledown(self, provider, request_l4, vllm_cmd):
        request_l4.scaling.serverless.scaledown_window = 120
        plan = provider.plan(request_l4, vllm_cmd)
        assert plan.metadata["scale_down_delay"] == "120"


# ---------------------------------------------------------------------------
# _configure_autoscaling() tests
# ---------------------------------------------------------------------------

class TestBasetenAutoscaling:
    @patch.dict("os.environ", {"BASETEN_API_KEY": "test-key"}, clear=True)
    def test_autoscaling_patch_request(self, provider):
        mock_resp = MagicMock(status_code=200)
        with patch("tuna.providers.baseten_provider.requests.patch", return_value=mock_resp) as mock_patch:
            provider._configure_autoscaling(
                "model123",
                concurrency_target=64,
                scale_down_delay=120,
            )

        mock_patch.assert_called_once()
        call_args = mock_patch.call_args
        assert call_args[0][0] == "https://api.baseten.co/v1/models/model123/environments/production"
        assert call_args[1]["json"] == {
            "autoscaling_settings": {
                "concurrency_target": 64,
                "scale_down_delay": 120,
            }
        }
        assert call_args[1]["headers"]["Authorization"] == "Api-Key test-key"

    @patch.dict("os.environ", {"BASETEN_API_KEY": "test-key"}, clear=True)
    def test_autoscaling_failure_is_non_fatal(self, provider):
        """Autoscaling API failure should log a warning, not raise."""
        mock_resp = MagicMock(status_code=500, text="Internal Server Error")
        with patch("tuna.providers.baseten_provider.requests.patch", return_value=mock_resp):
            # Should not raise
            provider._configure_autoscaling(
                "model123",
                concurrency_target=32,
                scale_down_delay=60,
            )

    @patch.dict("os.environ", {"BASETEN_API_KEY": "test-key"}, clear=True)
    def test_autoscaling_network_error_is_non_fatal(self, provider):
        """Network errors during autoscaling should be caught."""
        import requests as req_lib
        with patch(
            "tuna.providers.baseten_provider.requests.patch",
            side_effect=req_lib.ConnectionError("network down"),
        ):
            # Should not raise
            provider._configure_autoscaling(
                "model123",
                concurrency_target=32,
                scale_down_delay=60,
            )

    @patch.dict("os.environ", {}, clear=True)
    def test_autoscaling_skipped_without_api_key(self, provider):
        """Without API key, _configure_autoscaling should return silently."""
        with patch("tuna.providers.baseten_provider.requests.patch") as mock_patch:
            provider._configure_autoscaling(
                "model123",
                concurrency_target=32,
                scale_down_delay=60,
            )
        mock_patch.assert_not_called()

    @patch.dict("os.environ", {"BASETEN_API_KEY": "test-key"}, clear=True)
    def test_deploy_calls_autoscaling(self, provider):
        """deploy() should call _configure_autoscaling after successful truss push."""
        mock_proc = MagicMock(
            returncode=0,
            stdout="https://app.baseten.co/models/abc123/logs/def456\n",
            stderr="",
        )
        mock_autoscale_resp = MagicMock(status_code=200)

        plan = ProviderPlan(
            provider="baseten",
            rendered_script="model_name: test",
            metadata={
                "service_name": "test-serverless",
                "model_name": "Qwen/Qwen3-0.6B",
                "concurrency_target": "64",
                "scale_down_delay": "120",
            },
        )

        with (
            patch("tuna.providers.baseten_provider.subprocess.run", return_value=mock_proc),
            patch("tuna.providers.baseten_provider.requests.patch", return_value=mock_autoscale_resp) as mock_patch,
        ):
            result = provider.deploy(plan)

        assert result.error is None
        assert result.endpoint_url is not None
        mock_patch.assert_called_once()
        call_json = mock_patch.call_args[1]["json"]
        assert call_json["autoscaling_settings"]["concurrency_target"] == 64
        assert call_json["autoscaling_settings"]["scale_down_delay"] == 120

    @patch.dict("os.environ", {"BASETEN_API_KEY": "test-key"}, clear=True)
    def test_deploy_uses_defaults_when_metadata_missing(self, provider):
        """deploy() should use defaults if autoscaling metadata is absent."""
        mock_proc = MagicMock(
            returncode=0,
            stdout="https://app.baseten.co/models/abc123/logs/def456\n",
            stderr="",
        )
        mock_autoscale_resp = MagicMock(status_code=200)

        plan = ProviderPlan(
            provider="baseten",
            rendered_script="model_name: test",
            metadata={
                "service_name": "test-serverless",
                "model_name": "m",
            },
        )

        with (
            patch("tuna.providers.baseten_provider.subprocess.run", return_value=mock_proc),
            patch("tuna.providers.baseten_provider.requests.patch", return_value=mock_autoscale_resp) as mock_patch,
        ):
            result = provider.deploy(plan)

        assert result.error is None
        call_json = mock_patch.call_args[1]["json"]
        assert call_json["autoscaling_settings"]["concurrency_target"] == 32
        assert call_json["autoscaling_settings"]["scale_down_delay"] == 60
