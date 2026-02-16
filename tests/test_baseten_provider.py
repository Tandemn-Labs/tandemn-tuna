"""Tests for tuna.providers.baseten_provider — plan() and preflight()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml

from tuna.models import DeployRequest
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
        assert parsed["resources"]["accelerator"] == "H100_MIG"

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
