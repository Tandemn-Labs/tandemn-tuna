"""Tests for CloudRunProvider preflight checks."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tuna.catalog import provider_regions
from tuna.models import DeployRequest
from tuna.providers.cloudrun_provider import (
    CloudRunProvider,
    REQUIRED_APIS,
)


@pytest.fixture
def provider():
    return CloudRunProvider()


@pytest.fixture
def request_l4():
    return DeployRequest(
        model_name="Qwen/Qwen3-0.6B",
        gpu="L4",
        region="us-central1",
        serverless_provider="cloudrun",
        service_name="test-svc",
    )


# ---------------------------------------------------------------------------
# _check_gcloud_installed
# ---------------------------------------------------------------------------

class TestPreflightGcloudCheck:
    def test_gcloud_found(self, provider):
        mock_proc = MagicMock(returncode=0, stdout="Google Cloud SDK 496.0.0\nsome other line\n")
        with patch("tuna.providers.cloudrun_provider.subprocess.run", return_value=mock_proc):
            check = provider._check_gcloud_installed()
        assert check.passed is True
        assert check.name == "gcloud_installed"
        assert "Google Cloud SDK 496.0.0" in check.message

    def test_gcloud_not_found(self, provider):
        with patch("tuna.providers.cloudrun_provider.subprocess.run", side_effect=FileNotFoundError):
            check = provider._check_gcloud_installed()
        assert check.passed is False
        assert check.name == "gcloud_installed"
        assert "not found" in check.message
        assert check.fix_command is not None


# ---------------------------------------------------------------------------
# _check_adc
# ---------------------------------------------------------------------------

class TestPreflightADCCheck:
    def test_adc_file_exists(self, provider):
        with patch("tuna.providers.cloudrun_provider.Path.exists", return_value=True):
            check = provider._check_adc()
        assert check.passed is True
        assert check.name == "credentials"

    def test_adc_missing_fallback_fails(self, provider):
        mock_proc = MagicMock(returncode=1, stdout="")
        with (
            patch("tuna.providers.cloudrun_provider.Path.exists", return_value=False),
            patch("tuna.providers.cloudrun_provider.subprocess.run", return_value=mock_proc),
        ):
            check = provider._check_adc()
        assert check.passed is False
        assert "gcloud auth application-default login" in check.fix_command

    def test_adc_missing_fallback_succeeds(self, provider):
        mock_proc = MagicMock(returncode=0, stdout="ya29.some-token\n")
        with (
            patch("tuna.providers.cloudrun_provider.Path.exists", return_value=False),
            patch("tuna.providers.cloudrun_provider.subprocess.run", return_value=mock_proc),
        ):
            check = provider._check_adc()
        assert check.passed is True
        assert "access token" in check.message


# ---------------------------------------------------------------------------
# _check_project
# ---------------------------------------------------------------------------

class TestPreflightProject:
    def test_project_active(self, provider):
        mock_proc = MagicMock(
            returncode=0,
            stdout=json.dumps({"lifecycleState": "ACTIVE", "projectId": "my-proj"}),
        )
        with patch("tuna.providers.cloudrun_provider.subprocess.run", return_value=mock_proc):
            check = provider._check_project("my-proj")
        assert check.passed is True
        assert "my-proj" in check.message
        assert "active" in check.message.lower()

    def test_project_not_found(self, provider):
        mock_proc = MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")
        with patch("tuna.providers.cloudrun_provider.subprocess.run", return_value=mock_proc):
            check = provider._check_project("bad-proj")
        assert check.passed is False
        assert "bad-proj" in check.message

    def test_project_not_active(self, provider):
        mock_proc = MagicMock(
            returncode=0,
            stdout=json.dumps({"lifecycleState": "DELETE_REQUESTED"}),
        )
        with patch("tuna.providers.cloudrun_provider.subprocess.run", return_value=mock_proc):
            check = provider._check_project("dying-proj")
        assert check.passed is False
        assert "DELETE_REQUESTED" in check.message


# ---------------------------------------------------------------------------
# _check_billing
# ---------------------------------------------------------------------------

class TestPreflightBilling:
    def test_billing_enabled(self, provider):
        mock_proc = MagicMock(
            returncode=0,
            stdout=json.dumps({"billingEnabled": True}),
        )
        with patch("tuna.providers.cloudrun_provider.subprocess.run", return_value=mock_proc):
            check = provider._check_billing("my-proj")
        assert check.passed is True
        assert "enabled" in check.message.lower()

    def test_billing_disabled(self, provider):
        mock_proc = MagicMock(
            returncode=0,
            stdout=json.dumps({"billingEnabled": False}),
        )
        with patch("tuna.providers.cloudrun_provider.subprocess.run", return_value=mock_proc):
            check = provider._check_billing("my-proj")
        assert check.passed is False
        assert check.fix_command is not None

    def test_billing_check_fails(self, provider):
        mock_proc = MagicMock(returncode=1, stdout="", stderr="error")
        with patch("tuna.providers.cloudrun_provider.subprocess.run", return_value=mock_proc):
            check = provider._check_billing("my-proj")
        assert check.passed is False


# ---------------------------------------------------------------------------
# _check_and_enable_apis
# ---------------------------------------------------------------------------

class TestPreflightAPIs:
    def test_all_apis_already_enabled(self, provider):
        mock_proc = MagicMock(
            returncode=0,
            stdout="run.googleapis.com\niam.googleapis.com\ncompute.googleapis.com\n",
        )
        with patch("tuna.providers.cloudrun_provider.subprocess.run", return_value=mock_proc):
            check = provider._check_and_enable_apis("my-proj")
        assert check.passed is True
        assert check.auto_fixed is False

    def test_auto_enable_succeeds(self, provider):
        list_proc = MagicMock(
            returncode=0,
            stdout="compute.googleapis.com\n",  # missing both required APIs
        )
        enable_proc = MagicMock(returncode=0)

        def side_effect(cmd, **kwargs):
            if "list" in cmd:
                return list_proc
            return enable_proc

        with patch("tuna.providers.cloudrun_provider.subprocess.run", side_effect=side_effect):
            check = provider._check_and_enable_apis("my-proj")
        assert check.passed is True
        assert check.auto_fixed is True
        assert "Auto-enabled" in check.message

    def test_auto_enable_fails(self, provider):
        list_proc = MagicMock(
            returncode=0,
            stdout="compute.googleapis.com\n",
        )
        enable_proc = MagicMock(returncode=1)

        def side_effect(cmd, **kwargs):
            if "list" in cmd:
                return list_proc
            return enable_proc

        with patch("tuna.providers.cloudrun_provider.subprocess.run", side_effect=side_effect):
            check = provider._check_and_enable_apis("my-proj")
        assert check.passed is False
        assert check.fix_command is not None

    def test_list_apis_fails(self, provider):
        mock_proc = MagicMock(returncode=1, stdout="", stderr="permission denied")
        with patch("tuna.providers.cloudrun_provider.subprocess.run", return_value=mock_proc):
            check = provider._check_and_enable_apis("my-proj")
        assert check.passed is False


# ---------------------------------------------------------------------------
# _check_gpu_region
# ---------------------------------------------------------------------------

class TestPreflightGpuRegion:
    def test_valid_region(self, provider):
        regions = provider_regions("L4", "cloudrun")
        check = provider._check_gpu_region("nvidia-l4", "us-central1", regions)
        assert check.passed is True
        assert "nvidia-l4" in check.message
        assert "us-central1" in check.message

    def test_invalid_region(self, provider):
        regions = provider_regions("L4", "cloudrun")
        check = provider._check_gpu_region("nvidia-l4", "antarctica-south1", regions)
        assert check.passed is False
        assert "antarctica-south1" in check.message
        assert "us-central1" in check.fix_command

    def test_unknown_gpu_skips(self, provider):
        check = provider._check_gpu_region("nvidia-future-gpu-9000", "us-central1")
        assert check.passed is True
        assert "skipped" in check.message.lower()


# ---------------------------------------------------------------------------
# Full preflight() integration
# ---------------------------------------------------------------------------

class TestPreflightIntegration:
    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "my-proj"}, clear=False)
    def test_all_checks_pass(self, provider, request_l4):
        """All checks pass when gcloud works and everything is configured."""
        gcloud_proc = MagicMock(returncode=0, stdout="Google Cloud SDK 496.0.0\n")
        project_proc = MagicMock(
            returncode=0,
            stdout=json.dumps({"lifecycleState": "ACTIVE"}),
        )
        billing_proc = MagicMock(
            returncode=0,
            stdout=json.dumps({"billingEnabled": True}),
        )
        apis_proc = MagicMock(
            returncode=0,
            stdout="run.googleapis.com\niam.googleapis.com\n",
        )

        call_count = 0

        def side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if "--version" in cmd:
                return gcloud_proc
            if "billing" in cmd:
                return billing_proc
            if "projects" in cmd and "describe" in cmd:
                return project_proc
            if "services" in cmd and "list" in cmd:
                return apis_proc
            return MagicMock(returncode=0, stdout="")

        with (
            patch("tuna.providers.cloudrun_provider.subprocess.run", side_effect=side_effect),
            patch("tuna.providers.cloudrun_provider.Path.exists", return_value=True),
        ):
            result = provider.preflight(request_l4)

        assert result.ok is True
        assert len(result.failed) == 0
        assert result.provider == "cloudrun"
        # Should have: gcloud_installed, credentials, project, billing, apis, gpu_region
        assert len(result.checks) == 6

    def test_gcloud_missing_short_circuits(self, provider, request_l4):
        """If gcloud is missing, we stop after the first check."""
        with patch("tuna.providers.cloudrun_provider.subprocess.run", side_effect=FileNotFoundError):
            result = provider.preflight(request_l4)

        assert result.ok is False
        assert len(result.checks) == 1
        assert result.checks[0].name == "gcloud_installed"
        assert result.checks[0].passed is False

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "my-proj"}, clear=False)
    def test_project_not_found_short_circuits(self, provider, request_l4):
        """If the project doesn't exist, we stop before billing/API checks."""
        gcloud_proc = MagicMock(returncode=0, stdout="Google Cloud SDK 496.0.0\n")
        project_proc = MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")

        def side_effect(cmd, **kwargs):
            if "--version" in cmd:
                return gcloud_proc
            if "projects" in cmd and "describe" in cmd:
                return project_proc
            return MagicMock(returncode=0, stdout="")

        with (
            patch("tuna.providers.cloudrun_provider.subprocess.run", side_effect=side_effect),
            patch("tuna.providers.cloudrun_provider.Path.exists", return_value=True),
        ):
            result = provider.preflight(request_l4)

        assert result.ok is False
        # gcloud_installed, credentials, project (failed) — then stopped
        assert len(result.checks) == 3
        assert result.checks[2].name == "project"
        assert result.checks[2].passed is False

    @patch.dict("os.environ", {}, clear=True)
    def test_no_project_configured_short_circuits(self, provider, request_l4):
        """If no project is set in env or gcloud, we stop with a project failure."""
        gcloud_version_proc = MagicMock(returncode=0, stdout="Google Cloud SDK 496.0.0\n")
        gcloud_config_proc = MagicMock(returncode=0, stdout="(unset)\n")

        def side_effect(cmd, **kwargs):
            if "--version" in cmd:
                return gcloud_version_proc
            if "config" in cmd and "get-value" in cmd:
                return gcloud_config_proc
            return MagicMock(returncode=0, stdout="")

        with (
            patch("tuna.providers.cloudrun_provider.subprocess.run", side_effect=side_effect),
            patch("tuna.providers.cloudrun_provider.Path.exists", return_value=True),
        ):
            result = provider.preflight(request_l4)

        assert result.ok is False
        # gcloud_installed, credentials, project (failed — no project configured)
        assert len(result.checks) == 3
        assert result.checks[2].name == "project"
        assert result.checks[2].passed is False

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "my-proj"}, clear=False)
    def test_preflight_result_properties(self, provider, request_l4):
        """Test PreflightResult.ok and .failed properties."""
        gcloud_proc = MagicMock(returncode=0, stdout="Google Cloud SDK 496.0.0\n")
        project_proc = MagicMock(
            returncode=0,
            stdout=json.dumps({"lifecycleState": "ACTIVE"}),
        )
        billing_proc = MagicMock(
            returncode=0,
            stdout=json.dumps({"billingEnabled": False}),
        )
        apis_proc = MagicMock(
            returncode=0,
            stdout="run.googleapis.com\niam.googleapis.com\n",
        )

        def side_effect(cmd, **kwargs):
            if "--version" in cmd:
                return gcloud_proc
            if "billing" in cmd:
                return billing_proc
            if "projects" in cmd and "describe" in cmd:
                return project_proc
            if "services" in cmd and "list" in cmd:
                return apis_proc
            return MagicMock(returncode=0, stdout="")

        with (
            patch("tuna.providers.cloudrun_provider.subprocess.run", side_effect=side_effect),
            patch("tuna.providers.cloudrun_provider.Path.exists", return_value=True),
        ):
            result = provider.preflight(request_l4)

        assert result.ok is False
        assert len(result.failed) == 1
        assert result.failed[0].name == "billing"
