"""Tests for --serverless-only mode."""

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from tuna.models import (
    DeployRequest,
    DeploymentResult,
    HybridDeployment,
    PreflightCheck,
    PreflightResult,
    ProviderPlan,
)
from tuna.orchestrator import launch_serverless_only, status_hybrid, destroy_hybrid, _warmup_serverless
from tuna.state import DeploymentRecord, save_deployment, load_deployment


# ---------------------------------------------------------------------------
# launch_serverless_only
# ---------------------------------------------------------------------------


class TestLaunchServerlessOnly:
    def _mock_provider(self, *, preflight_ok=True, deploy_error=None):
        provider = MagicMock()
        provider.name.return_value = "modal"
        provider.vllm_version.return_value = "0.15.1"
        provider.preflight.return_value = PreflightResult(
            provider="modal",
            checks=[PreflightCheck(name="ok", passed=preflight_ok, message="ok" if preflight_ok else "fail")],
        )
        plan = ProviderPlan(provider="modal", rendered_script="# script", metadata={"app_name": "test-serverless"})
        provider.plan.return_value = plan
        if deploy_error:
            provider.deploy.return_value = DeploymentResult(
                provider="modal", error=deploy_error,
            )
        else:
            provider.deploy.return_value = DeploymentResult(
                provider="modal",
                endpoint_url="https://modal.run/test",
                metadata={"app_name": "test-serverless"},
            )
        return provider

    @patch("tuna.orchestrator._warmup_serverless", return_value=True)
    @patch("tuna.orchestrator.get_provider")
    def test_success(self, mock_get_provider, mock_warmup):
        mock_prov = self._mock_provider()
        mock_get_provider.return_value = mock_prov

        request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B", gpu="L4",
            service_name="test-svc", serverless_provider="modal",
            serverless_only=True,
        )
        result = launch_serverless_only(request)

        assert result.serverless is not None
        assert result.serverless.endpoint_url == "https://modal.run/test"
        assert result.serverless.error is None
        assert result.router_url == "https://modal.run/test"
        assert result.spot is None
        assert result.router is None

        mock_prov.preflight.assert_called_once()
        mock_prov.plan.assert_called_once()
        mock_prov.deploy.assert_called_once()
        mock_warmup.assert_called_once_with("https://modal.run/test/health")

    @patch("tuna.orchestrator.get_provider")
    def test_preflight_fail(self, mock_get_provider):
        mock_prov = self._mock_provider(preflight_ok=False)
        mock_get_provider.return_value = mock_prov

        request = DeployRequest(
            model_name="m", gpu="g", service_name="test-svc",
            serverless_provider="modal", serverless_only=True,
        )
        result = launch_serverless_only(request)

        assert result.serverless is not None
        assert result.serverless.error is not None
        assert "Preflight failed" in result.serverless.error
        mock_prov.deploy.assert_not_called()

    @patch("tuna.orchestrator.get_provider")
    def test_deploy_error(self, mock_get_provider):
        mock_prov = self._mock_provider(deploy_error="deploy crashed")
        mock_get_provider.return_value = mock_prov

        request = DeployRequest(
            model_name="m", gpu="g", service_name="test-svc",
            serverless_provider="modal", serverless_only=True,
        )
        result = launch_serverless_only(request)

        assert result.serverless is not None
        assert result.serverless.error == "deploy crashed"
        assert result.router_url is None  # Not set on error

    @patch("tuna.orchestrator.get_provider")
    def test_deploy_exception_preserves_metadata(self, mock_get_provider):
        """When deploy() raises (not returns error), plan metadata is preserved."""
        mock_prov = self._mock_provider()
        mock_prov.deploy.side_effect = RuntimeError("connection reset")
        mock_get_provider.return_value = mock_prov

        request = DeployRequest(
            model_name="m", gpu="g", service_name="test-svc",
            serverless_provider="modal", serverless_only=True,
        )
        result = launch_serverless_only(request)

        assert result.serverless is not None
        assert "connection reset" in result.serverless.error
        assert result.serverless.metadata.get("app_name") == "test-serverless"
        assert result.router_url is None

    @patch("tuna.orchestrator.get_provider")
    def test_plan_exception_returns_error(self, mock_get_provider):
        """When plan() raises, error result is returned (no metadata to preserve)."""
        mock_prov = self._mock_provider()
        mock_prov.plan.side_effect = ValueError("bad GPU")
        mock_get_provider.return_value = mock_prov

        request = DeployRequest(
            model_name="m", gpu="g", service_name="test-svc",
            serverless_provider="modal", serverless_only=True,
        )
        result = launch_serverless_only(request)

        assert result.serverless is not None
        assert "bad GPU" in result.serverless.error
        mock_prov.deploy.assert_not_called()


# ---------------------------------------------------------------------------
# _warmup_serverless
# ---------------------------------------------------------------------------


class TestWarmupServerless:
    @patch("tuna.orchestrator.requests")
    def test_healthy_on_first_try(self, mock_requests):
        mock_resp = MagicMock(status_code=200)
        mock_requests.get.return_value = mock_resp

        result = _warmup_serverless("https://example.com/health", timeout=10)

        assert result is True
        mock_requests.get.assert_called_once_with("https://example.com/health", timeout=10)

    @patch("tuna.orchestrator.time.sleep")
    @patch("tuna.orchestrator.requests")
    def test_healthy_after_retries(self, mock_requests, mock_sleep):
        mock_fail = MagicMock(status_code=503)
        mock_ok = MagicMock(status_code=200)
        mock_requests.get.side_effect = [mock_fail, mock_fail, mock_ok]

        result = _warmup_serverless("https://example.com/health", timeout=60)

        assert result is True
        assert mock_requests.get.call_count == 3

    @patch("tuna.orchestrator.requests")
    def test_timeout(self, mock_requests):
        mock_requests.get.side_effect = Exception("connection refused")

        # Use a tiny timeout so it exits after one failed attempt
        result = _warmup_serverless("https://example.com/health", timeout=0)

        assert result is False


# ---------------------------------------------------------------------------
# status_hybrid — serverless-only path
# ---------------------------------------------------------------------------


class TestStatusServerlessOnly:
    @patch("tuna.orchestrator.get_provider")
    def test_returns_serverless_only_mode(self, mock_get_provider):
        mock_modal = MagicMock()
        mock_modal.status.return_value = {"provider": "modal", "status": "running", "endpoint_url": "https://modal.run"}
        mock_get_provider.return_value = mock_modal

        record = DeploymentRecord(
            service_name="test-svc",
            serverless_provider_name="modal",
            serverless_provider="modal",
            spot_provider_name=None,
            router_endpoint=None,
        )
        result = status_hybrid("test-svc", record=record)

        assert result["mode"] == "serverless-only"
        assert result["router"] is None
        assert result["spot"] is None
        assert result["serverless"]["status"] == "running"
        mock_modal.status.assert_called_once_with("test-svc")


# ---------------------------------------------------------------------------
# destroy_hybrid — serverless-only path
# ---------------------------------------------------------------------------


class TestDestroyServerlessOnly:
    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.orchestrator.cluster_down")
    @patch("tuna.orchestrator.get_provider")
    def test_skips_spot_and_router(self, mock_get_provider, mock_cluster_down, mock_cleanup):
        mock_modal = MagicMock()
        mock_modal.name.return_value = "modal"
        mock_get_provider.return_value = mock_modal

        record = DeploymentRecord(
            service_name="test-svc",
            serverless_provider_name="modal",
            serverless_metadata={"app_name": "test-svc-serverless"},
            spot_provider_name=None,
            spot_metadata={},
            router_metadata={},
        )
        destroy_hybrid("test-svc", record=record)

        # Serverless should be destroyed
        mock_modal.destroy.assert_called_once()
        # cluster_down should NOT be called — no router was deployed
        mock_cluster_down.assert_not_called()
        # No spot provider should be looked up
        mock_get_provider.assert_called_once_with("modal")


# ---------------------------------------------------------------------------
# save_deployment — serverless-only sets spot_provider_name=None
# ---------------------------------------------------------------------------


class TestSaveDeploymentServerlessOnly:
    def test_spot_provider_name_is_none(self, tmp_path):
        db_path = tmp_path / "test.db"
        request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B", gpu="L4",
            service_name="so-test", serverless_only=True,
        )
        result = HybridDeployment(
            serverless=DeploymentResult(
                provider="modal",
                endpoint_url="https://modal.run/test",
                metadata={"app_name": "so-test-serverless"},
            ),
            router_url="https://modal.run/test",
        )
        save_deployment(request, result, db_path=db_path)

        record = load_deployment("so-test", db_path=db_path)
        assert record is not None
        assert record.spot_provider_name is None
        assert record.router_endpoint is None
        assert record.serverless_provider_name == "modal"
        assert record.router_url == "https://modal.run/test"

    def test_hybrid_still_sets_spot_provider(self, tmp_path):
        db_path = tmp_path / "test.db"
        request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B", gpu="L4",
            service_name="hybrid-test", serverless_only=False,
        )
        result = HybridDeployment(
            serverless=DeploymentResult(provider="modal", endpoint_url="https://modal.run"),
            spot=DeploymentResult(provider="skyserve", endpoint_url="http://spot:30001"),
            router=DeploymentResult(provider="router", endpoint_url="http://10.0.0.1:8080"),
            router_url="http://10.0.0.1:8080",
        )
        save_deployment(request, result, db_path=db_path)

        record = load_deployment("hybrid-test", db_path=db_path)
        assert record.spot_provider_name == "skyserve"


# ---------------------------------------------------------------------------
# CLI — --serverless-only flag
# ---------------------------------------------------------------------------


class TestCliServerlessOnlyFlag:
    """Use the real CLI parser to ensure the flag actually exists."""

    def test_flag_parsed(self):
        """--serverless-only sets args.serverless_only = True on the real parser."""
        import sys
        from unittest.mock import patch as _patch

        captured_args = {}

        def fake_deploy(args):
            captured_args.update(vars(args))

        with _patch("tuna.__main__.cmd_deploy", fake_deploy), \
             _patch.object(sys, "argv", ["tuna", "deploy", "--model", "m", "--gpu", "g", "--serverless-only"]):
            from tuna.__main__ import main
            main()

        assert captured_args["serverless_only"] is True

    def test_flag_default_false(self):
        """Omitting --serverless-only defaults to False on the real parser."""
        import sys
        from unittest.mock import patch as _patch

        captured_args = {}

        def fake_deploy(args):
            captured_args.update(vars(args))

        with _patch("tuna.__main__.cmd_deploy", fake_deploy), \
             _patch.object(sys, "argv", ["tuna", "deploy", "--model", "m", "--gpu", "g"]):
            from tuna.__main__ import main
            main()

        assert captured_args["serverless_only"] is False


# ---------------------------------------------------------------------------
# cmd_cost — serverless-only path
# ---------------------------------------------------------------------------


class TestCostServerlessOnly:
    @patch("tuna.providers.registry.ensure_providers_for_deployment")
    @patch("tuna.state.load_deployment")
    @patch("tuna.__main__._print_serverless_only_cost")
    def test_serverless_only_cost_called(self, mock_print_cost, mock_load, mock_ensure):
        record = DeploymentRecord(
            service_name="so-test",
            serverless_provider_name="modal",
            serverless_provider="modal",
            spot_provider_name=None,
            router_endpoint=None,
            gpu="L4",
            gpu_count=1,
            created_at="2026-01-01T00:00:00+00:00",
        )
        mock_load.return_value = record

        from tuna.__main__ import cmd_cost
        args = argparse.Namespace(service_name="so-test")
        cmd_cost(args)

        mock_print_cost.assert_called_once_with(record)

    @patch("tuna.providers.registry.ensure_providers_for_deployment")
    @patch("tuna.state.load_deployment")
    @patch("tuna.orchestrator.status_hybrid")
    def test_hybrid_cost_still_uses_router(self, mock_status, mock_load, mock_ensure):
        """Hybrid deployments should NOT take the serverless-only cost path."""
        record = DeploymentRecord(
            service_name="hybrid-test",
            serverless_provider_name="modal",
            serverless_provider="modal",
            spot_provider_name="skyserve",
            router_endpoint="http://10.0.0.1:8080",
            gpu="L4",
            gpu_count=1,
        )
        mock_load.return_value = record
        mock_status.return_value = {
            "router": {"route_stats": {"total": 10, "gpu_seconds_serverless": 100, "gpu_seconds_spot": 200,
                                       "spot_ready_seconds": 300, "uptime_seconds": 400,
                                       "pct_spot": 66, "pct_serverless": 34}},
        }

        from tuna.__main__ import cmd_cost
        args = argparse.Namespace(service_name="hybrid-test")
        cmd_cost(args)

        # Should have called status_hybrid for the hybrid path
        mock_status.assert_called_once()
