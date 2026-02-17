"""Tests for tuna.orchestrator — unit tests with mocked subprocess/HTTP."""

import subprocess
from unittest.mock import MagicMock, patch

from tuna.models import DeployRequest, DeploymentResult, HybridDeployment, ProviderPlan
from tuna.models import PreflightCheck, PreflightResult
from tuna.orchestrator import (
    build_vllm_cmd,
    destroy_hybrid,
    launch_hybrid,
    push_url_to_router,
    status_hybrid,
    _find_controller_cluster,
    _launch_router_on_controller,
)
from tuna.state import DeploymentRecord


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
    @patch("tuna.orchestrator.requests")
    def test_pushes_serverless_url(self, mock_requests):
        mock_requests.post.return_value = MagicMock(status_code=200)
        result = push_url_to_router(
            "http://router:8080", serverless_url="https://modal.run"
        )
        assert result is True
        mock_requests.post.assert_called_once()
        call_kwargs = mock_requests.post.call_args
        assert call_kwargs[1]["json"]["serverless_url"] == "https://modal.run"

    @patch("tuna.orchestrator.requests")
    def test_pushes_spot_url(self, mock_requests):
        mock_requests.post.return_value = MagicMock(status_code=200)
        result = push_url_to_router(
            "http://router:8080", spot_url="http://spot:30001"
        )
        assert result is True
        call_kwargs = mock_requests.post.call_args
        assert call_kwargs[1]["json"]["spot_url"] == "http://spot:30001"

    @patch("tuna.orchestrator.requests")
    def test_empty_payload_returns_true(self, mock_requests):
        result = push_url_to_router("http://router:8080")
        assert result is True
        mock_requests.post.assert_not_called()

    @patch("tuna.orchestrator.requests")
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

    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.orchestrator.subprocess")
    @patch("tuna.orchestrator.get_provider")
    def test_calls_provider_destroy_for_spot(self, mock_get_provider, mock_subprocess, mock_cleanup):
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

    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.orchestrator.subprocess")
    @patch("tuna.orchestrator.get_provider")
    def test_calls_provider_destroy_for_serverless(self, mock_get_provider, mock_subprocess, mock_cleanup):
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

    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.orchestrator.subprocess")
    @patch("tuna.orchestrator.get_provider")
    def test_router_teardown_uses_sky_down(self, mock_get_provider, mock_subprocess, mock_cleanup):
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

    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.orchestrator.subprocess")
    @patch("tuna.orchestrator.get_provider")
    def test_record_provider_names_used(self, mock_get_provider, mock_subprocess, mock_cleanup):
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

    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.orchestrator.subprocess")
    @patch("tuna.orchestrator.get_provider")
    def test_fallback_without_record(self, mock_get_provider, mock_subprocess, mock_cleanup):
        """Without a record (no provider names), skips spot/serverless teardown."""
        destroy_hybrid("my-svc")

        # No providers should be looked up since record has no provider names
        mock_get_provider.assert_not_called()


class TestStatusHybrid:
    def _make_record(self, **kwargs):
        defaults = dict(
            service_name="my-svc",
            serverless_provider_name="modal",
            spot_provider_name="skyserve",
        )
        defaults.update(kwargs)
        return DeploymentRecord(**defaults)

    @patch("tuna.orchestrator._get_cluster_ip", return_value=None)
    @patch("tuna.orchestrator.get_provider")
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

    @patch("tuna.orchestrator._get_cluster_ip", return_value=None)
    @patch("tuna.orchestrator.get_provider")
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

    @patch("tuna.orchestrator._get_cluster_ip", return_value=None)
    @patch("tuna.orchestrator.get_provider")
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

    @patch("tuna.orchestrator._get_cluster_ip", return_value=None)
    @patch("tuna.orchestrator.get_provider")
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

    @patch("tuna.orchestrator._get_cluster_ip", return_value=None)
    @patch("tuna.orchestrator.get_provider")
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

    @patch("tuna.orchestrator._get_cluster_ip", return_value="10.0.0.5")
    @patch("tuna.orchestrator.get_provider")
    @patch("tuna.orchestrator.requests")
    def test_colocated_router_uses_controller_ip(self, mock_requests, mock_get_provider, mock_get_ip):
        """Colocated router reads controller cluster name and port from metadata."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "healthy"}
        mock_requests.get.return_value = mock_resp

        mock_spot = MagicMock()
        mock_spot.status.return_value = {"provider": "skyserve", "status": "running"}
        mock_modal = MagicMock()
        mock_modal.status.return_value = {"provider": "modal", "status": "running"}
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        record = self._make_record(
            router_metadata={"colocated": "true", "cluster_name": "sky-serve-controller-abc", "router_port": "8080"},
        )
        result = status_hybrid("my-svc", record=record)

        mock_get_ip.assert_called_with("sky-serve-controller-abc")
        assert result["router"]["url"] == "http://10.0.0.5:8080"


class TestFindControllerCluster:
    @patch("tuna.orchestrator.subprocess")
    def test_finds_controller(self, mock_subprocess):
        mock_subprocess.run.return_value = MagicMock(
            stdout=(
                "NAME                         LAUNCHED    RESOURCES\n"
                "sky-serve-controller-abc123  2 hrs ago   AWS(m4.xlarge)\n"
            ),
            returncode=0,
        )
        result = _find_controller_cluster()
        assert result == "sky-serve-controller-abc123"

    @patch("tuna.orchestrator.subprocess")
    def test_returns_none_when_no_controller(self, mock_subprocess):
        mock_subprocess.run.return_value = MagicMock(
            stdout="NAME      LAUNCHED    RESOURCES\nmy-vm    1 hr ago    AWS\n",
            returncode=0,
        )
        result = _find_controller_cluster()
        assert result is None

    @patch("tuna.orchestrator.subprocess")
    def test_returns_none_on_exception(self, mock_subprocess):
        mock_subprocess.run.side_effect = subprocess.TimeoutExpired("sky", 30)
        result = _find_controller_cluster()
        assert result is None


class TestLaunchRouterOnController:
    @patch("tuna.orchestrator._open_port_on_cluster", return_value=True)
    @patch("tuna.orchestrator._get_ssh_user", return_value="ubuntu")
    @patch("tuna.orchestrator._get_ssh_key_path", return_value="/home/user/.ssh/sky-key")
    @patch("tuna.orchestrator._get_cluster_ip", return_value="10.0.0.1")
    @patch("tuna.orchestrator.subprocess")
    def test_success(self, mock_subprocess, mock_ip, mock_key, mock_user, mock_port):
        mock_subprocess.run.return_value = MagicMock(returncode=0, stderr="")
        mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
        request = DeployRequest(model_name="m", gpu="g")
        result = _launch_router_on_controller(request, "sky-serve-controller-x")

        assert result.error is None
        assert result.endpoint_url == "http://10.0.0.1:8080"
        assert result.metadata["colocated"] == "true"
        assert result.metadata["cluster_name"] == "sky-serve-controller-x"
        # Verify SCP was called
        scp_call = mock_subprocess.run.call_args_list[0]
        assert "scp" in scp_call[0][0]

    @patch("tuna.orchestrator._get_cluster_ip", return_value=None)
    def test_no_ip(self, mock_ip):
        request = DeployRequest(model_name="m", gpu="g")
        result = _launch_router_on_controller(request, "sky-serve-controller-x")
        assert result.error is not None
        assert "Could not resolve IP" in result.error

    @patch("tuna.orchestrator._open_port_on_cluster", return_value=True)
    @patch("tuna.orchestrator._get_ssh_user", return_value="ubuntu")
    @patch("tuna.orchestrator._get_ssh_key_path", return_value="/home/user/.ssh/sky-key")
    @patch("tuna.orchestrator._get_cluster_ip", return_value="10.0.0.1")
    @patch("tuna.orchestrator.subprocess")
    def test_scp_failure(self, mock_subprocess, mock_ip, mock_key, mock_user, mock_port):
        mock_subprocess.run.return_value = MagicMock(returncode=1, stderr="Permission denied")
        mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
        request = DeployRequest(model_name="m", gpu="g")
        result = _launch_router_on_controller(request, "sky-serve-controller-x")
        assert result.error is not None
        assert "SCP failed" in result.error

    @patch("tuna.orchestrator._open_port_on_cluster", return_value=True)
    @patch("tuna.orchestrator._get_ssh_user", return_value="ubuntu")
    @patch("tuna.orchestrator._get_ssh_key_path", return_value="/home/user/.ssh/sky-key")
    @patch("tuna.orchestrator._get_cluster_ip", return_value="10.0.0.1")
    @patch("tuna.orchestrator.subprocess")
    def test_serverless_url_in_env(self, mock_subprocess, mock_ip, mock_key, mock_user, mock_port):
        mock_subprocess.run.return_value = MagicMock(returncode=0, stderr="")
        mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
        request = DeployRequest(model_name="m", gpu="g")
        _launch_router_on_controller(
            request, "sky-serve-controller-x",
            serverless_url="https://modal.run/abc",
        )
        # The SSH start command should include the serverless URL
        ssh_call = mock_subprocess.run.call_args_list[2]  # 3rd call: scp, pip install, gunicorn start
        ssh_cmd = ssh_call[0][0][-1]  # last arg is the remote command
        assert "SERVERLESS_BASE_URL='https://modal.run/abc'" in ssh_cmd


class TestDestroyHybridColocated:
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

    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.orchestrator._get_ssh_user", return_value="ubuntu")
    @patch("tuna.orchestrator._get_ssh_key_path", return_value="/home/user/.ssh/sky-key")
    @patch("tuna.orchestrator._get_cluster_ip", return_value="10.0.0.1")
    @patch("tuna.orchestrator.subprocess")
    @patch("tuna.orchestrator.get_provider")
    def test_colocated_router_skips_sky_down(
        self, mock_get_provider, mock_subprocess, mock_ip, mock_key, mock_user, mock_cleanup,
    ):
        """Colocated router should use SSH pkill, not sky down."""
        mock_spot = MagicMock()
        mock_spot.name.return_value = "skyserve"
        mock_modal = MagicMock()
        mock_modal.name.return_value = "modal"
        mock_get_provider.side_effect = lambda name: {"skyserve": mock_spot, "modal": mock_modal}[name]

        record = self._make_record(
            router_metadata={"colocated": "true", "cluster_name": "sky-serve-controller-abc"},
        )
        destroy_hybrid("my-svc", record=record)

        # Verify no "sky down *-router" call
        for call in mock_subprocess.run.call_args_list:
            args = call[0][0]
            if isinstance(args, list) and "sky" in args and "down" in args:
                assert "my-svc-router" not in args, "Should not call sky down on router when colocated"

        # Verify SSH pkill was called
        found_pkill = False
        for call in mock_subprocess.run.call_args_list:
            args = call[0][0]
            if isinstance(args, list) and "ssh" in args:
                cmd = args[-1]
                if "pkill" in cmd and "gunicorn" in cmd:
                    found_pkill = True
        assert found_pkill, "Should SSH pkill the colocated gunicorn process"


class TestLaunchHybridPreflightGate:
    @patch("tuna.orchestrator.get_provider")
    def test_serverless_preflight_failure_aborts_before_spot(self, mock_get_provider):
        """If serverless preflight fails, launch_hybrid should return immediately without launching spot."""
        mock_serverless = MagicMock()
        mock_serverless.vllm_version.return_value = "0.15.1"
        mock_serverless.auth_token.return_value = ""
        mock_serverless.preflight.return_value = PreflightResult(
            provider="baseten",
            checks=[PreflightCheck(
                name="api_key",
                passed=False,
                message="BASETEN_API_KEY environment variable not set",
                fix_command="export BASETEN_API_KEY=<your-api-key>",
            )],
        )

        mock_spot = MagicMock()
        mock_get_provider.side_effect = lambda name: {
            "baseten": mock_serverless,
            "skyserve": mock_spot,
        }[name]

        request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="T4",
            service_name="test-svc",
            serverless_provider="baseten",
        )
        result = launch_hybrid(request)

        assert result.serverless is not None
        assert result.serverless.error is not None
        assert "Preflight failed" in result.serverless.error
        assert result.spot is None
        assert result.router is None
        mock_spot.deploy.assert_not_called()

    @patch("tuna.orchestrator.get_provider")
    def test_preflight_failure_includes_service_name_metadata(self, mock_get_provider):
        """Preflight failure result should include service_name in metadata for destroy."""
        mock_serverless = MagicMock()
        mock_serverless.vllm_version.return_value = "0.15.1"
        mock_serverless.auth_token.return_value = ""
        mock_serverless.preflight.return_value = PreflightResult(
            provider="modal",
            checks=[PreflightCheck(name="check", passed=False, message="fail")],
        )
        mock_get_provider.side_effect = lambda name: {
            "modal": mock_serverless,
            "skyserve": MagicMock(),
        }[name]

        request = DeployRequest(
            model_name="m", gpu="g", service_name="test-svc",
            serverless_provider="modal",
        )
        result = launch_hybrid(request)

        assert result.serverless.metadata.get("service_name") == "test-svc-serverless"


def _make_mock_provider(name, plan_metadata=None, deploy_raises=None):
    """Create a mock provider with plan/deploy/preflight wired up."""
    provider = MagicMock()
    provider.name.return_value = name
    provider.vllm_version.return_value = "0.15.1"
    provider.auth_token.return_value = ""
    provider.preflight.return_value = PreflightResult(
        provider=name, checks=[PreflightCheck(name="ok", passed=True, message="ok")]
    )
    plan = ProviderPlan(
        provider=name,
        rendered_script="# script",
        metadata=plan_metadata or {},
    )
    provider.plan.return_value = plan
    if deploy_raises:
        provider.deploy.side_effect = deploy_raises
    else:
        provider.deploy.return_value = DeploymentResult(
            provider=name,
            endpoint_url="https://example.com",
            metadata=plan_metadata or {},
        )
    return provider


class TestPartialFailureMetadata:
    """Tests that error-path DeploymentResults preserve plan metadata."""

    @patch("tuna.orchestrator.push_url_to_router", return_value=True)
    @patch("tuna.orchestrator._find_controller_cluster", return_value=None)
    @patch("tuna.orchestrator._launch_router_vm")
    @patch("tuna.orchestrator.get_provider")
    def test_serverless_deploy_error_preserves_metadata(
        self, mock_get_provider, mock_launch_router, mock_find_ctrl, mock_push,
    ):
        """When serverless deploy() raises, the error result should include plan metadata."""
        mock_serverless = _make_mock_provider(
            "modal",
            plan_metadata={"app_name": "test-svc-serverless"},
            deploy_raises=RuntimeError("Modal crashed"),
        )
        mock_spot = _make_mock_provider(
            "skyserve",
            plan_metadata={"service_name": "test-svc-spot"},
        )
        mock_get_provider.side_effect = lambda name: {
            "modal": mock_serverless, "skyserve": mock_spot,
        }[name]
        mock_launch_router.return_value = DeploymentResult(
            provider="router", endpoint_url="http://1.2.3.4:8080",
            metadata={"cluster_name": "ctrl"},
        )

        request = DeployRequest(
            model_name="m", gpu="g", service_name="test-svc",
            serverless_provider="modal",
        )
        result = launch_hybrid(request, separate_router_vm=True)

        assert result.serverless.error is not None
        assert "Modal crashed" in result.serverless.error
        assert result.serverless.metadata.get("app_name") == "test-svc-serverless"

    @patch("tuna.orchestrator.push_url_to_router", return_value=True)
    @patch("tuna.orchestrator._find_controller_cluster", return_value=None)
    @patch("tuna.orchestrator._launch_router_vm")
    @patch("tuna.orchestrator.get_provider")
    def test_spot_deploy_error_preserves_metadata(
        self, mock_get_provider, mock_launch_router, mock_find_ctrl, mock_push,
    ):
        """When spot deploy() raises, the error result should include plan metadata."""
        mock_serverless = _make_mock_provider(
            "modal",
            plan_metadata={"app_name": "test-svc-serverless"},
        )
        mock_spot = _make_mock_provider(
            "skyserve",
            plan_metadata={"service_name": "test-svc-spot"},
            deploy_raises=RuntimeError("SkyPilot crashed"),
        )
        mock_get_provider.side_effect = lambda name: {
            "modal": mock_serverless, "skyserve": mock_spot,
        }[name]
        mock_launch_router.return_value = DeploymentResult(
            provider="router", endpoint_url="http://1.2.3.4:8080",
            metadata={"cluster_name": "ctrl"},
        )

        request = DeployRequest(
            model_name="m", gpu="g", service_name="test-svc",
            serverless_provider="modal",
        )
        result = launch_hybrid(request, separate_router_vm=True)

        assert result.spot.error is not None
        assert "SkyPilot crashed" in result.spot.error
        assert result.spot.metadata.get("service_name") == "test-svc-spot"


class TestRouterFailureCollectsAll:
    """Router failure in separate_router_vm mode should still collect serverless/spot results."""

    @patch("tuna.orchestrator.push_url_to_router", return_value=True)
    @patch("tuna.orchestrator.get_provider")
    @patch("tuna.orchestrator._launch_router_vm")
    def test_router_failure_still_collects_serverless_spot(
        self, mock_launch_router, mock_get_provider, mock_push,
    ):
        mock_launch_router.return_value = DeploymentResult(
            provider="router", error="sky launch failed",
            metadata={"cluster_name": "test-router"},
        )
        mock_serverless = _make_mock_provider(
            "modal", plan_metadata={"app_name": "test-svc-serverless"},
        )
        mock_spot = _make_mock_provider(
            "skyserve", plan_metadata={"service_name": "test-svc-spot"},
        )
        mock_get_provider.side_effect = lambda name: {
            "modal": mock_serverless, "skyserve": mock_spot,
        }[name]

        request = DeployRequest(
            model_name="m", gpu="g", service_name="test-svc",
            serverless_provider="modal",
        )
        result = launch_hybrid(request, separate_router_vm=True)

        # Router failed, but serverless and spot should still be collected
        assert result.router is not None
        assert result.router.error is not None
        assert result.serverless is not None
        assert result.serverless.endpoint_url == "https://example.com"
        assert result.spot is not None
        assert result.spot.endpoint_url == "https://example.com"


class TestDestroyWithStatusLookup:
    """Tests that destroy_hybrid uses status() lookup when metadata is missing."""

    def _make_record(self, **kwargs):
        defaults = dict(
            service_name="my-svc",
            serverless_provider_name="modal",
            serverless_metadata={},
            spot_provider_name="skyserve",
            spot_metadata={"service_name": "my-svc-spot"},
        )
        defaults.update(kwargs)
        return DeploymentRecord(**defaults)

    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.orchestrator.subprocess")
    @patch("tuna.orchestrator.get_provider")
    def test_destroy_baseten_uses_status_lookup_for_model_id(
        self, mock_get_provider, mock_subprocess, mock_cleanup,
    ):
        """When baseten metadata has no model_id, destroy should call status() to find it."""
        mock_baseten = MagicMock()
        mock_baseten.name.return_value = "baseten"
        mock_baseten.status.return_value = {
            "provider": "baseten",
            "model_id": "mdl_abc123",
            "status": "running",
        }
        mock_spot = MagicMock()
        mock_spot.name.return_value = "skyserve"
        mock_get_provider.side_effect = lambda name: {
            "baseten": mock_baseten, "skyserve": mock_spot,
        }[name]

        record = self._make_record(
            serverless_provider_name="baseten",
            serverless_metadata={},  # no model_id!
        )
        destroy_hybrid("my-svc", record=record)

        # status() should have been called to find the model_id
        mock_baseten.status.assert_called_once_with("my-svc")
        # destroy() should have been called with the discovered model_id
        mock_baseten.destroy.assert_called_once()
        meta = mock_baseten.destroy.call_args[0][0].metadata
        assert meta["model_id"] == "mdl_abc123"

    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.orchestrator.subprocess")
    @patch("tuna.orchestrator.get_provider")
    def test_destroy_runpod_uses_status_lookup_for_endpoint_id(
        self, mock_get_provider, mock_subprocess, mock_cleanup,
    ):
        """When runpod metadata has no endpoint_id, destroy should call status() to find it."""
        mock_runpod = MagicMock()
        mock_runpod.name.return_value = "runpod"
        mock_runpod.status.return_value = {
            "provider": "runpod",
            "endpoint_id": "ep_xyz789",
            "status": "running",
        }
        mock_spot = MagicMock()
        mock_spot.name.return_value = "skyserve"
        mock_get_provider.side_effect = lambda name: {
            "runpod": mock_runpod, "skyserve": mock_spot,
        }[name]

        record = self._make_record(
            serverless_provider_name="runpod",
            serverless_metadata={},  # no endpoint_id!
        )
        destroy_hybrid("my-svc", record=record)

        mock_runpod.status.assert_called_once_with("my-svc")
        mock_runpod.destroy.assert_called_once()
        meta = mock_runpod.destroy.call_args[0][0].metadata
        assert meta["endpoint_id"] == "ep_xyz789"

    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.orchestrator.subprocess")
    @patch("tuna.orchestrator.get_provider")
    def test_destroy_modal_with_empty_metadata_uses_default_app_name(
        self, mock_get_provider, mock_subprocess, mock_cleanup,
    ):
        """Modal destroy with empty metadata should fall back to default app_name."""
        mock_modal = MagicMock()
        mock_modal.name.return_value = "modal"
        mock_spot = MagicMock()
        mock_spot.name.return_value = "skyserve"
        mock_get_provider.side_effect = lambda name: {
            "modal": mock_modal, "skyserve": mock_spot,
        }[name]

        record = self._make_record(
            serverless_provider_name="modal",
            serverless_metadata={},  # empty!
        )
        destroy_hybrid("my-svc", record=record)

        mock_modal.destroy.assert_called_once()
        meta = mock_modal.destroy.call_args[0][0].metadata
        assert meta["app_name"] == "my-svc-serverless"
        assert meta["service_name"] == "my-svc-serverless"


class TestCmdDeployKeyboardInterrupt:
    """Tests that cmd_deploy saves deployment state even on interruption."""

    @patch("tuna.providers.registry.ensure_provider_registered")
    @patch("tuna.state.save_deployment")
    @patch("tuna.orchestrator.launch_hybrid", side_effect=KeyboardInterrupt)
    def test_ctrl_c_during_deploy_saves_record(
        self, mock_launch, mock_save, mock_ensure,
    ):
        """KeyboardInterrupt during launch_hybrid should still call save_deployment."""
        import argparse
        from tuna.__main__ import cmd_deploy

        args = argparse.Namespace(
            model="m", gpu="g", gpu_count=1, tp_size=1, max_model_len=4096,
            serverless_provider="modal", spots_cloud="aws", region=None,
            concurrency=None, workers_max=None, no_scale_to_zero=False,
            scaling_policy=None, service_name="test-svc", public=False,
            use_different_vm_for_lb=False, gcp_project=None, gcp_region=None,
            cold_start_mode="fast_boot",
        )

        try:
            cmd_deploy(args)
        except SystemExit:
            pass  # Expected — total failure exits

        mock_save.assert_called_once()
        _, result = mock_save.call_args[0]
        assert isinstance(result, HybridDeployment)
