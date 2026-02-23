"""Tests for tuna.spot.sky_launcher — plan() only, no real deploys."""

import yaml
from unittest.mock import MagicMock, patch

from sky import ClusterStatus
from sky.serve import ServiceStatus

from tuna.models import DeployRequest, DeploymentResult
from tuna.providers.base import InferenceProvider
from tuna.scaling import ScalingPolicy, SpotScaling, ServerlessScaling
from tuna.spot.sky_launcher import SkyLauncher


class TestSkyLauncherProvider:
    def test_is_inference_provider(self):
        assert issubclass(SkyLauncher, InferenceProvider)
        assert isinstance(SkyLauncher(), InferenceProvider)

    def test_name(self):
        launcher = SkyLauncher()
        assert launcher.name() == "skyserve"


class TestSkyLauncherPlan:
    def setup_method(self):
        self.launcher = SkyLauncher()
        self.request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="L40S",
            service_name="test-svc",
        )
        self.vllm_cmd = (
            "vllm serve Qwen/Qwen3-0.6B "
            "--host 0.0.0.0 --port 8001 --max-model-len 4096 "
            "--served-model-name Qwen/Qwen3-0.6B --tensor-parallel-size 1 "
            "--disable-log-requests --uvicorn-log-level info --enforce-eager"
        )

    def test_plan_provider(self):
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        assert plan.provider == "skyserve"

    def test_plan_service_name(self):
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        assert plan.metadata["service_name"] == "test-svc-spot"

    def test_plan_rendered_is_valid_yaml(self):
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert "service" in parsed
        assert "resources" in parsed
        assert "run" in parsed

    def test_plan_gpu_in_resources(self):
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert "L40S:1" in parsed["resources"]["accelerators"]

    def test_plan_use_spot(self):
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["resources"]["use_spot"] is True

    def test_plan_scale_to_zero(self):
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["service"]["replica_policy"]["min_replicas"] == 0

    def test_plan_no_scale_to_zero(self):
        self.request.scaling.spot.min_replicas = 1
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["service"]["replica_policy"]["min_replicas"] == 1

    def test_plan_vllm_cmd_in_run(self):
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert "Qwen/Qwen3-0.6B" in parsed["run"]

    def test_plan_port(self):
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["resources"]["ports"] == 8001

    def test_plan_with_region(self):
        self.request.region = "us-west-2"
        self.request.spots_cloud = "aws"
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        any_of = parsed["resources"]["any_of"]
        assert any_of[0]["infra"] == "aws/us-west-2"

    def test_plan_with_azure_region(self):
        self.request.region = "eastus"
        self.request.spots_cloud = "azure"
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert parsed["resources"]["any_of"][0]["infra"] == "azure/eastus"

    def test_plan_without_region_pins_cloud(self):
        self.request.region = None
        self.request.spots_cloud = "aws"
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert "any_of" not in parsed["resources"]
        assert parsed["resources"]["cloud"] == "aws"

    def test_plan_multi_gpu(self):
        self.request.gpu_count = 4
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert "L40S:4" in parsed["resources"]["accelerators"]


class TestSkyLauncherDeploy:
    def setup_method(self):
        self.launcher = SkyLauncher()

    @patch("tuna.spot.sky_launcher.serve_up", return_value=("my-svc-spot", "http://1.2.3.4:30001"))
    @patch("tuna.spot.sky_launcher.task_from_yaml_str")
    def test_deploy_success(self, mock_task, mock_serve_up):
        from tuna.models import ProviderPlan
        plan = ProviderPlan(
            provider="skyserve",
            rendered_script="resources:\n  accelerators: L40S:1\n",
            metadata={"service_name": "my-svc-spot"},
        )
        result = self.launcher.deploy(plan)

        assert result.error is None
        assert result.endpoint_url == "http://1.2.3.4:30001"
        assert result.health_url == "http://1.2.3.4:30001/health"
        assert result.metadata["service_name"] == "my-svc-spot"

    @patch("tuna.spot.sky_launcher.serve_up", return_value=("my-svc-spot", ""))
    @patch("tuna.spot.sky_launcher.task_from_yaml_str")
    def test_deploy_no_endpoint(self, mock_task, mock_serve_up):
        from tuna.models import ProviderPlan
        plan = ProviderPlan(
            provider="skyserve",
            rendered_script="resources:\n  accelerators: L40S:1\n",
            metadata={"service_name": "my-svc-spot"},
        )
        result = self.launcher.deploy(plan)

        assert result.error is not None
        assert "Endpoint not yet available" in result.error

    @patch("tuna.spot.sky_launcher.serve_up", side_effect=RuntimeError("launch failed"))
    @patch("tuna.spot.sky_launcher.task_from_yaml_str")
    def test_deploy_exception(self, mock_task, mock_serve_up):
        from tuna.models import ProviderPlan
        plan = ProviderPlan(
            provider="skyserve",
            rendered_script="resources:\n  accelerators: L40S:1\n",
            metadata={"service_name": "my-svc-spot"},
        )
        result = self.launcher.deploy(plan)

        assert result.error is not None
        assert "launch failed" in result.error


class TestSkyLauncherDestroy:
    def setup_method(self):
        self.launcher = SkyLauncher()
        self.result = DeploymentResult(
            provider="skyserve",
            metadata={"service_name": "test-svc-spot"},
        )

    @patch("tuna.spot.sky_launcher.SkyLauncher._controller_is_init", return_value=False)
    @patch("tuna.spot.sky_launcher.serve_status", return_value=[])
    @patch("tuna.spot.sky_launcher.serve_down")
    def test_destroy_confirms_service_gone(self, mock_down, mock_status, mock_ctrl):
        """destroy() should verify the service is actually deleted."""
        self.launcher.destroy(self.result)

        mock_down.assert_called_once_with("test-svc-spot")
        mock_status.assert_called_once_with("test-svc-spot")

    @patch("tuna.spot.sky_launcher.time")
    @patch("tuna.spot.sky_launcher.cluster_status")
    @patch("tuna.spot.sky_launcher.serve_status")
    @patch("tuna.spot.sky_launcher.serve_down")
    def test_destroy_retries_when_controller_init(
        self, mock_down, mock_serve_status, mock_cluster_status, mock_time,
    ):
        """When controller is INIT, sky reports no services but destroy should retry."""
        # Build a mock controller entry in INIT state
        ctrl_entry = MagicMock()
        ctrl_entry.name = "sky-serve-controller-abc"
        ctrl_entry.status = ClusterStatus.INIT

        # Build a mock service entry that appears after controller is ready
        svc_entry = {"status": ServiceStatus.READY, "name": "test-svc-spot"}

        # Sequence: 1st empty (but controller INIT), 2nd found, 3rd gone (no controller)
        mock_serve_status.side_effect = [
            [],                 # 1st check: empty (controller INIT)
            [svc_entry],        # 2nd check: service found (not gone)
            [],                 # 3rd check: empty (gone for real)
        ]
        mock_cluster_status.side_effect = [
            [ctrl_entry],       # 1st: controller in INIT
            [],                 # 3rd: no controller
        ]

        self.launcher.destroy(self.result)

        assert mock_down.call_count == 3

    @patch("tuna.spot.sky_launcher.SkyLauncher._controller_is_init", return_value=False)
    @patch("tuna.spot.sky_launcher.time")
    @patch("tuna.spot.sky_launcher.serve_status")
    @patch("tuna.spot.sky_launcher.serve_down")
    def test_destroy_waits_for_shutting_down(
        self, mock_down, mock_serve_status, mock_time, mock_ctrl,
    ):
        """If service is SHUTTING_DOWN, destroy should wait and recheck."""
        mock_serve_status.side_effect = [
            [{"status": ServiceStatus.SHUTTING_DOWN}],  # still shutting down
            [],                                          # gone
        ]
        self.launcher.destroy(self.result)

        mock_time.sleep.assert_called()

    @patch("tuna.spot.sky_launcher.serve_down")
    def test_destroy_no_service_name(self, mock_serve_down):
        """destroy() with no service_name in metadata should skip."""
        result = DeploymentResult(provider="skyserve", metadata={})
        self.launcher.destroy(result)
        mock_serve_down.assert_not_called()


class TestSkyLauncherStatus:
    def setup_method(self):
        self.launcher = SkyLauncher()

    @patch("tuna.spot.sky_launcher.serve_status")
    def test_status_returns_structured_dict(self, mock_serve_status):
        mock_serve_status.return_value = [
            {"status": ServiceStatus.READY, "endpoint": "http://1.2.3.4:30001"},
        ]
        result = self.launcher.status("my-svc")

        assert result["provider"] == "skyserve"
        assert result["service_name"] == "my-svc-spot"
        assert result["status"] == "READY"
        assert result["endpoint"] == "http://1.2.3.4:30001"

    @patch("tuna.spot.sky_launcher.serve_status", return_value=[])
    def test_status_not_found(self, mock_serve_status):
        result = self.launcher.status("my-svc")

        assert result["status"] == "NOT_FOUND"

    @patch("tuna.spot.sky_launcher.serve_status", side_effect=RuntimeError("oops"))
    def test_status_error(self, mock_serve_status):
        result = self.launcher.status("my-svc")

        assert "error" in result
        assert "oops" in result["error"]
