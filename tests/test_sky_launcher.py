"""Tests for tuna.spot.sky_launcher — plan() only, no real deploys."""

import subprocess
import yaml
from unittest.mock import MagicMock, patch, call

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

    def test_plan_multi_gpu(self):
        self.request.gpu_count = 4
        plan = self.launcher.plan(self.request, self.vllm_cmd)
        parsed = yaml.safe_load(plan.rendered_script)
        assert "L40S:4" in parsed["resources"]["accelerators"]


class TestSkyLauncherDestroy:
    def setup_method(self):
        self.launcher = SkyLauncher()
        self.result = DeploymentResult(
            provider="skyserve",
            metadata={"service_name": "test-svc-spot"},
        )

    @patch("tuna.spot.sky_launcher.subprocess")
    def test_destroy_confirms_service_gone(self, mock_subprocess):
        """destroy() should verify the service is actually deleted."""
        # sky serve down succeeds, status confirms gone
        mock_subprocess.run.side_effect = [
            MagicMock(returncode=0),  # sky serve down
            MagicMock(stdout="No existing services.", stderr="", returncode=0),  # status check
        ]
        self.launcher.destroy(self.result)

        assert mock_subprocess.run.call_count == 2

    @patch("tuna.spot.sky_launcher.time")
    @patch("tuna.spot.sky_launcher.subprocess")
    def test_destroy_retries_when_controller_init(self, mock_subprocess, mock_time):
        """When controller is INIT, sky serve down fails silently — destroy should retry."""
        mock_subprocess.run.side_effect = [
            MagicMock(returncode=0),  # 1st sky serve down (silently fails)
            MagicMock(stdout="test-svc-spot  READY", stderr="", returncode=0),  # still there
            MagicMock(returncode=0),  # 2nd sky serve down (works now)
            MagicMock(stdout="No existing services.", stderr="", returncode=0),  # confirmed gone
        ]
        self.launcher.destroy(self.result)

        # Should have called sky serve down twice
        down_calls = [
            c for c in mock_subprocess.run.call_args_list
            if "down" in str(c)
        ]
        assert len(down_calls) == 2

    @patch("tuna.spot.sky_launcher.time")
    @patch("tuna.spot.sky_launcher.subprocess")
    def test_destroy_waits_for_shutting_down(self, mock_subprocess, mock_time):
        """If service is SHUTTING_DOWN, destroy should wait and recheck."""
        mock_subprocess.run.side_effect = [
            MagicMock(returncode=0),  # sky serve down
            MagicMock(stdout="test-svc-spot  SHUTTING_DOWN", stderr="", returncode=0),  # shutting down
            MagicMock(returncode=0),  # retry sky serve down
            MagicMock(stdout="No existing services.", stderr="", returncode=0),  # gone
        ]
        self.launcher.destroy(self.result)

        mock_time.sleep.assert_called()

    @patch("tuna.spot.sky_launcher.subprocess")
    def test_destroy_no_service_name(self, mock_subprocess):
        """destroy() with no service_name in metadata should skip."""
        result = DeploymentResult(provider="skyserve", metadata={})
        self.launcher.destroy(result)

        mock_subprocess.run.assert_not_called()
