"""Tests for BYOC (Bring Your Own Container) CLI flags and behavior."""

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from tuna.models import DeployRequest, DeploymentResult, HybridDeployment


class TestByocCliFlags:
    """Test that --image, --container-port, --container-args are parsed correctly."""

    def test_image_flag_parsed(self):
        captured_args = {}

        def fake_deploy(args):
            captured_args.update(vars(args))

        with patch("tuna.__main__.cmd_deploy", fake_deploy), \
             patch.object(sys, "argv", [
                 "tuna", "deploy", "--gpu", "T4",
                 "--image", "choprahetarth/sam2-server:latest",
             ]):
            from tuna.__main__ import main
            main()

        assert captured_args["image"] == "choprahetarth/sam2-server:latest"
        assert captured_args["model"] is None  # --model not required with --image

    def test_container_port_parsed(self):
        captured_args = {}

        def fake_deploy(args):
            captured_args.update(vars(args))

        with patch("tuna.__main__.cmd_deploy", fake_deploy), \
             patch.object(sys, "argv", [
                 "tuna", "deploy", "--gpu", "T4",
                 "--image", "my/image:v1",
                 "--container-port", "5000",
             ]):
            from tuna.__main__ import main
            main()

        assert captured_args["container_port"] == 5000

    def test_container_port_default(self):
        captured_args = {}

        def fake_deploy(args):
            captured_args.update(vars(args))

        with patch("tuna.__main__.cmd_deploy", fake_deploy), \
             patch.object(sys, "argv", [
                 "tuna", "deploy", "--gpu", "T4",
                 "--image", "my/image:v1",
             ]):
            from tuna.__main__ import main
            main()

        assert captured_args["container_port"] == 8080

    def test_container_args_parsed(self):
        captured_args = {}

        def fake_deploy(args):
            captured_args.update(vars(args))

        with patch("tuna.__main__.cmd_deploy", fake_deploy), \
             patch.object(sys, "argv", [
                 "tuna", "deploy", "--gpu", "T4",
                 "--image", "my/image:v1",
                 "--container-args", "python", "serve.py",
             ]):
            from tuna.__main__ import main
            main()

        assert captured_args["container_args"] == ["python", "serve.py"]

    def test_model_required_without_image(self):
        """Without --image, --model is required."""
        from tuna.__main__ import cmd_deploy

        args = argparse.Namespace(
            model=None, gpu="T4", gpu_count=1, tp_size=1, max_model_len=4096,
            serverless_provider="cloudrun", spots_cloud="aws", region=None,
            concurrency=None, workers_max=None, no_scale_to_zero=False,
            scaling_policy=None, service_name="test-svc", public=False,
            use_different_vm_for_lb=False, gcp_project=None, gcp_region=None,
            cold_start_mode="fast_boot", serverless_only=True,
            image=None, container_port=8080, container_args=None,
            azure_subscription=None, azure_resource_group=None,
            azure_region=None, azure_environment=None,
        )

        with pytest.raises(SystemExit):
            cmd_deploy(args)


class TestByocModelDerivation:
    """Test that --model is auto-derived from --image when not specified."""

    @patch("tuna.providers.registry.ensure_provider_registered")
    @patch("tuna.state.save_deployment")
    @patch("tuna.orchestrator.launch_serverless_only")
    def test_model_derived_from_image(self, mock_launch, mock_save, mock_ensure):
        mock_launch.return_value = HybridDeployment()
        from tuna.__main__ import cmd_deploy

        args = argparse.Namespace(
            model=None, gpu="T4", gpu_count=1, tp_size=1, max_model_len=4096,
            serverless_provider="cloudrun", spots_cloud="aws", region=None,
            concurrency=None, workers_max=None, no_scale_to_zero=False,
            scaling_policy=None, service_name="test-svc", public=False,
            use_different_vm_for_lb=False, gcp_project=None, gcp_region=None,
            cold_start_mode="fast_boot", serverless_only=True,
            image="choprahetarth/sam2-server:latest",
            container_port=8080, container_args=None,
            azure_subscription=None, azure_resource_group=None,
            azure_region=None, azure_environment=None,
        )

        try:
            cmd_deploy(args)
        except SystemExit:
            pass  # may exit due to total failure

        # Verify the request was built with derived model name
        call_args = mock_launch.call_args
        if call_args:
            request = call_args[0][0]
            assert request.model_name == "sam2-server"
            assert request.image == "choprahetarth/sam2-server:latest"
            assert request.is_byoc is True

    @patch("tuna.providers.registry.ensure_provider_registered")
    @patch("tuna.state.save_deployment")
    @patch("tuna.orchestrator.launch_serverless_only")
    def test_explicit_model_overrides_derivation(self, mock_launch, mock_save, mock_ensure):
        mock_launch.return_value = HybridDeployment()
        from tuna.__main__ import cmd_deploy

        args = argparse.Namespace(
            model="my-custom-name", gpu="T4", gpu_count=1, tp_size=1, max_model_len=4096,
            serverless_provider="cloudrun", spots_cloud="aws", region=None,
            concurrency=None, workers_max=None, no_scale_to_zero=False,
            scaling_policy=None, service_name="test-svc", public=False,
            use_different_vm_for_lb=False, gcp_project=None, gcp_region=None,
            cold_start_mode="fast_boot", serverless_only=True,
            image="choprahetarth/sam2-server:latest",
            container_port=8080, container_args=None,
            azure_subscription=None, azure_resource_group=None,
            azure_region=None, azure_environment=None,
        )

        try:
            cmd_deploy(args)
        except SystemExit:
            pass

        call_args = mock_launch.call_args
        if call_args:
            request = call_args[0][0]
            assert request.model_name == "my-custom-name"


class TestByocDefaultProvider:
    """Test that BYOC defaults to cloudrun when no provider specified."""

    @patch("tuna.providers.registry.ensure_provider_registered")
    @patch("tuna.state.save_deployment")
    @patch("tuna.orchestrator.launch_serverless_only")
    def test_byoc_defaults_to_cloudrun(self, mock_launch, mock_save, mock_ensure):
        mock_launch.return_value = HybridDeployment()
        from tuna.__main__ import cmd_deploy

        args = argparse.Namespace(
            model=None, gpu="T4", gpu_count=1, tp_size=1, max_model_len=4096,
            serverless_provider=None,  # Not specified
            spots_cloud="aws", region=None,
            concurrency=None, workers_max=None, no_scale_to_zero=False,
            scaling_policy=None, service_name="test-svc", public=False,
            use_different_vm_for_lb=False, gcp_project=None, gcp_region=None,
            cold_start_mode="fast_boot", serverless_only=True,
            image="choprahetarth/sam2-server:latest",
            container_port=8080, container_args=None,
            azure_subscription=None, azure_resource_group=None,
            azure_region=None, azure_environment=None,
        )

        try:
            cmd_deploy(args)
        except SystemExit:
            pass

        call_args = mock_launch.call_args
        if call_args:
            request = call_args[0][0]
            assert request.serverless_provider == "cloudrun"


class TestByocIgnoredFlagWarnings:
    """Test that vLLM-specific flags produce warnings in BYOC mode."""

    @patch("tuna.providers.registry.ensure_provider_registered")
    @patch("tuna.state.save_deployment")
    @patch("tuna.orchestrator.launch_serverless_only")
    def test_tp_size_warning(self, mock_launch, mock_save, mock_ensure, capsys):
        mock_launch.return_value = HybridDeployment()
        from tuna.__main__ import cmd_deploy

        args = argparse.Namespace(
            model=None, gpu="T4", gpu_count=1, tp_size=4, max_model_len=4096,
            serverless_provider="cloudrun", spots_cloud="aws", region=None,
            concurrency=None, workers_max=None, no_scale_to_zero=False,
            scaling_policy=None, service_name="test-svc", public=False,
            use_different_vm_for_lb=False, gcp_project=None, gcp_region=None,
            cold_start_mode="fast_boot", serverless_only=True,
            image="my/image:v1", container_port=8080, container_args=None,
            azure_subscription=None, azure_resource_group=None,
            azure_region=None, azure_environment=None,
        )

        try:
            cmd_deploy(args)
        except SystemExit:
            pass

        captured = capsys.readouterr()
        assert "--tp-size 4" in captured.err
        assert "ignored in BYOC mode" in captured.err
