"""Tests for tuna.sky_sdk — thin SkyPilot SDK wrappers."""

from unittest.mock import MagicMock, patch

from tuna.sky_sdk import (
    cluster_down,
    cluster_launch,
    cluster_status,
    serve_down,
    serve_status,
    serve_up,
    task_from_yaml_str,
)


class TestServeUp:
    @patch("tuna.sky_sdk.sky")
    def test_returns_name_and_endpoint(self, mock_sky):
        mock_task = MagicMock()
        mock_sky.serve.up.return_value = "req-1"
        mock_sky.get.return_value = ("my-service", "http://1.2.3.4:30001")

        name, endpoint = serve_up(mock_task, "my-service")

        mock_sky.serve.up.assert_called_once_with(mock_task, "my-service")
        mock_sky.get.assert_called_once_with("req-1")
        assert name == "my-service"
        assert endpoint == "http://1.2.3.4:30001"


class TestServeDown:
    @patch("tuna.sky_sdk.sky")
    def test_calls_down_and_blocks(self, mock_sky):
        mock_sky.serve.down.return_value = "req-2"
        mock_sky.get.return_value = None

        serve_down("my-service")

        mock_sky.serve.down.assert_called_once_with("my-service", purge=False)
        mock_sky.get.assert_called_once_with("req-2")

    @patch("tuna.sky_sdk.sky")
    def test_purge_flag(self, mock_sky):
        mock_sky.serve.down.return_value = "req-3"
        mock_sky.get.return_value = None

        serve_down("my-service", purge=True)

        mock_sky.serve.down.assert_called_once_with("my-service", purge=True)


class TestServeStatus:
    @patch("tuna.sky_sdk.sky")
    def test_returns_list_of_dicts(self, mock_sky):
        mock_sky.serve.status.return_value = "req-4"
        mock_sky.get.return_value = [{"name": "svc-1", "status": "READY"}]

        result = serve_status("svc-1")

        mock_sky.serve.status.assert_called_once_with("svc-1")
        assert result == [{"name": "svc-1", "status": "READY"}]

    @patch("tuna.sky_sdk.sky")
    def test_no_args_queries_all(self, mock_sky):
        mock_sky.serve.status.return_value = "req-5"
        mock_sky.get.return_value = []

        result = serve_status()

        mock_sky.serve.status.assert_called_once_with(None)
        assert result == []


class TestClusterLaunch:
    @patch("tuna.sky_sdk.sky")
    def test_returns_job_id_and_handle(self, mock_sky):
        mock_task = MagicMock()
        mock_handle = MagicMock()
        mock_handle.head_ip = "10.0.0.1"
        mock_sky.launch.return_value = "req-6"
        mock_sky.get.return_value = (42, mock_handle)

        job_id, handle = cluster_launch(mock_task, cluster_name="my-cluster", down=True)

        mock_sky.launch.assert_called_once_with(
            mock_task, cluster_name="my-cluster", down=True,
        )
        assert job_id == 42
        assert handle.head_ip == "10.0.0.1"


class TestClusterStatus:
    @patch("tuna.sky_sdk.sky")
    def test_returns_status_responses(self, mock_sky):
        entry = MagicMock()
        entry.name = "sky-serve-controller-abc"
        mock_sky.status.return_value = "req-7"
        mock_sky.get.return_value = [entry]

        result = cluster_status()

        mock_sky.status.assert_called_once_with(None)
        assert len(result) == 1
        assert result[0].name == "sky-serve-controller-abc"

    @patch("tuna.sky_sdk.sky")
    def test_with_cluster_names(self, mock_sky):
        mock_sky.status.return_value = "req-8"
        mock_sky.get.return_value = []

        cluster_status(cluster_names=["my-cluster"])

        mock_sky.status.assert_called_once_with(["my-cluster"])


class TestClusterDown:
    @patch("tuna.sky_sdk.sky")
    def test_calls_down_and_blocks(self, mock_sky):
        mock_sky.down.return_value = "req-9"
        mock_sky.get.return_value = None

        cluster_down("my-cluster")

        mock_sky.down.assert_called_once_with("my-cluster", purge=False)
        mock_sky.get.assert_called_once_with("req-9")


class TestTaskFromYamlStr:
    @patch("tuna.sky_sdk.sky")
    def test_parses_yaml_and_creates_task(self, mock_sky):
        mock_task = MagicMock()
        mock_sky.Task.from_yaml_config.return_value = mock_task

        yaml_str = "resources:\n  accelerators: L40S:1\nrun: echo hello\n"
        result = task_from_yaml_str(yaml_str)

        mock_sky.Task.from_yaml_config.assert_called_once_with(
            {"resources": {"accelerators": "L40S:1"}, "run": "echo hello"}
        )
        assert result is mock_task
