"""Tests for `tuna destroy` CLI — --all flag and argument validation."""

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from tuna.state import DeploymentRecord


def _make_record(name: str) -> DeploymentRecord:
    return DeploymentRecord(
        service_name=name,
        status="active",
        model_name="Qwen/Qwen3-0.6B",
        gpu="L4",
    )


def _make_args(service_name=None, destroy_all=False) -> argparse.Namespace:
    return argparse.Namespace(service_name=service_name, destroy_all=destroy_all)


class TestDestroyAll:
    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.state.list_deployments", return_value=[])
    def test_destroy_all_no_active(self, mock_list, mock_cleanup, capsys):
        from tuna.__main__ import cmd_destroy

        cmd_destroy(_make_args(destroy_all=True))

        mock_list.assert_called_once_with(status="active")
        assert "No active deployments to destroy." in capsys.readouterr().out

    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.state.update_deployment_status")
    @patch("tuna.orchestrator.destroy_hybrid")
    @patch("tuna.providers.registry.ensure_providers_for_deployment")
    @patch("tuna.state.list_deployments")
    def test_destroy_all_tears_down_each(
        self, mock_list, mock_ensure, mock_destroy, mock_update, mock_cleanup, capsys
    ):
        from tuna.__main__ import cmd_destroy

        records = [_make_record("svc-a"), _make_record("svc-b")]
        mock_list.return_value = records

        cmd_destroy(_make_args(destroy_all=True))

        assert mock_destroy.call_count == 2
        mock_destroy.assert_any_call("svc-a", record=records[0],
                                     skip_controller_cleanup=True)
        mock_destroy.assert_any_call("svc-b", record=records[1],
                                     skip_controller_cleanup=True)
        assert mock_update.call_count == 2
        mock_update.assert_any_call("svc-a", "destroyed")
        mock_update.assert_any_call("svc-b", "destroyed")
        mock_cleanup.assert_called_once()
        out = capsys.readouterr().out
        assert "Destroyed: svc-a" in out
        assert "Destroyed: svc-b" in out

    @patch("tuna.orchestrator._cleanup_serve_controller")
    @patch("tuna.state.update_deployment_status")
    @patch("tuna.orchestrator.destroy_hybrid")
    @patch("tuna.providers.registry.ensure_providers_for_deployment")
    @patch("tuna.state.list_deployments")
    def test_destroy_all_continues_on_failure(
        self, mock_list, mock_ensure, mock_destroy, mock_update, mock_cleanup, capsys
    ):
        from tuna.__main__ import cmd_destroy

        records = [_make_record("svc-fail"), _make_record("svc-ok")]
        mock_list.return_value = records
        mock_destroy.side_effect = [RuntimeError("boom"), None]

        with pytest.raises(SystemExit, match="1"):
            cmd_destroy(_make_args(destroy_all=True))

        # Second deployment still attempted
        assert mock_destroy.call_count == 2
        # Only the successful one gets status update
        mock_update.assert_called_once_with("svc-ok", "destroyed")
        # Controller cleanup still called even with errors
        mock_cleanup.assert_called_once()
        err = capsys.readouterr().err
        assert "Failed to destroy svc-fail" in err


class TestDestroyArgValidation:
    """Argparse mutual exclusion: --service-name and --all can't coexist,
    and at least one is required."""

    def test_destroy_all_with_service_name_errors(self):
        with patch("sys.argv", ["tuna", "destroy", "--all", "--service-name", "foo"]):
            from tuna.__main__ import main
            with pytest.raises(SystemExit, match="2"):
                main()

    def test_destroy_neither_flag_errors(self):
        with patch("sys.argv", ["tuna", "destroy"]):
            from tuna.__main__ import main
            with pytest.raises(SystemExit, match="2"):
                main()


class TestDestroySingle:
    @patch("tuna.state.update_deployment_status")
    @patch("tuna.orchestrator.destroy_hybrid")
    @patch("tuna.providers.registry.ensure_providers_for_deployment")
    @patch("tuna.state.load_deployment")
    def test_destroy_single_unchanged(
        self, mock_load, mock_ensure, mock_destroy, mock_update, capsys
    ):
        from tuna.__main__ import cmd_destroy

        record = _make_record("my-svc")
        mock_load.return_value = record

        cmd_destroy(_make_args(service_name="my-svc"))

        mock_load.assert_called_once_with("my-svc")
        mock_ensure.assert_called_once_with(record)
        mock_destroy.assert_called_once_with("my-svc", record=record)
        mock_update.assert_called_once_with("my-svc", "destroyed")
        assert "Done." in capsys.readouterr().out
