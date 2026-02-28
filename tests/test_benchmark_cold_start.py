"""Tests for tuna.benchmark.cold_start — core benchmark logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tuna.benchmark.cold_start import (
    RunResult,
    _find_existing_deployment,
    _mean_run,
    _record_to_result,
    _wait_for_cold,
)
from tuna.state import DeploymentRecord


# -- _record_to_result --


def test_record_to_result_maps_fields():
    record = DeploymentRecord(
        service_name="svc-1",
        serverless_provider_name="modal",
        serverless_endpoint="https://modal.run/ep",
        serverless_metadata={"app_name": "my-app"},
    )
    result = _record_to_result(record)
    assert result.provider == "modal"
    assert result.endpoint_url == "https://modal.run/ep"
    assert result.health_url == "https://modal.run/ep/health"
    assert result.metadata == {"app_name": "my-app"}


def test_record_to_result_none_endpoint():
    record = DeploymentRecord(
        service_name="svc-2",
        serverless_provider_name="runpod",
        serverless_endpoint=None,
    )
    result = _record_to_result(record)
    assert result.endpoint_url is None
    assert result.health_url is None


def test_record_to_result_empty_metadata():
    record = DeploymentRecord(
        service_name="svc-3",
        serverless_provider_name="modal",
        serverless_endpoint="https://modal.run/ep",
        serverless_metadata=None,
    )
    result = _record_to_result(record)
    assert result.metadata == {}


# -- _mean_run --


def _make_run(**kwargs) -> RunResult:
    defaults = dict(
        scenario="warm_cold_start",
        provider="modal",
        gpu="T4",
        total_s=10.0,
    )
    defaults.update(kwargs)
    return RunResult(**defaults)


def test_mean_all_values():
    r1 = _make_run(total_s=10.0, health_ready_s=5.0, container_boot_s=2.0, model_load_s=3.0)
    r2 = _make_run(total_s=20.0, health_ready_s=7.0, container_boot_s=4.0, model_load_s=5.0)
    avg = _mean_run([r1, r2])
    assert avg.total_s == 15.0
    assert avg.health_ready_s == 6.0
    assert avg.container_boot_s == 3.0
    assert avg.model_load_s == 4.0


def test_mean_some_none():
    r1 = _make_run(total_s=10.0, container_boot_s=2.0, model_load_s=None)
    r2 = _make_run(total_s=20.0, container_boot_s=None, model_load_s=6.0)
    avg = _mean_run([r1, r2])
    assert avg.total_s == 15.0
    assert avg.container_boot_s == 2.0  # only one non-None value
    assert avg.model_load_s == 6.0  # only one non-None value


def test_mean_all_none():
    r1 = _make_run(total_s=10.0, container_boot_s=None, model_load_s=None)
    r2 = _make_run(total_s=20.0, container_boot_s=None, model_load_s=None)
    avg = _mean_run([r1, r2])
    assert avg.container_boot_s is None
    assert avg.model_load_s is None


def test_mean_single_run():
    r = _make_run(total_s=10.0, health_ready_s=5.0)
    avg = _mean_run([r])
    assert avg is r  # single run returns itself


# -- _wait_for_cold --


@patch("tuna.benchmark.cold_start.is_cold")
@patch("tuna.benchmark.cold_start.time.sleep")
def test_wait_for_cold_consecutive(mock_sleep, mock_is_cold):
    """3 consecutive cold checks → True."""
    mock_is_cold.return_value = True
    result = _wait_for_cold("modal", "http://h", {}, timeout=60, cooldown=0)
    assert result is True
    assert mock_is_cold.call_count == 3


@patch("tuna.benchmark.cold_start.is_cold")
@patch("tuna.benchmark.cold_start.time.sleep")
def test_wait_for_cold_runpod_immediate(mock_sleep, mock_is_cold):
    """RunPod: single cold check → True (no consecutive needed)."""
    mock_is_cold.return_value = True
    result = _wait_for_cold("runpod", "http://h", {}, timeout=60, cooldown=0)
    assert result is True
    assert mock_is_cold.call_count == 1


@patch("tuna.benchmark.cold_start.is_cold")
@patch("tuna.benchmark.cold_start.time.sleep")
@patch("tuna.benchmark.cold_start.time.monotonic")
def test_wait_for_cold_timeout(mock_mono, mock_sleep, mock_is_cold):
    """Never cold → returns False after timeout."""
    # Simulate: start at 0, each call advances 10s, timeout at 30
    call_count = 0
    def mono_side():
        nonlocal call_count
        call_count += 1
        return call_count * 10.0
    mock_mono.side_effect = mono_side
    mock_is_cold.return_value = False
    result = _wait_for_cold("modal", "http://h", {}, timeout=30, cooldown=0)
    assert result is False


@patch("tuna.benchmark.cold_start.is_cold")
@patch("tuna.benchmark.cold_start.time.sleep")
def test_wait_for_cold_interrupted(mock_sleep, mock_is_cold):
    """Cold interrupted by warm → resets counter, eventually succeeds."""
    # cold, cold, warm, cold, cold, cold → success
    mock_is_cold.side_effect = [True, True, False, True, True, True]
    result = _wait_for_cold("modal", "http://h", {}, timeout=300, cooldown=0)
    assert result is True
    assert mock_is_cold.call_count == 6


# -- _find_existing_deployment --


@patch("tuna.benchmark.cold_start.list_deployments")
def test_find_existing_deployment_match(mock_list):
    record = DeploymentRecord(
        service_name="svc-1",
        serverless_provider_name="modal",
        model_name="Qwen/Qwen3-0.6B",
    )
    mock_list.return_value = [record]
    found = _find_existing_deployment("modal", "Qwen/Qwen3-0.6B")
    assert found is record


@patch("tuna.benchmark.cold_start.list_deployments")
def test_find_existing_deployment_no_match(mock_list):
    record = DeploymentRecord(
        service_name="svc-1",
        serverless_provider_name="runpod",
        model_name="Qwen/Qwen3-0.6B",
    )
    mock_list.return_value = [record]
    found = _find_existing_deployment("modal", "Qwen/Qwen3-0.6B")
    assert found is None


@patch("tuna.benchmark.cold_start.list_deployments")
def test_find_existing_deployment_empty(mock_list):
    mock_list.return_value = []
    found = _find_existing_deployment("modal", "Qwen/Qwen3-0.6B")
    assert found is None
