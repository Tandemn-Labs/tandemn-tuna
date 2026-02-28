"""Tests for tuna.benchmark.providers — provider-specific helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tuna.benchmark.providers import (
    get_auth_headers,
    is_cold,
    supports_log_phases,
    trigger_cold_start,
    validate_provider,
)


# -- validate_provider --


def test_azure_rejected():
    with pytest.raises(ValueError, match="30\\+ min"):
        validate_provider("azure")


def test_modal_allowed():
    validate_provider("modal")  # should not raise


def test_runpod_allowed():
    validate_provider("runpod")


# -- get_auth_headers --


def test_runpod_bearer_auth(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key-123")
    assert get_auth_headers("runpod") == {"Authorization": "Bearer rp-key-123"}


def test_baseten_api_key_auth(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "bt-key-456")
    assert get_auth_headers("baseten") == {"Authorization": "Api-Key bt-key-456"}


def test_modal_no_auth():
    assert get_auth_headers("modal") == {}


def test_runpod_missing_key_raises(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
        get_auth_headers("runpod")


def test_baseten_missing_key_raises(monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="BASETEN_API_KEY"):
        get_auth_headers("baseten")


# -- is_cold --


@patch("tuna.benchmark.providers.requests.get")
def test_runpod_cold_workers_zero(mock_get):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"workers": {"ready": 0, "running": 0, "initializing": 0}}
    mock_get.return_value = resp
    assert is_cold("runpod", "http://health", {}) is True


@patch("tuna.benchmark.providers.requests.get")
def test_runpod_warm_workers_ready(mock_get):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"workers": {"ready": 1, "running": 0, "initializing": 0}}
    mock_get.return_value = resp
    assert is_cold("runpod", "http://health", {}) is False


@patch("tuna.benchmark.providers.requests.get")
def test_runpod_warm_workers_initializing(mock_get):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"workers": {"ready": 0, "running": 0, "initializing": 1}}
    mock_get.return_value = resp
    assert is_cold("runpod", "http://health", {}) is False


@patch("tuna.benchmark.providers.requests.get")
def test_modal_cold_connection_error(mock_get):
    import requests as req

    mock_get.side_effect = req.ConnectionError("refused")
    assert is_cold("modal", "http://health", {}) is True


@patch("tuna.benchmark.providers.requests.get")
def test_modal_cold_503(mock_get):
    resp = MagicMock(status_code=503)
    mock_get.return_value = resp
    assert is_cold("modal", "http://health", {}) is True


@patch("tuna.benchmark.providers.requests.get")
def test_modal_warm_200(mock_get):
    resp = MagicMock(status_code=200)
    mock_get.return_value = resp
    assert is_cold("modal", "http://health", {}) is False


# -- trigger_cold_start --


@patch("tuna.benchmark.providers.requests.post")
def test_runpod_trigger_posts_inference(mock_post):
    trigger_cold_start(
        "runpod",
        "https://api.runpod.ai/v2/abc",
        "https://api.runpod.ai/v2/abc/health",
        "Qwen/Qwen3-0.6B",
        {"Authorization": "Bearer key"},
    )
    mock_post.assert_called_once()
    call_url = mock_post.call_args[0][0]
    assert call_url.endswith("/v1/chat/completions")
    body = mock_post.call_args[1]["json"]
    assert body["model"] == "Qwen/Qwen3-0.6B"


@patch("tuna.benchmark.providers.requests.get")
def test_modal_trigger_gets_health(mock_get):
    trigger_cold_start(
        "modal",
        "https://modal.run/endpoint",
        "https://modal.run/endpoint/health",
        "Qwen/Qwen3-0.6B",
        {},
    )
    mock_get.assert_called_once()
    assert mock_get.call_args[0][0] == "https://modal.run/endpoint/health"


# -- supports_log_phases --


def test_modal_supports_logs():
    assert supports_log_phases("modal") is True


def test_runpod_no_logs():
    assert supports_log_phases("runpod") is False


def test_cloudrun_supports_logs():
    assert supports_log_phases("cloudrun") is True
