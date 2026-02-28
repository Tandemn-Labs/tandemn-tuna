"""Tests for tuna.benchmark.log_watchers — log phase extraction."""

from __future__ import annotations

import pytest

from tuna.benchmark.log_watchers import (
    BasetenLogWatcher,
    CloudRunLogWatcher,
    CerebriumLogWatcher,
    LogPhases,
    LogWatcher,
    ModalLogWatcher,
    create_log_watcher,
)


# -- LogWatcher._process_line --


class _FakeWatcher(LogWatcher):
    """Concrete subclass for unit-testing _process_line."""

    def _stream_lines(self):
        return iter([])


def test_log_watcher_first_line_is_container_start():
    w = _FakeWatcher()
    w._process_line(1000.0, "some random line")
    assert w.phases.container_start == 1000.0


def test_log_watcher_pattern_model_load():
    w = _FakeWatcher()
    w._process_line(1000.0, "starting up")
    w._process_line(1005.0, "Loading model weights...")
    assert w.phases.model_load_start == 1005.0


def test_log_watcher_pattern_ready():
    w = _FakeWatcher()
    w._process_line(1000.0, "starting up")
    w._process_line(1010.0, "Uvicorn running on 0.0.0.0:8000")
    assert w.phases.ready == 1010.0


def test_log_watcher_full_sequence():
    w = _FakeWatcher()
    w._process_line(100.0, "container init")
    w._process_line(105.0, "Starting to load model")
    w._process_line(120.0, "Application startup complete")

    assert w.phases.container_start == 100.0
    assert w.phases.model_load_start == 105.0
    assert w.phases.ready == 120.0


def test_log_watcher_only_first_match_recorded():
    """Second matching line should not overwrite the first."""
    w = _FakeWatcher()
    w._process_line(100.0, "Loading model first time")
    w._process_line(110.0, "Loading model second time")
    assert w.phases.model_load_start == 100.0


# -- create_log_watcher factory --


def test_create_log_watcher_modal():
    watcher = create_log_watcher("modal", {"app_name": "my-app"})
    assert isinstance(watcher, ModalLogWatcher)
    assert watcher.app_name == "my-app"


def test_create_log_watcher_cloudrun():
    meta = {"service_name": "svc", "project_id": "proj", "region": "us-central1"}
    watcher = create_log_watcher("cloudrun", meta)
    assert isinstance(watcher, CloudRunLogWatcher)


def test_create_log_watcher_cerebrium():
    watcher = create_log_watcher("cerebrium", {"service_name": "my-svc"})
    assert isinstance(watcher, CerebriumLogWatcher)


def test_create_log_watcher_baseten():
    meta = {"model_id": "abc123", "deployment_id": "dep456"}
    watcher = create_log_watcher("baseten", meta)
    assert isinstance(watcher, BasetenLogWatcher)
    assert watcher.model_id == "abc123"
    assert watcher.deployment_id == "dep456"


def test_create_log_watcher_baseten_missing_ids():
    assert create_log_watcher("baseten", {}) is None


def test_create_log_watcher_runpod_returns_none():
    assert create_log_watcher("runpod", {}) is None


def test_create_log_watcher_modal_missing_app_name():
    assert create_log_watcher("modal", {}) is None
