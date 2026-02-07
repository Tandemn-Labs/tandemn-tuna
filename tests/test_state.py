"""Tests for tandemn.state — SQLite deployment persistence."""

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from tandemn.models import DeployRequest, DeploymentResult, HybridDeployment
from tandemn.state import (
    DeploymentRecord,
    _connect,
    _db_path,
    _state_dir,
    list_deployments,
    load_deployment,
    save_deployment,
    update_deployment_status,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_deployments.db"


def _make_request(**kwargs):
    defaults = dict(
        model_name="Qwen/Qwen3-0.6B",
        gpu="L40S",
        service_name="test-svc",
    )
    defaults.update(kwargs)
    return DeployRequest(**defaults)


def _make_result(
    serverless=True,
    spot=True,
    router=True,
):
    sl = (
        DeploymentResult(
            provider="modal",
            endpoint_url="https://modal.run/test",
            metadata={"app_name": "test-svc-serverless"},
        )
        if serverless
        else None
    )
    sp = (
        DeploymentResult(
            provider="skyserve",
            endpoint_url="http://spot:30001",
            metadata={"service_name": "test-svc-spot"},
        )
        if spot
        else None
    )
    rt = (
        DeploymentResult(
            provider="router",
            endpoint_url="http://10.0.0.1:8080",
            metadata={"cluster_name": "test-svc-router"},
        )
        if router
        else None
    )
    return HybridDeployment(
        serverless=sl,
        spot=sp,
        router=rt,
        router_url="http://10.0.0.1:8080" if router else None,
    )


class TestSaveAndLoad:
    def test_save_then_load(self, db_path):
        req = _make_request()
        result = _make_result()
        save_deployment(req, result, db_path=db_path)

        record = load_deployment("test-svc", db_path=db_path)
        assert record is not None
        assert record.service_name == "test-svc"
        assert record.status == "active"
        assert record.model_name == "Qwen/Qwen3-0.6B"
        assert record.gpu == "L40S"
        assert record.serverless_provider == "modal"
        assert record.spots_cloud == "aws"
        assert record.router_url == "http://10.0.0.1:8080"

    def test_load_nonexistent_returns_none(self, db_path):
        record = load_deployment("no-such-svc", db_path=db_path)
        assert record is None

    def test_metadata_roundtrip(self, db_path):
        req = _make_request()
        result = _make_result()
        save_deployment(req, result, db_path=db_path)

        record = load_deployment("test-svc", db_path=db_path)
        assert record.serverless_metadata == {"app_name": "test-svc-serverless"}
        assert record.spot_metadata == {"service_name": "test-svc-spot"}
        assert record.router_metadata == {"cluster_name": "test-svc-router"}

    def test_partial_results_spot_none(self, db_path):
        req = _make_request()
        result = _make_result(spot=False)
        save_deployment(req, result, db_path=db_path)

        record = load_deployment("test-svc", db_path=db_path)
        assert record.spot_provider_name is None
        assert record.spot_endpoint is None
        assert record.spot_metadata == {}

    def test_partial_results_serverless_none(self, db_path):
        req = _make_request()
        result = _make_result(serverless=False)
        save_deployment(req, result, db_path=db_path)

        record = load_deployment("test-svc", db_path=db_path)
        assert record.serverless_provider_name is None
        assert record.serverless_endpoint is None

    def test_save_replaces_existing(self, db_path):
        req = _make_request()
        result1 = _make_result()
        save_deployment(req, result1, db_path=db_path)

        result2 = _make_result(spot=False)
        save_deployment(req, result2, db_path=db_path)

        record = load_deployment("test-svc", db_path=db_path)
        assert record.spot_provider_name is None

    def test_request_json_stored(self, db_path):
        req = _make_request()
        result = _make_result()
        save_deployment(req, result, db_path=db_path)

        record = load_deployment("test-svc", db_path=db_path)
        assert record.request_json
        data = json.loads(record.request_json)
        assert data["model_name"] == "Qwen/Qwen3-0.6B"

    def test_provider_names_stored(self, db_path):
        req = _make_request()
        result = _make_result()
        save_deployment(req, result, db_path=db_path)

        record = load_deployment("test-svc", db_path=db_path)
        assert record.serverless_provider_name == "modal"
        assert record.spot_provider_name == "skyserve"

    def test_endpoints_stored(self, db_path):
        req = _make_request()
        result = _make_result()
        save_deployment(req, result, db_path=db_path)

        record = load_deployment("test-svc", db_path=db_path)
        assert record.serverless_endpoint == "https://modal.run/test"
        assert record.spot_endpoint == "http://spot:30001"
        assert record.router_endpoint == "http://10.0.0.1:8080"

    def test_gpu_count_stored(self, db_path):
        req = _make_request(gpu_count=4)
        result = _make_result()
        save_deployment(req, result, db_path=db_path)

        record = load_deployment("test-svc", db_path=db_path)
        assert record.gpu_count == 4

    def test_region_stored(self, db_path):
        req = _make_request(region="us-west-2")
        result = _make_result()
        save_deployment(req, result, db_path=db_path)

        record = load_deployment("test-svc", db_path=db_path)
        assert record.region == "us-west-2"


class TestUpdateStatus:
    def test_update_to_destroyed(self, db_path):
        req = _make_request()
        result = _make_result()
        save_deployment(req, result, db_path=db_path)

        update_deployment_status("test-svc", "destroyed", db_path=db_path)

        record = load_deployment("test-svc", db_path=db_path)
        assert record.status == "destroyed"

    def test_update_nonexistent_is_noop(self, db_path):
        # Should not raise
        update_deployment_status("no-such-svc", "destroyed", db_path=db_path)
        record = load_deployment("no-such-svc", db_path=db_path)
        assert record is None

    def test_updated_at_changes(self, db_path):
        req = _make_request()
        result = _make_result()
        save_deployment(req, result, db_path=db_path)

        record_before = load_deployment("test-svc", db_path=db_path)
        update_deployment_status("test-svc", "destroyed", db_path=db_path)
        record_after = load_deployment("test-svc", db_path=db_path)

        assert record_after.updated_at >= record_before.updated_at


class TestListDeployments:
    def test_list_all(self, db_path):
        for name in ("svc-a", "svc-b", "svc-c"):
            req = _make_request(service_name=name)
            result = _make_result()
            save_deployment(req, result, db_path=db_path)

        records = list_deployments(db_path=db_path)
        assert len(records) == 3
        names = {r.service_name for r in records}
        assert names == {"svc-a", "svc-b", "svc-c"}

    def test_filter_by_status(self, db_path):
        for name in ("svc-a", "svc-b"):
            req = _make_request(service_name=name)
            result = _make_result()
            save_deployment(req, result, db_path=db_path)

        update_deployment_status("svc-a", "destroyed", db_path=db_path)

        active = list_deployments(status="active", db_path=db_path)
        assert len(active) == 1
        assert active[0].service_name == "svc-b"

        destroyed = list_deployments(status="destroyed", db_path=db_path)
        assert len(destroyed) == 1
        assert destroyed[0].service_name == "svc-a"

    def test_list_empty(self, db_path):
        records = list_deployments(db_path=db_path)
        assert records == []


class TestConnectionAndSchema:
    def test_wal_mode_enabled(self, db_path):
        conn = _connect(db_path)
        try:
            result = conn.execute("PRAGMA journal_mode").fetchone()
            assert result[0] == "wal"
        finally:
            conn.close()

    def test_idempotent_schema_creation(self, db_path):
        conn1 = _connect(db_path)
        conn1.close()
        conn2 = _connect(db_path)
        conn2.close()
        # Should not raise

    def test_env_var_override(self, tmp_path):
        custom_dir = tmp_path / "custom_state"
        with patch.dict(os.environ, {"TANDEMN_STATE_DIR": str(custom_dir)}):
            assert _state_dir() == custom_dir
            assert _db_path() == custom_dir / "deployments.db"

    def test_default_state_dir(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove TANDEMN_STATE_DIR if present
            os.environ.pop("TANDEMN_STATE_DIR", None)
            assert _state_dir() == Path.home() / ".tandemn"

    def test_creates_parent_directories(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "test.db"
        conn = _connect(deep_path)
        conn.close()
        assert deep_path.exists()
