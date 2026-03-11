"""Tests for tuna.router.meta_lb — routing logic without real backends."""

import json
import time

import pytest
import requests as req_lib

from tuna.router import meta_lb


def _mock_response(mocker, *, content=b"ok", status_code=200, headers=None):
    """Create a mock response compatible with streaming (iter_content/close)."""
    if headers is None:
        headers = {}
    resp = mocker.Mock(status_code=status_code, headers=headers)
    resp.iter_content.return_value = iter([content])
    return resp


@pytest.fixture
def client():
    """Flask test client with clean state per test."""
    meta_lb.app.config["TESTING"] = True

    # Reset mutable state
    meta_lb._serverless_base_url = ""
    meta_lb._skyserve_base_url = ""
    meta_lb._spot_state = meta_lb.SpotState.COLD
    meta_lb._warming_thread = None
    meta_lb._last_probe_ts = None
    meta_lb._last_probe_err = None
    meta_lb._req_total = 0
    meta_lb._req_to_spot = 0
    meta_lb._req_to_serverless = 0
    meta_lb._recent_routes.clear()
    meta_lb._gpu_seconds_spot = 0.0
    meta_lb._gpu_seconds_serverless = 0.0
    meta_lb._spot_ready_cumulative_s = 0.0
    meta_lb._spot_ready_since = None
    meta_lb._last_real_request_ts = 0.0

    # Disable auth for tests
    original_key = meta_lb.API_KEY
    meta_lb.API_KEY = ""

    with meta_lb.app.test_client() as c:
        yield c

    meta_lb.API_KEY = original_key


class TestRouterHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/router/health")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "skyserve_ready" in data

    def test_health_shows_urls(self, client):
        meta_lb.set_serverless_url("https://modal.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        resp = client.get("/router/health")
        data = json.loads(resp.data)
        assert data["serverless_base_url"] == "https://modal.example.com"
        assert data["skyserve_base_url"] == "http://spot.example.com"

    def test_health_shows_spot_state(self, client):
        resp = client.get("/router/health")
        data = json.loads(resp.data)
        assert data["spot_state"] == "cold"
        assert data["skyserve_ready"] is False

    def test_health_shows_ready_when_spot_ready(self, client):
        meta_lb._set_state(meta_lb.SpotState.READY)
        resp = client.get("/router/health")
        data = json.loads(resp.data)
        assert data["spot_state"] == "ready"
        assert data["skyserve_ready"] is True


class TestHealthNoProbe:
    def test_health_never_hits_skyserve_lb(self, client, mocker):
        """Verify /router/health never sends HTTP to SkyServe LB."""
        meta_lb.set_spot_url("http://spot.example.com")
        mock_get = mocker.patch.object(req_lib, "get")

        resp = client.get("/router/health")
        assert resp.status_code == 200
        mock_get.assert_not_called()

    def test_health_never_hits_skyserve_lb_when_ready(self, client, mocker):
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_state(meta_lb.SpotState.READY)
        mock_get = mocker.patch.object(req_lib, "get")

        resp = client.get("/router/health")
        assert resp.status_code == 200
        mock_get.assert_not_called()


class TestRouterConfig:
    def test_push_serverless_url(self, client):
        resp = client.post(
            "/router/config",
            json={"serverless_url": "https://modal.example.com"},
        )
        assert resp.status_code == 200
        assert meta_lb._get_serverless_url() == "https://modal.example.com"

    def test_push_spot_url(self, client):
        resp = client.post(
            "/router/config",
            json={"spot_url": "http://spot.example.com"},
        )
        assert resp.status_code == 200
        assert meta_lb._get_skyserve_url() == "http://spot.example.com"

    def test_push_both_urls(self, client):
        resp = client.post(
            "/router/config",
            json={
                "serverless_url": "https://modal.example.com",
                "spot_url": "http://spot.example.com",
            },
        )
        assert resp.status_code == 200
        assert meta_lb._get_serverless_url() == "https://modal.example.com"
        assert meta_lb._get_skyserve_url() == "http://spot.example.com"

    def test_push_strips_trailing_slash(self, client):
        client.post(
            "/router/config",
            json={"serverless_url": "https://modal.example.com/"},
        )
        assert meta_lb._get_serverless_url() == "https://modal.example.com"


class TestProxy503:
    def test_no_backends_returns_503(self, client):
        resp = client.get("/v1/chat/completions")
        assert resp.status_code == 503

    def test_only_spot_not_ready_returns_503(self, client):
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_state(meta_lb.SpotState.COLD)
        resp = client.get("/v1/chat/completions")
        assert resp.status_code == 503


class TestRoutingDecision:
    def test_routes_to_serverless_when_spot_not_ready(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_state(meta_lb.SpotState.COLD)

        # Prevent warming thread from actually running
        mocker.patch.object(meta_lb, "_enter_warming")

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = _mock_response(
            mocker, content=b'{"ok": true}', status_code=200,
            headers={"content-type": "application/json"},
        )

        resp = client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 200

        called_url = mock_request.call_args[1]["url"]
        assert called_url.startswith("http://serverless.example.com/")
        assert meta_lb._req_to_serverless == 1

    def test_routes_to_spot_when_ready(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_state(meta_lb.SpotState.READY)

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = _mock_response(
            mocker, content=b'{"ok": true}', status_code=200,
            headers={"content-type": "application/json"},
        )

        resp = client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 200

        called_url = mock_request.call_args[1]["url"]
        assert called_url.startswith("http://spot.example.com/")
        assert meta_lb._req_to_spot == 1

    def test_preserves_query_string(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = _mock_response(
            mocker, content=b"ok", status_code=200,
            headers={"content-type": "text/plain"},
        )

        resp = client.get("/v1/models?foo=bar")
        called_url = mock_request.call_args[1]["url"]
        assert "foo=bar" in called_url

    def test_triggers_warming_on_serverless_route_when_spot_cold(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_state(meta_lb.SpotState.COLD)

        mock_warming = mocker.patch.object(meta_lb, "_enter_warming")
        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = _mock_response(mocker)

        client.get("/v1/models")
        mock_warming.assert_called_once()

    def test_no_warming_when_already_warming(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._spot_state = meta_lb.SpotState.WARMING

        mock_warming = mocker.patch.object(meta_lb, "_enter_warming")
        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = _mock_response(mocker)

        client.get("/v1/models")
        mock_warming.assert_not_called()


class TestAuth:
    def test_rejects_missing_key(self, client):
        meta_lb.API_KEY = "secret123"
        meta_lb.set_serverless_url("http://serverless.example.com")

        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_accepts_correct_key(self, client, mocker):
        meta_lb.API_KEY = "secret123"
        meta_lb.set_serverless_url("http://serverless.example.com")

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = _mock_response(mocker)

        resp = client.get("/v1/models", headers={"x-api-key": "secret123"})
        assert resp.status_code == 200

    def test_accepts_bearer_token(self, client, mocker):
        meta_lb.API_KEY = "secret123"
        meta_lb.set_serverless_url("http://serverless.example.com")

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = _mock_response(mocker)

        resp = client.get(
            "/v1/models", headers={"Authorization": "Bearer secret123"}
        )
        assert resp.status_code == 200


class TestRouteStats:
    def test_stats_increment(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = _mock_response(mocker)

        for _ in range(5):
            client.get("/test")

        stats = meta_lb._route_stats()
        assert stats["total"] == 5
        assert stats["serverless"] == 5
        assert stats["spot"] == 0


class TestSpotFailoverRetry:
    def test_spot_connection_error_retries_on_serverless(self, client, mocker):
        """When spot fails with connection error, retry on serverless."""
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_state(meta_lb.SpotState.READY)

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        # First call (spot) fails, second call (serverless) succeeds
        mock_request.side_effect = [
            req_lib.ConnectionError("spot died"),
            _mock_response(
                mocker, content=b'{"ok":true}', status_code=200,
                headers={"content-type": "application/json"},
            ),
        ]
        resp = client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 200
        assert not meta_lb._is_ready()  # spot marked down
        assert meta_lb._spot_state == meta_lb.SpotState.COLD

    def test_spot_5xx_retries_on_serverless(self, client, mocker):
        """When spot returns 500, retry on serverless."""
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_state(meta_lb.SpotState.READY)

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.side_effect = [
            _mock_response(mocker, content=b"error", status_code=500),
            _mock_response(
                mocker, content=b'{"ok":true}', status_code=200,
                headers={"content-type": "application/json"},
            ),
        ]
        resp = client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 200

    def test_spot_failure_no_serverless_returns_502(self, client, mocker):
        """When spot fails and no serverless configured, return 502."""
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_state(meta_lb.SpotState.READY)

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.side_effect = req_lib.ConnectionError("spot died")
        resp = client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 502

    def test_serverless_failure_no_retry(self, client, mocker):
        """When serverless fails, don't retry on spot."""
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb._set_state(meta_lb.SpotState.COLD)

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.side_effect = req_lib.ConnectionError("serverless died")
        resp = client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 502
        assert mock_request.call_count == 1  # no retry

    def test_spot_4xx_no_retry(self, client, mocker):
        """Client errors (4xx) from spot are NOT retried."""
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_state(meta_lb.SpotState.READY)

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = _mock_response(
            mocker, content=b"bad request", status_code=400,
            headers={"content-type": "text/plain"},
        )
        resp = client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 400
        assert mock_request.call_count == 1  # no retry


class TestSpotStateMachine:
    def test_cold_to_warming_to_ready(self):
        """Test COLD → WARMING → READY transitions."""
        meta_lb._spot_state = meta_lb.SpotState.COLD
        meta_lb._set_state(meta_lb.SpotState.WARMING)
        assert meta_lb._spot_state == meta_lb.SpotState.WARMING
        assert not meta_lb._is_ready()

        meta_lb._set_state(meta_lb.SpotState.READY)
        assert meta_lb._spot_state == meta_lb.SpotState.READY
        assert meta_lb._is_ready()

    def test_ready_to_cold(self):
        """Test READY → COLD on failure."""
        meta_lb._spot_state = meta_lb.SpotState.COLD
        meta_lb._spot_ready_since = None
        meta_lb._spot_ready_cumulative_s = 0.0
        meta_lb._set_state(meta_lb.SpotState.READY)
        assert meta_lb._spot_ready_since is not None

        meta_lb._set_state(meta_lb.SpotState.COLD, "connection failed")
        assert meta_lb._spot_state == meta_lb.SpotState.COLD
        assert meta_lb._spot_ready_since is None
        assert meta_lb._spot_ready_cumulative_s > 0.0

    def test_set_ready_backward_compat(self):
        """_set_ready(True/False) still works via SpotState."""
        meta_lb._spot_state = meta_lb.SpotState.COLD
        meta_lb._set_ready(True)
        assert meta_lb._spot_state == meta_lb.SpotState.READY
        assert meta_lb._is_ready()

        meta_lb._set_ready(False)
        assert meta_lb._spot_state == meta_lb.SpotState.COLD
        assert not meta_lb._is_ready()


class TestWarmingThread:
    def test_warming_starts_on_cold(self, mocker):
        """Background poke loop starts when state is COLD and keeps poking."""
        meta_lb._spot_state = meta_lb.SpotState.COLD
        meta_lb._warming_thread = None
        meta_lb._skyserve_base_url = "http://spot.example.com"

        mock_get = mocker.patch.object(req_lib, "get")
        mock_get.return_value = mocker.Mock(status_code=200)
        mocker.patch("time.sleep")

        # Use short timeout so thread finishes
        orig_timeout = meta_lb.WARMUP_TIMEOUT_SECONDS
        orig_interval = meta_lb.WARMUP_POKE_INTERVAL_SECONDS
        meta_lb.WARMUP_TIMEOUT_SECONDS = 2.0
        meta_lb.WARMUP_POKE_INTERVAL_SECONDS = 1.0
        try:
            meta_lb._enter_warming()
            assert meta_lb._warming_thread is not None
            # Simulate watcher marking READY while warming
            meta_lb._spot_state = meta_lb.SpotState.READY
            meta_lb._warming_thread.join(timeout=5)
            # Warming thread exits because state changed externally
            assert meta_lb._spot_state == meta_lb.SpotState.READY
        finally:
            meta_lb.WARMUP_TIMEOUT_SECONDS = orig_timeout
            meta_lb.WARMUP_POKE_INTERVAL_SECONDS = orig_interval

    def test_warming_does_not_start_when_already_warming(self, mocker):
        """If already warming, _enter_warming is a no-op."""
        meta_lb._spot_state = meta_lb.SpotState.WARMING
        meta_lb._skyserve_base_url = "http://spot.example.com"

        mock_thread = mocker.patch("threading.Thread")
        meta_lb._enter_warming()
        mock_thread.assert_not_called()

    def test_warming_does_not_start_when_ready(self, mocker):
        """If already ready, _enter_warming is a no-op."""
        meta_lb._spot_state = meta_lb.SpotState.READY
        meta_lb._skyserve_base_url = "http://spot.example.com"

        mock_thread = mocker.patch("threading.Thread")
        meta_lb._enter_warming()
        mock_thread.assert_not_called()

    def test_warming_timeout_returns_to_cold(self, mocker):
        """After timeout, warming transitions back to COLD."""
        meta_lb._spot_state = meta_lb.SpotState.COLD
        meta_lb._warming_thread = None
        meta_lb._skyserve_base_url = "http://spot.example.com"

        # Mock poke to always fail
        mock_get = mocker.patch.object(req_lib, "get")
        mock_get.side_effect = req_lib.ConnectionError("refused")
        mocker.patch("time.sleep")

        # Use very short timeout for test (3 attempts)
        orig_timeout = meta_lb.WARMUP_TIMEOUT_SECONDS
        orig_interval = meta_lb.WARMUP_POKE_INTERVAL_SECONDS
        meta_lb.WARMUP_TIMEOUT_SECONDS = 3.0
        meta_lb.WARMUP_POKE_INTERVAL_SECONDS = 1.0

        try:
            meta_lb._enter_warming()
            meta_lb._warming_thread.join(timeout=10)
            # After all attempts fail, state returns to COLD
            assert meta_lb._spot_state == meta_lb.SpotState.COLD
            assert mock_get.call_count == 3
        finally:
            meta_lb.WARMUP_TIMEOUT_SECONDS = orig_timeout
            meta_lb.WARMUP_POKE_INTERVAL_SECONDS = orig_interval

    def test_warming_exits_when_state_changes_externally(self, mocker):
        """Warming thread stops if state changes externally (e.g., spot-replicas)."""
        meta_lb._spot_state = meta_lb.SpotState.COLD
        meta_lb._warming_thread = None
        meta_lb._skyserve_base_url = "http://spot.example.com"

        call_count = 0

        def _fake_get(*a, **kw):
            nonlocal call_count
            call_count += 1
            # After first poke, externally set state to READY
            if call_count == 1:
                meta_lb._spot_state = meta_lb.SpotState.READY
            raise req_lib.ConnectionError("refused")

        mocker.patch.object(req_lib, "get", side_effect=_fake_get)
        mocker.patch("time.sleep")

        orig_timeout = meta_lb.WARMUP_TIMEOUT_SECONDS
        orig_interval = meta_lb.WARMUP_POKE_INTERVAL_SECONDS
        meta_lb.WARMUP_TIMEOUT_SECONDS = 10.0
        meta_lb.WARMUP_POKE_INTERVAL_SECONDS = 1.0
        try:
            meta_lb._enter_warming()
            meta_lb._warming_thread.join(timeout=5)
            # Should have exited early — only 1 poke before external state change
            assert call_count == 1
        finally:
            meta_lb.WARMUP_TIMEOUT_SECONDS = orig_timeout
            meta_lb.WARMUP_POKE_INTERVAL_SECONDS = orig_interval


class TestSpotReplicas:
    def test_replicas_zero_marks_cold(self, client):
        """POST /router/spot-replicas with 0 replicas: READY → COLD."""
        meta_lb._set_state(meta_lb.SpotState.READY)
        resp = client.post(
            "/router/spot-replicas",
            json={"replicas": 0},
        )
        assert resp.status_code == 200
        assert meta_lb._spot_state == meta_lb.SpotState.COLD

    def test_replicas_positive_marks_ready(self, client):
        """POST /router/spot-replicas with 1 replica: COLD → READY."""
        meta_lb._set_state(meta_lb.SpotState.COLD)
        resp = client.post(
            "/router/spot-replicas",
            json={"replicas": 1},
        )
        assert resp.status_code == 200
        assert meta_lb._spot_state == meta_lb.SpotState.READY

    def test_replicas_positive_during_warming_marks_ready(self, client):
        """POST with replicas=1 during WARMING transitions to READY."""
        meta_lb._spot_state = meta_lb.SpotState.WARMING
        resp = client.post(
            "/router/spot-replicas",
            json={"replicas": 1},
        )
        assert resp.status_code == 200
        assert meta_lb._spot_state == meta_lb.SpotState.READY

    def test_replicas_zero_during_cold_no_change(self, client):
        """POST with replicas=0 during COLD doesn't change state."""
        meta_lb._set_state(meta_lb.SpotState.COLD)
        resp = client.post(
            "/router/spot-replicas",
            json={"replicas": 0},
        )
        assert resp.status_code == 200
        assert meta_lb._spot_state == meta_lb.SpotState.COLD

    def test_replicas_requires_auth(self, client):
        """Spot-replicas endpoint requires auth."""
        meta_lb.API_KEY = "secret"
        resp = client.post(
            "/router/spot-replicas",
            json={"replicas": 1},
        )
        assert resp.status_code == 401
