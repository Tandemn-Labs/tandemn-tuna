"""Tests for tuna.router.meta_lb — routing logic without real backends."""

import json

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
    meta_lb._skyserve_ready = False
    meta_lb._last_probe_ts = None
    meta_lb._last_probe_err = None
    meta_lb._last_check_ts = 0.0
    meta_lb._last_poke_ts = 0.0
    meta_lb._req_total = 0
    meta_lb._req_to_spot = 0
    meta_lb._req_to_serverless = 0
    meta_lb._recent_routes.clear()
    meta_lb._gpu_seconds_spot = 0.0
    meta_lb._gpu_seconds_serverless = 0.0
    meta_lb._spot_ready_cumulative_s = 0.0
    meta_lb._spot_ready_since = None

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
        meta_lb._set_ready(False)
        resp = client.get("/v1/chat/completions")
        assert resp.status_code == 503


class TestRoutingDecision:
    def test_routes_to_serverless_when_spot_not_ready(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_ready(False)

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
        meta_lb._set_ready(True)

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
        meta_lb._set_ready(True)

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

    def test_spot_5xx_retries_on_serverless(self, client, mocker):
        """When spot returns 500, retry on serverless."""
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_ready(True)

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
        meta_lb._set_ready(True)

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.side_effect = req_lib.ConnectionError("spot died")
        resp = client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 502

    def test_serverless_failure_no_retry(self, client, mocker):
        """When serverless fails, don't retry on spot."""
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb._set_ready(False)

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.side_effect = req_lib.ConnectionError("serverless died")
        resp = client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 502
        assert mock_request.call_count == 1  # no retry

    def test_spot_4xx_no_retry(self, client, mocker):
        """Client errors (4xx) from spot are NOT retried."""
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._set_ready(True)

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = _mock_response(
            mocker, content=b"bad request", status_code=400,
            headers={"content-type": "text/plain"},
        )
        resp = client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 400
        assert mock_request.call_count == 1  # no retry
