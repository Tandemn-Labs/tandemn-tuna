"""Tests for tandemn.router.meta_lb — routing logic without real backends."""

import json

import pytest

from tandemn.router import meta_lb


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
        mock_request.return_value = mocker.Mock(
            content=b'{"ok": true}',
            status_code=200,
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
        mock_request.return_value = mocker.Mock(
            content=b'{"ok": true}',
            status_code=200,
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
        mock_request.return_value = mocker.Mock(
            content=b"ok",
            status_code=200,
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
        mock_request.return_value = mocker.Mock(
            content=b"ok",
            status_code=200,
            headers={},
        )

        resp = client.get("/v1/models", headers={"x-api-key": "secret123"})
        assert resp.status_code == 200

    def test_accepts_bearer_token(self, client, mocker):
        meta_lb.API_KEY = "secret123"
        meta_lb.set_serverless_url("http://serverless.example.com")

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = mocker.Mock(
            content=b"ok",
            status_code=200,
            headers={},
        )

        resp = client.get(
            "/v1/models", headers={"Authorization": "Bearer secret123"}
        )
        assert resp.status_code == 200


class TestRouteStats:
    def test_stats_increment(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")

        mock_request = mocker.patch.object(meta_lb.SESSION, "request")
        mock_request.return_value = mocker.Mock(
            content=b"ok", status_code=200, headers={},
        )

        for _ in range(5):
            client.get("/test")

        stats = meta_lb._route_stats()
        assert stats["total"] == 5
        assert stats["serverless"] == 5
        assert stats["spot"] == 0
