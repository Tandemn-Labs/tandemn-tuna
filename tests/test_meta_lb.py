"""Tests for tuna.router.meta_lb — routing logic without real backends."""

import asyncio
import json
import time

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from tuna.router import meta_lb


def _mock_response(mocker, *, content=b"ok", status_code=200, headers=None):
    """Create a mock httpx.Response compatible with async streaming."""
    if headers is None:
        headers = {}

    async def _aiter_bytes(chunk_size=4096):
        yield content

    resp = mocker.AsyncMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = httpx.Headers(headers)
    resp.aiter_bytes = _aiter_bytes
    resp.aclose = mocker.AsyncMock()
    return resp


def _reset_state():
    """Reset all mutable module state for clean tests."""
    meta_lb._serverless_base_url = ""
    meta_lb._skyserve_base_url = ""
    meta_lb._spot_state = meta_lb.SpotState.COLD
    meta_lb._warming_task = None
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
    # Ensure we have a fresh asyncio lock (tests may run in different event loops)
    meta_lb._state_lock = asyncio.Lock()


@pytest_asyncio.fixture
async def client():
    """httpx async test client with clean state per test."""
    _reset_state()

    # Create a real httpx.AsyncClient for the module to use during tests
    meta_lb._http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=2.0, read=30.0, write=10.0, pool=5.0),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        follow_redirects=True,
    )

    # Disable auth for tests
    original_key = meta_lb.API_KEY
    meta_lb.API_KEY = ""

    async with AsyncClient(
        transport=ASGITransport(app=meta_lb.app),
        base_url="http://test",
    ) as c:
        yield c

    meta_lb.API_KEY = original_key
    if meta_lb._http_client:
        await meta_lb._http_client.aclose()
        meta_lb._http_client = None


class TestRouterHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/router/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "skyserve_ready" in data

    @pytest.mark.asyncio
    async def test_health_shows_urls(self, client):
        meta_lb.set_serverless_url("https://modal.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        resp = await client.get("/router/health")
        data = resp.json()
        assert data["serverless_base_url"] == "https://modal.example.com"
        assert data["skyserve_base_url"] == "http://spot.example.com"

    @pytest.mark.asyncio
    async def test_health_shows_spot_state(self, client):
        resp = await client.get("/router/health")
        data = resp.json()
        assert data["spot_state"] == "cold"
        assert data["skyserve_ready"] is False

    @pytest.mark.asyncio
    async def test_health_shows_ready_when_spot_ready(self, client):
        await meta_lb._set_state(meta_lb.SpotState.READY)
        resp = await client.get("/router/health")
        data = resp.json()
        assert data["spot_state"] == "ready"
        assert data["skyserve_ready"] is True


class TestHealthNoProbe:
    @pytest.mark.asyncio
    async def test_health_never_hits_skyserve_lb(self, client, mocker):
        """Verify /router/health never sends HTTP to SkyServe LB."""
        meta_lb.set_spot_url("http://spot.example.com")
        mock_get = mocker.patch.object(meta_lb._http_client, "get", new_callable=mocker.AsyncMock)

        resp = await client.get("/router/health")
        assert resp.status_code == 200
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_never_hits_skyserve_lb_when_ready(self, client, mocker):
        meta_lb.set_spot_url("http://spot.example.com")
        await meta_lb._set_state(meta_lb.SpotState.READY)
        mock_get = mocker.patch.object(meta_lb._http_client, "get", new_callable=mocker.AsyncMock)

        resp = await client.get("/router/health")
        assert resp.status_code == 200
        mock_get.assert_not_called()


class TestRouterConfig:
    @pytest.mark.asyncio
    async def test_push_serverless_url(self, client):
        resp = await client.post(
            "/router/config",
            json={"serverless_url": "https://modal.example.com"},
        )
        assert resp.status_code == 200
        assert await meta_lb._get_serverless_url() == "https://modal.example.com"

    @pytest.mark.asyncio
    async def test_push_spot_url(self, client):
        resp = await client.post(
            "/router/config",
            json={"spot_url": "http://spot.example.com"},
        )
        assert resp.status_code == 200
        assert await meta_lb._get_skyserve_url() == "http://spot.example.com"

    @pytest.mark.asyncio
    async def test_push_both_urls(self, client):
        resp = await client.post(
            "/router/config",
            json={
                "serverless_url": "https://modal.example.com",
                "spot_url": "http://spot.example.com",
            },
        )
        assert resp.status_code == 200
        assert await meta_lb._get_serverless_url() == "https://modal.example.com"
        assert await meta_lb._get_skyserve_url() == "http://spot.example.com"

    @pytest.mark.asyncio
    async def test_push_strips_trailing_slash(self, client):
        await client.post(
            "/router/config",
            json={"serverless_url": "https://modal.example.com/"},
        )
        assert await meta_lb._get_serverless_url() == "https://modal.example.com"


class TestProxy503:
    @pytest.mark.asyncio
    async def test_no_backends_returns_503(self, client):
        resp = await client.get("/v1/chat/completions")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_only_spot_not_ready_returns_503(self, client):
        meta_lb.set_spot_url("http://spot.example.com")
        await meta_lb._set_state(meta_lb.SpotState.COLD)
        resp = await client.get("/v1/chat/completions")
        assert resp.status_code == 503


class TestRoutingDecision:
    @pytest.mark.asyncio
    async def test_routes_to_serverless_when_spot_not_ready(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        await meta_lb._set_state(meta_lb.SpotState.COLD)

        # Prevent warming task from actually running
        mocker.patch.object(meta_lb, "_enter_warming", new_callable=mocker.AsyncMock)

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.return_value = _mock_response(
            mocker, content=b'{"ok": true}', status_code=200,
            headers={"content-type": "application/json"},
        )

        resp = await client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 200

        called_request = mock_send.call_args[0][0]
        assert str(called_request.url).startswith("http://serverless.example.com/")
        assert meta_lb._req_to_serverless == 1

    @pytest.mark.asyncio
    async def test_routes_to_spot_when_ready(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        await meta_lb._set_state(meta_lb.SpotState.READY)

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.return_value = _mock_response(
            mocker, content=b'{"ok": true}', status_code=200,
            headers={"content-type": "application/json"},
        )

        resp = await client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 200

        called_request = mock_send.call_args[0][0]
        assert str(called_request.url).startswith("http://spot.example.com/")
        assert meta_lb._req_to_spot == 1

    @pytest.mark.asyncio
    async def test_preserves_query_string(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.return_value = _mock_response(
            mocker, content=b"ok", status_code=200,
            headers={"content-type": "text/plain"},
        )

        resp = await client.get("/v1/models?foo=bar")
        called_request = mock_send.call_args[0][0]
        assert "foo=bar" in str(called_request.url)

    @pytest.mark.asyncio
    async def test_triggers_warming_on_serverless_route_when_spot_cold(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        await meta_lb._set_state(meta_lb.SpotState.COLD)

        mock_warming = mocker.patch.object(
            meta_lb, "_enter_warming", new_callable=mocker.AsyncMock,
        )
        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.return_value = _mock_response(mocker)

        await client.get("/v1/models")
        mock_warming.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_warming_when_already_warming(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        meta_lb._spot_state = meta_lb.SpotState.WARMING

        mock_warming = mocker.patch.object(
            meta_lb, "_enter_warming", new_callable=mocker.AsyncMock,
        )
        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.return_value = _mock_response(mocker)

        await client.get("/v1/models")
        mock_warming.assert_not_called()


class TestAuth:
    @pytest.mark.asyncio
    async def test_rejects_missing_key(self, client):
        meta_lb.API_KEY = "secret123"
        meta_lb.set_serverless_url("http://serverless.example.com")

        resp = await client.get("/v1/models")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_accepts_correct_key(self, client, mocker):
        meta_lb.API_KEY = "secret123"
        meta_lb.set_serverless_url("http://serverless.example.com")

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.return_value = _mock_response(mocker)

        resp = await client.get("/v1/models", headers={"x-api-key": "secret123"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_accepts_bearer_token(self, client, mocker):
        meta_lb.API_KEY = "secret123"
        meta_lb.set_serverless_url("http://serverless.example.com")

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.return_value = _mock_response(mocker)

        resp = await client.get(
            "/v1/models", headers={"Authorization": "Bearer secret123"}
        )
        assert resp.status_code == 200


class TestRouteStats:
    @pytest.mark.asyncio
    async def test_stats_increment(self, client, mocker):
        meta_lb.set_serverless_url("http://serverless.example.com")

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.return_value = _mock_response(mocker)

        for _ in range(5):
            await client.get("/test")

        stats = await meta_lb._route_stats()
        assert stats["total"] == 5
        assert stats["serverless"] == 5
        assert stats["spot"] == 0


class TestSpotFailoverRetry:
    @pytest.mark.asyncio
    async def test_spot_connection_error_retries_on_serverless(self, client, mocker):
        """When spot fails with connection error, retry on serverless."""
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        await meta_lb._set_state(meta_lb.SpotState.READY)

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        # First call (spot) fails, second call (serverless) succeeds
        mock_send.side_effect = [
            httpx.ConnectError("spot died"),
            _mock_response(
                mocker, content=b'{"ok":true}', status_code=200,
                headers={"content-type": "application/json"},
            ),
        ]
        resp = await client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 200
        assert not await meta_lb._is_ready()  # spot marked down
        assert meta_lb._spot_state == meta_lb.SpotState.COLD

    @pytest.mark.asyncio
    async def test_spot_5xx_retries_on_serverless(self, client, mocker):
        """When spot returns 500, retry on serverless."""
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        await meta_lb._set_state(meta_lb.SpotState.READY)

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.side_effect = [
            _mock_response(mocker, content=b"error", status_code=500),
            _mock_response(
                mocker, content=b'{"ok":true}', status_code=200,
                headers={"content-type": "application/json"},
            ),
        ]
        resp = await client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_spot_failure_no_serverless_returns_502(self, client, mocker):
        """When spot fails and no serverless configured, return 502."""
        meta_lb.set_spot_url("http://spot.example.com")
        await meta_lb._set_state(meta_lb.SpotState.READY)

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.side_effect = httpx.ConnectError("spot died")
        resp = await client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_serverless_failure_no_retry(self, client, mocker):
        """When serverless fails, don't retry on spot."""
        meta_lb.set_serverless_url("http://serverless.example.com")
        await meta_lb._set_state(meta_lb.SpotState.COLD)

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.side_effect = httpx.ConnectError("serverless died")
        resp = await client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 502
        assert mock_send.call_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_spot_4xx_no_retry(self, client, mocker):
        """Client errors (4xx) from spot are NOT retried."""
        meta_lb.set_serverless_url("http://serverless.example.com")
        meta_lb.set_spot_url("http://spot.example.com")
        await meta_lb._set_state(meta_lb.SpotState.READY)

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.return_value = _mock_response(
            mocker, content=b"bad request", status_code=400,
            headers={"content-type": "text/plain"},
        )
        resp = await client.post("/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 400
        assert mock_send.call_count == 1  # no retry


class TestSpotStateMachine:
    @pytest.mark.asyncio
    async def test_cold_to_warming_to_ready(self):
        _reset_state()
        meta_lb._spot_state = meta_lb.SpotState.COLD
        await meta_lb._set_state(meta_lb.SpotState.WARMING)
        assert meta_lb._spot_state == meta_lb.SpotState.WARMING
        assert not await meta_lb._is_ready()

        await meta_lb._set_state(meta_lb.SpotState.READY)
        assert meta_lb._spot_state == meta_lb.SpotState.READY
        assert await meta_lb._is_ready()

    @pytest.mark.asyncio
    async def test_ready_to_cold(self):
        _reset_state()
        meta_lb._spot_state = meta_lb.SpotState.COLD
        meta_lb._spot_ready_since = None
        meta_lb._spot_ready_cumulative_s = 0.0
        await meta_lb._set_state(meta_lb.SpotState.READY)
        assert meta_lb._spot_ready_since is not None

        await meta_lb._set_state(meta_lb.SpotState.COLD, "connection failed")
        assert meta_lb._spot_state == meta_lb.SpotState.COLD
        assert meta_lb._spot_ready_since is None
        assert meta_lb._spot_ready_cumulative_s > 0.0

    @pytest.mark.asyncio
    async def test_set_ready_backward_compat(self):
        _reset_state()
        meta_lb._spot_state = meta_lb.SpotState.COLD
        await meta_lb._set_ready(True)
        assert meta_lb._spot_state == meta_lb.SpotState.READY
        assert await meta_lb._is_ready()

        await meta_lb._set_ready(False)
        assert meta_lb._spot_state == meta_lb.SpotState.COLD
        assert not await meta_lb._is_ready()


class TestWarmingTask:
    @pytest.mark.asyncio
    async def test_warming_starts_on_cold(self, mocker):
        """Background poke loop starts when state is COLD and keeps poking."""
        _reset_state()
        meta_lb._spot_state = meta_lb.SpotState.COLD
        meta_lb._warming_task = None
        meta_lb._skyserve_base_url = "http://spot.example.com"

        # Create a real client for the warming loop
        meta_lb._http_client = httpx.AsyncClient()
        mock_get = mocker.patch.object(
            meta_lb._http_client, "get", new_callable=mocker.AsyncMock,
        )
        mock_resp = mocker.AsyncMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

        # Use short timeout so task finishes
        orig_timeout = meta_lb.WARMUP_TIMEOUT_SECONDS
        orig_interval = meta_lb.WARMUP_POKE_INTERVAL_SECONDS
        meta_lb.WARMUP_TIMEOUT_SECONDS = 2.0
        meta_lb.WARMUP_POKE_INTERVAL_SECONDS = 1.0
        try:
            await meta_lb._enter_warming()
            assert meta_lb._warming_task is not None
            # Wait for task to complete (readiness probe returns 200 → READY)
            await asyncio.wait_for(meta_lb._warming_task, timeout=5)
            assert meta_lb._spot_state == meta_lb.SpotState.READY
        finally:
            meta_lb.WARMUP_TIMEOUT_SECONDS = orig_timeout
            meta_lb.WARMUP_POKE_INTERVAL_SECONDS = orig_interval
            await meta_lb._http_client.aclose()

    @pytest.mark.asyncio
    async def test_warming_does_not_start_when_already_warming(self):
        """If already warming, _enter_warming is a no-op."""
        _reset_state()
        meta_lb._spot_state = meta_lb.SpotState.WARMING
        meta_lb._skyserve_base_url = "http://spot.example.com"

        old_task = meta_lb._warming_task
        await meta_lb._enter_warming()
        assert meta_lb._warming_task is old_task

    @pytest.mark.asyncio
    async def test_warming_does_not_start_when_ready(self):
        """If already ready, _enter_warming is a no-op."""
        _reset_state()
        meta_lb._spot_state = meta_lb.SpotState.READY
        meta_lb._skyserve_base_url = "http://spot.example.com"

        old_task = meta_lb._warming_task
        await meta_lb._enter_warming()
        assert meta_lb._warming_task is old_task

    @pytest.mark.asyncio
    async def test_warming_timeout_returns_to_cold(self, mocker):
        """After timeout, warming transitions back to COLD."""
        _reset_state()
        meta_lb._spot_state = meta_lb.SpotState.COLD
        meta_lb._warming_task = None
        meta_lb._skyserve_base_url = "http://spot.example.com"

        # Create a real client for the warming loop
        meta_lb._http_client = httpx.AsyncClient()
        mock_get = mocker.patch.object(
            meta_lb._http_client, "get", new_callable=mocker.AsyncMock,
        )
        mock_get.side_effect = httpx.ConnectError("refused")
        mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

        # Use very short timeout for test (3 attempts)
        orig_timeout = meta_lb.WARMUP_TIMEOUT_SECONDS
        orig_interval = meta_lb.WARMUP_POKE_INTERVAL_SECONDS
        meta_lb.WARMUP_TIMEOUT_SECONDS = 3.0
        meta_lb.WARMUP_POKE_INTERVAL_SECONDS = 1.0

        try:
            await meta_lb._enter_warming()
            await asyncio.wait_for(meta_lb._warming_task, timeout=10)
            # After all attempts fail, state returns to COLD
            assert meta_lb._spot_state == meta_lb.SpotState.COLD
            # 2 calls per iteration (poke + readiness probe) × 3 attempts = 6
            assert mock_get.call_count == 6
        finally:
            meta_lb.WARMUP_TIMEOUT_SECONDS = orig_timeout
            meta_lb.WARMUP_POKE_INTERVAL_SECONDS = orig_interval
            await meta_lb._http_client.aclose()

    @pytest.mark.asyncio
    async def test_warming_exits_when_state_changes_externally(self, mocker):
        """Warming task stops if state changes externally (e.g., spot-replicas)."""
        _reset_state()
        meta_lb._spot_state = meta_lb.SpotState.COLD
        meta_lb._warming_task = None
        meta_lb._skyserve_base_url = "http://spot.example.com"

        call_count = 0

        meta_lb._http_client = httpx.AsyncClient()

        async def _fake_get(*a, **kw):
            nonlocal call_count
            call_count += 1
            # After first poke, externally set state to READY
            if call_count == 1:
                meta_lb._spot_state = meta_lb.SpotState.READY
            raise httpx.ConnectError("refused")

        mocker.patch.object(meta_lb._http_client, "get", side_effect=_fake_get)
        mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

        orig_timeout = meta_lb.WARMUP_TIMEOUT_SECONDS
        orig_interval = meta_lb.WARMUP_POKE_INTERVAL_SECONDS
        meta_lb.WARMUP_TIMEOUT_SECONDS = 10.0
        meta_lb.WARMUP_POKE_INTERVAL_SECONDS = 1.0
        try:
            await meta_lb._enter_warming()
            await asyncio.wait_for(meta_lb._warming_task, timeout=5)
            # 2 calls in first iteration (poke + readiness), then state check exits
            assert call_count == 2
        finally:
            meta_lb.WARMUP_TIMEOUT_SECONDS = orig_timeout
            meta_lb.WARMUP_POKE_INTERVAL_SECONDS = orig_interval
            await meta_lb._http_client.aclose()


class TestSpotReplicas:
    @pytest.mark.asyncio
    async def test_replicas_zero_marks_cold(self, client):
        """POST /router/spot-replicas with 0 replicas: READY → COLD."""
        await meta_lb._set_state(meta_lb.SpotState.READY)
        resp = await client.post(
            "/router/spot-replicas",
            json={"replicas": 0},
        )
        assert resp.status_code == 200
        assert meta_lb._spot_state == meta_lb.SpotState.COLD

    @pytest.mark.asyncio
    async def test_replicas_positive_marks_ready(self, client):
        """POST /router/spot-replicas with 1 replica: COLD → READY."""
        await meta_lb._set_state(meta_lb.SpotState.COLD)
        resp = await client.post(
            "/router/spot-replicas",
            json={"replicas": 1},
        )
        assert resp.status_code == 200
        assert meta_lb._spot_state == meta_lb.SpotState.READY

    @pytest.mark.asyncio
    async def test_replicas_positive_during_warming_marks_ready(self, client):
        """POST with replicas=1 during WARMING transitions to READY."""
        meta_lb._spot_state = meta_lb.SpotState.WARMING
        resp = await client.post(
            "/router/spot-replicas",
            json={"replicas": 1},
        )
        assert resp.status_code == 200
        assert meta_lb._spot_state == meta_lb.SpotState.READY

    @pytest.mark.asyncio
    async def test_replicas_zero_during_cold_no_change(self, client):
        """POST with replicas=0 during COLD doesn't change state."""
        await meta_lb._set_state(meta_lb.SpotState.COLD)
        resp = await client.post(
            "/router/spot-replicas",
            json={"replicas": 0},
        )
        assert resp.status_code == 200
        assert meta_lb._spot_state == meta_lb.SpotState.COLD

    @pytest.mark.asyncio
    async def test_replicas_requires_auth(self, client):
        """Spot-replicas endpoint requires auth."""
        meta_lb.API_KEY = "secret"
        resp = await client.post(
            "/router/spot-replicas",
            json={"replicas": 1},
        )
        assert resp.status_code == 401
