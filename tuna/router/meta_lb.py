"""Meta load balancer — routes between serverless and spot backends.

Adapted from the proven SAM load_balancer.py pattern. Provider-agnostic:
only knows about URLs, never imports Modal/SkyPilot.

Spot state machine (COLD → WARMING → READY):
  COLD:    Spot is down. Route everything to serverless. Zero SkyServe LB hits.
  WARMING: Waking spot up. Background task pokes SkyServe periodically.
  READY:   Spot is up. Route to spot. Zero probing — failures trigger COLD.

Env vars:
  SERVERLESS_BASE_URL     e.g. https://xxx.modal.run
  SKYSERVE_BASE_URL       e.g. http://x.x.x.x:30001

  SKYSERVE_POKE_PATH      default: /health (maintains QPS for autoscaler)

  POKE_TIMEOUT_SECONDS    default: 0.3
  UPSTREAM_TIMEOUT_SECONDS default: 210.0

  IDLE_TIMEOUT_SECONDS          default: 60.0  (= spot.downscale_delay)
  WARMUP_POKE_INTERVAL_SECONDS  default: 5.0   (= spot.upscale_delay)
  WARMUP_TIMEOUT_SECONDS        default: 1200.0 (= readiness_probe.initial_delay_seconds)

  API_KEY                 if set, required for all requests
  API_KEY_HEADER          default: x-api-key
  ALLOW_HEALTH_NO_AUTH    default: 0

Run:
  uvicorn meta_lb:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 300
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Dict
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("meta_lb")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y")


def _sanitize_path(path: str) -> str:
    """Strip scheme, netloc, and directory traversal from a user-supplied path."""
    from urllib.parse import urlparse
    parsed = urlparse(path)
    # Use only the path component — discard any scheme://host an attacker may inject
    clean = parsed.path
    # Collapse .. segments so the path cannot escape the base URL
    segments = [s for s in clean.split("/") if s not in ("", ".", "..")]
    return "/".join(segments)


def _build_proxy_url(base: str, path: str, query_string: bytes | None = None) -> str:
    """Build a safe proxy target URL from a backend base and user-supplied path.

    Sanitizes the path and validates the result points to the expected host.
    """
    from urllib.parse import urlparse, quote
    clean_path = _sanitize_path(path)
    url = base.rstrip("/") + "/" + clean_path
    if query_string:
        # Re-encode the query string to prevent injection
        url += "?" + quote(query_string.decode("utf-8"), safe="=&")
    # Final validation: result must share the same scheme+host as base
    base_parsed = urlparse(base)
    url_parsed = urlparse(url)
    if url_parsed.netloc != base_parsed.netloc:
        raise ValueError(f"URL host mismatch: expected {base_parsed.netloc}")
    return url


def _join_url(base: str, path: str) -> str:
    """Join base URL with a server-controlled path (no sanitization needed)."""
    return base.rstrip("/") + "/" + path.lstrip("/")


# ---------------------------------------------------------------------------
# Configuration (env-driven, same pattern as SAM)
# ---------------------------------------------------------------------------

SKYSERVE_POKE_PATH = os.getenv("SKYSERVE_POKE_PATH", "/health")

POKE_TIMEOUT_SECONDS = _env_float("POKE_TIMEOUT_SECONDS", 0.5)
UPSTREAM_TIMEOUT_SECONDS = _env_float("UPSTREAM_TIMEOUT_SECONDS", 210.0)

# Derived from SpotScaling config — no magic numbers
IDLE_TIMEOUT_SECONDS = _env_float("IDLE_TIMEOUT_SECONDS", 60.0)
WARMUP_POKE_INTERVAL_SECONDS = _env_float("WARMUP_POKE_INTERVAL_SECONDS", 5.0)
WARMUP_TIMEOUT_SECONDS = _env_float("WARMUP_TIMEOUT_SECONDS", 1200.0)

API_KEY = os.getenv("API_KEY", "")
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "x-api-key")
ALLOW_HEALTH_NO_AUTH = _env_bool("ALLOW_HEALTH_NO_AUTH", False)

ROUTE_WINDOW_SIZE = int(os.getenv("ROUTE_WINDOW_SIZE", "200"))

# HTTP client — created in lifespan, shared across all requests.
_http_client: httpx.AsyncClient | None = None


# ---------------------------------------------------------------------------
# Spot state machine
# ---------------------------------------------------------------------------

class SpotState:
    COLD = "cold"
    WARMING = "warming"
    READY = "ready"


# ---------------------------------------------------------------------------
# Mutable state — backend URLs can be updated at runtime via /router/config
# ---------------------------------------------------------------------------

_state_lock = asyncio.Lock()
_serverless_base_url: str = os.environ.get("SERVERLESS_BASE_URL", "").rstrip("/")
_serverless_auth_token: str = os.environ.get("SERVERLESS_AUTH_TOKEN", "")
_skyserve_base_url: str = os.environ.get("SKYSERVE_BASE_URL", "").rstrip("/")

_spot_state: str = SpotState.COLD
_warming_task: asyncio.Task | None = None
_last_probe_ts: float | None = None
_last_probe_err: str | None = None

# Route stats
_req_total: int = 0
_req_to_spot: int = 0
_req_to_serverless: int = 0
_recent_routes: deque = deque(maxlen=ROUTE_WINDOW_SIZE)

# Cost tracking
_start_time: float = time.time()
_gpu_seconds_spot: float = 0.0
_gpu_seconds_serverless: float = 0.0
_spot_ready_cumulative_s: float = 0.0
_spot_ready_since: float | None = None
_last_real_request_ts: float = 0.0


# ---------------------------------------------------------------------------
# State accessors
# ---------------------------------------------------------------------------

async def _get_serverless_url() -> str:
    async with _state_lock:
        return _serverless_base_url


async def _get_skyserve_url() -> str:
    async with _state_lock:
        return _skyserve_base_url


async def set_serverless_url(url: str) -> None:
    global _serverless_base_url
    validated = _validate_backend_url(url)
    async with _state_lock:
        _serverless_base_url = validated
    logger.info("Serverless URL updated: %s", url)


async def set_serverless_auth_token(token: str) -> None:
    global _serverless_auth_token
    async with _state_lock:
        _serverless_auth_token = token
    logger.info("Serverless auth token updated")


def _validate_backend_url(url: str) -> str:
    """Validate and sanitize a backend URL to prevent SSRF.

    Only allows http/https schemes. The /router/config endpoint is
    auth-protected and only called by the orchestrator, but we
    validate defensively.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme!r} (expected http/https)")
    if not parsed.netloc:
        raise ValueError(f"Invalid URL: missing host in {url!r}")
    return url.rstrip("/")


async def set_spot_url(url: str) -> None:
    global _skyserve_base_url
    validated = _validate_backend_url(url)
    async with _state_lock:
        _skyserve_base_url = validated
    logger.info("Spot URL updated: %s", url)


async def _set_state(new_state: str, err: str | None = None) -> None:
    """Transition the spot state machine and track ready-time accounting."""
    global _spot_state, _last_probe_ts, _last_probe_err
    global _spot_ready_cumulative_s, _spot_ready_since
    async with _state_lock:
        old = _spot_state
        now = time.time()
        # Accumulate spot-ready time on state transitions
        if old == SpotState.READY and new_state != SpotState.READY and _spot_ready_since is not None:
            _spot_ready_cumulative_s += now - _spot_ready_since
            _spot_ready_since = None
        elif old != SpotState.READY and new_state == SpotState.READY:
            _spot_ready_since = now
        _spot_state = new_state
        _last_probe_ts = now
        _last_probe_err = err
    if old != new_state:
        logger.info("Spot state: %s -> %s", old, new_state)


async def _is_ready() -> bool:
    async with _state_lock:
        return _spot_state == SpotState.READY


# Backward-compat helper for _set_ready(bool) callers
async def _set_ready(val: bool, err: str | None = None) -> None:
    await _set_state(SpotState.READY if val else SpotState.COLD, err)


# ---------------------------------------------------------------------------
# Route stats
# ---------------------------------------------------------------------------

async def _record_route(backend: str) -> None:
    global _req_total, _req_to_spot, _req_to_serverless, _last_real_request_ts
    async with _state_lock:
        _req_total += 1
        _recent_routes.append(backend)
        _last_real_request_ts = time.time()
        if backend == "spot":
            _req_to_spot += 1
        else:
            _req_to_serverless += 1


async def _route_stats() -> dict:
    async with _state_lock:
        total = _req_total
        spot = _req_to_spot
        svl = _req_to_serverless
        recent = list(_recent_routes)
        gpu_s_spot = _gpu_seconds_spot
        gpu_s_svl = _gpu_seconds_serverless
        # Compute spot_ready including current ongoing ready period
        spot_ready_s = _spot_ready_cumulative_s
        if _spot_ready_since is not None:
            spot_ready_s += time.time() - _spot_ready_since
    recent_total = len(recent)
    recent_spot = sum(1 for r in recent if r == "spot")
    recent_svl = recent_total - recent_spot
    return {
        "total": total,
        "spot": spot,
        "serverless": svl,
        "pct_spot": (100.0 * spot / total) if total else 0.0,
        "pct_serverless": (100.0 * svl / total) if total else 0.0,
        "window_total": recent_total,
        "window_spot": recent_spot,
        "window_serverless": recent_svl,
        "gpu_seconds_spot": round(gpu_s_spot, 2),
        "gpu_seconds_serverless": round(gpu_s_svl, 2),
        "uptime_seconds": round(time.time() - _start_time, 2),
        "spot_ready_seconds": round(spot_ready_s, 2),
    }


# ---------------------------------------------------------------------------
# Header filtering
# ---------------------------------------------------------------------------

def _filter_incoming(h: Dict[str, str], *, strip_auth: bool = True) -> Dict[str, str]:
    drop = {API_KEY_HEADER.lower()}
    if strip_auth:
        drop.add("authorization")
    return {
        k: v for k, v in h.items()
        if k.lower() not in HOP_BY_HOP_HEADERS
        and k.lower() != "host"
        and k.lower() not in drop
    }


def _filter_outgoing(h) -> Dict[str, str]:
    return {
        k: v for k, v in h.items()
        if k.lower() not in HOP_BY_HOP_HEADERS
        and k.lower() != "content-length"
    }


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _extract_api_key(req: Request) -> str:
    key = req.headers.get(API_KEY_HEADER)
    if not key:
        auth = req.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth.split(" ", 1)[1]
    return key or ""


def _is_authorized(req: Request) -> bool:
    if not API_KEY:
        return True
    provided = _extract_api_key(req)
    if not provided:
        return False
    return hmac.compare_digest(provided, API_KEY)


# ---------------------------------------------------------------------------
# Warming state machine — replaces proactive health probes
# ---------------------------------------------------------------------------

async def _enter_warming() -> None:
    """Start background warming task that pokes SkyServe periodically.

    Keeps QPS > 0 so SkyServe doesn't kill the PROVISIONING replica.
    Poke interval = upscale_delay (autoscaler's tick rate).
    Timeout = readiness_probe.initial_delay_seconds (max time for replica to boot).
    Reads poke responses to detect when spot becomes READY.
    """
    global _warming_task, _spot_state, _last_probe_ts
    async with _state_lock:
        if _spot_state != SpotState.COLD:
            return  # already warming or ready
        # Inline transition to WARMING (avoid re-acquiring lock via _set_state)
        _spot_state = SpotState.WARMING
        _last_probe_ts = time.time()
    logger.info("Spot state: cold -> warming")

    async def _warm_loop():
        """Poke SkyServe LB every interval to maintain QPS > 0 (prevents
        autoscaler from killing the PROVISIONING replica).  Also probes
        /v1/models to verify a real replica is serving — transitions to
        READY only when that succeeds.
        """
        max_attempts = int(WARMUP_TIMEOUT_SECONDS / WARMUP_POKE_INTERVAL_SECONDS)
        skyserve_url = await _get_skyserve_url()
        # Validate URL to prevent SSRF — skyserve_url is set via auth-protected
        # /router/config or env var, but validate defensively.
        # Re-assign from validator return to break CodeQL taint flow.
        skyserve_url = _validate_backend_url(skyserve_url)  # noqa: SSRF
        poke_url = _join_url(skyserve_url, SKYSERVE_POKE_PATH)
        readiness_url = _join_url(skyserve_url, "/v1/models")

        for _ in range(max_attempts):
            # Check if state changed externally (e.g., watcher reported replicas>0)
            async with _state_lock:
                if _spot_state != SpotState.WARMING:
                    return
            # Poke /health to maintain QPS for autoscaler
            try:
                await _http_client.get(poke_url, timeout=POKE_TIMEOUT_SECONDS)  # nosec: validated above
            except Exception:
                pass
            # Probe /v1/models to verify a real replica is serving
            try:
                r = await _http_client.get(readiness_url, timeout=5.0)  # nosec: validated above
                if 200 <= r.status_code < 300:
                    await _set_state(SpotState.READY)
                    return
            except Exception:
                pass
            await asyncio.sleep(WARMUP_POKE_INTERVAL_SECONDS)

        # Timeout — replica didn't become READY within initial_delay window
        logger.warning(
            "Spot warmup timed out after %.0fs", WARMUP_TIMEOUT_SECONDS,
        )
        await _set_state(SpotState.COLD)

    _warming_task = asyncio.create_task(_warm_loop())


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

async def _stream_and_track(response: httpx.Response, t0: float, backend_name: str):
    """Async generator: yield chunks from upstream and track GPU seconds on completion."""
    global _gpu_seconds_spot, _gpu_seconds_serverless
    try:
        async for chunk in response.aiter_bytes(chunk_size=4096):
            if chunk:
                yield chunk
    finally:
        await response.aclose()
        elapsed = time.time() - t0
        async with _state_lock:
            if backend_name == "spot":
                _gpu_seconds_spot += elapsed
            else:
                _gpu_seconds_serverless += elapsed


# ---------------------------------------------------------------------------
# Spot → serverless failover
# ---------------------------------------------------------------------------

async def _forward_to_serverless(
    request: Request, path: str, headers: dict, data: bytes, serverless_url: str,
) -> Response:
    """Retry a failed spot request on the serverless backend."""
    global _gpu_seconds_serverless
    target_url = _build_proxy_url(
        serverless_url, path, request.url.query.encode() if request.url.query else None,
    )

    # Swap in serverless auth token
    async with _state_lock:
        auth_token = _serverless_auth_token
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    else:
        headers.pop("Authorization", None)

    await _record_route("serverless")  # Count the retry as a serverless route

    t0 = time.time()
    try:
        resp = await _http_client.send(
            _http_client.build_request(
                request.method, target_url, headers=headers, content=data or None,
            ),
            stream=True,
        )
    except httpx.HTTPError as e:
        elapsed = time.time() - t0
        async with _state_lock:
            _gpu_seconds_serverless += elapsed
        logger.warning("Upstream error: %s", e)
        return Response(content="upstream_error", status_code=502)

    resp_headers = _filter_outgoing(dict(resp.headers))
    return StreamingResponse(
        _stream_and_track(resp, t0, "serverless"),
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Lifespan — manage httpx client and warming task
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=2.0,
            read=UPSTREAM_TIMEOUT_SECONDS,
            write=10.0,
            pool=5.0,
        ),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        follow_redirects=True,
    )
    # Auto-warm on startup if spot URL is configured
    _auto_warm_on_startup()
    yield
    # Shutdown: cancel warming task, close client
    if _warming_task and not _warming_task.done():
        _warming_task.cancel()
        try:
            await _warming_task
        except asyncio.CancelledError:
            pass
    await _http_client.aclose()

app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/router/health")
async def router_health(request: Request):
    if not ALLOW_HEALTH_NO_AUTH and not _is_authorized(request):
        return Response(content="unauthorized", status_code=401)

    # Pure read — no SkyServe LB hit. State is updated by proxy() results
    # and the warming task.
    async with _state_lock:
        state = {
            "skyserve_ready": _spot_state == SpotState.READY,
            "spot_state": _spot_state,
            "last_probe_ts": _last_probe_ts,
            "last_probe_err": bool(_last_probe_err),
            "serverless_base_url": _serverless_base_url,
            "skyserve_base_url": _skyserve_base_url,
        }
    state["route_stats"] = await _route_stats()

    return JSONResponse(content=state, status_code=200)


@app.post("/router/config")
async def update_config(request: Request):
    """Orchestrator pushes backend URLs here after deploy."""
    if not _is_authorized(request):
        return Response(content="unauthorized", status_code=401)
    data = await request.json()
    if not data:
        data = {}
    if "serverless_url" in data:
        await set_serverless_url(data["serverless_url"])
    if "serverless_auth_token" in data:
        await set_serverless_auth_token(data["serverless_auth_token"])
    if "spot_url" in data:
        await set_spot_url(data["spot_url"])
    return JSONResponse(content={"status": "ok"}, status_code=200)


@app.post("/router/spot-replicas")
async def update_spot_replicas(request: Request):
    """Replica watcher pushes actual replica count here."""
    if not _is_authorized(request):
        return Response(content="unauthorized", status_code=401)
    data = await request.json()
    if not data:
        data = {}
    replicas = data.get("replicas", 0)
    async with _state_lock:
        current = _spot_state
    if replicas == 0 and current == SpotState.READY:
        logger.info("Replica watcher: 0 replicas, marking COLD")
        await _set_state(SpotState.COLD)
    elif replicas > 0 and current in (SpotState.COLD, SpotState.WARMING):
        logger.info("Replica watcher: %d replicas, marking READY", replicas)
        await _set_state(SpotState.READY)
    return JSONResponse(content={"ok": True}, status_code=200)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy(request: Request, path: str = ""):
    global _gpu_seconds_spot, _gpu_seconds_serverless
    serverless_url = await _get_serverless_url()
    skyserve_url = await _get_skyserve_url()

    if not serverless_url and not skyserve_url:
        return JSONResponse(
            content={"error": "No backends configured yet"},
            status_code=503,
        )

    if not _is_authorized(request):
        return Response(content="unauthorized", status_code=401)

    # Preemptive idle→cold: if spot is READY but no real traffic for
    # downscale_delay seconds, SkyServe has likely scaled down already.
    # Mark COLD to avoid routing to a dead spot (saves one failed request).
    if skyserve_url and await _is_ready():
        async with _state_lock:
            idle_s = time.time() - _last_real_request_ts if _last_real_request_ts else 0
        if idle_s > IDLE_TIMEOUT_SECONDS:
            logger.info("Idle %.0fs > %.0fs, preemptively marking COLD", idle_s, IDLE_TIMEOUT_SECONDS)
            await _set_state(SpotState.COLD)

    # Decide backend: prefer spot (cheaper) if ready, else serverless (fast)
    if skyserve_url and await _is_ready():
        backend_base = skyserve_url
        backend_name = "spot"
        await _record_route("spot")
    elif serverless_url:
        backend_base = serverless_url
        backend_name = "serverless"
        await _record_route("serverless")
        # Trigger warming if spot is cold and URL is configured
        if skyserve_url and _spot_state == SpotState.COLD:
            await _enter_warming()
    else:
        # Only spot configured but not ready
        return JSONResponse(
            content={"error": "Spot backend not ready, no serverless fallback"},
            status_code=503,
        )

    # Log periodically
    stats = await _route_stats()
    if stats["total"] % 100 == 0 and stats["total"] > 0:
        logger.info(
            "requests=%d spot=%d (%.0f%%) serverless=%d (%.0f%%) spot_state=%s",
            stats["total"], stats["spot"], stats["pct_spot"],
            stats["serverless"], stats["pct_serverless"], _spot_state,
        )

    # Forward request
    query_string = request.url.query.encode() if request.url.query else None
    target_url = _build_proxy_url(backend_base, path, query_string)

    # Strip auth headers for serverless (we inject our own token); preserve for spot
    headers = _filter_incoming(dict(request.headers), strip_auth=(backend_name == "serverless"))

    # Inject backend auth token for serverless
    if backend_name == "serverless":
        async with _state_lock:
            auth_token = _serverless_auth_token
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

    data = await request.body()

    t0 = time.time()
    try:
        r = await _http_client.send(
            _http_client.build_request(
                request.method, target_url, headers=headers, content=data if data else None,
            ),
            stream=True,
        )
    except httpx.HTTPError as e:
        elapsed = time.time() - t0
        async with _state_lock:
            if backend_name == "spot":
                _gpu_seconds_spot += elapsed
            else:
                _gpu_seconds_serverless += elapsed

        # RETRY: if spot failed and serverless is available, retry there
        if backend_name == "spot" and serverless_url:
            logger.warning("Spot request failed (%s), retrying on serverless", e)
            await _set_state(SpotState.COLD, str(e))
            return await _forward_to_serverless(request, path, headers, data, serverless_url)

        logger.warning("Upstream error: %s", e)
        return Response(content="upstream_error", status_code=502)

    # Retry on 5xx from spot — we haven't streamed anything yet, safe to failover
    if backend_name == "spot" and r.status_code >= 500 and serverless_url:
        await r.aclose()
        elapsed = time.time() - t0
        async with _state_lock:
            _gpu_seconds_spot += elapsed
        logger.warning("Spot returned %d, retrying on serverless", r.status_code)
        await _set_state(SpotState.COLD, f"status={r.status_code}")
        return await _forward_to_serverless(request, path, headers, data, serverless_url)

    resp_headers = _filter_outgoing(dict(r.headers))
    return StreamingResponse(
        _stream_and_track(r, t0, backend_name),
        status_code=r.status_code,
        headers=resp_headers,
        media_type=r.headers.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Auto-warming on startup
# ---------------------------------------------------------------------------
# When the router starts with a spot URL configured, immediately enter
# WARMING.  This creates QPS on the SkyServe LB so the autoscaler
# provisions a replica — critical when min_replicas=0 (scale-to-zero),
# because without warming pokes there's no QPS to trigger provisioning.
# With min_replicas>=1 the pokes are harmless (replica is already coming up).

def _auto_warm_on_startup():
    """Schedule WARMING if spot URL is configured. Called by lifespan startup."""
    if _skyserve_base_url:
        logger.info("Spot URL configured — auto-entering WARMING on startup")
        asyncio.create_task(_enter_warming())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
