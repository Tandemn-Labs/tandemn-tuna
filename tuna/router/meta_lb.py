"""Meta load balancer — routes between serverless and spot backends.

Adapted from the proven SAM load_balancer.py pattern. Provider-agnostic:
only knows about URLs, never imports Modal/SkyPilot.

Env vars:
  SERVERLESS_BASE_URL     e.g. https://xxx.modal.run
  SKYSERVE_BASE_URL       e.g. http://x.x.x.x:30001

  SKYSERVE_READY_PATH     default: /health
  SKYSERVE_POKE_PATH      default: /health (trigger scale-up)

  PROBE_TIMEOUT_SECONDS   default: 1.0
  POKE_TIMEOUT_SECONDS    default: 0.3
  UPSTREAM_TIMEOUT_SECONDS default: 210.0

  CHECK_MIN_INTERVAL_SECONDS default: 1.0
  POKE_MIN_INTERVAL_SECONDS  default: 0.5

  API_KEY                 if set, required for all requests
  API_KEY_HEADER          default: x-api-key
  ALLOW_HEALTH_NO_AUTH    default: 0

Run:
  gunicorn -w 2 -k gthread --threads 8 --timeout 300 --bind 0.0.0.0:8080 meta_lb:app
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import requests as req_lib
from flask import Flask, Response, request

app = Flask(__name__)
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
    from urllib.parse import urlparse, quote, urlencode, parse_qs
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

BG_MAX_WORKERS = int(os.getenv("BG_MAX_WORKERS", "4"))
EXECUTOR = ThreadPoolExecutor(max_workers=BG_MAX_WORKERS)

SKYSERVE_READY_PATH = os.getenv("SKYSERVE_READY_PATH", "/health")
SKYSERVE_POKE_PATH = os.getenv("SKYSERVE_POKE_PATH", "/health")

PROBE_TIMEOUT_SECONDS = _env_float("PROBE_TIMEOUT_SECONDS", 1.0)
POKE_TIMEOUT_SECONDS = _env_float("POKE_TIMEOUT_SECONDS", 0.3)
UPSTREAM_TIMEOUT_SECONDS = _env_float("UPSTREAM_TIMEOUT_SECONDS", 210.0)
CHECK_MIN_INTERVAL_SECONDS = _env_float("CHECK_MIN_INTERVAL_SECONDS", 1.0)
POKE_MIN_INTERVAL_SECONDS = _env_float("POKE_MIN_INTERVAL_SECONDS", 0.5)

API_KEY = os.getenv("API_KEY", "")
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "x-api-key")
ALLOW_HEALTH_NO_AUTH = _env_bool("ALLOW_HEALTH_NO_AUTH", False)

ROUTE_WINDOW_SIZE = int(os.getenv("ROUTE_WINDOW_SIZE", "200"))

SESSION = req_lib.Session()

# ---------------------------------------------------------------------------
# Mutable state — backend URLs can be updated at runtime via /router/config
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_serverless_base_url: str = os.environ.get("SERVERLESS_BASE_URL", "").rstrip("/")
_serverless_auth_token: str = os.environ.get("SERVERLESS_AUTH_TOKEN", "")
_skyserve_base_url: str = os.environ.get("SKYSERVE_BASE_URL", "").rstrip("/")
_skyserve_ready: bool = False
_last_probe_ts: float | None = None
_last_probe_err: str | None = None
_last_check_ts: float = 0.0
_last_poke_ts: float = 0.0

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


# ---------------------------------------------------------------------------
# State accessors
# ---------------------------------------------------------------------------

def _get_serverless_url() -> str:
    with _state_lock:
        return _serverless_base_url


def _get_skyserve_url() -> str:
    with _state_lock:
        return _skyserve_base_url


def set_serverless_url(url: str) -> None:
    global _serverless_base_url
    with _state_lock:
        _serverless_base_url = url.rstrip("/")
    logger.info("Serverless URL updated: %s", url)


def set_serverless_auth_token(token: str) -> None:
    global _serverless_auth_token
    with _state_lock:
        _serverless_auth_token = token
    logger.info("Serverless auth token updated")


def set_spot_url(url: str) -> None:
    global _skyserve_base_url
    with _state_lock:
        _skyserve_base_url = url.rstrip("/")
    logger.info("Spot URL updated: %s", url)


def _set_ready(val: bool, err: str | None = None) -> None:
    global _skyserve_ready, _last_probe_ts, _last_probe_err
    global _spot_ready_cumulative_s, _spot_ready_since
    with _state_lock:
        now = time.time()
        # Accumulate spot-ready time on state transitions
        if _skyserve_ready and not val and _spot_ready_since is not None:
            _spot_ready_cumulative_s += now - _spot_ready_since
            _spot_ready_since = None
        elif not _skyserve_ready and val:
            _spot_ready_since = now
        _skyserve_ready = val
        _last_probe_ts = now
        _last_probe_err = err


def _is_ready() -> bool:
    with _state_lock:
        return _skyserve_ready


# ---------------------------------------------------------------------------
# Route stats
# ---------------------------------------------------------------------------

def _record_route(backend: str) -> None:
    global _req_total, _req_to_spot, _req_to_serverless
    with _state_lock:
        _req_total += 1
        _recent_routes.append(backend)
        if backend == "spot":
            _req_to_spot += 1
        else:
            _req_to_serverless += 1


def _route_stats() -> dict:
    with _state_lock:
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

def _extract_api_key(req) -> str:
    key = req.headers.get(API_KEY_HEADER)
    if not key:
        auth = req.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth.split(" ", 1)[1]
    return key or ""


def _is_authorized(req) -> bool:
    if not API_KEY:
        return True
    provided = _extract_api_key(req)
    if not provided:
        return False
    return hmac.compare_digest(provided, API_KEY)


# ---------------------------------------------------------------------------
# Background probes
# ---------------------------------------------------------------------------

def _check_skyserve_ready_async() -> None:
    skyserve_url = _get_skyserve_url()
    if not skyserve_url:
        return

    global _last_check_ts
    now = time.time()
    with _state_lock:
        if now - _last_check_ts < CHECK_MIN_INTERVAL_SECONDS:
            return
        _last_check_ts = now

    ready_url = _join_url(skyserve_url, SKYSERVE_READY_PATH)

    def _do():
        try:
            r = req_lib.get(ready_url, timeout=PROBE_TIMEOUT_SECONDS)
            if 200 <= r.status_code < 300:
                _set_ready(True, None)
            else:
                _set_ready(False, f"status={r.status_code}")
        except Exception as e:
            _set_ready(False, str(e))

    threading.Thread(target=_do, daemon=True).start()


def _check_skyserve_ready_sync() -> None:
    """Synchronous readiness check — used by /router/health to avoid stale state."""
    skyserve_url = _get_skyserve_url()
    if not skyserve_url:
        return
    ready_url = _join_url(skyserve_url, SKYSERVE_READY_PATH)
    try:
        r = req_lib.get(ready_url, timeout=PROBE_TIMEOUT_SECONDS)
        if 200 <= r.status_code < 300:
            _set_ready(True, None)
        else:
            _set_ready(False, f"status={r.status_code}")
    except Exception as e:
        _set_ready(False, str(e))


def _poke_skyserve_async() -> None:
    skyserve_url = _get_skyserve_url()
    if not skyserve_url:
        return

    global _last_poke_ts
    now = time.time()
    with _state_lock:
        if now - _last_poke_ts < POKE_MIN_INTERVAL_SECONDS:
            return
        _last_poke_ts = now

    poke_url = _join_url(skyserve_url, SKYSERVE_POKE_PATH)

    def _do():
        try:
            req_lib.get(poke_url, timeout=POKE_TIMEOUT_SECONDS)
        except Exception:
            pass

    EXECUTOR.submit(_do)


# ---------------------------------------------------------------------------
# Spot → serverless failover
# ---------------------------------------------------------------------------

def _forward_to_serverless(path, headers, data, serverless_url):
    """Retry a failed spot request on the serverless backend."""
    global _gpu_seconds_serverless
    target_url = _build_proxy_url(serverless_url, path, request.query_string or None)

    # Swap in serverless auth token
    with _state_lock:
        auth_token = _serverless_auth_token
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    else:
        headers.pop("Authorization", None)

    _record_route("serverless")  # Count the retry as a serverless route

    t0 = time.time()
    try:
        r = SESSION.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=data if data else None,
            allow_redirects=True,
            timeout=(2.0, UPSTREAM_TIMEOUT_SECONDS),
            stream=True,
        )
    except req_lib.RequestException as e:
        elapsed = time.time() - t0
        with _state_lock:
            _gpu_seconds_serverless += elapsed
        logger.warning("Upstream error: %s", e)
        return Response("upstream_error", status=502)

    resp_headers = _filter_outgoing(r.headers)

    def generate():
        global _gpu_seconds_serverless
        try:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk
        finally:
            r.close()
            elapsed = time.time() - t0
            with _state_lock:
                _gpu_seconds_serverless += elapsed

    return Response(
        response=generate(),
        status=r.status_code,
        headers=resp_headers,
        content_type=r.headers.get("content-type"),
        direct_passthrough=True,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/router/health", methods=["GET"])
def router_health():
    if not ALLOW_HEALTH_NO_AUTH and not _is_authorized(request):
        return Response("unauthorized", status=401)

    # Refresh spot readiness so cost stats aren't stale
    _check_skyserve_ready_sync()

    # Snapshot state and stats separately to avoid nested lock acquisition
    with _state_lock:
        state = {
            "skyserve_ready": _skyserve_ready,
            "last_probe_ts": _last_probe_ts,
            "last_probe_err": _last_probe_err,
            "serverless_base_url": _serverless_base_url,
            "skyserve_base_url": _skyserve_base_url,
        }
    state["route_stats"] = _route_stats()

    return Response(
        response=json.dumps(state),
        status=200,
        mimetype="application/json",
    )


@app.route("/router/config", methods=["POST"])
def update_config():
    """Orchestrator pushes backend URLs here after deploy."""
    if not _is_authorized(request):
        return Response("unauthorized", status=401)
    data = request.get_json(silent=True) or {}
    if "serverless_url" in data:
        set_serverless_url(data["serverless_url"])
    if "serverless_auth_token" in data:
        set_serverless_auth_token(data["serverless_auth_token"])
    if "spot_url" in data:
        set_spot_url(data["spot_url"])
    return Response(
        response=json.dumps({"status": "ok"}),
        status=200,
        mimetype="application/json",
    )


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
def proxy(path: str):
    global _gpu_seconds_spot, _gpu_seconds_serverless
    serverless_url = _get_serverless_url()
    skyserve_url = _get_skyserve_url()

    if not serverless_url and not skyserve_url:
        return Response(
            json.dumps({"error": "No backends configured yet"}),
            status=503,
            mimetype="application/json",
        )

    if not _is_authorized(request):
        return Response("unauthorized", status=401)

    # Decide backend: prefer spot (cheaper) if ready, else serverless (fast)
    if skyserve_url and _is_ready():
        backend_base = skyserve_url
        backend_name = "spot"
        _record_route("spot")
    elif serverless_url:
        backend_base = serverless_url
        backend_name = "serverless"
        _record_route("serverless")
        # Poke spot to trigger scale-up during cold start
        if skyserve_url:
            _poke_skyserve_async()
            _check_skyserve_ready_async()
    else:
        # Only spot configured but not ready
        return Response(
            json.dumps({"error": "Spot backend not ready, no serverless fallback"}),
            status=503,
            mimetype="application/json",
        )

    # Log periodically
    stats = _route_stats()
    if stats["total"] % 100 == 0 and stats["total"] > 0:
        logger.info(
            "requests=%d spot=%d (%.0f%%) serverless=%d (%.0f%%) spot_ready=%s",
            stats["total"], stats["spot"], stats["pct_spot"],
            stats["serverless"], stats["pct_serverless"], _is_ready(),
        )

    # Forward request
    target_url = _build_proxy_url(backend_base, path, request.query_string or None)

    # Strip auth headers for serverless (we inject our own token); preserve for spot
    headers = _filter_incoming(dict(request.headers), strip_auth=(backend_name == "serverless"))

    # Inject backend auth token for serverless
    if backend_name == "serverless":
        with _state_lock:
            auth_token = _serverless_auth_token
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

    data = request.get_data()

    t0 = time.time()
    try:
        r = SESSION.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=data if data else None,
            allow_redirects=True,
            timeout=(2.0, UPSTREAM_TIMEOUT_SECONDS),
            stream=True,
        )
    except req_lib.RequestException as e:
        elapsed = time.time() - t0
        with _state_lock:
            if backend_name == "spot":
                _gpu_seconds_spot += elapsed
            else:
                _gpu_seconds_serverless += elapsed

        # RETRY: if spot failed and serverless is available, retry there
        if backend_name == "spot" and serverless_url:
            logger.warning("Spot request failed (%s), retrying on serverless", e)
            _set_ready(False, str(e))
            return _forward_to_serverless(path, headers, data, serverless_url)

        logger.warning("Upstream error: %s", e)
        return Response("upstream_error", status=502)

    # Retry on 5xx from spot — we haven't streamed anything yet, safe to failover
    if backend_name == "spot" and r.status_code >= 500 and serverless_url:
        r.close()
        elapsed = time.time() - t0
        with _state_lock:
            _gpu_seconds_spot += elapsed
        logger.warning("Spot returned %d, retrying on serverless", r.status_code)
        _set_ready(False, f"status={r.status_code}")
        return _forward_to_serverless(path, headers, data, serverless_url)

    resp_headers = _filter_outgoing(r.headers)

    def generate():
        global _gpu_seconds_spot, _gpu_seconds_serverless
        try:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk
        finally:
            r.close()
            elapsed = time.time() - t0
            with _state_lock:
                if backend_name == "spot":
                    _gpu_seconds_spot += elapsed
                else:
                    _gpu_seconds_serverless += elapsed

    return Response(
        response=generate(),
        status=r.status_code,
        headers=resp_headers,
        content_type=r.headers.get("content-type"),
        direct_passthrough=True,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
