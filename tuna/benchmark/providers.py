"""Provider-specific helpers for cold start benchmarking."""

from __future__ import annotations

import os

import requests

LOG_CAPABLE_PROVIDERS = frozenset({"modal", "cloudrun", "cerebrium", "baseten"})
EXCLUDED_PROVIDERS = frozenset({"azure"})
EXCLUDED_REASON = {
    "azure": (
        "Azure Container Apps ManagedEnvironment takes 30+ min to create/delete, "
        "making cold start benchmarking impractical."
    ),
}


def validate_provider(provider: str) -> None:
    """Reject providers unsuitable for cold start benchmarking."""
    if provider in EXCLUDED_PROVIDERS:
        raise ValueError(EXCLUDED_REASON[provider])


def get_auth_headers(provider_name: str) -> dict[str, str]:
    """Return auth headers required for a provider's endpoints."""
    if provider_name == "runpod":
        key = os.environ.get("RUNPOD_API_KEY", "")
        if not key:
            raise RuntimeError("RUNPOD_API_KEY required for RunPod benchmarking")
        return {"Authorization": f"Bearer {key}"}
    if provider_name == "baseten":
        key = os.environ.get("BASETEN_API_KEY", "")
        if not key:
            raise RuntimeError("BASETEN_API_KEY required for Baseten benchmarking")
        return {"Authorization": f"Api-Key {key}"}
    return {}


def is_cold(
    provider_name: str,
    health_url: str,
    auth_headers: dict[str, str],
    metadata: dict | None = None,
) -> bool:
    """Check whether a provider endpoint is currently scaled to zero."""
    if provider_name == "runpod":
        return _is_cold_runpod(health_url, auth_headers)
    if provider_name == "baseten" and metadata:
        return _is_cold_baseten(metadata)
    return _is_cold_http(health_url, auth_headers)


def _is_cold_runpod(health_url: str, auth_headers: dict[str, str]) -> bool:
    """RunPod /health returns 200 with workers JSON even when cold."""
    try:
        resp = requests.get(health_url, headers=auth_headers, timeout=10)
        if resp.status_code != 200:
            return True
        w = resp.json().get("workers", {})
        return (
            w.get("ready", 0) == 0
            and w.get("running", 0) == 0
            and w.get("initializing", 0) == 0
        )
    except (requests.RequestException, ValueError):
        return True


def _is_cold_baseten(metadata: dict) -> bool:
    """Check Baseten model status via REST API — doesn't wake the container."""
    model_id = metadata.get("model_id", "")
    if not model_id:
        return True
    api_key = os.environ.get("BASETEN_API_KEY", "")
    if not api_key:
        return True
    try:
        resp = requests.get(
            f"https://api.baseten.co/v1/models/{model_id}",
            headers={"Authorization": f"Api-Key {api_key}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return True
        data = resp.json()
        # Check if any replicas are running in the production environment
        for env in data.get("model", {}).get("environments", []):
            if env.get("name") == "production":
                replicas = env.get("current_replicas", 0)
                return replicas == 0
        return True
    except (requests.RequestException, ValueError, KeyError):
        return True


def _is_cold_cerebrium(metadata: dict) -> bool:
    """Check if Cerebrium endpoint is cold via HTTP.

    Cerebrium's CLI (``cerebrium apps get``) doesn't expose current
    replica count, so we fall back to a single HTTP health check.
    The quiet-period logic in ``_wait_for_cold`` ensures we don't
    poll too frequently and reset the scaledown timer.
    """
    svc = metadata.get("service_name", "")
    project_id = metadata.get("project_id", "")
    region = metadata.get("region", "us-east-1")
    if not svc or not project_id:
        return True
    health_url = f"https://api.aws.{region}.cerebrium.ai/v4/{project_id}/{svc}/health"
    return _is_cold_http(health_url, {})


def _is_cold_http(health_url: str, auth_headers: dict[str, str]) -> bool:
    """Generic cold check: non-200 or connection error means cold."""
    try:
        resp = requests.get(health_url, headers=auth_headers, timeout=5)
        return resp.status_code != 200
    except requests.RequestException:
        return True


def trigger_cold_start(
    provider_name: str,
    endpoint_url: str,
    health_url: str,
    model: str,
    auth_headers: dict[str, str],
) -> None:
    """Send a request that triggers the provider to boot from cold."""
    if provider_name == "runpod":
        # RunPod health endpoint doesn't boot workers — must POST inference
        url = endpoint_url.rstrip("/")
        if not url.endswith("/v1/chat/completions"):
            url = f"{url}/v1/chat/completions"
        try:
            requests.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "stream": False,
                },
                headers={**auth_headers, "Content-Type": "application/json"},
                timeout=600,
            )
        except requests.RequestException:
            pass
    else:
        try:
            requests.get(health_url, headers=auth_headers, timeout=600)
        except requests.RequestException:
            pass


def supports_log_phases(provider_name: str) -> bool:
    """Whether we have a verified log watcher for this provider."""
    return provider_name in LOG_CAPABLE_PROVIDERS
