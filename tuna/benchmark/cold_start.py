"""Core cold start benchmark orchestration and data models."""

from __future__ import annotations

import csv
import io
import json
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

from tuna.benchmark.log_watchers import create_log_watcher
from tuna.benchmark.providers import (
    get_auth_headers,
    is_cold,
    supports_log_phases,
    trigger_cold_start,
    validate_provider,
)
from tuna.models import DeploymentResult
from tuna.state import DeploymentRecord, list_deployments, load_deployment


@dataclass
class RunResult:
    """Result of a single cold start measurement."""

    scenario: str  # "fresh_cold_start" or "warm_cold_start"
    provider: str
    gpu: str
    total_s: float
    health_ready_s: Optional[float] = None
    first_inference_s: Optional[float] = None
    ttft_s: Optional[float] = None
    container_boot_s: Optional[float] = None
    model_load_s: Optional[float] = None
    deploy_time_s: Optional[float] = None
    error: Optional[str] = None


def _record_to_result(record: DeploymentRecord) -> DeploymentResult:
    """Map a DeploymentRecord to a DeploymentResult for benchmark use."""
    endpoint = record.serverless_endpoint
    return DeploymentResult(
        provider=record.serverless_provider_name or "",
        endpoint_url=endpoint,
        health_url=f"{endpoint}/health" if endpoint else None,
        metadata=dict(record.serverless_metadata or {}),
    )


def _find_existing_deployment(
    provider: str, model: str,
) -> DeploymentRecord | None:
    """Find the most recent active deployment matching provider + model."""
    records = list_deployments(status="active")
    for r in records:
        if r.serverless_provider_name == provider and r.model_name == model:
            return r
    return None


def _wait_for_cold(
    provider_name: str,
    health_url: str,
    auth_headers: dict[str, str],
    timeout: float = 300,
    cooldown: float = 120,
    consecutive_required: int = 3,
    metadata: dict | None = None,
) -> bool:
    """Wait for the endpoint to scale to zero.

    Waits ``cooldown`` seconds without sending any requests (so health
    polls don't reset the provider's scaledown timer), then does a
    single check.  If still warm, goes quiet again for another full
    cooldown cycle.  RunPod uses JSON worker status so a single cold
    check is enough.
    """
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        # Quiet period — no requests, let the scaledown timer expire
        quiet_end = time.monotonic() + cooldown
        print(f"  Quiet period ({cooldown:.0f}s)...", flush=True)
        while time.monotonic() < quiet_end:
            if time.monotonic() - start >= timeout:
                break
            time.sleep(min(15, max(quiet_end - time.monotonic(), 0)))
            if time.monotonic() < quiet_end:
                print(
                    f"  Quiet period... {time.monotonic() - start:.0f}s elapsed",
                    flush=True,
                )

        if time.monotonic() - start >= timeout:
            break

        # Single check after quiet period
        cold = is_cold(provider_name, health_url, auth_headers, metadata=metadata)
        if cold:
            elapsed = time.monotonic() - start
            print(f"  Scale-to-zero confirmed after {elapsed:.0f}s")
            return True

        print(
            f"  Still warm after {time.monotonic() - start:.0f}s, "
            "restarting quiet period...",
            flush=True,
        )

    print(
        f"  WARNING: scale-to-zero not confirmed after {timeout:.0f}s",
        file=sys.stderr,
    )
    return False


def _wait_for_health(
    health_url: str,
    auth_headers: dict[str, str],
    timeout: float = 600,
) -> float | None:
    """Wait for health endpoint to return 200. Returns monotonic duration or None on timeout."""
    import requests as req

    start = time.monotonic()
    last_progress = start

    while time.monotonic() - start < timeout:
        try:
            resp = req.get(health_url, headers=auth_headers, timeout=10)
            if resp.status_code == 200:
                return time.monotonic() - start
        except Exception:
            pass

        now = time.monotonic()
        if now - last_progress >= 15:
            print(
                f"  Waiting for health... {now - start:.0f}s elapsed",
                flush=True,
            )
            last_progress = now
        time.sleep(2)

    return None


def _measure_ttft(
    endpoint_url: str,
    model: str,
    auth_headers: dict[str, str],
) -> tuple[float | None, float | None]:
    """Send an inference request. Returns (ttft_s, total_inference_s) as monotonic durations."""
    import requests

    url = endpoint_url.rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url = f"{url}/v1/chat/completions"

    start = time.monotonic()
    ttft = None
    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "stream": True,
            },
            headers={**auth_headers, "Content-Type": "application/json"},
            timeout=600,
            stream=True,
        )
        if resp.status_code != 200:
            print(f"  Warning: inference returned {resp.status_code}", file=sys.stderr)
            return None, None
        for chunk in resp.iter_lines():
            if ttft is None and chunk:
                ttft = time.monotonic() - start
        total = time.monotonic() - start
        return ttft, total
    except requests.RequestException as e:
        print(f"  Warning: inference request failed: {e}", file=sys.stderr)
        return None, None


def _single_run(
    provider_name: str,
    endpoint_url: str,
    health_url: str,
    model: str,
    gpu: str,
    auth_headers: dict[str, str],
    metadata: dict,
    scenario: str,
) -> RunResult:
    """Execute one cold start measurement cycle."""
    watcher = None
    if supports_log_phases(provider_name):
        watcher = create_log_watcher(provider_name, metadata)
    if watcher:
        watcher.start()

    t0 = time.monotonic()
    trigger_thread = threading.Thread(
        target=trigger_cold_start,
        args=(provider_name, endpoint_url, health_url, model, auth_headers),
        daemon=True,
    )
    trigger_thread.start()

    health_ready_s = _wait_for_health(health_url, auth_headers, timeout=600)
    ttft_s, inference_s = _measure_ttft(endpoint_url, model, auth_headers)
    total_s = time.monotonic() - t0

    # Phase breakdown: prefer log watcher (more precise), fall back to HTTP timing
    container_boot_s = None
    model_load_s = None
    if watcher:
        watcher.stop()
        p = watcher.phases
        if p.container_start and p.model_load_start:
            container_boot_s = p.model_load_start - p.container_start
        if p.model_load_start and p.ready:
            model_load_s = p.ready - p.model_load_start

    return RunResult(
        scenario=scenario,
        provider=provider_name,
        gpu=gpu,
        total_s=total_s,
        health_ready_s=health_ready_s,
        first_inference_s=inference_s,
        ttft_s=ttft_s,
        container_boot_s=container_boot_s,
        model_load_s=model_load_s,
    )


def _mean_run(runs: list[RunResult]) -> RunResult:
    """Compute mean of multiple runs, handling None values."""
    if len(runs) == 1:
        return runs[0]

    def _avg(values: list[float | None]) -> float | None:
        non_none = [v for v in values if v is not None]
        return statistics.mean(non_none) if non_none else None

    return RunResult(
        scenario=runs[0].scenario,
        provider=runs[0].provider,
        gpu=runs[0].gpu,
        total_s=statistics.mean([r.total_s for r in runs]),
        health_ready_s=_avg([r.health_ready_s for r in runs]),
        first_inference_s=_avg([r.first_inference_s for r in runs]),
        ttft_s=_avg([r.ttft_s for r in runs]),
        container_boot_s=_avg([r.container_boot_s for r in runs]),
        model_load_s=_avg([r.model_load_s for r in runs]),
        deploy_time_s=_avg([r.deploy_time_s for r in runs]),
    )


def run_warm_cold_start(
    provider: str,
    gpu: str,
    model: str,
    endpoint_url: str,
    health_url: str,
    metadata: dict,
    repeat: int = 3,
    idle_wait: int = 300,
) -> list[RunResult]:
    """Benchmark cold start on an existing (warm) deployment."""
    validate_provider(provider)
    auth_headers = get_auth_headers(provider)
    results: list[RunResult] = []

    for i in range(repeat):
        print(f"\n--- Warm cold start run {i + 1}/{repeat} ---")

        print("  Waiting for scale-to-zero...")
        if not _wait_for_cold(provider, health_url, auth_headers, timeout=idle_wait, metadata=metadata):
            print(f"  Skipping run {i + 1}: endpoint did not scale to zero")
            continue

        print("  Triggering cold start...")
        result = _single_run(
            provider, endpoint_url, health_url, model, gpu,
            auth_headers, metadata, "warm_cold_start",
        )
        results.append(result)
        print(f"  Total: {result.total_s:.1f}s")

    return results


def _teardown(service_name: str) -> None:
    """Tear down a deployment by service name."""
    print("  Tearing down deployment...")
    try:
        from tuna.orchestrator import destroy_hybrid
        from tuna.providers.registry import ensure_providers_for_deployment
        from tuna.state import update_deployment_status

        record = load_deployment(service_name)
        if record:
            ensure_providers_for_deployment(record)
        destroy_hybrid(service_name, record=record)
        update_deployment_status(service_name, "destroyed")
    except Exception as e:
        print(f"  Warning: teardown failed: {e}", file=sys.stderr)


def run_fresh_cold_start(
    provider: str,
    gpu: str,
    model: str,
    max_model_len: int = 512,
    no_teardown: bool = False,
) -> list[RunResult]:
    """Deploy fresh, measure cold start, optionally teardown."""
    from tuna.models import DeployRequest
    from tuna.orchestrator import launch_serverless_only
    from tuna.scaling import ScalingPolicy, ServerlessScaling, SpotScaling

    validate_provider(provider)

    scaling = ScalingPolicy(
        serverless=ServerlessScaling(scaledown_window=30),
        spot=SpotScaling(),
    )
    request = DeployRequest(
        model_name=model,
        gpu=gpu,
        serverless_provider=provider,
        max_model_len=max_model_len,
        cold_start_mode="fast_boot",
        public=True,
        serverless_only=True,
        scaling=scaling,
    )

    from tuna.providers.registry import ensure_provider_registered

    ensure_provider_registered(provider)

    # Try to start log watcher BEFORE deploy with pre-known metadata.
    # Works for providers where service_name/app_name is enough (Modal, Cerebrium, CloudRun).
    # For providers needing post-deploy IDs (Baseten), this returns None and we retry after.
    pre_metadata = {"service_name": f"{request.service_name}-serverless"}
    watcher = None
    if supports_log_phases(provider):
        watcher = create_log_watcher(provider, pre_metadata)
        if watcher:
            watcher.start()

    print(f"Deploying {model} on {provider} ({gpu})...")
    t_deploy_start = time.monotonic()
    result = launch_serverless_only(request)
    deploy_time = time.monotonic() - t_deploy_start
    print(f"  Deploy completed in {deploy_time:.1f}s")

    # Save to state so destroy_hybrid can find it
    from tuna.state import save_deployment

    save_deployment(request, result)

    if not result.serverless or not result.serverless.endpoint_url:
        if watcher:
            watcher.stop()
        err = "Deployment failed — no endpoint returned"
        if result.serverless and result.serverless.error:
            err = result.serverless.error
        print(f"  ERROR: {err}", file=sys.stderr)
        return [
            RunResult(
                scenario="fresh_cold_start",
                provider=provider,
                gpu=gpu,
                total_s=deploy_time,
                deploy_time_s=deploy_time,
                error=err,
            )
        ]

    endpoint_url = result.serverless.endpoint_url
    health_url = result.serverless.health_url or f"{endpoint_url}/health"
    metadata = dict(result.serverless.metadata or {})
    auth_headers = get_auth_headers(provider)

    # If pre-deploy watcher wasn't possible (e.g. Baseten needs model_id),
    # start it now with real metadata — container is still booting.
    if watcher is None and supports_log_phases(provider):
        watcher = create_log_watcher(provider, metadata)
        if watcher:
            watcher.start()

    # The deploy CLI returns fast but the container may still be booting.
    # Wait for health to be ready — this is the real cold start.
    t0 = time.monotonic()
    print("  Waiting for container to be ready...")
    health_ready_s = _wait_for_health(health_url, auth_headers, timeout=600)

    # Extract log phases (wall-clock diffs)
    container_boot_s = None
    model_load_s = None
    if watcher:
        watcher.stop()
        p = watcher.phases
        if p.container_start and p.model_load_start:
            container_boot_s = p.model_load_start - p.container_start
        if p.model_load_start and p.ready:
            model_load_s = p.ready - p.model_load_start

    if health_ready_s is None:
        total_s = time.monotonic() - t0 + deploy_time
        run = RunResult(
            scenario="fresh_cold_start",
            provider=provider,
            gpu=gpu,
            total_s=total_s,
            deploy_time_s=deploy_time,
            container_boot_s=container_boot_s,
            model_load_s=model_load_s,
            error="Health endpoint never became ready (timeout 600s)",
        )
        if not no_teardown:
            _teardown(request.service_name)
        return [run]

    print("  Measuring first inference...")
    ttft_s, inference_s = _measure_ttft(endpoint_url, model, auth_headers)
    total_s = time.monotonic() - t0 + deploy_time

    run = RunResult(
        scenario="fresh_cold_start",
        provider=provider,
        gpu=gpu,
        total_s=total_s,
        deploy_time_s=deploy_time,
        health_ready_s=health_ready_s + deploy_time,
        first_inference_s=inference_s,
        ttft_s=ttft_s,
        container_boot_s=container_boot_s,
        model_load_s=model_load_s,
    )

    if not no_teardown:
        _teardown(request.service_name)

    return [run]


def run_auto(
    provider: str,
    gpu: str,
    model: str,
    scenario: str = "both",
    repeat: int = 3,
    idle_wait: int = 300,
    max_model_len: int = 512,
    no_teardown: bool = False,
) -> list[RunResult]:
    """Auto-detect existing deployment or deploy fresh."""
    validate_provider(provider)
    results: list[RunResult] = []

    # For "both": keep fresh deploy alive so warm phase can use it
    keep_for_warm = scenario == "both"

    if scenario in ("fresh-cold", "both"):
        results.extend(
            run_fresh_cold_start(
                provider, gpu, model, max_model_len,
                no_teardown=no_teardown or keep_for_warm,
            )
        )

    if scenario in ("warm-cold", "both"):
        record = _find_existing_deployment(provider, model)
        if record and record.serverless_endpoint:
            dr = _record_to_result(record)
            results.extend(
                run_warm_cold_start(
                    provider,
                    gpu,
                    model,
                    dr.endpoint_url,
                    dr.health_url or f"{dr.endpoint_url}/health",
                    dr.metadata,
                    repeat=repeat,
                    idle_wait=idle_wait,
                )
            )
            # Teardown after warm phase completes (unless --no-teardown)
            if not no_teardown and keep_for_warm:
                _teardown(record.service_name)
        elif scenario == "warm-cold":
            print(
                f"No active deployment found for {provider}/{model}. "
                "Deploy first or use --scenario fresh-cold.",
                file=sys.stderr,
            )

    return results


def print_summary(results: list[RunResult], output: str = "table") -> None:
    """Print results in the requested format."""
    if output == "json":
        print(json.dumps([asdict(r) for r in results], indent=2))
    elif output == "csv":
        _print_csv(results)
    else:
        _print_table(results)


def _print_table(results: list[RunResult]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Cold Start Benchmark Results")
    table.add_column("Provider")
    table.add_column("GPU")
    table.add_column("Scenario")
    has_deploy = any(r.deploy_time_s for r in results)
    if has_deploy:
        table.add_column("Deploy")
    table.add_column("Container Boot")
    table.add_column("Model Load")
    table.add_column("Health Ready")
    table.add_column("First Inference")
    table.add_column("Total")

    has_errors = any(r.error for r in results)
    if has_errors:
        table.add_column("Error")

    for r in results:
        row = [
            r.provider,
            r.gpu,
            r.scenario,
        ]
        if has_deploy:
            row.append(f"{r.deploy_time_s:.1f}s" if r.deploy_time_s else "\u2014")
        row.extend([
            f"{r.container_boot_s:.1f}s" if r.container_boot_s else "\u2014",
            f"{r.model_load_s:.1f}s" if r.model_load_s else "\u2014",
            f"{r.health_ready_s:.1f}s" if r.health_ready_s else "\u2014",
            f"{r.first_inference_s:.1f}s" if r.first_inference_s else "\u2014",
            f"{r.total_s:.1f}s",
        ])
        if has_errors:
            row.append(r.error or "")
        table.add_row(*row)
    Console().print(table)


def _print_csv(results: list[RunResult]) -> None:
    fieldnames = list(asdict(results[0]).keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(asdict(r))
