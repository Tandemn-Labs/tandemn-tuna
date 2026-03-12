"""aiperf-powered load test with tuna cost sidecar.

Runs ``aiperf profile`` as a subprocess for production-grade LLM metrics
(TTFT, ITL, token throughput) while polling the tuna router's /router/health
endpoint in parallel for cost tracking (spot vs serverless split).
"""

from __future__ import annotations

import asyncio
import csv
import glob
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import httpx


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AiperfReport:
    """Combined aiperf metrics + tuna cost metrics."""

    # aiperf performance metrics
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    itl_p50_ms: float = 0.0
    itl_p90_ms: float = 0.0
    itl_p99_ms: float = 0.0
    output_token_throughput: float = 0.0   # tokens/sec
    request_throughput: float = 0.0        # req/sec
    request_latency_p50_ms: float = 0.0
    request_latency_p90_ms: float = 0.0
    request_latency_p99_ms: float = 0.0
    total_requests: int = 0
    success_count: int = 0
    failure_count: int = 0
    actual_duration_s: float = 0.0

    # tuna cost metrics (from /router/health sidecar)
    spot_requests: int = 0
    serverless_requests: int = 0
    pct_spot: float = 0.0
    gpu_seconds_spot: float = 0.0
    gpu_seconds_serverless: float = 0.0
    spot_ready_seconds: float = 0.0
    uptime_seconds: float = 0.0
    failover_events: int = 0

    # derived cost (filled by caller with real prices)
    estimated_cost_spot: float = 0.0
    estimated_cost_serverless: float = 0.0
    estimated_cost_hybrid: float = 0.0
    counterfactual_all_serverless: float = 0.0
    counterfactual_all_on_demand: float = 0.0
    savings_vs_serverless: float = 0.0
    savings_vs_on_demand: float = 0.0

    # config
    model: str = ""
    concurrency: int = 0
    request_rate: float = 0.0


# ---------------------------------------------------------------------------
# Cost sidecar — polls /router/health while aiperf runs
# ---------------------------------------------------------------------------

@dataclass
class _RouterSnapshot:
    ts: float
    total: int
    spot: int
    serverless: int
    gpu_s_spot: float
    gpu_s_svl: float


async def _cost_sidecar(
    endpoint_url: str,
    api_key: str | None,
    stop: asyncio.Event,
    snapshots: list[_RouterSnapshot],
    interval: float = 30.0,
) -> None:
    """Poll /router/health and record snapshots until stopped."""
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    url = endpoint_url.rstrip("/") + "/router/health"

    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(15)) as client:
        while not stop.is_set():
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    body = resp.json()
                    s = body.get("route_stats", body)
                    snapshots.append(_RouterSnapshot(
                        ts=time.time(),
                        total=int(s.get("total", 0)),
                        spot=int(s.get("spot", 0)),
                        serverless=int(s.get("serverless", 0)),
                        gpu_s_spot=float(s.get("gpu_seconds_spot", 0.0)),
                        gpu_s_svl=float(s.get("gpu_seconds_serverless", 0.0)),
                    ))
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass


def _count_failovers(snapshots: list[_RouterSnapshot]) -> int:
    """Count times spot routing rate dropped from >50% to <20%."""
    events = 0
    for i in range(1, len(snapshots)):
        prev, curr = snapshots[i - 1], snapshots[i]
        if prev.total > 0 and curr.total > 0:
            if prev.spot / prev.total > 0.5 and curr.spot / curr.total < 0.2:
                events += 1
    return events


# ---------------------------------------------------------------------------
# aiperf output parsing
# ---------------------------------------------------------------------------

def _parse_aiperf_output(artifact_dir: str) -> dict:
    """Parse aiperf's output.json from the artifact directory."""
    # aiperf writes to artifacts/<model>-<endpoint_type>-<mode>/output.json
    pattern = os.path.join(artifact_dir, "*", "output.json")
    files = glob.glob(pattern)
    if not files:
        # Try top-level
        top = os.path.join(artifact_dir, "output.json")
        if os.path.exists(top):
            files = [top]
    if not files:
        return {}

    with open(files[0]) as f:
        return json.load(f)


def _extract_metric(data: dict, metric_name: str, stat: str = "avg") -> float:
    """Extract a metric value from aiperf's nested output format."""
    # aiperf output structure varies — try common paths
    for path in [
        ["data", "records", metric_name, stat],
        ["records", metric_name, stat],
        [metric_name, stat],
    ]:
        obj = data
        for key in path:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                obj = None
                break
        if obj is not None:
            try:
                return float(obj)
            except (TypeError, ValueError):
                pass
    return 0.0


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _find_aiperf_bin() -> str | None:
    """Find the aiperf binary — check venv first, then system PATH."""
    import shutil
    # Check venv bin dir (where uv pip install puts it)
    venv_bin = os.path.join(sys.prefix, "bin", "aiperf")
    if os.path.isfile(venv_bin) and os.access(venv_bin, os.X_OK):
        return venv_bin
    # Fall back to system PATH
    found = shutil.which("aiperf")
    return found


def aiperf_available() -> bool:
    """Check if aiperf CLI is installed."""
    return _find_aiperf_bin() is not None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_aiperf_benchmark(
    endpoint_url: str,
    model: str,
    duration_s: float,
    concurrency: int = 30,
    request_rate: float | None = None,
    request_rate_mode: str = "poisson",
    streaming: bool = True,
    isl: int = 550,
    osl: int = 150,
    api_key: str | None = None,
    output_dir: str | None = None,
    ui_mode: str = "simple",
    extra_args: list[str] | None = None,
) -> AiperfReport:
    """Run aiperf benchmark with parallel cost sidecar.

    Returns a combined report with aiperf metrics + tuna cost tracking.
    """
    artifact_dir = output_dir or tempfile.mkdtemp(prefix="tuna_aiperf_")

    # Build aiperf command
    aiperf_bin = _find_aiperf_bin()
    if not aiperf_bin:
        print("Error: aiperf not installed. Install with: uv pip install aiperf", file=sys.stderr)
        sys.exit(1)

    cmd = [
        aiperf_bin, "profile",
        "--model", model,
        "--url", endpoint_url.rstrip("/"),
        "--endpoint-type", "chat",
        "--benchmark-duration", str(int(duration_s)),
        "--isl", str(isl),
        "--osl", str(osl),
        "--output-artifact-dir", artifact_dir,
        "--ui-type", ui_mode,
        "--use-legacy-max-tokens",
    ]

    if streaming:
        cmd.append("--streaming")

    if request_rate is not None:
        cmd += ["--request-rate", str(request_rate)]
        cmd += ["--request-rate-mode", request_rate_mode]
        if concurrency:
            cmd += ["--concurrency", str(concurrency)]
    else:
        cmd += ["--concurrency", str(concurrency)]

    if api_key:
        cmd += ["--api-key", api_key]

    if extra_args:
        cmd += extra_args

    print(f"Running: {shlex.join(cmd)}", flush=True)
    print(f"Artifacts: {artifact_dir}", flush=True)

    # Run aiperf + cost sidecar in parallel
    snapshots: list[_RouterSnapshot] = []
    stop = asyncio.Event()
    t_start = time.monotonic()

    async def _run():
        # Start cost sidecar
        sidecar = asyncio.create_task(
            _cost_sidecar(endpoint_url, api_key, stop, snapshots)
        )

        # Run aiperf subprocess (stream output to terminal)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        await proc.wait()

        # Stop sidecar
        stop.set()
        sidecar.cancel()
        try:
            await sidecar
        except (asyncio.CancelledError, Exception):
            pass

        return proc.returncode

    returncode = asyncio.run(_run())
    actual_duration = time.monotonic() - t_start

    if returncode != 0:
        print(f"Warning: aiperf exited with code {returncode}", file=sys.stderr)

    # Parse aiperf results
    aiperf_data = _parse_aiperf_output(artifact_dir)

    # Extract metrics
    report = AiperfReport(
        model=model,
        concurrency=concurrency,
        request_rate=request_rate or 0.0,
        actual_duration_s=round(actual_duration, 1),
        # TTFT
        ttft_p50_ms=_extract_metric(aiperf_data, "time_to_first_token", "p50") * 1000,
        ttft_p90_ms=_extract_metric(aiperf_data, "time_to_first_token", "p90") * 1000,
        ttft_p99_ms=_extract_metric(aiperf_data, "time_to_first_token", "p99") * 1000,
        # ITL
        itl_p50_ms=_extract_metric(aiperf_data, "inter_token_latency", "p50") * 1000,
        itl_p90_ms=_extract_metric(aiperf_data, "inter_token_latency", "p90") * 1000,
        itl_p99_ms=_extract_metric(aiperf_data, "inter_token_latency", "p99") * 1000,
        # Throughput
        output_token_throughput=_extract_metric(aiperf_data, "output_token_throughput", "avg"),
        request_throughput=_extract_metric(aiperf_data, "request_throughput", "avg"),
        # Request latency
        request_latency_p50_ms=_extract_metric(aiperf_data, "request_latency", "p50") * 1000,
        request_latency_p90_ms=_extract_metric(aiperf_data, "request_latency", "p90") * 1000,
        request_latency_p99_ms=_extract_metric(aiperf_data, "request_latency", "p99") * 1000,
        # Counts
        total_requests=int(_extract_metric(aiperf_data, "request_count", "avg") or
                           _extract_metric(aiperf_data, "total_requests", "avg")),
    )

    # Fill cost metrics from sidecar snapshots
    if len(snapshots) >= 2:
        first, last = snapshots[0], snapshots[-1]
        report.gpu_seconds_spot = round(max(0.0, last.gpu_s_spot - first.gpu_s_spot), 2)
        report.gpu_seconds_serverless = round(max(0.0, last.gpu_s_svl - first.gpu_s_svl), 2)
        report.spot_requests = max(0, last.spot - first.spot)
        report.serverless_requests = max(0, last.serverless - first.serverless)
        total_reqs = report.spot_requests + report.serverless_requests
        report.pct_spot = round(report.spot_requests / total_reqs * 100, 1) if total_reqs else 0.0
        report.failover_events = _count_failovers(snapshots)

    # Get final router state for uptime/spot_ready
    if snapshots:
        try:
            headers = {}
            if api_key:
                headers["x-api-key"] = api_key
            import requests as req_lib
            resp = req_lib.get(
                endpoint_url.rstrip("/") + "/router/health",
                headers=headers, timeout=10,
            )
            if resp.status_code == 200:
                s = resp.json().get("route_stats", {})
                report.uptime_seconds = s.get("uptime_seconds", 0.0)
                report.spot_ready_seconds = s.get("spot_ready_seconds", 0.0)
        except Exception:
            pass

    # Total requests from sidecar if aiperf count is missing
    if report.total_requests == 0:
        report.total_requests = report.spot_requests + report.serverless_requests
    report.success_count = report.total_requests - report.failure_count

    return report


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_aiperf_summary(report: AiperfReport, output: str = "table") -> None:
    """Print the combined report."""
    if output == "json":
        print(json.dumps(asdict(report), indent=2))
    elif output == "csv":
        d = asdict(report)
        writer = csv.DictWriter(sys.stdout, fieldnames=list(d.keys()))
        writer.writeheader()
        writer.writerow(d)
    else:
        _print_aiperf_table(report)


def _print_aiperf_table(report: AiperfReport) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Performance table
    perf = Table(title="Performance (aiperf)", show_header=True, header_style="bold", expand=True)
    perf.add_column("Metric")
    perf.add_column("Value", justify="right")

    perf.add_row("Duration", f"{report.actual_duration_s:.0f}s")
    perf.add_row("Model", report.model)
    perf.add_row("Concurrency", str(report.concurrency) if report.concurrency else f"{report.request_rate} req/s")
    perf.add_row("Total Requests", f"{report.total_requests:,}")
    perf.add_row("Failures", f"{report.failure_count:,}")
    perf.add_row("")
    perf.add_row("TTFT p50 / p90 / p99",
                 f"{report.ttft_p50_ms:.0f}ms / {report.ttft_p90_ms:.0f}ms / {report.ttft_p99_ms:.0f}ms")
    perf.add_row("ITL p50 / p90 / p99",
                 f"{report.itl_p50_ms:.1f}ms / {report.itl_p90_ms:.1f}ms / {report.itl_p99_ms:.1f}ms")
    perf.add_row("Request Latency p50 / p90 / p99",
                 f"{report.request_latency_p50_ms:.0f}ms / {report.request_latency_p90_ms:.0f}ms / {report.request_latency_p99_ms:.0f}ms")
    perf.add_row("Output Token Throughput", f"{report.output_token_throughput:.1f} tokens/sec")
    perf.add_row("Request Throughput", f"{report.request_throughput:.2f} req/sec")

    console.print(perf)
    console.print()

    # Cost table
    cost = Table(title="Cost (tuna router)", show_header=True, header_style="bold", expand=True)
    cost.add_column("Metric")
    cost.add_column("Value", justify="right")

    pct_svl = 100 - report.pct_spot if report.pct_spot else 0
    cost.add_row("Spot Requests", f"{report.spot_requests:,} ({report.pct_spot:.1f}%)")
    cost.add_row("Serverless Requests", f"{report.serverless_requests:,} ({pct_svl:.1f}%)")
    cost.add_row("GPU-sec Spot", f"{report.gpu_seconds_spot:.1f}s")
    cost.add_row("GPU-sec Serverless", f"{report.gpu_seconds_serverless:.1f}s")
    cost.add_row("Spot Ready Time", f"{report.spot_ready_seconds:.0f}s")
    cost.add_row("Failover Events", str(report.failover_events))
    cost.add_row("")
    cost.add_row("[bold]Hybrid Cost[/bold]", f"[bold]${report.estimated_cost_hybrid:.2f}[/bold]")
    cost.add_row("All-Serverless Would Be", f"${report.counterfactual_all_serverless:.2f}")
    cost.add_row("All-On-Demand Would Be", f"${report.counterfactual_all_on_demand:.2f}")

    if report.savings_vs_serverless > 0:
        cost.add_row("[green]Savings vs Serverless[/green]",
                     f"[green]${report.savings_vs_serverless:.2f} ({report.savings_vs_serverless / report.counterfactual_all_serverless * 100:.0f}%)[/green]"
                     if report.counterfactual_all_serverless > 0 else "$0.00")
    else:
        cost.add_row("Savings vs Serverless",
                     f"${report.savings_vs_serverless:.2f} (spot more expensive at this traffic)")

    if report.savings_vs_on_demand > 0:
        cost.add_row("[green]Savings vs On-Demand[/green]",
                     f"[green]${report.savings_vs_on_demand:.2f} ({report.savings_vs_on_demand / report.counterfactual_all_on_demand * 100:.0f}%)[/green]"
                     if report.counterfactual_all_on_demand > 0 else "$0.00")

    console.print(cost)
