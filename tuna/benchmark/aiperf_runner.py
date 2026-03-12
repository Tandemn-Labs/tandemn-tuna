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
            except Exception as e:
                print(f"Warning: cost sidecar poll failed: {e}", file=sys.stderr)
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
    """Parse aiperf's JSON export from the artifact directory."""
    # aiperf 0.6.0 writes profile_export_aiperf.json at the top level
    candidates = [
        os.path.join(artifact_dir, "profile_export_aiperf.json"),
        os.path.join(artifact_dir, "output.json"),
    ]
    # Also check subdirectories
    for pattern in [
        os.path.join(artifact_dir, "*", "profile_export_aiperf.json"),
        os.path.join(artifact_dir, "*", "output.json"),
    ]:
        candidates.extend(glob.glob(pattern))

    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    return {}


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
    profile: str | None = None,
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

    # Generate trace file if a traffic profile is requested (day-cycle, spike, etc.)
    trace_file = None
    if profile:
        from tuna.benchmark.trace_generator import generate_trace, print_trace_summary
        entries = generate_trace(
            duration_s=duration_s, profile=profile,
            isl=isl, osl=osl, seed=42,
        )
        if not entries:
            print(f"Error: profile '{profile}' generated 0 requests", file=sys.stderr)
            sys.exit(1)
        print_trace_summary(entries, duration_s, profile)
        print()

        trace_file = os.path.join(artifact_dir, "trace.jsonl")
        os.makedirs(artifact_dir, exist_ok=True)
        with open(trace_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    cmd = [
        aiperf_bin, "profile",
        "--model", model,
        "--url", endpoint_url.rstrip("/"),
        "--endpoint-type", "chat",
        "--output-artifact-dir", artifact_dir,
        "--ui-type", ui_mode,
        "--use-legacy-max-tokens",
    ]

    if trace_file:
        # Fixed-schedule mode: aiperf replays trace timestamps
        cmd += [
            "--input-file", trace_file,
            "--custom-dataset-type", "mooncake-trace",
            "--fixed-schedule",
            "--fixed-schedule-auto-offset",
        ]
    else:
        # Standard mode: duration + rate/concurrency
        cmd += [
            "--benchmark-duration", str(int(duration_s)),
            "--isl", str(isl),
            "--osl", str(osl),
        ]

    if streaming:
        cmd.append("--streaming")

    if trace_file:
        # Trace mode: concurrency is a ceiling, not a driver
        if concurrency:
            cmd += ["--concurrency", str(concurrency)]
    elif request_rate is not None:
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

    # Log command without API key
    safe_cmd = [c if i == 0 or cmd[i - 1] != "--api-key" else "***" for i, c in enumerate(cmd)]
    print(f"Running: {shlex.join(safe_cmd)}", flush=True)
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
        # TTFT (aiperf reports in ms already)
        ttft_p50_ms=_extract_metric(aiperf_data, "time_to_first_token", "p50"),
        ttft_p90_ms=_extract_metric(aiperf_data, "time_to_first_token", "p90"),
        ttft_p99_ms=_extract_metric(aiperf_data, "time_to_first_token", "p99"),
        # ITL
        itl_p50_ms=_extract_metric(aiperf_data, "inter_token_latency", "p50"),
        itl_p90_ms=_extract_metric(aiperf_data, "inter_token_latency", "p90"),
        itl_p99_ms=_extract_metric(aiperf_data, "inter_token_latency", "p99"),
        # Throughput
        output_token_throughput=_extract_metric(aiperf_data, "output_token_throughput", "avg"),
        request_throughput=_extract_metric(aiperf_data, "request_throughput", "avg"),
        # Request latency
        # Request latency (aiperf reports in ms already)
        request_latency_p50_ms=_extract_metric(aiperf_data, "request_latency", "p50"),
        request_latency_p90_ms=_extract_metric(aiperf_data, "request_latency", "p90"),
        request_latency_p99_ms=_extract_metric(aiperf_data, "request_latency", "p99"),
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
        except Exception as e:
            print(f"Warning: final router health fetch failed: {e}", file=sys.stderr)

    # Total requests from sidecar if aiperf count is missing
    if report.total_requests == 0:
        report.total_requests = report.spot_requests + report.serverless_requests
    report.success_count = report.total_requests - report.failure_count

    # Compute costs using catalog prices
    try:
        from tuna.catalog import get_provider_price, fetch_spot_prices, fetch_on_demand_prices
        # Infer GPU and provider from deployment state
        from tuna.state import list_deployments
        records = list_deployments()
        active = [r for r in records if r.status == "active"]
        if active:
            rec = active[0]
            svl_price = get_provider_price(rec.gpu, rec.serverless_provider)
            spot_prices = fetch_spot_prices(cloud=rec.spots_cloud)
            spot_entry = spot_prices.get(rec.gpu)
            spot_price = spot_entry.price_per_gpu_hour if spot_entry else 0.0
            od_prices = fetch_on_demand_prices(cloud=rec.spots_cloud)
            od_entry = od_prices.get(rec.gpu)
            od_price = od_entry.price_per_gpu_hour if od_entry else 0.0

            report.estimated_cost_serverless = round(
                (report.gpu_seconds_serverless / 3600) * svl_price, 4)
            report.estimated_cost_spot = round(
                (report.spot_ready_seconds / 3600) * spot_price, 4)
            report.estimated_cost_hybrid = round(
                report.estimated_cost_spot + report.estimated_cost_serverless, 4)
            report.counterfactual_all_serverless = round(
                ((report.gpu_seconds_spot + report.gpu_seconds_serverless) / 3600) * svl_price, 4)
            report.counterfactual_all_on_demand = round(
                (report.uptime_seconds / 3600) * od_price, 4)
            report.savings_vs_serverless = round(
                report.counterfactual_all_serverless - report.estimated_cost_hybrid, 4)
            report.savings_vs_on_demand = round(
                report.counterfactual_all_on_demand - report.estimated_cost_hybrid, 4)
    except Exception as e:
        print(f"Warning: cost computation failed: {e}", file=sys.stderr)

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
