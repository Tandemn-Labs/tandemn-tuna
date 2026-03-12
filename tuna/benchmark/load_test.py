"""Load test benchmark for the tuna meta load balancer router."""

from __future__ import annotations

import asyncio
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import httpx

_COMPLETIONS_PATH = "/v1/chat/completions"
_STATS_PATH = "/router/health"

# Rough cost rates (USD per GPU-second)
_COST_PER_GPU_S_SPOT = 0.0003        # ~$1.08/hr
_COST_PER_GPU_S_SERVERLESS = 0.0008  # ~$2.88/hr

VALID_PROFILES = ("day-cycle", "flat", "spike", "ramp")


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

def parse_duration(s: str) -> float:
    """Parse duration strings: '10h', '30m', '90s', or bare seconds."""
    s = s.strip()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


# ---------------------------------------------------------------------------
# Traffic profiles
# ---------------------------------------------------------------------------

def _concurrency_for_profile(
    profile: str, elapsed: float, duration: float, max_users: int
) -> int:
    """Return target concurrent users for the profile at elapsed time."""
    pct = min(1.0, elapsed / duration) if duration > 0 else 1.0

    if profile == "flat":
        return max_users

    if profile == "ramp":
        # Linear ramp from 1 to max_users over full duration
        return max(1, round(max_users * pct))

    if profile == "spike":
        # Low baseline (10%) with 10x bursts at 20%, 40%, 60%, 80% of duration
        baseline = max(1, max_users // 10)
        for center in (0.20, 0.40, 0.60, 0.80):
            if abs(pct - center) <= 0.025:
                return min(max_users, baseline * 10)
        return baseline

    if profile == "day-cycle":
        # Realistic day-cycle with 3 zero-traffic gaps that trigger spot scale-down.
        # Each gap is ~5% of duration (~6 min in a 2hr run) to exceed SkyPilot's
        # 300s downscale_delay.
        #
        # Timeline (2hr example):
        #   0:00-0:12  (0-10%)    Ramp up 1→max             [serverless absorbs, spot cold]
        #   0:12-0:30  (10-25%)   Morning peak, max users   [spot takes over]
        #   0:30-0:36  (25-30%)   Wind down to 0
        #   0:36-0:42  (30-35%)   ☆ ZERO TRAFFIC (gap 1)   [spot scales down]
        #   0:42-0:44  (35-37%)   Recovery ramp              [serverless absorbs again]
        #   0:44-1:00  (37-50%)   Steady state, 70% users
        #   1:00-1:10  (50-58%)   Afternoon spike, max users
        #   1:10-1:16  (58-63%)   Wind down to 0
        #   1:16-1:22  (63-68%)   ☆ ZERO TRAFFIC (gap 2)   [spot scales down]
        #   1:22-1:24  (68-70%)   Recovery ramp              [serverless absorbs again]
        #   1:24-1:42  (70-85%)   Evening steady, 70% users
        #   1:42-1:48  (85-90%)   Wind down to 0
        #   1:48-1:54  (90-95%)   ☆ ZERO TRAFFIC (gap 3)   [spot scales down]
        #   1:54-2:00  (95-100%)  Final trickle, 20% users  [serverless handles tail]

        # Phase 1: Ramp up (0-10%)
        if pct < 0.10:
            frac = pct / 0.10
            return max(1, round(max_users * frac))
        # Phase 2: Morning peak (10-25%)
        if pct < 0.25:
            return max_users
        # Phase 3: Wind down to gap 1 (25-30%)
        if pct < 0.30:
            frac = (pct - 0.25) / 0.05
            return max(0, round(max_users * (1.0 - frac)))
        # Phase 4: ☆ ZERO TRAFFIC gap 1 (30-35%)
        if pct < 0.35:
            return 0
        # Phase 5: Recovery ramp (35-37%)
        if pct < 0.37:
            frac = (pct - 0.35) / 0.02
            return max(1, round(max_users * 0.7 * frac))
        # Phase 6: Steady state (37-50%)
        if pct < 0.50:
            return max(1, round(max_users * 0.7))
        # Phase 7: Afternoon spike (50-58%)
        if pct < 0.58:
            return max_users
        # Phase 8: Wind down to gap 2 (58-63%)
        if pct < 0.63:
            frac = (pct - 0.58) / 0.05
            return max(0, round(max_users * (1.0 - frac)))
        # Phase 9: ☆ ZERO TRAFFIC gap 2 (63-68%)
        if pct < 0.68:
            return 0
        # Phase 10: Recovery ramp (68-70%)
        if pct < 0.70:
            frac = (pct - 0.68) / 0.02
            return max(1, round(max_users * 0.7 * frac))
        # Phase 11: Evening steady (70-85%)
        if pct < 0.85:
            return max(1, round(max_users * 0.7))
        # Phase 12: Wind down to gap 3 (85-90%)
        if pct < 0.90:
            frac = (pct - 0.85) / 0.05
            return max(0, round(max_users * 0.7 * (1.0 - frac)))
        # Phase 13: ☆ ZERO TRAFFIC gap 3 (90-95%)
        if pct < 0.95:
            return 0
        # Phase 14: Final trickle (95-100%)
        return max(1, round(max_users * 0.2))

    raise ValueError(f"Unknown profile: {profile!r}")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Per-request measurement."""

    timestamp: float    # time.time() at request start
    latency_s: float    # end-to-end request duration
    tokens: int         # completion_tokens from response (0 if unknown)
    backend: str        # "unknown" — router does not expose per-request backend
    success: bool
    error: Optional[str] = None


@dataclass
class _RouterSnapshot:
    """Snapshot of /router/health route_stats at a point in time."""

    ts: float
    total: int
    spot: int
    serverless: int
    gpu_s_spot: float
    gpu_s_svl: float


@dataclass
class LoadTestReport:
    """Aggregate load test summary."""

    profile: str
    duration_s: float
    max_users: int
    model: str
    total_requests: int
    success_count: int
    failure_count: int
    failure_rate_pct: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    throughput_rps: float
    spot_requests: int
    serverless_requests: int
    gpu_seconds_spot: float
    gpu_seconds_serverless: float
    estimated_cost_usd: float
    failover_events: int
    actual_duration_s: float


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list[float], p: float) -> float:
    """p-th percentile via linear interpolation (p in 0..100)."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


def _count_failovers(snapshots: list[_RouterSnapshot]) -> int:
    """Count times spot routing rate dropped from >50% to <20% (failover to serverless)."""
    events = 0
    for i in range(1, len(snapshots)):
        prev, curr = snapshots[i - 1], snapshots[i]
        if prev.total > 0 and curr.total > 0:
            if prev.spot / prev.total > 0.5 and curr.spot / curr.total < 0.2:
                events += 1
    return events


def _compute_report(
    results: list[RequestResult],
    snapshots: list[_RouterSnapshot],
    profile: str,
    duration_s: float,
    max_users: int,
    model: str,
    actual_duration_s: float,
) -> LoadTestReport:
    """Compute aggregate report from raw request results and router snapshots."""
    total = len(results)
    ok = [r for r in results if r.success]
    latencies = sorted(r.latency_s for r in ok)

    gpu_s_spot = gpu_s_svl = 0.0
    spot_reqs = svl_reqs = 0
    if len(snapshots) >= 2:
        first, last = snapshots[0], snapshots[-1]
        gpu_s_spot = max(0.0, last.gpu_s_spot - first.gpu_s_spot)
        gpu_s_svl = max(0.0, last.gpu_s_svl - first.gpu_s_svl)
        spot_reqs = max(0, last.spot - first.spot)
        svl_reqs = max(0, last.serverless - first.serverless)

    return LoadTestReport(
        profile=profile,
        duration_s=duration_s,
        max_users=max_users,
        model=model,
        total_requests=total,
        success_count=len(ok),
        failure_count=total - len(ok),
        failure_rate_pct=round(100.0 * (total - len(ok)) / total, 2) if total else 0.0,
        p50_latency_s=round(_percentile(latencies, 50), 3),
        p95_latency_s=round(_percentile(latencies, 95), 3),
        p99_latency_s=round(_percentile(latencies, 99), 3),
        throughput_rps=round(total / actual_duration_s, 2) if actual_duration_s > 0 else 0.0,
        spot_requests=spot_reqs,
        serverless_requests=svl_reqs,
        gpu_seconds_spot=round(gpu_s_spot, 2),
        gpu_seconds_serverless=round(gpu_s_svl, 2),
        estimated_cost_usd=round(
            gpu_s_spot * _COST_PER_GPU_S_SPOT + gpu_s_svl * _COST_PER_GPU_S_SERVERLESS, 4
        ),
        failover_events=_count_failovers(snapshots),
        actual_duration_s=round(actual_duration_s, 1),
    )


# ---------------------------------------------------------------------------
# Async core
# ---------------------------------------------------------------------------

async def _do_request(
    client: httpx.AsyncClient, url: str, model: str
) -> RequestResult:
    """Send one chat completion request and return the measurement."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 32,
        "stream": False,
    }
    ts = time.time()
    t0 = time.monotonic()
    try:
        resp = await client.post(url, json=payload)
        latency = time.monotonic() - t0
        if resp.status_code != 200:
            return RequestResult(
                timestamp=ts, latency_s=latency, tokens=0,
                backend="unknown", success=False, error=f"HTTP {resp.status_code}",
            )
        data = resp.json()
        tokens = 0
        if isinstance(data, dict) and "usage" in data:
            tokens = int((data["usage"] or {}).get("completion_tokens", 0) or 0)
        return RequestResult(
            timestamp=ts, latency_s=latency, tokens=tokens,
            backend="unknown", success=True,
        )
    except Exception as exc:  # httpx errors; CancelledError (BaseException) propagates
        return RequestResult(
            timestamp=ts, latency_s=time.monotonic() - t0, tokens=0,
            backend="unknown", success=False, error=str(exc),
        )


async def _user_loop(
    stop: asyncio.Event,
    results: list[RequestResult],
    client: httpx.AsyncClient,
    url: str,
    model: str,
    think_min: float,
    think_max: float,
    log_fp,
) -> None:
    """Simulated user: send request, think, repeat until stopped."""
    while not stop.is_set():
        r = await _do_request(client, url, model)
        results.append(r)
        if log_fp is not None:
            log_fp.write(json.dumps(asdict(r)) + "\n")
        think = random.uniform(think_min, think_max)
        try:
            await asyncio.wait_for(stop.wait(), timeout=think)
        except asyncio.TimeoutError:
            pass


async def _stats_poller(
    stop: asyncio.Event,
    client: httpx.AsyncClient,
    url: str,
    snapshots: list[_RouterSnapshot],
    interval: float,
) -> None:
    """Periodically poll /router/health and record route_stats snapshots."""
    while not stop.is_set():
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                body = resp.json()
                # /router/health nests stats under "route_stats"
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
            print(f"Warning: stats poller failed: {e}", file=sys.stderr)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def _run_async(
    endpoint_url: str,
    duration_s: float,
    max_users: int,
    profile: str,
    model: str,
    log_file: Optional[str],
    stats_interval: float,
    think_min: float,
    think_max: float,
    api_key: Optional[str],
    request_timeout: float,
) -> LoadTestReport:
    url = endpoint_url.rstrip("/") + _COMPLETIONS_PATH
    stats_url = endpoint_url.rstrip("/") + _STATS_PATH
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key

    results: list[RequestResult] = []
    snapshots: list[_RouterSnapshot] = []
    stop = asyncio.Event()
    users: dict[int, asyncio.Task] = {}
    next_id = 0

    log_fp = open(log_file, "w") if log_file else None  # noqa: SIM115
    try:
        async with httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(request_timeout),
            follow_redirects=True,
        ) as client:
            poller = asyncio.create_task(
                _stats_poller(stop, client, stats_url, snapshots, stats_interval)
            )
            t_start = time.monotonic()
            last_progress = t_start

            while True:
                elapsed = time.monotonic() - t_start
                if elapsed >= duration_s:
                    break

                target = _concurrency_for_profile(profile, elapsed, duration_s, max_users)

                # Scale up
                while len(users) < target:
                    uid = next_id
                    next_id += 1
                    users[uid] = asyncio.create_task(
                        _user_loop(stop, results, client, url, model,
                                   think_min, think_max, log_fp),
                        name=f"user-{uid}",
                    )

                # Scale down — cancel most recently created users first
                while len(users) > target:
                    uid, task = next(reversed(users.items()))
                    task.cancel()
                    del users[uid]

                # Prune naturally finished tasks
                done = [uid for uid, t in users.items() if t.done()]
                for uid in done:
                    del users[uid]

                # Progress report every 60s
                now = time.monotonic()
                if now - last_progress >= 60:
                    last_progress = now
                    n_fail = sum(1 for r in results if not r.success)
                    print(
                        f"  [{elapsed:.0f}s/{duration_s:.0f}s] "
                        f"users={len(users)} requests={len(results)} failures={n_fail}",
                        flush=True,
                    )

                await asyncio.sleep(1.0)

            actual_duration = time.monotonic() - t_start

            # Stop all user tasks
            stop.set()
            if users:
                await asyncio.gather(*users.values(), return_exceptions=True)

            poller.cancel()
            try:
                await poller
            except (asyncio.CancelledError, Exception):
                pass

    finally:
        if log_fp:
            log_fp.flush()
            log_fp.close()

    return _compute_report(
        results, snapshots, profile, duration_s, max_users, model, actual_duration
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_load_test(
    endpoint_url: str,
    duration_s: float,
    max_users: int,
    profile: str = "day-cycle",
    model: str = "Qwen/Qwen3-0.6B",
    log_file: Optional[str] = None,
    stats_interval: float = 30.0,
    think_min: float = 5.0,
    think_max: float = 10.0,
    api_key: Optional[str] = None,
    request_timeout: float = 120.0,
) -> LoadTestReport:
    """Run a load test against the router and return the aggregate report."""
    return asyncio.run(
        _run_async(
            endpoint_url=endpoint_url,
            duration_s=duration_s,
            max_users=max_users,
            profile=profile,
            model=model,
            log_file=log_file,
            stats_interval=stats_interval,
            think_min=think_min,
            think_max=think_max,
            api_key=api_key,
            request_timeout=request_timeout,
        )
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(report: LoadTestReport, output: str = "table") -> None:
    """Print the load test report in the requested format."""
    if output == "json":
        print(json.dumps(asdict(report), indent=2))
    elif output == "csv":
        _print_csv(report)
    else:
        _print_table(report)


def _print_table(report: LoadTestReport) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Load Test Results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    rows = [
        ("Profile", report.profile),
        ("Actual Duration", f"{report.actual_duration_s:.0f}s"),
        ("Max Users", str(report.max_users)),
        ("Model", report.model),
        ("Total Requests", str(report.total_requests)),
        ("Successes", str(report.success_count)),
        ("Failures", f"{report.failure_count} ({report.failure_rate_pct:.1f}%)"),
        ("p50 Latency", f"{report.p50_latency_s:.3f}s"),
        ("p95 Latency", f"{report.p95_latency_s:.3f}s"),
        ("p99 Latency", f"{report.p99_latency_s:.3f}s"),
        ("Throughput", f"{report.throughput_rps:.2f} req/s"),
        ("Spot Requests", str(report.spot_requests)),
        ("Serverless Requests", str(report.serverless_requests)),
        ("GPU-s Spot", f"{report.gpu_seconds_spot:.1f}s"),
        ("GPU-s Serverless", f"{report.gpu_seconds_serverless:.1f}s"),
        ("Estimated Cost", f"${report.estimated_cost_usd:.4f}"),
        ("Failover Events", str(report.failover_events)),
    ]
    for k, v in rows:
        table.add_row(k, v)

    Console().print(table)


def _print_csv(report: LoadTestReport) -> None:
    d = asdict(report)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(d.keys()))
    writer.writeheader()
    writer.writerow(d)
