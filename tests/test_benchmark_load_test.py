"""Tests for tuna.benchmark.load_test — profile logic, stats, and report generation."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tuna.benchmark.load_test import (
    LoadTestReport,
    RequestResult,
    _RouterSnapshot,
    _compute_report,
    _concurrency_for_profile,
    _count_failovers,
    _do_request,
    _parse_duration,
    _percentile,
    print_summary,
)


# ---------------------------------------------------------------------------
# _parse_duration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s,expected", [
    ("10h", 36000.0),
    ("30m", 1800.0),
    ("90s", 90.0),
    ("3600", 3600.0),
    ("1.5h", 5400.0),
    ("0.5m", 30.0),
    ("  5m  ", 300.0),
])
def test_parse_duration(s, expected):
    assert _parse_duration(s) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _concurrency_for_profile — flat
# ---------------------------------------------------------------------------

def test_flat_always_max():
    for elapsed in (0.0, 50.0, 100.0):
        assert _concurrency_for_profile("flat", elapsed, 100.0, 20) == 20


def test_flat_single_user():
    assert _concurrency_for_profile("flat", 0.0, 100.0, 1) == 1


# ---------------------------------------------------------------------------
# _concurrency_for_profile — ramp
# ---------------------------------------------------------------------------

def test_ramp_starts_at_1():
    assert _concurrency_for_profile("ramp", 0.0, 100.0, 50) >= 1


def test_ramp_ends_at_max():
    assert _concurrency_for_profile("ramp", 100.0, 100.0, 50) == 50


def test_ramp_midpoint():
    assert _concurrency_for_profile("ramp", 50.0, 100.0, 100) == 50


def test_ramp_monotone():
    prev = 0
    for t in range(0, 101, 10):
        cur = _concurrency_for_profile("ramp", float(t), 100.0, 100)
        assert cur >= prev
        prev = cur


# ---------------------------------------------------------------------------
# _concurrency_for_profile — spike
# ---------------------------------------------------------------------------

def test_spike_baseline_between_bursts():
    # At 10% — no burst center nearby (20%, 40%, 60%, 80%)
    result = _concurrency_for_profile("spike", 10.0, 100.0, 100)
    assert result == 10  # 10% of 100


def test_spike_burst_at_20pct():
    result = _concurrency_for_profile("spike", 20.0, 100.0, 100)
    assert result == 100  # 10x burst


def test_spike_burst_at_40pct():
    result = _concurrency_for_profile("spike", 40.0, 100.0, 100)
    assert result == 100


def test_spike_burst_at_60pct():
    result = _concurrency_for_profile("spike", 60.0, 100.0, 100)
    assert result == 100


def test_spike_burst_at_80pct():
    result = _concurrency_for_profile("spike", 80.0, 100.0, 100)
    assert result == 100


def test_spike_no_burst_at_30pct():
    result = _concurrency_for_profile("spike", 30.0, 100.0, 100)
    assert result == 10  # baseline


def test_spike_small_max_users():
    # max_users=5 → baseline=1 (max(1, 5//10))
    assert _concurrency_for_profile("spike", 10.0, 100.0, 5) == 1
    assert _concurrency_for_profile("spike", 20.0, 100.0, 5) == 5


# ---------------------------------------------------------------------------
# _concurrency_for_profile — day-cycle
# ---------------------------------------------------------------------------

def test_day_cycle_ramp_start():
    # At 0%: near 10% of max
    result = _concurrency_for_profile("day-cycle", 0.0, 100.0, 100)
    assert 1 <= result <= 15


def test_day_cycle_ramp_midpoint():
    # At 7.5% (half of ramp phase): ~55% of max
    result = _concurrency_for_profile("day-cycle", 7.5, 100.0, 100)
    assert 40 <= result <= 65


def test_day_cycle_peak():
    # At 25%: peak phase → 100% of max
    assert _concurrency_for_profile("day-cycle", 25.0, 100.0, 100) == 100


def test_day_cycle_steady():
    # At 45%: steady phase → 70% of max
    assert _concurrency_for_profile("day-cycle", 45.0, 100.0, 100) == 70


def test_day_cycle_spike():
    # At 60%: spike phase → 100% of max
    assert _concurrency_for_profile("day-cycle", 60.0, 100.0, 100) == 100


def test_day_cycle_low():
    # At 90%: low phase → 20% of max
    assert _concurrency_for_profile("day-cycle", 90.0, 100.0, 100) == 20


def test_day_cycle_wind_down():
    # At 72.5% (midpoint of wind-down 65-80%): between 70% and 20% → ~45%
    result = _concurrency_for_profile("day-cycle", 72.5, 100.0, 100)
    assert 30 <= result <= 60


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="Unknown profile"):
        _concurrency_for_profile("bad-profile", 0.0, 100.0, 10)


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------

def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_percentile_single():
    assert _percentile([5.0], 50) == 5.0
    assert _percentile([5.0], 99) == 5.0


def test_percentile_median_odd():
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)


def test_percentile_p95():
    data = [float(x) for x in range(1, 101)]  # 1..100
    assert 94.0 <= _percentile(data, 95) <= 96.0


def test_percentile_p99():
    data = [float(x) for x in range(1, 101)]
    assert _percentile(data, 99) >= 98.0


# ---------------------------------------------------------------------------
# _count_failovers
# ---------------------------------------------------------------------------

def _snap(ts, total, spot, serverless, gpu_s_spot=0.0, gpu_s_svl=0.0):
    return _RouterSnapshot(
        ts=ts, total=total, spot=spot, serverless=serverless,
        gpu_s_spot=gpu_s_spot, gpu_s_svl=gpu_s_svl,
    )


def test_count_failovers_empty():
    assert _count_failovers([]) == 0


def test_count_failovers_single():
    assert _count_failovers([_snap(0, 100, 80, 20)]) == 0


def test_count_failovers_no_failover():
    # Spot rate stays at 80% throughout
    snaps = [_snap(0, 100, 80, 20), _snap(30, 200, 160, 40)]
    assert _count_failovers(snaps) == 0


def test_count_failovers_detects_one():
    # snap1: 80% spot; snap2: 8.2% spot (flood of serverless reqs)
    # cumulative totals: prev.spot/prev.total = 80/100 = 0.8 > 0.5
    # curr.spot/curr.total = 82/1000 = 0.082 < 0.2  → failover
    snaps = [_snap(0, 100, 80, 20), _snap(30, 1000, 82, 918)]
    assert _count_failovers(snaps) == 1


def test_count_failovers_not_triggered_below_threshold():
    # prev=60% spot, curr=35% spot — curr not < 20%, so no failover
    snaps = [_snap(0, 100, 60, 40), _snap(30, 200, 70, 130)]
    assert _count_failovers(snaps) == 0


# ---------------------------------------------------------------------------
# _compute_report
# ---------------------------------------------------------------------------

def _make_results(latencies: list[float], n_fail: int = 0) -> list[RequestResult]:
    results = []
    for i, lat in enumerate(latencies):
        results.append(RequestResult(
            timestamp=float(i), latency_s=lat, tokens=10,
            backend="unknown", success=True,
        ))
    for i in range(n_fail):
        results.append(RequestResult(
            timestamp=float(len(latencies) + i), latency_s=0.1, tokens=0,
            backend="unknown", success=False, error="HTTP 500",
        ))
    return results


def test_compute_report_basic():
    results = _make_results([1.0, 2.0, 3.0, 4.0, 5.0], n_fail=1)
    report = _compute_report(results, [], "flat", 60.0, 5, "test-model", 60.0)
    assert report.total_requests == 6
    assert report.success_count == 5
    assert report.failure_count == 1
    assert report.failure_rate_pct == pytest.approx(100 * 1 / 6, rel=0.01)
    assert report.throughput_rps == pytest.approx(6 / 60, rel=0.01)
    assert report.failover_events == 0
    assert report.profile == "flat"
    assert report.model == "test-model"


def test_compute_report_no_results():
    report = _compute_report([], [], "flat", 60.0, 5, "model", 60.0)
    assert report.total_requests == 0
    assert report.failure_rate_pct == 0.0
    assert report.p50_latency_s == 0.0
    assert report.throughput_rps == 0.0


def test_compute_report_latency_percentiles():
    # 100 requests with latency 0.01, 0.02, ..., 1.00
    latencies = [i / 100.0 for i in range(1, 101)]
    report = _compute_report(_make_results(latencies), [], "flat", 100.0, 5, "m", 100.0)
    assert report.p50_latency_s == pytest.approx(0.505, abs=0.02)
    assert report.p95_latency_s >= 0.90
    assert report.p99_latency_s >= 0.95


def test_compute_report_with_router_snapshots():
    snaps = [
        _RouterSnapshot(ts=0, total=0, spot=0, serverless=0, gpu_s_spot=0.0, gpu_s_svl=0.0),
        _RouterSnapshot(ts=60, total=100, spot=70, serverless=30, gpu_s_spot=10.0, gpu_s_svl=5.0),
    ]
    report = _compute_report(_make_results([0.5] * 100), snaps, "flat", 60.0, 10, "m", 60.0)
    assert report.spot_requests == 70
    assert report.serverless_requests == 30
    assert report.gpu_seconds_spot == pytest.approx(10.0)
    assert report.gpu_seconds_serverless == pytest.approx(5.0)
    assert report.estimated_cost_usd > 0


def test_compute_report_cost_formula():
    snaps = [
        _RouterSnapshot(ts=0, total=0, spot=0, serverless=0, gpu_s_spot=0.0, gpu_s_svl=0.0),
        _RouterSnapshot(ts=60, total=10, spot=5, serverless=5, gpu_s_spot=100.0, gpu_s_svl=100.0),
    ]
    report = _compute_report([], snaps, "flat", 60.0, 1, "m", 60.0)
    expected = 100.0 * 0.0003 + 100.0 * 0.0008  # 0.03 + 0.08 = 0.11
    assert report.estimated_cost_usd == pytest.approx(expected, rel=0.001)


# ---------------------------------------------------------------------------
# _do_request — mock HTTP
# ---------------------------------------------------------------------------

def test_do_request_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Hello!"}}],
        "usage": {"completion_tokens": 15, "prompt_tokens": 10},
    }

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    async def _run():
        return await _do_request(mock_client, "http://x/v1/chat/completions", "my-model")

    result = asyncio.run(_run())
    assert result.success is True
    assert result.tokens == 15
    assert result.latency_s >= 0.0
    assert result.backend == "unknown"
    assert result.error is None


def test_do_request_http_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 503

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    async def _run():
        return await _do_request(mock_client, "http://x/v1/chat/completions", "m")

    result = asyncio.run(_run())
    assert result.success is False
    assert "503" in result.error
    assert result.tokens == 0


def test_do_request_network_exception():
    import httpx as _httpx

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=_httpx.ConnectError("connection refused"))

    async def _run():
        return await _do_request(mock_client, "http://x/v1/chat/completions", "m")

    result = asyncio.run(_run())
    assert result.success is False
    assert result.error is not None
    assert result.tokens == 0


def test_do_request_missing_usage():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "Hi"}}]}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    async def _run():
        return await _do_request(mock_client, "http://x/v1/chat/completions", "m")

    result = asyncio.run(_run())
    assert result.success is True
    assert result.tokens == 0


# ---------------------------------------------------------------------------
# print_summary — json and csv output
# ---------------------------------------------------------------------------

def _make_report(**overrides) -> LoadTestReport:
    defaults = dict(
        profile="flat", duration_s=60.0, max_users=5, model="test-model",
        total_requests=10, success_count=9, failure_count=1,
        failure_rate_pct=10.0, p50_latency_s=0.5, p95_latency_s=0.9,
        p99_latency_s=1.0, throughput_rps=5.0,
        spot_requests=7, serverless_requests=3,
        gpu_seconds_spot=1.0, gpu_seconds_serverless=0.5,
        estimated_cost_usd=0.0007, failover_events=0,
        actual_duration_s=60.0,
    )
    defaults.update(overrides)
    return LoadTestReport(**defaults)


def test_print_summary_json(capsys):
    report = _make_report(total_requests=42, profile="ramp")
    print_summary(report, output="json")
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["total_requests"] == 42
    assert data["profile"] == "ramp"


def test_print_summary_json_all_fields(capsys):
    report = _make_report()
    print_summary(report, output="json")
    data = json.loads(capsys.readouterr().out)
    for field in ("p50_latency_s", "p95_latency_s", "p99_latency_s",
                  "throughput_rps", "estimated_cost_usd", "failover_events"):
        assert field in data


def test_print_summary_csv(capsys):
    report = _make_report()
    print_summary(report, output="csv")
    out = capsys.readouterr().out
    lines = [l for l in out.strip().split("\n") if l]
    assert len(lines) == 2  # header + 1 data row
    assert "profile" in lines[0]
    assert "flat" in lines[1]
