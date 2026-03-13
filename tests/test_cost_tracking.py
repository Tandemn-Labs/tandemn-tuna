"""Tests for cost tracking: router GPU-seconds, catalog on-demand prices, cost computation."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from tuna.router import meta_lb
from tuna.catalog import (
    OnDemandPrice,
    fetch_on_demand_prices,
    get_provider_price,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _reset_meta_lb():
    """Reset meta_lb cost-tracking state for each test."""
    meta_lb._gpu_seconds_spot = 0.0
    meta_lb._gpu_seconds_serverless = 0.0
    meta_lb._spot_ready_cumulative_s = 0.0
    meta_lb._spot_ready_since = None
    meta_lb._spot_state = meta_lb.SpotState.COLD
    meta_lb._req_total = 0
    meta_lb._req_to_spot = 0
    meta_lb._req_to_serverless = 0
    meta_lb._recent_routes.clear()
    meta_lb._last_probe_ts = None
    meta_lb._last_probe_err = None
    meta_lb._state_lock = asyncio.Lock()
    yield


# ---------------------------------------------------------------------------
# Router cost tracking tests
# ---------------------------------------------------------------------------

class TestRouterCostTracking:
    """Test the cost-tracking globals in meta_lb."""

    @pytest.mark.asyncio
    async def test_route_stats_includes_cost_fields(self, _reset_meta_lb):
        stats = await meta_lb._route_stats()
        assert "gpu_seconds_spot" in stats
        assert "gpu_seconds_serverless" in stats
        assert "uptime_seconds" in stats
        assert "spot_ready_seconds" in stats

    @pytest.mark.asyncio
    async def test_spot_ready_accumulates_on_transition(self, _reset_meta_lb):
        # Transition to ready
        await meta_lb._set_ready(True)
        await asyncio.sleep(0.05)
        # Transition to not-ready — should accumulate
        await meta_lb._set_ready(False)

        assert meta_lb._spot_ready_cumulative_s > 0.0
        assert meta_lb._spot_ready_since is None

    @pytest.mark.asyncio
    async def test_spot_ready_includes_current_period(self, _reset_meta_lb):
        # Transition to ready and stay ready
        await meta_lb._set_ready(True)
        await asyncio.sleep(0.05)

        # _route_stats should include the ongoing ready period
        stats = await meta_lb._route_stats()
        assert stats["spot_ready_seconds"] > 0.0
        # _spot_ready_since should still be set (spot is still ready)
        assert meta_lb._spot_ready_since is not None

    @pytest.mark.asyncio
    async def test_repeated_ready_true_does_not_double_count(self, _reset_meta_lb):
        await meta_lb._set_ready(True)
        since1 = meta_lb._spot_ready_since
        await asyncio.sleep(0.02)
        # Calling _set_ready(True) again while already ready shouldn't reset _spot_ready_since
        await meta_lb._set_ready(True)
        assert meta_lb._spot_ready_since == since1

    @pytest.mark.asyncio
    async def test_gpu_seconds_accumulate_on_proxy(self, _reset_meta_lb, mocker):
        """Verify GPU-seconds accumulate when proxy() handles a request."""
        meta_lb._serverless_base_url = "http://serverless.example.com"
        meta_lb._skyserve_base_url = ""
        original_key = meta_lb.API_KEY
        meta_lb.API_KEY = ""

        # Create a real httpx client and mock its send method
        meta_lb._http_client = httpx.AsyncClient()

        async def _aiter_bytes(chunk_size=4096):
            yield b'{"ok": true}'

        mock_resp = mocker.AsyncMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = httpx.Headers({"content-type": "application/json"})
        mock_resp.aiter_bytes = _aiter_bytes
        mock_resp.aclose = mocker.AsyncMock()

        mock_send = mocker.patch.object(
            meta_lb._http_client, "send", new_callable=mocker.AsyncMock,
        )
        mock_send.return_value = mock_resp

        async with AsyncClient(
            transport=ASGITransport(app=meta_lb.app),
            base_url="http://test",
        ) as client:
            await client.post("/v1/chat/completions", json={"prompt": "hi"})

        meta_lb.API_KEY = original_key
        assert meta_lb._gpu_seconds_serverless > 0.0
        await meta_lb._http_client.aclose()


# ---------------------------------------------------------------------------
# Catalog on-demand price tests
# ---------------------------------------------------------------------------

class TestFetchOnDemandPrices:
    @patch("sky.catalog.list_accelerators")
    def test_returns_on_demand_prices(self, mock_list):
        mock_info = MagicMock()
        mock_info.accelerator_count = 1
        mock_info.price = 2.50
        mock_info.instance_type = "p4d.24xlarge"
        mock_info.region = "us-east-1"

        mock_list.return_value = {"H100": [mock_info]}

        result = fetch_on_demand_prices(cloud="aws")

        assert "H100" in result
        assert isinstance(result["H100"], OnDemandPrice)
        assert result["H100"].price_per_gpu_hour == 2.50
        assert result["H100"].instance_type == "p4d.24xlarge"
        assert result["H100"].region == "us-east-1"
        assert result["H100"].cloud == "aws"

    def test_graceful_when_skypilot_missing(self):
        with patch.dict("sys.modules", {"sky": None, "sky.catalog": None}):
            result = fetch_on_demand_prices(cloud="aws")
        assert result == {}

    @patch("sky.catalog.list_accelerators")
    def test_skips_nan_prices(self, mock_list):
        mock_info = MagicMock()
        mock_info.accelerator_count = 1
        mock_info.price = float("nan")
        mock_info.instance_type = "p4d.24xlarge"
        mock_info.region = "us-east-1"

        mock_list.return_value = {"H100": [mock_info]}

        result = fetch_on_demand_prices(cloud="aws")
        assert "H100" not in result

    @patch("sky.catalog.list_accelerators")
    def test_picks_cheapest_on_demand(self, mock_list):
        expensive = MagicMock()
        expensive.accelerator_count = 1
        expensive.price = 5.00
        expensive.instance_type = "p4d.24xlarge"
        expensive.region = "us-east-1"

        cheap = MagicMock()
        cheap.accelerator_count = 1
        cheap.price = 3.00
        cheap.instance_type = "p4de.24xlarge"
        cheap.region = "us-west-2"

        mock_list.return_value = {"H100": [expensive, cheap]}

        result = fetch_on_demand_prices(cloud="aws")
        assert result["H100"].price_per_gpu_hour == 3.00


class TestGetProviderPrice:
    def test_known_gpu_provider(self):
        assert get_provider_price("H100", "modal") == 3.95

    def test_unknown_returns_zero(self):
        assert get_provider_price("FAKE_GPU", "modal") == 0.0

    def test_unknown_provider_returns_zero(self):
        assert get_provider_price("H100", "nonexistent") == 0.0

    def test_l4_modal(self):
        assert get_provider_price("L4", "modal") == 0.80


# ---------------------------------------------------------------------------
# Cost computation tests
# ---------------------------------------------------------------------------

class TestCostComputation:
    """Test the cost formulas used by cmd_cost (unit tests on the math)."""

    ROUTER_CPU_COST_PER_HOUR = 0.04

    def test_actual_cost_calculation(self):
        gpu_sec_svl = 1560.0
        gpu_sec_spot = 0.0
        spot_ready_s = 2 * 3600 + 28 * 60  # 2h 28m
        uptime_s = 2 * 3600 + 34 * 60  # 2h 34m
        gpu_count = 1
        serverless_price = 3.95  # H100 on modal
        spot_price = 1.20

        actual_svl = (gpu_sec_svl / 3600) * serverless_price
        actual_spot = (spot_ready_s / 3600) * spot_price * gpu_count
        actual_router = (uptime_s / 3600) * self.ROUTER_CPU_COST_PER_HOUR
        actual_total = actual_svl + actual_spot + actual_router

        assert actual_svl == pytest.approx((1560 / 3600) * 3.95, rel=1e-6)
        assert actual_spot == pytest.approx((8880 / 3600) * 1.20, rel=1e-6)
        assert actual_total == pytest.approx(actual_svl + actual_spot + actual_router, rel=1e-6)

    def test_counterfactual_all_serverless(self):
        gpu_sec_svl = 1000.0
        gpu_sec_spot = 500.0
        serverless_price = 3.95

        all_serverless = ((gpu_sec_svl + gpu_sec_spot) / 3600) * serverless_price
        assert all_serverless == pytest.approx((1500 / 3600) * 3.95, rel=1e-6)

    def test_counterfactual_all_on_demand(self):
        uptime_s = 9240.0  # 2h 34m
        on_demand_price = 3.50
        gpu_count = 2

        all_on_demand = (uptime_s / 3600) * on_demand_price * gpu_count
        assert all_on_demand == pytest.approx((9240 / 3600) * 3.50 * 2, rel=1e-6)

    def test_savings_positive_when_spot_cheaper(self):
        # Spot-heavy workload: lots of GPU-seconds routed through cheap spot
        gpu_sec_svl = 500.0
        gpu_sec_spot = 5000.0
        spot_ready_s = 1.5 * 3600  # 1.5h ready
        uptime_s = 2 * 3600
        gpu_count = 1
        serverless_price = 3.95
        spot_price = 1.20

        actual_svl = (gpu_sec_svl / 3600) * serverless_price
        actual_spot = (spot_ready_s / 3600) * spot_price * gpu_count
        actual_router = (uptime_s / 3600) * self.ROUTER_CPU_COST_PER_HOUR
        actual_total = actual_svl + actual_spot + actual_router

        all_serverless = ((gpu_sec_svl + gpu_sec_spot) / 3600) * serverless_price

        savings = all_serverless - actual_total
        assert savings > 0, f"Expected positive savings, got {savings}"

    def test_zero_requests_gives_zero_costs(self):
        gpu_sec_svl = 0.0
        gpu_sec_spot = 0.0
        spot_ready_s = 0.0
        uptime_s = 60.0  # 1 minute uptime
        serverless_price = 3.95
        spot_price = 1.20
        gpu_count = 1

        actual_svl = (gpu_sec_svl / 3600) * serverless_price
        actual_spot = (spot_ready_s / 3600) * spot_price * gpu_count
        actual_router = (uptime_s / 3600) * self.ROUTER_CPU_COST_PER_HOUR

        assert actual_svl == 0.0
        assert actual_spot == 0.0
        assert actual_router == pytest.approx((60 / 3600) * 0.04, rel=1e-6)
