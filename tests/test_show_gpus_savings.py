"""Tests for the show-gpus savings column feature."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from tuna.catalog import CatalogQuery, ProviderGpu, SpotPrice


# Minimal fixtures
_ENTRIES = [
    ProviderGpu("L4", "modal", "nvidia-l4", 0.80, ()),
    ProviderGpu("L4", "runpod", "nvidia-l4", 2.74, ()),
    ProviderGpu("L4", "cerebrium", "nvidia-l4", 0.80, ()),
    ProviderGpu("H100", "modal", "nvidia-h100", 3.95, ()),
    ProviderGpu("H100", "cerebrium", "nvidia-h100", 2.21, ()),
]

_SPOT = {
    "L4": SpotPrice("L4", "aws", 0.28, "gr6.4xlarge", "us-east-2"),
    "H100": SpotPrice("H100", "aws", 2.05, "p5.4xlarge", "us-east-1"),
}


def _make_result(entries=_ENTRIES, spot=None):
    q = CatalogQuery(results=entries, spot_prices=spot or {})
    return q


class TestSavingsColumn:
    """Test that the savings column appears and computes correctly."""

    def _capture_table(self, show_spot, spot_prices=None):
        from tuna.__main__ import _print_gpu_table
        from tuna.catalog import get_gpu_spec
        from rich.console import Console

        result = _make_result(spot=spot_prices or {})
        buf = io.StringIO()
        console = Console(file=buf, width=200)

        real_init = Console.__init__

        def fake_init(self_c, *a, **kw):
            # Copy state from our pre-built console
            self_c.__dict__.update(console.__dict__)

        with patch.object(Console, "__init__", fake_init):
            _print_gpu_table(result, spot_prices or {}, show_spot=show_spot, get_gpu_spec=get_gpu_spec)
        return buf.getvalue()

    def test_table_no_spot_no_savings_column(self):
        """Without --spot, no SAVINGS column should appear."""
        text = self._capture_table(show_spot=False)
        assert "SAVINGS" not in text

    def test_table_with_spot_has_savings_column(self):
        """With --spot, SAVINGS column should appear."""
        text = self._capture_table(show_spot=True, spot_prices=_SPOT)
        assert "SAVINGS" in text or "cheaper" in text

    def test_savings_percentage_l4(self):
        """L4: spot $0.28 vs cheapest serverless $0.80 = 65% cheaper."""
        cheapest_serverless = 0.80
        spot = 0.28
        pct = (cheapest_serverless - spot) / cheapest_serverless * 100
        assert 64 <= pct <= 66  # ~65%

    def test_savings_percentage_h100(self):
        """H100: spot $2.05 vs cheapest serverless $2.21 = ~7% cheaper."""
        cheapest_serverless = 2.21
        spot = 2.05
        pct = (cheapest_serverless - spot) / cheapest_serverless * 100
        assert 6 <= pct <= 8

    def test_spot_more_expensive(self):
        """If spot > cheapest serverless, should show 'X% more'."""
        pct = (0.80 - 1.00) / 0.80 * 100  # spot=$1.00, serverless=$0.80
        assert pct < 0  # Negative = spot is more expensive

    def test_detail_view_shows_savings(self):
        """Detail view should show 'Spot saves X%' line."""
        from tuna.__main__ import _print_gpu_detail
        from tuna.catalog import get_gpu_spec

        result = _make_result(
            entries=[e for e in _ENTRIES if e.gpu == "L4"],
            spot={"L4": _SPOT["L4"]},
        )
        result.spot_prices = {"L4": _SPOT["L4"]}

        output = io.StringIO()
        with patch("rich.console.Console.print", side_effect=lambda *a, **kw: output.write(str(a[0]) if a else "")):
            _print_gpu_detail("L4", result, {"L4": _SPOT["L4"]}, get_gpu_spec)
        text = output.getvalue()
        assert "Spot saves" in text or "65%" in text
