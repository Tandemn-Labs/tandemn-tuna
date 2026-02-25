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

_SPOT_GCP = {
    "L4": SpotPrice("L4", "gcp", 0.22, "n1-highmem-4", "me-west1-b"),
    "H100": SpotPrice("H100", "gcp", 3.20, "a3-highgpu-1g", "us-central1-a"),
}


def _make_result(entries=_ENTRIES, spot=None):
    return CatalogQuery(results=entries, spot_prices=spot or {})


class TestSavingsColumn:
    """Test that the savings column appears and computes correctly."""

    def _capture_table(self, show_spot, spot_prices_by_cloud=None):
        from tuna.__main__ import _print_gpu_table
        from tuna.catalog import get_gpu_spec
        from rich.console import Console

        result = _make_result()
        buf = io.StringIO()
        console = Console(file=buf, width=200)

        real_init = Console.__init__

        def fake_init(self_c, *a, **kw):
            self_c.__dict__.update(console.__dict__)

        with patch.object(Console, "__init__", fake_init):
            _print_gpu_table(result, spot_prices_by_cloud or {}, show_spot=show_spot, get_gpu_spec=get_gpu_spec)
        return buf.getvalue()

    def _capture_detail(self, gpu, entries, spot_prices_by_cloud):
        from tuna.__main__ import _print_gpu_detail
        from tuna.catalog import get_gpu_spec
        from rich.console import Console

        result = _make_result(entries=entries)
        buf = io.StringIO()
        console = Console(file=buf, width=200)

        def fake_init(self_c, *a, **kw):
            self_c.__dict__.update(console.__dict__)

        with patch.object(Console, "__init__", fake_init):
            _print_gpu_detail(gpu, result, spot_prices_by_cloud, get_gpu_spec)
        return buf.getvalue()

    def test_table_no_spot_no_savings_column(self):
        """Without --spot, no SAVINGS column should appear."""
        text = self._capture_table(show_spot=False)
        assert "SAVINGS" not in text

    def test_table_with_spot_has_savings_column(self):
        """With --spot, SAVINGS column should appear."""
        text = self._capture_table(show_spot=True, spot_prices_by_cloud={"aws": _SPOT})
        assert "SAVINGS" in text or "cheaper" in text

    def test_savings_percentage_l4(self):
        """L4: spot $0.28 vs cheapest serverless $0.80 = 65% cheaper."""
        text = self._capture_table(show_spot=True, spot_prices_by_cloud={"aws": _SPOT})
        assert "65% cheaper" in text

    def test_savings_percentage_h100(self):
        """H100: spot $2.05 vs cheapest serverless $2.21 = ~7% cheaper."""
        text = self._capture_table(show_spot=True, spot_prices_by_cloud={"aws": _SPOT})
        assert "7% cheaper" in text

    def test_spot_more_expensive(self):
        """If spot > cheapest serverless, should show 'X% more'."""
        spot_expensive = {
            "L4": SpotPrice("L4", "aws", 1.00, "gr6.4xlarge", "us-east-2"),
        }
        text = self._capture_table(show_spot=True, spot_prices_by_cloud={"aws": spot_expensive})
        assert "25% more" in text

    def test_both_aws_and_gcp_columns(self):
        """Both AWS SPOT and GCP SPOT columns should appear."""
        text = self._capture_table(
            show_spot=True,
            spot_prices_by_cloud={"aws": _SPOT, "gcp": _SPOT_GCP},
        )
        assert "AWS SPOT" in text
        assert "GCP SPOT" in text

    def test_savings_uses_cheapest_spot(self):
        """Savings should use the cheapest spot across clouds (GCP L4 $0.22 < AWS $0.28)."""
        text = self._capture_table(
            show_spot=True,
            spot_prices_by_cloud={"aws": _SPOT, "gcp": _SPOT_GCP},
        )
        # GCP L4 at $0.22 vs cheapest serverless $0.80 = 72.5% cheaper
        assert "72% cheaper" in text or "73% cheaper" in text

    def test_detail_view_shows_savings(self):
        """Detail view should show savings line."""
        text = self._capture_detail(
            "L4",
            entries=[e for e in _ENTRIES if e.gpu == "L4"],
            spot_prices_by_cloud={"aws": {"L4": _SPOT["L4"]}},
        )
        assert "saves" in text or "65%" in text

    def test_detail_view_shows_both_clouds(self):
        """Detail view should list both aws spot and gcp spot."""
        text = self._capture_detail(
            "L4",
            entries=[e for e in _ENTRIES if e.gpu == "L4"],
            spot_prices_by_cloud={
                "aws": {"L4": _SPOT["L4"]},
                "gcp": {"L4": _SPOT_GCP["L4"]},
            },
        )
        assert "aws spot" in text
        assert "gcp spot" in text

    def test_detail_view_spot_more_expensive(self):
        """Detail view should show 'more expensive' when spot costs more."""
        spot_expensive = {"L4": SpotPrice("L4", "aws", 1.00, "gr6.4xlarge", "us-east-2")}
        text = self._capture_detail(
            "L4",
            entries=[e for e in _ENTRIES if e.gpu == "L4"],
            spot_prices_by_cloud={"aws": spot_expensive},
        )
        assert "more" in text

    def test_spot_savings_pct_helper(self):
        """Test the shared helper function directly."""
        from tuna.__main__ import _spot_savings_pct

        assert _spot_savings_pct(0.28, [0.80, 2.74]) == pytest.approx(65.0, abs=1)
        assert _spot_savings_pct(2.05, [2.21, 3.95]) == pytest.approx(7.2, abs=1)
        assert _spot_savings_pct(1.00, [0.80]) == pytest.approx(-25.0, abs=1)
        assert _spot_savings_pct(0.80, [0.80]) == pytest.approx(0.0)
        assert _spot_savings_pct(0.0, [0.80]) is None
        assert _spot_savings_pct(0.50, []) is None
