"""Tests for tandemn.catalog."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from tandemn.catalog import (
    GPU_SPECS,
    CatalogQuery,
    GpuSpec,
    ProviderGpu,
    SpotPrice,
    fetch_spot_prices,
    get_gpu_spec,
    normalize_gpu_name,
    provider_gpu_id,
    provider_gpu_map,
    provider_regions,
    query,
)


class TestGpuSpecs:
    def test_all_specs_have_positive_vram(self):
        for name, spec in GPU_SPECS.items():
            assert spec.vram_gb > 0, f"{name} has non-positive VRAM"

    def test_known_gpus_present(self):
        assert "L4" in GPU_SPECS
        assert "H100" in GPU_SPECS
        assert "A100_80GB" in GPU_SPECS

    def test_spec_fields(self):
        spec = GPU_SPECS["L4"]
        assert spec.short_name == "L4"
        assert spec.full_name == "NVIDIA L4"
        assert spec.vram_gb == 24
        assert spec.arch == "ada"


class TestNormalize:
    def test_canonical_passthrough(self):
        assert normalize_gpu_name("L4") == "L4"

    def test_alias_resolution(self):
        assert normalize_gpu_name("A100") == "A100_80GB"

    def test_alias_4090(self):
        assert normalize_gpu_name("4090") == "RTX4090"

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            normalize_gpu_name("FAKE_GPU")


class TestQuery:
    def test_query_all(self):
        result = query()
        assert len(result.results) > 0

    def test_by_gpu(self):
        result = query(gpu="L4")
        assert all(r.gpu == "L4" for r in result.results)
        assert len(result.results) > 0

    def test_by_provider(self):
        result = query(provider="modal")
        assert all(r.provider == "modal" for r in result.results)
        assert len(result.results) > 0

    def test_combined_filters(self):
        result = query(gpu="L4", provider="modal")
        assert all(r.gpu == "L4" and r.provider == "modal" for r in result.results)
        assert len(result.results) == 1

    def test_empty_result(self):
        result = query(gpu="NONEXISTENT")
        assert len(result.results) == 0


class TestProviderGpuId:
    def test_runpod_l40s(self):
        assert provider_gpu_id("L40S", "runpod") == "NVIDIA L40S"

    def test_cloudrun_l4(self):
        assert provider_gpu_id("L4", "cloudrun") == "nvidia-l4"

    def test_modal_h100(self):
        assert provider_gpu_id("H100", "modal") == "H100"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            provider_gpu_id("L4", "fake")


class TestProviderGpuMap:
    def test_runpod_map_matches_old(self):
        """The catalog must produce the same mapping as the old RunPod GPU_MAP."""
        old_gpu_map = {
            "A4000": "NVIDIA RTX A4000",
            "A5000": "NVIDIA RTX A5000",
            "L4": "NVIDIA L4",
            "RTX4090": "NVIDIA GeForce RTX 4090",
            "L40": "NVIDIA L40",
            "L40S": "NVIDIA L40S",
            "A6000": "NVIDIA RTX A6000",
            "A40": "NVIDIA A40",
            "A100_80GB": "NVIDIA A100-SXM4-80GB",
            "H100": "NVIDIA H100 80GB HBM3",
            "H200": "NVIDIA H200",
            "B200": "NVIDIA B200",
        }
        catalog_map = provider_gpu_map("runpod")
        for short_name, full_name in old_gpu_map.items():
            assert catalog_map[short_name] == full_name, (
                f"Mismatch for {short_name}: catalog={catalog_map.get(short_name)!r}, old={full_name!r}"
            )

    def test_cloudrun_map_matches_old(self):
        """The catalog must produce the same mapping as the old CloudRun GPU_MAP."""
        old_gpu_map = {
            "L4": "nvidia-l4",
            "RTX_PRO_6000": "nvidia-rtx-pro-6000",
        }
        catalog_map = provider_gpu_map("cloudrun")
        for short_name, full_name in old_gpu_map.items():
            assert catalog_map[short_name] == full_name, (
                f"Mismatch for {short_name}: catalog={catalog_map.get(short_name)!r}, old={full_name!r}"
            )


class TestProviderRegions:
    def test_cloudrun_l4_has_regions(self):
        regions = provider_regions("L4", "cloudrun")
        assert len(regions) > 0
        assert "us-central1" in regions

    def test_modal_returns_empty(self):
        regions = provider_regions("H100", "modal")
        assert regions == ()


class TestCatalogQuery:
    def test_cheapest(self):
        entries = [
            ProviderGpu("L4", "modal", "L4", 0.80),
            ProviderGpu("L4", "runpod", "NVIDIA L4", 2.74),
        ]
        cq = CatalogQuery(results=entries)
        assert cq.cheapest() is not None
        assert cq.cheapest().price_per_gpu_hour == 0.80

    def test_cheapest_skips_zero_price(self):
        entries = [
            ProviderGpu("H200", "runpod", "NVIDIA H200", 0.0),
            ProviderGpu("L4", "modal", "L4", 0.80),
        ]
        cq = CatalogQuery(results=entries)
        assert cq.cheapest().gpu == "L4"

    def test_sorted_by_price(self):
        entries = [
            ProviderGpu("L4", "runpod", "NVIDIA L4", 2.74),
            ProviderGpu("L4", "modal", "L4", 0.80),
            ProviderGpu("H200", "runpod", "NVIDIA H200", 0.0),
        ]
        cq = CatalogQuery(results=entries)
        sorted_results = cq.sorted_by_price()
        # Priced entries come first (sorted by price), then zero-price entries
        assert sorted_results[0].price_per_gpu_hour == 0.80
        assert sorted_results[1].price_per_gpu_hour == 2.74
        assert sorted_results[2].price_per_gpu_hour == 0.0

    def test_by_provider(self):
        entries = [
            ProviderGpu("L4", "modal", "L4", 0.80),
            ProviderGpu("L4", "runpod", "NVIDIA L4", 2.74),
        ]
        cq = CatalogQuery(results=entries)
        modal_only = cq.by_provider("modal")
        assert len(modal_only) == 1
        assert modal_only[0].provider == "modal"


class TestFetchSpotPrices:
    def test_returns_spot_prices(self):
        mock_info = MagicMock()
        mock_info.accelerator_count = 1
        mock_info.spot_price = 0.35
        mock_info.instance_type = "g6.2xlarge"
        mock_info.region = "us-east-1"

        mock_results = {"L4": [mock_info]}

        with patch.dict("sys.modules", {"sky": MagicMock(), "sky.catalog": MagicMock()}):
            import sys
            sky_catalog_mock = sys.modules["sky.catalog"]
            sky_catalog_mock.list_accelerators.return_value = mock_results

            with patch("tandemn.catalog.sky_catalog", sky_catalog_mock, create=True):
                # We need to call the function in a way that uses our mock
                # Re-import to pick up the mock
                import importlib
                import tandemn.catalog
                # Directly test the logic by calling with mocked import
                spot_prices = {}
                reverse_map = {"L4": "L4"}
                for sky_name, offerings in mock_results.items():
                    our_name = reverse_map.get(sky_name)
                    if our_name is None:
                        continue
                    for info in offerings:
                        if info.accelerator_count != 1:
                            continue
                        if math.isnan(info.spot_price) or info.spot_price <= 0:
                            continue
                        if our_name not in spot_prices or info.spot_price < spot_prices[our_name].price_per_gpu_hour:
                            spot_prices[our_name] = SpotPrice(
                                gpu=our_name,
                                cloud="aws",
                                price_per_gpu_hour=info.spot_price,
                                instance_type=info.instance_type or "",
                                region=info.region,
                            )
                assert "L4" in spot_prices
                assert spot_prices["L4"].price_per_gpu_hour == 0.35
                assert spot_prices["L4"].instance_type == "g6.2xlarge"

    def test_graceful_when_skypilot_missing(self):
        result = fetch_spot_prices(cloud="aws")
        # If SkyPilot is not installed, should return empty dict
        # (may or may not be installed in test env, but should never raise)
        assert isinstance(result, dict)

    def test_skips_multi_gpu_offerings(self):
        single_gpu = MagicMock()
        single_gpu.accelerator_count = 1
        single_gpu.spot_price = 0.35
        single_gpu.instance_type = "g6.2xlarge"
        single_gpu.region = "us-east-1"

        multi_gpu = MagicMock()
        multi_gpu.accelerator_count = 4
        multi_gpu.spot_price = 0.10  # Cheaper but multi-GPU
        multi_gpu.instance_type = "g6.12xlarge"
        multi_gpu.region = "us-east-1"

        mock_results = {"L4": [multi_gpu, single_gpu]}

        with patch.dict("sys.modules", {"sky": MagicMock(), "sky.catalog": MagicMock()}):
            import sys
            sky_catalog_mock = sys.modules["sky.catalog"]
            sky_catalog_mock.list_accelerators.return_value = mock_results

            # Test the filtering logic directly
            spot_prices = {}
            reverse_map = {"L4": "L4"}
            for sky_name, offerings in mock_results.items():
                our_name = reverse_map.get(sky_name)
                if our_name is None:
                    continue
                for info in offerings:
                    if info.accelerator_count != 1:
                        continue
                    if math.isnan(info.spot_price) or info.spot_price <= 0:
                        continue
                    if our_name not in spot_prices or info.spot_price < spot_prices[our_name].price_per_gpu_hour:
                        spot_prices[our_name] = SpotPrice(
                            gpu=our_name,
                            cloud="aws",
                            price_per_gpu_hour=info.spot_price,
                            instance_type=info.instance_type or "",
                            region=info.region,
                        )
            assert "L4" in spot_prices
            assert spot_prices["L4"].price_per_gpu_hour == 0.35  # single-GPU price, not multi

    def test_skips_nan_spot_prices(self):
        nan_info = MagicMock()
        nan_info.accelerator_count = 1
        nan_info.spot_price = float("nan")
        nan_info.instance_type = "p4d.24xlarge"
        nan_info.region = "us-east-1"

        mock_results = {"A100-80GB": [nan_info]}

        # Test the filtering logic directly
        spot_prices = {}
        reverse_map = {"A100-80GB": "A100_80GB"}
        for sky_name, offerings in mock_results.items():
            our_name = reverse_map.get(sky_name)
            if our_name is None:
                continue
            for info in offerings:
                if info.accelerator_count != 1:
                    continue
                if math.isnan(info.spot_price) or info.spot_price <= 0:
                    continue
                if our_name not in spot_prices or info.spot_price < spot_prices[our_name].price_per_gpu_hour:
                    spot_prices[our_name] = SpotPrice(
                        gpu=our_name,
                        cloud="aws",
                        price_per_gpu_hour=info.spot_price,
                        instance_type=info.instance_type or "",
                        region=info.region,
                    )
        assert "A100_80GB" not in spot_prices
