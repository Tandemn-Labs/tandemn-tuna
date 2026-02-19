"""GPU pricing catalog — single source of truth for GPU specs, provider mappings, and pricing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuSpec:
    """Hardware facts for a GPU type."""
    short_name: str       # "L4", "H100", "A100_80GB"
    full_name: str        # "NVIDIA L4", "NVIDIA H100 80GB HBM3"
    vram_gb: int          # 24, 80, etc.
    arch: str             # "ada", "ampere", "hopper", "blackwell"


@dataclass(frozen=True)
class ProviderGpu:
    """One GPU offering from one serverless provider."""
    gpu: str              # References GpuSpec.short_name
    provider: str         # "modal", "runpod", "cloudrun"
    provider_gpu_id: str  # Provider-specific ID (e.g. "nvidia-l4" for Cloud Run)
    price_per_gpu_hour: float  # USD, 0.0 = unknown/not listed
    regions: tuple[str, ...] = ()  # Empty = all regions


@dataclass(frozen=True)
class SpotPrice:
    """A spot GPU price from SkyPilot catalog."""
    gpu: str              # GpuSpec.short_name
    cloud: str            # "aws"
    price_per_gpu_hour: float
    instance_type: str    # e.g. "p5.4xlarge"
    region: str           # Cheapest region


@dataclass(frozen=True)
class OnDemandPrice:
    """An on-demand GPU price from SkyPilot catalog."""
    gpu: str              # GpuSpec.short_name
    cloud: str            # "aws"
    price_per_gpu_hour: float
    instance_type: str    # e.g. "p5.4xlarge"
    region: str           # Cheapest region


@dataclass
class CatalogQuery:
    """Result of a catalog query with convenience methods."""
    results: list[ProviderGpu]
    spot_prices: dict[str, SpotPrice] = field(default_factory=dict)  # gpu -> SpotPrice

    def cheapest(self) -> ProviderGpu | None:
        priced = [r for r in self.results if r.price_per_gpu_hour > 0]
        if not priced:
            return None
        return min(priced, key=lambda r: r.price_per_gpu_hour)

    def by_provider(self, provider: str) -> list[ProviderGpu]:
        return [r for r in self.results if r.provider == provider]

    def sorted_by_price(self) -> list[ProviderGpu]:
        return sorted(self.results, key=lambda r: (r.price_per_gpu_hour == 0, r.price_per_gpu_hour))


# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

GPU_SPECS: dict[str, GpuSpec] = {
    "T4": GpuSpec("T4", "NVIDIA T4", 16, "turing"),
    "A10": GpuSpec("A10", "NVIDIA A10", 24, "ampere"),
    "A10G": GpuSpec("A10G", "NVIDIA A10G", 24, "ampere"),
    "L4": GpuSpec("L4", "NVIDIA L4", 24, "ada"),
    "A4000": GpuSpec("A4000", "NVIDIA RTX A4000", 16, "ampere"),
    "A5000": GpuSpec("A5000", "NVIDIA RTX A5000", 24, "ampere"),
    "A6000": GpuSpec("A6000", "NVIDIA RTX A6000", 48, "ampere"),
    "RTX4090": GpuSpec("RTX4090", "NVIDIA GeForce RTX 4090", 24, "ada"),
    "A40": GpuSpec("A40", "NVIDIA A40", 48, "ampere"),
    "L40": GpuSpec("L40", "NVIDIA L40", 48, "ada"),
    "L40S": GpuSpec("L40S", "NVIDIA L40S", 48, "ada"),
    "A100_40GB": GpuSpec("A100_40GB", "NVIDIA A100 40GB", 40, "ampere"),
    "A100_80GB": GpuSpec("A100_80GB", "NVIDIA A100 80GB SXM", 80, "ampere"),
    "H100_MIG": GpuSpec("H100_MIG", "NVIDIA H100 MIG", 40, "hopper"),
    "H100": GpuSpec("H100", "NVIDIA H100 80GB HBM3", 80, "hopper"),
    "H200": GpuSpec("H200", "NVIDIA H200", 141, "hopper"),
    "B200": GpuSpec("B200", "NVIDIA B200", 192, "blackwell"),
    "RTX_PRO_6000": GpuSpec("RTX_PRO_6000", "NVIDIA RTX PRO 6000", 32, "blackwell"),
}

GPU_ALIASES: dict[str, str] = {
    "A100": "A100_80GB",      # RunPod "A100" is the 80GB variant
    "4090": "RTX4090",         # RunPod uses "4090"
}

_PROVIDER_GPUS: list[ProviderGpu] = [
    # Modal
    ProviderGpu("T4", "modal", "T4", 0.59),
    ProviderGpu("A10G", "modal", "A10G", 1.10),
    ProviderGpu("L4", "modal", "L4", 0.80),
    ProviderGpu("A40", "modal", "A40", 1.10),
    ProviderGpu("L40S", "modal", "L40S", 1.60),
    ProviderGpu("A100_40GB", "modal", "A100_40GB", 1.82),
    ProviderGpu("A100_80GB", "modal", "A100_80GB", 2.78),
    ProviderGpu("H100", "modal", "H100", 3.95),
    ProviderGpu("B200", "modal", "B200", 5.49),

    # RunPod (serverless) — prices converted from per-second to per-hour
    ProviderGpu("A4000", "runpod", "NVIDIA RTX A4000", 0.43),
    ProviderGpu("A5000", "runpod", "NVIDIA RTX A5000", 0.58),
    ProviderGpu("L4", "runpod", "NVIDIA L4", 2.74),
    ProviderGpu("RTX4090", "runpod", "NVIDIA GeForce RTX 4090", 1.01),
    ProviderGpu("A6000", "runpod", "NVIDIA RTX A6000", 0.79),
    ProviderGpu("L40", "runpod", "NVIDIA L40", 1.15),
    ProviderGpu("L40S", "runpod", "NVIDIA L40S", 1.58),
    ProviderGpu("A40", "runpod", "NVIDIA A40", 0.79),
    ProviderGpu("A100_80GB", "runpod", "NVIDIA A100-SXM4-80GB", 1.12),
    ProviderGpu("H100", "runpod", "NVIDIA H100 80GB HBM3", 4.97),
    ProviderGpu("H200", "runpod", "NVIDIA H200", 0.0),
    ProviderGpu("B200", "runpod", "NVIDIA B200", 0.0),

    # Cloud Run
    ProviderGpu(
        "L4", "cloudrun", "nvidia-l4", 0.84,
        regions=(
            "asia-east1", "asia-northeast1", "asia-south1", "asia-southeast1",
            "europe-west1", "europe-west4", "me-west1",
            "us-central1", "us-east1", "us-east4", "us-west1", "us-west4",
        ),
    ),
    ProviderGpu(
        "RTX_PRO_6000", "cloudrun", "nvidia-rtx-pro-6000", 0.84,
        regions=("us-central1",),
    ),

    # Azure Container Apps (prices from Azure Retail Prices API, eastus region)
    ProviderGpu(
        "T4", "azure", "Consumption-GPU-NC8as-T4", 0.26,
        regions=(
            "australiaeast", "brazilsouth", "canadacentral", "canadaeast",
            "centralindia", "centralus", "eastasia", "eastus", "eastus2",
            "francecentral", "germanywestcentral", "japaneast", "koreacentral",
            "northcentralus", "northeurope", "southcentralus", "southeastasia",
            "swedencentral", "uksouth", "westeurope", "westus", "westus2", "westus3",
        ),
    ),
    ProviderGpu(
        "A100_80GB", "azure", "Consumption-GPU-NC24-A100", 1.90,
        regions=(
            "australiaeast", "brazilsouth", "canadacentral", "canadaeast",
            "centralindia", "centralus", "eastasia", "eastus", "eastus2",
            "francecentral", "germanywestcentral", "japaneast", "koreacentral",
            "northcentralus", "northeurope", "southcentralus", "southeastasia",
            "swedencentral", "uksouth", "westeurope", "westus", "westus2", "westus3",
        ),
    ),

    # Baseten
    ProviderGpu("T4", "baseten", "T4", 0.63),
    ProviderGpu("L4", "baseten", "L4", 0.85),
    ProviderGpu("A10G", "baseten", "A10G", 1.21),
    ProviderGpu("A100_80GB", "baseten", "A100", 4.00),
    ProviderGpu("H100_MIG", "baseten", "H100_MIG", 3.75),
    ProviderGpu("H100", "baseten", "H100", 6.50),
    ProviderGpu("B200", "baseten", "B200", 9.98),
]

_SKYPILOT_GPU_NAME_MAP: dict[str, str] = {
    "T4": "T4",
    "L4": "L4",
    "L40S": "L40S",
    "A10G": "A10G",
    "A100_40GB": "A100",
    "A100_80GB": "A100-80GB",
    "H100": "H100",
    "H200": "H200",
    "B200": "B200",
}


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

def get_gpu_spec(name: str) -> GpuSpec:
    """Lookup hardware spec by short name. Raises KeyError if unknown."""
    return GPU_SPECS[name]


def normalize_gpu_name(name: str) -> str:
    """Resolve aliases to canonical name. Raises KeyError if completely unknown."""
    if name in GPU_SPECS:
        return name
    if name in GPU_ALIASES:
        return GPU_ALIASES[name]
    raise KeyError(name)


def query(
    gpu: str | None = None,
    provider: str | None = None,
    min_vram_gb: int | None = None,
    max_price: float | None = None,
    include_spot: bool = False,
    spot_cloud: str = "aws",
) -> CatalogQuery:
    """Query the catalog with optional filters. Returns CatalogQuery wrapper."""
    results = list(_PROVIDER_GPUS)

    if gpu is not None:
        results = [r for r in results if r.gpu == gpu]
    if provider is not None:
        results = [r for r in results if r.provider == provider]
    if min_vram_gb is not None:
        results = [r for r in results if GPU_SPECS.get(r.gpu, GpuSpec("", "", 0, "")).vram_gb >= min_vram_gb]
    if max_price is not None:
        results = [r for r in results if 0 < r.price_per_gpu_hour <= max_price]

    spot_prices: dict[str, SpotPrice] = {}
    if include_spot:
        spot_prices = fetch_spot_prices(cloud=spot_cloud)

    return CatalogQuery(results=results, spot_prices=spot_prices)


def provider_gpu_id(gpu: str, provider: str) -> str:
    """Get provider-specific GPU identifier. Raises KeyError if not found."""
    for entry in _PROVIDER_GPUS:
        if entry.gpu == gpu and entry.provider == provider:
            return entry.provider_gpu_id
    raise KeyError(f"No {provider} offering for GPU {gpu!r}")


def provider_gpu_map(provider: str) -> dict[str, str]:
    """Return full {short_name: provider_id} dict for a provider."""
    return {e.gpu: e.provider_gpu_id for e in _PROVIDER_GPUS if e.provider == provider}


def provider_regions(gpu: str, provider: str) -> tuple[str, ...]:
    """Return region availability for a GPU on a provider. Empty = all regions."""
    for entry in _PROVIDER_GPUS:
        if entry.gpu == gpu and entry.provider == provider:
            return entry.regions
    return ()


def fetch_spot_prices(cloud: str = "aws") -> dict[str, SpotPrice]:
    """Fetch live spot prices from SkyPilot catalog.

    Returns {our_short_name: SpotPrice} for the cheapest offering per GPU.
    Uses sky.catalog.list_accelerators() under the hood.
    Gracefully returns {} if SkyPilot not installed or catalog unavailable.
    """
    try:
        import sky.catalog as sky_catalog
    except ImportError:
        return {}

    # Build reverse map: SkyPilot name -> our short name
    reverse_map = {v: k for k, v in _SKYPILOT_GPU_NAME_MAP.items()}

    try:
        results = sky_catalog.list_accelerators(
            gpus_only=True,
            clouds=cloud,
            all_regions=False,
            require_price=True,
        )
    except Exception:
        return {}

    spot_prices: dict[str, SpotPrice] = {}
    for sky_name, offerings in results.items():
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
                    cloud=cloud,
                    price_per_gpu_hour=info.spot_price,
                    instance_type=info.instance_type or "",
                    region=info.region,
                )
    return spot_prices


def fetch_on_demand_prices(cloud: str = "aws") -> dict[str, OnDemandPrice]:
    """Fetch on-demand prices from SkyPilot catalog.

    Same structure as fetch_spot_prices() but uses info.price (on-demand)
    instead of info.spot_price.

    Returns {our_short_name: OnDemandPrice} for the cheapest offering per GPU.
    Gracefully returns {} if SkyPilot not installed or catalog unavailable.
    """
    try:
        import sky.catalog as sky_catalog
    except ImportError:
        return {}

    # Build reverse map: SkyPilot name -> our short name
    reverse_map = {v: k for k, v in _SKYPILOT_GPU_NAME_MAP.items()}

    try:
        results = sky_catalog.list_accelerators(
            gpus_only=True,
            clouds=cloud,
            all_regions=False,
            require_price=True,
        )
    except Exception:
        return {}

    on_demand_prices: dict[str, OnDemandPrice] = {}
    for sky_name, offerings in results.items():
        our_name = reverse_map.get(sky_name)
        if our_name is None:
            continue
        for info in offerings:
            if info.accelerator_count != 1:
                continue
            if math.isnan(info.price) or info.price <= 0:
                continue
            if our_name not in on_demand_prices or info.price < on_demand_prices[our_name].price_per_gpu_hour:
                on_demand_prices[our_name] = OnDemandPrice(
                    gpu=our_name,
                    cloud=cloud,
                    price_per_gpu_hour=info.price,
                    instance_type=info.instance_type or "",
                    region=info.region,
                )
    return on_demand_prices


def get_provider_price(gpu: str, provider: str) -> float:
    """Get the static serverless price for a GPU+provider combo. Returns 0.0 if not found."""
    for entry in _PROVIDER_GPUS:
        if entry.gpu == gpu and entry.provider == provider:
            return entry.price_per_gpu_hour
    return 0.0
