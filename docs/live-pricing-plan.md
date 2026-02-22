# Live Pricing System — Design Plan

> Nightly automated price updates for all serverless GPU providers.

## Problem

All serverless pricing in `tuna/catalog.py` is hardcoded. Prices go stale when providers change rates. Users can't trust `tuna show-gpus` without manually verifying each provider's website.

## Solution

A nightly GitHub Action scrapes/fetches prices from every serverless provider, writes `tuna/prices.json`, and commits it. The catalog loads this file at import time as an overlay on the static fallback prices.

## Architecture

```
  GitHub Actions (4 AM UTC daily)
          │
          ▼
  scripts/update_prices.py
          │
    ┌─────┼─────┬──────────┬──────────┬──────────┬───────────┐
    ▼     ▼     ▼          ▼          ▼          ▼           ▼
  Modal RunPod Baseten  Cerebrium  CloudRun   Azure     SkyPilot
  (web) (GQL)  (web)    (web)     (API)      (API)     (SDK)
    │     │     │          │          │          │           │
    └─────┴─────┴──────────┴──────────┴──────────┴───────────┘
          │
          ▼
    tuna/prices.json  ──(git commit)──▶  repo
          │
          ▼
    tuna/catalog.py loads at import
```

**Why JSON-in-repo, not a database:**
- Zero runtime dependencies — no SQLite, no network call at query time
- Git history IS the time series — `git log -p tuna/prices.json` shows every price change
- Transparent — users can inspect exactly what prices Tuna uses
- Offline works — prices.json ships with the package
- CI-friendly — GitHub Action commits the update automatically

---

## Provider-by-Provider Plan

### Tier 1: Public API (no auth, reliable)

| Provider | Method | Endpoint | Auth | Notes |
|----------|--------|----------|------|-------|
| **Azure** | REST API | `https://prices.azure.com/api/retail/prices` | None | Azure Retail Prices API is fully public. Filter by `serviceName` and `meterName` for Container Apps GPU SKUs. |
| **Cloud Run** | REST API | `https://cloudpricingcalculator.appspot.com/static/data/pricelist.json` or Cloud Billing API | None (public JSON) or API key | Google publishes a public pricing JSON. Alternatively, the Cloud Billing Catalog API (`cloudbilling.googleapis.com`) is free but needs a GCP project. |
| **SkyPilot** | Python SDK | `sky.catalog.list_accelerators()` | None | Already integrated for spot prices. Extend to include on-demand prices for all clouds. |

### Tier 2: Public GraphQL / REST (no auth for pricing)

| Provider | Method | Endpoint | Auth | Notes |
|----------|--------|----------|------|-------|
| **RunPod** | GraphQL | `POST https://api.runpod.io/graphql` | None for `gpuTypes` query | The `gpuTypes` query returns `communityPrice` and `securePrice` for all GPUs. This is the **serverless pod pricing** (secure = serverless). No API key needed for this read-only query. |

### Tier 3: Web Scraping (no API, prices on website)

| Provider | Method | Source Page | Auth | Notes |
|----------|--------|-------------|------|-------|
| **Modal** | Web scrape | `https://modal.com/pricing` | None | No public pricing API. Prices are on the website. Scrape the pricing table or use the known pricing page. Prices change infrequently (quarterly). |
| **Baseten** | Web scrape | `https://www.baseten.co/pricing` | None | No public pricing API. The pricing page lists GPU-hour rates. Prices change infrequently. |
| **Cerebrium** | Web scrape | `https://www.cerebrium.ai/pricing` | None | No public pricing API. Pricing page lists per-second rates by GPU. Convert to per-hour. |

### Tier 4: Future Providers (not yet integrated)

| Provider | Method | Endpoint | Auth | Notes |
|----------|--------|----------|------|-------|
| **Beam Cloud** | REST API | `https://api.beam.cloud/v1` | TBD | Per-millisecond billing. Check if pricing endpoint is public. |
| **Koyeb** | REST API | `https://app.koyeb.com/v1` | API key | Has a public pricing page, may need scraping. |
| **Fireworks AI** | Web scrape | `https://fireworks.ai/pricing` | None | Serverless LLM inference. Pricing on website. |
| **Together AI** | Web scrape | `https://www.together.ai/pricing` | None | Serverless inference. Pricing on website. |
| **Replicate** | REST API | `https://api.replicate.com/v1` | API key | Has `GET /v1/hardware` endpoint listing GPU prices. Needs auth token. |

---

## Data Format

### `tuna/prices.json`

```json
{
  "schema_version": 1,
  "updated_at": "2026-02-22T04:00:00Z",
  "providers": {
    "modal": {
      "source": "web_scrape",
      "fetched_at": "2026-02-22T04:00:12Z",
      "gpus": [
        {"gpu": "T4", "price_per_gpu_hour": 0.59, "provider_gpu_id": "T4"},
        {"gpu": "L4", "price_per_gpu_hour": 0.80, "provider_gpu_id": "L4"},
        {"gpu": "H100", "price_per_gpu_hour": 3.95, "provider_gpu_id": "H100"}
      ]
    },
    "runpod": {
      "source": "graphql_api",
      "fetched_at": "2026-02-22T04:00:08Z",
      "gpus": [
        {"gpu": "A100_80GB", "price_per_gpu_hour": 1.12, "provider_gpu_id": "NVIDIA A100-SXM4-80GB"},
        {"gpu": "H100", "price_per_gpu_hour": 4.97, "provider_gpu_id": "NVIDIA H100 80GB HBM3"}
      ]
    },
    "azure": {
      "source": "retail_prices_api",
      "fetched_at": "2026-02-22T04:00:05Z",
      "gpus": [
        {"gpu": "T4", "price_per_gpu_hour": 0.26, "provider_gpu_id": "Consumption-GPU-NC8as-T4", "regions": ["eastus", "westus2"]},
        {"gpu": "A100_80GB", "price_per_gpu_hour": 1.90, "provider_gpu_id": "Consumption-GPU-NC24-A100", "regions": ["eastus", "westus2"]}
      ]
    },
    "spot": {
      "source": "skypilot_catalog",
      "fetched_at": "2026-02-22T04:00:15Z",
      "clouds": {
        "aws": [
          {"gpu": "T4", "price_per_gpu_hour": 0.06, "instance_type": "Standard_NC4as_T4_v3", "region": "eastus2"},
          {"gpu": "A100_80GB", "price_per_gpu_hour": 0.40, "instance_type": "Standard_NC24ads_A100_v4", "region": "westus3"}
        ],
        "azure": [
          {"gpu": "T4", "price_per_gpu_hour": 0.06, "instance_type": "Standard_NC4as_T4_v3", "region": "eastus"}
        ]
      }
    }
  }
}
```

### Key design decisions:
- **`schema_version`** — allows future format changes without breaking old installs
- **`source`** per provider — makes it clear how each price was obtained
- **`fetched_at`** per provider — individual timestamps (some may fail, others succeed)
- **`spot` is nested by cloud** — matches our multi-cloud spot support
- **Regions are optional** — only included for providers where it matters (Azure, Cloud Run)

---

## Catalog Integration

In `tuna/catalog.py`, add a thin overlay loader:

```python
import json
from pathlib import Path

_PRICES_FILE = Path(__file__).parent / "prices.json"

def _load_live_prices() -> dict:
    """Load nightly-scraped prices. Returns {} if file missing or invalid."""
    if not _PRICES_FILE.exists():
        return {}
    try:
        data = json.loads(_PRICES_FILE.read_text())
        if data.get("schema_version") != 1:
            return {}
        return data
    except Exception:
        return {}

_LIVE_PRICES = _load_live_prices()
```

**Merge strategy in `query()`:**
- Live prices **override** static prices when available
- Static prices remain as fallback (if scraper fails for a provider, stale-but-correct > missing)
- `show-gpus` output shows `"updated: 6h ago"` from the `fetched_at` timestamp
- A provider entry in `prices.json` with `"gpus": []` (empty) means "scrape succeeded but found nothing" — distinct from the provider key being absent (scrape failed, use static fallback)

---

## Scraper Script

### `scripts/update_prices.py`

```
scripts/
  update_prices.py          # Main entrypoint
  scrapers/
    __init__.py
    base.py                 # BaseScraper protocol
    runpod.py               # GraphQL API
    azure.py                # Retail Prices API
    cloudrun.py             # Google pricing JSON
    modal.py                # Web scrape
    baseten.py              # Web scrape
    cerebrium.py            # Web scrape
    skypilot_spot.py        # SkyPilot SDK
```

**Each scraper is a single async function:**

```python
# scripts/scrapers/base.py
from typing import TypedDict

class GpuPrice(TypedDict):
    gpu: str                    # Tuna canonical name (e.g. "A100_80GB")
    price_per_gpu_hour: float
    provider_gpu_id: str        # Provider's own GPU identifier
    regions: list[str]          # Optional, empty = all

class ScraperResult(TypedDict):
    source: str                 # "graphql_api", "retail_prices_api", "web_scrape", "skypilot_catalog"
    gpus: list[GpuPrice]
```

**Adding a new provider = one file:**

```python
# scripts/scrapers/newprovider.py
async def fetch(session: aiohttp.ClientSession) -> ScraperResult:
    """Fetch pricing for NewProvider. Returns ScraperResult."""
    # ... fetch from API or scrape website ...
    return {"source": "web_scrape", "gpus": [...]}
```

Then register it in `update_prices.py`:

```python
SCRAPERS = {
    "modal": scrapers.modal.fetch,
    "runpod": scrapers.runpod.fetch,
    "baseten": scrapers.baseten.fetch,
    "cerebrium": scrapers.cerebrium.fetch,
    "cloudrun": scrapers.cloudrun.fetch,
    "azure": scrapers.azure.fetch,
    "newprovider": scrapers.newprovider.fetch,  # <-- one line
}
```

---

## GitHub Action

### `.github/workflows/update-prices.yml`

```yaml
name: Update GPU Prices
on:
  schedule:
    - cron: '0 4 * * *'    # 4 AM UTC daily
  workflow_dispatch: {}      # Manual trigger button

jobs:
  update-prices:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v5

      - name: Install scraper deps
        run: uv pip install aiohttp beautifulsoup4

      - name: Install tuna (for SkyPilot spot prices)
        run: uv pip install -e ".[all]"

      - name: Run price scraper
        run: uv run python scripts/update_prices.py

      - name: Check for changes
        id: diff
        run: |
          git diff --quiet tuna/prices.json && echo "changed=false" >> $GITHUB_OUTPUT || echo "changed=true" >> $GITHUB_OUTPUT

      - name: Commit and push
        if: steps.diff.outputs.changed == 'true'
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add tuna/prices.json
          git commit -m "chore: update GPU prices $(date -u +%Y-%m-%d)"
          git push
```

**Key behaviors:**
- Runs at 4 AM UTC daily (off-peak, all providers awake)
- Only commits if prices actually changed (avoids noise)
- `workflow_dispatch` allows manual trigger for testing
- Failures are silent (the previous prices.json remains — stale prices > no prices)
- No secrets needed (all APIs are public for price reads)

---

## Rollout Plan

### Phase 1: Foundation (do first)
1. Create `tuna/prices.json` with current hardcoded prices (seed file)
2. Add `_load_live_prices()` to `catalog.py`
3. Wire `show-gpus` to show `"updated: X ago"` timestamp
4. Create `scripts/update_prices.py` skeleton + `scrapers/` directory

### Phase 2: API-based scrapers (easiest, most reliable)
1. `scrapers/runpod.py` — GraphQL `gpuTypes` query (no auth, highest confidence)
2. `scrapers/azure.py` — Azure Retail Prices API (no auth, official)
3. `scrapers/skypilot_spot.py` — `sky.catalog.list_accelerators()` for all clouds

### Phase 3: Web scrapers (need maintenance)
1. `scrapers/modal.py` — scrape `modal.com/pricing`
2. `scrapers/baseten.py` — scrape `baseten.co/pricing`
3. `scrapers/cerebrium.py` — scrape `cerebrium.ai/pricing`
4. `scrapers/cloudrun.py` — Google's public pricing JSON or Cloud Billing API

### Phase 4: New providers (expand)
1. Add provider to `scrapers/newprovider.py`
2. Add one line to `SCRAPERS` dict
3. Add GPU name mappings to `catalog.py` if new GPU types
4. Done — next nightly run picks it up

---

## Failure Modes & Mitigations

| Failure | Impact | Mitigation |
|---------|--------|------------|
| API returns error | One provider missing from update | Skip that provider, keep previous prices.json entry |
| Website layout changes | Web scraper breaks | Static fallback prices in catalog.py still work. Alert via CI failure notification. |
| Price is obviously wrong (e.g. $0.00, $999) | Bad data in prices.json | Sanity check: reject prices outside 0.01-100.00 range, reject >50% change from previous |
| GitHub Action fails entirely | prices.json not updated | Previous commit's prices.json is still valid. No data loss. |
| Provider removes a GPU | Stale GPU entry | Scraper returns empty for that GPU, overlay removes it from live prices |

### Sanity Checks (in `update_prices.py`)

```python
def validate_price(gpu: str, price: float, previous: float | None) -> bool:
    """Reject obviously wrong prices."""
    if price <= 0 or price > 100:
        return False
    if previous and abs(price - previous) / previous > 0.5:
        # >50% change — flag for review but still accept
        logger.warning(f"{gpu}: price changed {previous:.2f} -> {price:.2f} (>50%)")
    return True
```

---

## What We DON'T Need

- **SQLite database** — git history gives us time-series for free
- **Financial Greeks / volatility** — serverless prices change quarterly, not hourly. Overkill.
- **User API keys for pricing** — all pricing sources are public
- **Real-time price fetching at CLI time** — nightly is fresh enough for serverless (prices don't change intraday)
- **Complex merge logic** — simple overlay: live > static > missing

---

## Open Questions

1. **Should prices.json ship in the PyPI package?** Yes — gives users instant pricing without waiting for first nightly run.
2. **Should we alert on price changes?** Maybe — a GitHub Action that comments on the commit with a diff table would be nice.
3. **Should web scrapers use a headless browser?** No — start with `beautifulsoup4` + `aiohttp`. Only add Playwright if a provider requires JS rendering.
