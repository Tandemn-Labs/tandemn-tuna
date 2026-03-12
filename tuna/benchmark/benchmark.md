# Cold Start Benchmarking

Measure cold start latency across serverless GPU providers.

## Quick Start

```bash
# Single provider
tuna benchmark cold-start --provider modal --gpu L4 --model Qwen/Qwen3-0.6B

# Multiple providers
tuna benchmark cold-start --provider modal,runpod,baseten --gpu L4 --model Qwen/Qwen3-4B-Instruct-2507

# All providers (modal, runpod, cloudrun, baseten, cerebrium)
tuna benchmark cold-start --provider all --gpu L4 --model Qwen/Qwen3-0.6B
```

## Scenarios

- **fresh-cold** — Deploy from scratch, measure full cold start (image pull + model load + first inference). Tears down after.
- **warm-cold** — Uses an existing deployment, waits for scale-to-zero, then measures reboot time. Repeats `--repeat` times.
- **both** (default) — Runs fresh-cold first, then warm-cold on the same deployment.

## What Gets Measured

**Time to Inference** — the only metric that matters. Measures wall-clock time from trigger (deploy or cold start) to a successful chat completion response.

The benchmark runs multiple iterations:
- **Run 1 (Fresh)** — Deploy from scratch → first successful inference
- **Run 2** — Scale to zero → trigger → inference
- **Run 3** — Scale to zero → trigger → inference
- **Avg (Warm)** — Mean of runs 2+

## Flags

```
--provider      Provider(s): single name, comma-separated, or 'all'
--gpu           GPU type (default: T4)
--model         HuggingFace model (default: Qwen/Qwen3-0.6B)
--scenario      fresh-cold | warm-cold | both (default: both)
--repeat        Number of warm-cold cycles (default: 3)
--idle-wait     Seconds to wait for scale-to-zero (default: 300)
--output        table | json | csv (default: table)
--no-teardown   Keep deployment alive after benchmarking
--max-model-len vLLM max model length (default: 512)
```

For Cloud Run specifically:
```
--gcp-project   GCP project ID
--gcp-region    GCP region (default: us-central1)
```

## Using an Existing Deployment

If you already have a deployment running, skip the deploy step:

```bash
# By service name (from `tuna list`)
tuna benchmark cold-start --provider modal --service-name my-deployment --gpu L4

# By endpoint URL
tuna benchmark cold-start --provider runpod --endpoint-url https://api.runpod.ai/v2/xxx/openai --gpu L4
```

These only work with a single provider.

## Preflight Checks

When using multiple providers, all providers are validated before any deployment starts (fail fast). This checks API keys, SDK installs, and provider-specific requirements. If any provider fails preflight, nothing gets deployed.

```bash
# Check a single provider manually
tuna check --provider cloudrun --gpu L4
```

## Required Environment Variables

| Provider | Variable | Where to get it |
|----------|----------|-----------------|
| RunPod | `RUNPOD_API_KEY` | https://www.runpod.io/console/user/settings |
| Baseten | `BASETEN_API_KEY` | https://app.baseten.co/settings/api_keys |
| Cerebrium | `CEREBRIUM_API_KEY` | Cerebrium dashboard |
| Cloud Run | `GOOGLE_CLOUD_PROJECT` | `gcloud config get-value project` |
| Modal | — | `modal token new` (one-time setup) |
| All | `HF_TOKEN` (optional) | https://huggingface.co/settings/tokens |

## Output Formats

```bash
# Rich table (default)
tuna benchmark cold-start --provider modal --output table

# JSON (for programmatic use)
tuna benchmark cold-start --provider modal --output json

# CSV (for spreadsheets)
tuna benchmark cold-start --provider modal --output csv
```

## Benchmark Results (2026-03-02)

Model: **Qwen/Qwen3-4B-Instruct-2507** (8 GB, BF16) on **NVIDIA L4** GPUs.

Scenario: `--scenario both --repeat 3` — deploy fresh, measure first inference, then 3 warm cold start cycles (scale-to-zero → trigger → inference).

### Time to Inference (ranked by warm cold start)

| Provider | Run 1 (Fresh) | Run 2 | Run 3 | Run 4 | Avg (Warm) |
|----------|--------------|-------|-------|-------|------------|
| Cerebrium | 140.9s | 66.8s | 65.2s | 58.4s | **63.5s** |
| Modal | 90.1s | 95.3s | 78.3s | 56.1s | **76.6s** |
| Baseten | 213.4s | 99.7s | 103.6s | 316.9s | **173.4s** |
| RunPod | 298.8s | 649.5s | 641.3s | 626.6s | **639.1s** |
| Cloud Run | 477.9s | — | — | — | N/A |

### Detailed breakdown

| Provider | Scenario | Deploy | Health Ready | 1st Inference | Total |
|----------|----------|--------|-------------|---------------|-------|
| Modal | fresh | 87.8s | 88.7s | 1.4s | 90.1s |
| Modal | warm 1 | — | 93.6s | 1.7s | 95.3s |
| Modal | warm 2 | — | 76.8s | 1.5s | 78.3s |
| Modal | warm 3 | — | 54.7s | 1.5s | 56.1s |
| RunPod | fresh | 5.6s | 6.1s | 292.7s | 298.8s |
| RunPod | warm 1 | — | — | 38.7s | 649.5s |
| RunPod | warm 2 | — | — | 40.4s | 641.3s |
| RunPod | warm 3 | — | — | 22.7s | 626.6s |
| Baseten | fresh | 22.7s | 212.2s | 1.2s | 213.4s |
| Baseten | warm 1 | — | 98.4s | 1.3s | 99.7s |
| Baseten | warm 2 | — | 102.3s | 1.3s | 103.6s |
| Baseten | warm 3 | — | 314.9s | 1.9s | 316.9s |
| Cerebrium | fresh | 133.1s | 139.6s | 1.3s | 140.9s |
| Cerebrium | warm 1 | — | 65.5s | 1.2s | 66.8s |
| Cerebrium | warm 2 | — | 64.1s | 1.1s | 65.2s |
| Cerebrium | warm 3 | — | 57.2s | 1.1s | 58.4s |
| Cloud Run | fresh | 475.9s | 476.7s | 1.2s | 477.9s |
| Cloud Run | warm 1-3 | — | — | — | skipped |

### Notes

- **Cerebrium** has the fastest and most consistent warm cold starts (~58-67s). Weights are cached on their infra.
- **Modal** is close behind (~56-95s) and gets faster with each run — BDN (Baseten Distribution Network) bakes weights into the container image.
- **Baseten** is usually ~100s but had a 317s outlier on warm run 3. Uses their own weight distribution system.
- **RunPod** is slow (~640s) because it downloads model weights from HuggingFace on every cold start. RunPod's "Cached Models" feature is console-only (no API), and network volumes pin workers to a single datacenter — reducing GPU availability and increasing queue times. Cold start optimization for RunPod is not yet supported. RunPod's health endpoint also returns 200 before vLLM is ready, so the "Health Ready" column is misleading (real latency shows up in "1st Inference" for RunPod).
- **Cloud Run** never scaled to zero within the 300s `--idle-wait` window, so warm runs were skipped. Would need `--idle-wait 600`+ or a longer `scaledown_window`. Cloud Run's minimum instance keepalive is longer than other providers.

## Excluded Providers

**Azure** is excluded from benchmarking — Container Apps environments take 30+ min to create/delete, making cold start benchmarking impractical.

---

# Load Test Benchmarking

Measure throughput, latency (TTFT, ITL), and cost savings of the hybrid router under sustained traffic.

## Setup

The load test uses [aiperf](https://github.com/ai-dynamo/aiperf) (by NVIDIA) for production-grade LLM metrics. Install it separately — it conflicts with truss (baseten) on `rich` versions, so it can't be a pip extra:

```bash
# After uv sync, install aiperf into the venv
uv pip install aiperf

# Verify
.venv/bin/aiperf --version
```

> **Note:** `uv sync` wipes pip-installed packages. Re-run `uv pip install aiperf` after each sync.
> **Note:** Use `.venv/bin/python -m tuna` (not `uv run`) when aiperf is installed, or `export PATH=".venv/bin:$PATH"`.

If aiperf is not installed, the load test falls back to the built-in engine (basic latency metrics, no TTFT/ITL).

## Quick Start

```bash
# Deploy a hybrid setup first
tuna deploy --model Qwen/Qwen3-0.6B --gpu L4 --serverless-provider cerebrium

# Run load test with aiperf (30 concurrent, Poisson arrivals, streaming)
tuna benchmark load-test \
  --endpoint-url http://router:8080 \
  --duration 30m \
  --concurrency 30 \
  --model Qwen/Qwen3-0.6B \
  --api-key <key>
```

## Engines

| Engine | Metrics | Install |
|--------|---------|---------|
| `aiperf` (default if installed) | TTFT, ITL, token throughput, p50/p90/p99 | `uv pip install aiperf` |
| `tuna` (built-in fallback) | End-to-end latency, p50/p95/p99, day-cycle profiles | Included |

Force a specific engine with `--engine aiperf` or `--engine tuna`.

## Flags

```
--endpoint-url      Router URL (required)
--duration          Duration: 8h, 30m, 90s (default: 10m)
--model             Model name (default: Qwen/Qwen3-0.6B)
--output            table | json | csv (default: table)
--api-key           Router auth key

# aiperf engine
--engine aiperf     Use aiperf (default if installed)
--concurrency       Peak concurrent requests (default: 30)
--request-rate      Target req/s (mutually exclusive with --concurrency)
--request-rate-mode poisson | constant | gamma (default: poisson)
--streaming         Enable streaming for TTFT (default: true)
--no-streaming      Disable streaming
--isl               Mean input tokens (default: 550)
--osl               Mean output tokens (default: 150)
--ui-mode           dashboard | simple | headless (default: simple)
--aiperf-args       Extra args passed to aiperf (quoted string)

# tuna engine (built-in)
--engine tuna       Use built-in load generator
--max-users         Peak concurrent users (default: 5)
--profile           day-cycle | flat | spike | ramp (default: day-cycle)
--log-file          JSONL per-request log
```

## Examples

```bash
# High-throughput benchmark (target 10 RPS with Poisson arrivals)
tuna benchmark load-test \
  --endpoint-url http://router:8080 \
  --duration 2h \
  --request-rate 10 \
  --model Qwen/Qwen3-0.6B

# Use aiperf's live dashboard
tuna benchmark load-test \
  --endpoint-url http://router:8080 \
  --duration 1h \
  --concurrency 20 \
  --ui-mode dashboard

# Built-in engine with day-cycle profile (scale-to-zero testing)
tuna benchmark load-test \
  --engine tuna \
  --endpoint-url http://router:8080 \
  --duration 8h \
  --max-users 5 \
  --profile day-cycle

# Pass extra aiperf flags
tuna benchmark load-test \
  --endpoint-url http://router:8080 \
  --concurrency 50 \
  --aiperf-args "--warmup-duration 30 --num-profile-runs 3"
```

## Cost Tracking

The load test automatically tracks cost alongside performance by polling the router's `/router/health` endpoint every 30 seconds while the benchmark runs. This provides:

- **Spot vs Serverless split** — what % of requests went to each backend
- **GPU-seconds** — actual compute time on each backend
- **Hybrid cost** — what the deployment actually cost (spot uptime + serverless compute)
- **Counterfactual comparisons:**
  - "All-Serverless" — what it would have cost if every request went to serverless
  - "All-On-Demand" — what it would have cost with a dedicated GPU
- **Savings** — how much the hybrid approach saved vs each alternative

Cost is computed using real provider prices from the tuna catalog (`tuna show-gpus --spot`).

### When does spot save money vs serverless?

Spot GPUs bill for **uptime** (idle or busy). Serverless bills for **compute seconds** only. The breakeven is:

```
GPU utilization >= spot_price / serverless_price
```

For L4: `$0.216 / $0.799 = 27%`. Below 27% utilization, serverless is cheaper. Above 27%, spot wins.

| Concurrent users | Avg GPU util | Spot vs Serverless |
|-----------------|-------------|-------------------|
| 5 (think 7.5s) | ~20% | Serverless cheaper |
| 20 (think 3s) | ~55% | **Spot 50% cheaper** |
| 50 (think 2s) | ~80% | **Spot 70% cheaper** |

Use `--concurrency 30+` for realistic startup traffic to see meaningful cost savings.

### Standalone cost dashboard

For a running deployment (no benchmark needed):

```bash
tuna cost --service-name my-deployment
```
