# Cold Start Benchmark Script

## Context
tandemn-tuna just went public. Concrete cold start numbers in the README are more compelling for adoption than adding providers. This script measures time-to-first-token for the hybrid architecture to validate the core value proposition: serverless covers traffic instantly while spot boots up cheaply.

New providers are coming soon — the benchmark needs to support running across multiple providers and GPUs in a single invocation, producing comparable results.

## What it measures

Two scenarios:
1. **Fresh cold start**: `tuna deploy` → router ready → first token received
2. **Warm-then-cold start**: send requests (warm up) → wait for scale-down → send request → first token (faster because provider caches warm)

Three targets per scenario:
- **Router** (the real user path — auto-routes to serverless/spot)
- **Serverless direct** (bypass router, measure true streaming TTFT)
- **Spot direct** (if ready — usually not during fresh cold start)

## Important finding: router does not stream

The router proxy in `meta_lb.py:435` uses `SESSION.request()` without `stream=True` — it buffers the full response before returning. So through the router, TTFT ≈ total response time. Direct serverless endpoint gives true streaming TTFT. The benchmark labels these differently.

## File structure

```
bench/
  cold_start.py     # Main benchmark script (~400 lines)
  compare.py        # Reads results dir, prints comparison table (~80 lines, deferred)
  matrix.yaml       # Example benchmark matrix
  results/          # Timestamped JSON output files (gitignored)
```

No new dependencies — only uses `requests` + `pyyaml` (both already core deps).

## Implementation

### CLI interface

Three modes of operation:

```bash
# 1. Single run — fresh deploy, benchmark, teardown
python bench/cold_start.py --mode fresh --model Qwen/Qwen3-0.6B --gpu T4 --provider modal

# 2. Single run — against existing deployment
python bench/cold_start.py --mode existing \
  --router-url http://x.x.x.x:8080 \
  --serverless-url https://xxx.modal.run

# 3. Matrix run — all provider×GPU combinations from YAML
python bench/cold_start.py --matrix bench/matrix.yaml
```

**Arguments:**
- `--mode`: `fresh` (deploy+measure+teardown) or `existing` (use live deployment)
- `--matrix`: path to YAML matrix file (overrides `--mode`, `--gpu`, `--provider`)
- `--scenario`: `fresh-cold`, `warm-cold`, or `both` (default)
- `--model`: default `Qwen/Qwen3-0.6B`
- `--gpu`: default `T4`
- `--max-model-len`: default `512` (minimizes allocation time)
- `--provider`: default `modal` (the serverless provider)
- `--router-url`: required for `existing` mode
- `--serverless-url`: optional, enables direct-endpoint comparison
- `--idle-wait`: seconds to wait for scale-down (default `120`)
- `--repeat`: repetitions for warm-cold (default `3`)
- `--output-dir`: directory for JSON results (default `bench/results/`)
- `--no-teardown`: skip cleanup for debugging
- `-v`: verbose logging

### Matrix YAML format

```yaml
# bench/matrix.yaml
model: Qwen/Qwen3-0.6B
max_model_len: 512
scenarios: [fresh-cold, warm-cold]
repeat: 3

matrix:
  - provider: modal
    gpus: [T4, L4, L40S]
  - provider: runpod
    gpus: [T4, L4]
  - provider: cloudrun
    gpus: [L4]
```

The script iterates over each `(provider, gpu)` combination sequentially: deploy → benchmark → teardown → next. Each combination writes a separate result file to `--output-dir`.

### Core functions

**`measure_ttft(url, model, stream=True, timeout=300) -> (ttft_s, total_s)`**
- Sends `POST /v1/chat/completions` with `{"model": model, "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8, "stream": true}`
- Parses SSE stream, records time to first `data:` chunk containing `delta.content`
- `max_tokens=8` isolates cold start from generation time
- Provider-agnostic: works with any backend serving the OpenAI-compatible API

**`wait_for_health(url, timeout=300, interval=5) -> float`**
- Polls `GET {url}/router/health` until 200
- Returns seconds waited

**`wait_for_scaledown(serverless_url, timeout=180, interval=10) -> float`**
- Polls `GET {serverless_url}/health` until it fails (container scaled down)
- Returns seconds waited

**`poll_spot_readiness(router_url, timeout=600) -> float | None`**
- Polls `/router/health`, checks `skyserve_ready` field
- Returns seconds until spot ready, or None if timeout

**`run_single_benchmark(provider, gpu, model, ...) -> list[dict]`**
- Orchestrates one full (provider, gpu) combination
- Handles deploy → measure fresh-cold → measure warm-cold → teardown
- Returns list of result dicts (one per scenario×target)

**`run_matrix(matrix_path) -> list[dict]`**
- Parses YAML, loops over combinations calling `run_single_benchmark()`
- Writes per-combination results to `output_dir/`
- Prints progress: `[2/7] runpod × L4 — fresh cold start...`

### Fresh cold start flow

1. Build `DeployRequest` using `tuna.models.DeployRequest` + `tuna.scaling.default_scaling_policy()`
   - Sets `public=True` (no auth for benchmark), `max_model_len=512`, `cold_start_mode="fast_boot"`
2. Call `tuna.orchestrator.launch_hybrid(request)` — record deploy time
3. Extract `router_url`, `serverless_url` from `HybridDeployment`
4. `wait_for_health(router_url)` — record router ready time
5. `measure_ttft(router_url)` — router TTFT (non-streaming, labeled as such)
6. `measure_ttft(serverless_url, stream=True)` — direct serverless streaming TTFT
7. Background: `poll_spot_readiness(router_url)` — records when spot comes online
8. Cleanup via `tuna.orchestrator.destroy_hybrid()` in `try/finally` + `atexit`

### Warm-then-cold start flow

1. Send 3 warmup requests (1s spacing) to ensure containers are hot
2. Wait for scale-down: `wait_for_scaledown(serverless_url)` or sleep `--idle-wait`
3. `measure_ttft(router_url)` — cold request through router
4. `measure_ttft(serverless_url, stream=True)` — cold request direct (if URL provided)
5. Repeat `--repeat` times, report min/median/max

### Cleanup safety

```python
_active_deployment = None  # module-level

def _cleanup():
    if _active_deployment:
        destroy_hybrid(_active_deployment.service_name)

atexit.register(_cleanup)
signal.signal(signal.SIGINT, lambda *_: sys.exit(1))  # triggers atexit
```

### Output format

Each result row is self-describing with `provider` and `gpu` fields (not just in config), so matrix results can be compared directly.

**Per-combination file** → `bench/results/2026-02-09T12-00-00_modal_T4.json`:

```json
{
  "benchmark": "cold_start",
  "timestamp": "2026-02-09T12:00:00Z",
  "config": {
    "model": "Qwen/Qwen3-0.6B",
    "gpu": "T4",
    "provider": "modal",
    "max_model_len": 512
  },
  "results": [
    {
      "scenario": "fresh_cold_start",
      "target": "router",
      "provider": "modal",
      "gpu": "T4",
      "deploy_time_s": 45.2,
      "ttft_s": 8.3,
      "deploy_to_first_token_s": 53.5,
      "streaming": false
    },
    {
      "scenario": "fresh_cold_start",
      "target": "serverless_direct",
      "provider": "modal",
      "gpu": "T4",
      "ttft_s": 6.1,
      "streaming": true
    },
    {
      "scenario": "warm_cold_start",
      "target": "router",
      "provider": "modal",
      "gpu": "T4",
      "idle_wait_s": 90,
      "ttft_s": [5.2, 5.8, 6.1],
      "streaming": false
    },
    {
      "scenario": "warm_cold_start",
      "target": "serverless_direct",
      "provider": "modal",
      "gpu": "T4",
      "ttft_s": [3.8, 4.1, 4.5],
      "streaming": true
    }
  ]
}
```

**Human-readable summary** (printed to stderr during run):

```
=== Cold Start Benchmark ===
Model: Qwen/Qwen3-0.6B

[1/5] modal × T4
  FRESH: deploy 45.2s | router TTFT 8.3s | serverless TTFT 6.1s | spot ready 187.4s
  WARM:  router TTFT 5.2/5.8/6.1s | serverless TTFT 3.8/4.1/4.5s

[2/5] modal × L4
  FRESH: deploy 42.1s | router TTFT 7.1s | serverless TTFT 5.3s | spot ready 165.2s
  WARM:  router TTFT 4.8/5.1/5.5s | serverless TTFT 3.2/3.6/3.9s

...

=== Summary: Fresh Cold Start TTFT (serverless direct, streaming) ===
              T4      L4      L40S
modal         6.1s    5.3s    4.2s
runpod        7.3s    6.9s    -
cloudrun      -       8.1s    -
```

### Adding a new provider

When a new provider is added to tuna:
1. Add it to `PROVIDER_MODULES` in `registry.py` (already the pattern)
2. Add a row in `matrix.yaml` with the GPUs it supports
3. Run `python bench/cold_start.py --matrix bench/matrix.yaml`
4. Results appear in `bench/results/` — no code changes to the benchmark needed

The benchmark is provider-agnostic by design — it only talks to URLs via the OpenAI-compatible API. As long as the new provider serves vLLM (which they all do), it works.

## Key files referenced

- `tuna/orchestrator.py` — `launch_hybrid()`, `destroy_hybrid()` for deploy/teardown
- `tuna/router/meta_lb.py` — `/router/health` response schema, proxy behavior (non-streaming)
- `tuna/models.py` — `DeployRequest`, `HybridDeployment` dataclasses
- `tuna/scaling.py` — `default_scaling_policy()`, `scaledown_window` parameter
- `tuna/state.py` — `save_deployment()`, `update_deployment_status()`

## Verification

1. `python bench/cold_start.py --mode fresh --model Qwen/Qwen3-0.6B --gpu T4 --provider modal` — deploys, measures, tears down cleanly
2. `python bench/cold_start.py --mode existing --router-url <URL> --scenario warm-cold` — works against a live deployment
3. `python bench/cold_start.py --matrix bench/matrix.yaml` — runs full matrix, writes per-combination JSON files to `bench/results/`
4. JSON output is valid, each row has `provider` + `gpu` fields
5. Ctrl+C during fresh/matrix mode triggers cleanup (no orphan deployments)
6. Adding a new provider to `matrix.yaml` requires zero code changes to the benchmark
