# Tandemn: Hybrid GPU Inference Orchestrator

Deploy LLM inference across **serverless** (Modal) + **spot instances** (AWS via SkyServe) with automatic load balancing. A single endpoint routes traffic efficiently, paying serverless rates during traffic spikes and spot rates during steady periods.

## When to Use This

- **Bursty traffic patterns**: Traffic that spikes unpredictably. Serverless scales instantly (no cold-start penalty), spot instances provide cheap sustained capacity.
- **Cost-sensitive inference**: Want 40-60% cost reduction vs pure serverless or pure on-demand.
- **Variable request rates**: 5-500 QPS throughout the day. Hybrid load balancing absorbs the variance.
- **Model serving with SLOs**: Need high availability even when spot instances are interrupted or scaling up.

**Not ideal for**:
- Constant steady-state traffic (pure spot is cheaper)
- Latency-critical endpoints (<100ms p99 requirement)
- Very low traffic (<5 QPS average; serverless fixed costs dominate)

## Architecture

```
┌─────────────────────────────────────────┐
│         User Traffic                    │
│     (any HTTP requests)                 │
└────────────────────┬────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   Router (c6i.large)  │◄─── SkyPilot-managed VM
         │   Load Balancer       │     on AWS
         └───────────┬───────────┘
                     │
        ┌────────────┴─────────────┐
        │                          │
   ┌────▼────────┐          ┌─────▼──────┐
   │Modal Serverless        │ SkyServe   │
   │  (vLLM 1-3 containers) │ (GPU fleet)│
   │  •Fast cold start      │ •Cheaper   │
   │  •Per-second billing   │ •Slower    │
   │  •Always ready         │  cold start│
   └────────────────┘       └────────────┘
        ~65s cold start         ~5min cold start
        $1.95/hr per L40S       $0.65/hr per L40S
```

The router automatically:
- Routes traffic to Modal while SkyServe is starting up
- Shifts traffic to SkyServe once it's ready (cheaper)
- Falls back to Modal if SkyServe has issues or high latency
- Scales Modal down (to 0) when SkyServe is serving

## Installation

### Requirements

- Python 3.10+
- AWS account (for SkyServe spot instances)
- Modal account + API key
- SkyPilot CLI

### Setup

1. **Clone and install**:
   ```bash
   git clone https://github.com/Tandemn-Labs/hybrid-router.git
   cd hybrid-router
   pip install -e .
   ```

2. **Configure credentials**:
   ```bash
   # Modal API key
   modal token new

   # AWS credentials (for SkyServe spot instances)
   aws configure

   # SkyPilot (manages spot instances)
   sky check
   ```

3. **Verify installation**:
   ```bash
   tandemn --help
   ```

## Usage

### Basic Deployment

Deploy a model with one command:

```bash
tandemn deploy \
  --model Qwen/Qwen3-0.6B \
  --gpu L40S \
  --serverless-provider modal
```

Output:
```
Deploying Qwen/Qwen3-0.6B on L40S
Service name: tandemn-abc12345
Serverless provider: modal
Spot cloud: aws

============================================================
DEPLOYMENT RESULT
============================================================
  Router:     http://x.x.x.x:8080
  Serverless: https://xxx.modal.run
  Spot:       launching in background...

All traffic -> http://x.x.x.x:8080
============================================================
```

### Advanced Options

```bash
tandemn deploy \
  --model meta-llama/Llama-2-7b-hf \
  --gpu A100-80GB \
  --serverless-provider modal \
  --max-model-len 8192 \
  --tp-size 1 \
  --concurrency 64 \
  --region us-east-1
```

**Options**:
- `--model`: HuggingFace model ID or local path (required)
- `--gpu`: GPU type, e.g. `L40S`, `A100-80GB`, `H100` (required)
- `--gpu-count`: Number of GPUs (default: 1)
- `--serverless-provider`: `modal` (default; more providers planned)
- `--max-model-len`: KV cache size (default: 4096)
- `--tp-size`: Tensor parallelism (default: 1)
- `--concurrency`: Request queue depth (default: 32)
- `--region`: AWS region for spot instances
- `--spots-cloud`: Cloud provider for spot GPUs (default: `aws`)
- `--cold-start-mode`: `fast_boot` (default) or `no_fast_boot`
- `--no-scale-to-zero`: Keep spot replicas warm when idle (default: scale to zero)
- `--service-name`: Custom service name (default: auto-generated)

### Testing the Endpoint

```bash
# Health check (use the router IP from deployment output)
curl http://<router-ip>:8080/health

# Send an inference request (vLLM OpenAI API format)
curl -X POST http://<router-ip>:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Once upon a time",
    "max_tokens": 50
  }'

# Streaming
curl -N -X POST http://<router-ip>:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "The future of AI",
    "max_tokens": 100,
    "stream": true
  }'
```

### Monitor Routing

The router exposes a health endpoint to check current status:

```bash
curl http://<router-ip>:8080/router/health

# Response:
# {
#   "skyserve_ready": true,
#   "last_probe_ts": 1706000000.0,
#   "last_probe_err": null,
#   "serverless_base_url": "https://xxx.modal.run",
#   "skyserve_base_url": "http://x.x.x.x:30001",
#   "route_stats": { ... }
# }
```

## Examples

### Example 1: Bursty Traffic (Business Hours + Off-Peak)

Model: `meta-llama/Llama-2-7b-hf`
- 9am–5pm: 50 QPS (business hours)
- 5pm–9am: 2 QPS (off-peak)

```bash
tandemn deploy \
  --model meta-llama/Llama-2-7b-hf \
  --gpu L40S \
  --serverless-provider modal
```

**Expected cost/month**: ~$1,700
- Modal absorbs 50 QPS during ramp-up (first 5 min daily)
- SkyServe handles steady 50 QPS for 8 hours
- Both scale to zero during off-peak

**Pure Modal** (same load): ~$3,200/month
**Pure Spot** (same load): ~$2,500/month (but risky: 5-min cold starts on every spike, no fallback)

### Example 2: Burst Processing (Late-Night Batch)

Model: `Qwen/Qwen3-72B` (requires A100-80GB)
- Most of the day: 1 QPS
- 11pm–12am: 100 QPS (batch processing window)

```bash
tandemn deploy \
  --model Qwen/Qwen3-72B \
  --gpu A100-80GB \
  --serverless-provider modal \
  --max-model-len 2048
```

**Expected cost/month**: ~$800
- SkyServe handles the 11pm batch, then scales to 0
- Modal covers daytime trickle (1 QPS per-second billing = cheap)

### Example 3: Production with High Availability

Model: `mistralai/Mistral-7B-Instruct` (small, fast)
- Constant 20 QPS minimum
- Spikes to 100 QPS unpredictably
- Must maintain <200ms p99 latency

```bash
tandemn deploy \
  --model mistralai/Mistral-7B-Instruct \
  --gpu L40S \
  --serverless-provider modal \
  --no-scale-to-zero \
  --concurrency 128
```

**Why `--no-scale-to-zero`**:
- SkyServe keeps 1 replica warm continuously (faster restart on spike)
- Modal acts as a buffer for overflow
- Router prioritizes SkyServe (cheaper), falls back to Modal if needed

**Expected cost/month**: ~$2,100
- Pure on-demand: ~$4,500/month
- Cost savings: ~53%

## Troubleshooting

### Endpoint not responding

```bash
# Check if both backends are alive
curl http://<router-ip>:8080/router/health

# Check if Modal deployment succeeded
modal app list

# Check SkyServe status
sky status --refresh
```

### High latency

Check which backend is serving:
```bash
# Check which backend the router is using
curl http://<router-ip>:8080/router/health
```

If Modal is responding (should be ~65ms overhead), wait for SkyServe to finish booting. If SkyServe is responding but slow, check EC2 instance type and network.

### Cost higher than expected

```bash
# Check how much time Modal/SkyServe are active
sky logs <cluster-name> --all
```

If Modal is always on: ensure `--no-scale-to-zero` is not set, or increase request volume.
If SkyServe is always on: don't use `--no-scale-to-zero` for batch workloads.

## API Reference

### Inference Endpoints (proxied to backends)

All vLLM endpoints are proxied through the router to the active backend:

- `POST /v1/completions` — Text generation
- `POST /v1/chat/completions` — Chat API
- `GET /health` — Health check (proxied to active backend)

### Router Endpoints

- `GET /router/health` — Router status, backend URLs, and routing stats
- `POST /router/config` — Update backend URLs (used internally by orchestrator)

## Project Status

**MVP** (current):
- ✅ Modal serverless backend
- ✅ SkyServe spot backend
- ✅ Intelligent router with latency-based fallback
- ✅ Single-command deployment

**Roadmap**:
- Cerebrium serverless backend
- Google Cloud Run backend
- Automatic "cheapest provider" selector
- Cost monitoring dashboard
- Multi-model endpoints

## Contributing

Issues and PRs welcome. See [ARCHITECTURE.md](ARCHITECTURE.md) for design details.

## License

MIT
