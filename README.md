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
   git clone https://github.com/your-org/serverless-spot.git
   cd serverless-spot
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
Deploying hybrid inference endpoint...

📡 Serverless (Modal):     https://xxx.modal.run
   Status: Ready in ~30s

📍 Spot (SkyServe):        http://x.x.x.x:30001
   Status: Starting (~5 min, cheaper once ready)

🔀 Router:                 http://localhost:8080
   Status: Ready
   Currently routing to: Modal (spot warming up)

✅ All traffic → http://localhost:8080
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
  --region us-east-1 \
  --scale-to-zero true
```

**Options**:
- `--model`: HuggingFace model ID or local path
- `--gpu`: `L40S` (default), `A100-80GB`, `H100`
- `--serverless-provider`: `modal`, `cerebrium`, `runpod` (planned)
- `--max-model-len`: KV cache size (for long contexts)
- `--tp-size`: Tensor parallelism (use 1 for single GPU)
- `--concurrency`: Request queue depth
- `--region`: AWS region for spot instances
- `--scale-to-zero`: Tear down spot when idle (saves money, slower restart)

### Testing the Endpoint

```bash
# Health check
curl http://localhost:8080/health

# Send an inference request (vLLM OpenAI API format)
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Once upon a time",
    "max_tokens": 50
  }'

# Streaming
curl -N -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "The future of AI",
    "max_tokens": 100,
    "stream": true
  }'
```

### Monitor Routing

The router exposes a config endpoint to check current status:

```bash
curl http://localhost:8080/router/config

# Response:
# {
#   "serverless_url": "https://xxx.modal.run",
#   "spot_url": "http://x.x.x.x:30001",
#   "current_primary": "spot",
#   "health": {
#     "serverless": "healthy",
#     "spot": "healthy"
#   }
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
  --serverless-provider modal \
  --scale-to-zero true
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
  --scale-to-zero true \
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
  --scale-to-zero false \
  --concurrency 128
```

**Why scale-to-zero=false**:
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
curl http://localhost:8080/router/config

# Check router logs
docker logs <router-container-id>

# Check if Modal deployment succeeded
modal app list

# Check SkyServe status
sky status --refresh
```

### High latency

Check which backend is serving:
```bash
# Add a request header to trace routing
curl -v -X POST http://localhost:8080/v1/completions \
  -H "X-Debug: true" \
  -d '...'
```

If Modal is responding (should be ~65ms overhead), wait for SkyServe to finish booting. If SkyServe is responding but slow, check EC2 instance type and network.

### Cost higher than expected

```bash
# Check how much time Modal/SkyServe are active
sky logs <cluster-name> --all
```

If Modal is always on: set `--scale-to-zero true` or increase request volume.
If SkyServe is always on: set `--scale-to-zero true` if batch workloads.

## API Reference

### Inference Endpoints

All endpoints support vLLM's OpenAI-compatible API:

- `POST /v1/completions` — Text generation
- `POST /v1/chat/completions` — Chat API
- `GET /health` — Health check
- `GET /router/config` — Routing config

### Management Endpoints

- `POST /router/config` — Update backend URLs (internal)

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
