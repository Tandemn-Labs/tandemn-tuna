# Tandemn

Hybrid serverless + spot GPU inference. One endpoint, automatic routing between serverless backends (Modal, RunPod, Cloud Run) and spot instances (AWS via SkyPilot). Automatically selects the cheapest serverless provider for your GPU.

## Who This Is For

Tandemn is for teams running LLM inference that want lower GPU costs without giving up availability.

**How it saves costs**: Serverless GPUs are always ready but expensive per-hour. Spot instances are cheap but take minutes to start and can be interrupted. Tandemn runs both behind a single endpoint — the router serves requests from serverless while spot instances boot, then shifts traffic to spot once ready. When spot is handling traffic, serverless scales to zero and you stop paying for it. If spot gets interrupted, the router falls back to serverless automatically.

The result: you pay spot rates for steady traffic and only pay serverless rates during cold starts and failover. `tandemn cost` tracks actual spend vs. what you'd pay on pure serverless or pure on-demand, in real time.

This is useful when you have:
- **Bursty or variable traffic** — serverless absorbs spikes instantly, spot handles the baseline
- **Cost-sensitive workloads** — spot GPUs cost a fraction of on-demand; serverless only runs when needed
- **Availability requirements** — spot interruptions are handled automatically, no manual failover

## Quick Start

```bash
pip install tandemn
modal token new && aws configure && sky check
tandemn deploy --model Qwen/Qwen3-0.6B --gpu L4
curl http://<router-ip>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello"}]}'
```

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
   ┌────▼──────────────┐    ┌─────▼──────┐
   │ Serverless         │    │ SkyServe   │
   │ (Modal / RunPod /  │    │ (GPU fleet)│
   │  Cloud Run)        │    │ •Spot      │
   │ •Fast cold start   │    │  pricing   │
   │ •Per-second billing│    │ •Slower    │
   │ •Always ready      │    │  cold start│
   └────────────────────┘    └────────────┘
```

The router:
- Routes to serverless while spot instances are starting up
- Shifts traffic to spot once ready (cheaper)
- Falls back to serverless if spot has issues or high latency
- Scales serverless down to zero when spot is serving

## CLI Reference

| Command | Description |
|---------|-------------|
| `deploy` | Deploy a model across serverless + spot |
| `destroy` | Tear down a deployment |
| `status` | Check deployment status |
| `cost` | Show cost dashboard for a deployment |
| `list` | List all deployments |
| `show-gpus` | GPU pricing across providers |
| `check` | Validate provider setup |

### `deploy` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace model ID (e.g. `Qwen/Qwen3-0.6B`) |
| `--gpu` | *(required)* | GPU type (e.g. `L4`, `L40S`, `A100`, `H100`) |
| `--gpu-count` | `1` | Number of GPUs |
| `--serverless-provider` | auto (cheapest for GPU) | `modal`, `runpod`, or `cloudrun` |
| `--spots-cloud` | `aws` | Cloud provider for spot GPUs |
| `--region` | — | Cloud region for spot instances |
| `--tp-size` | `1` | Tensor parallelism degree |
| `--max-model-len` | `4096` | Maximum sequence length (context window) |
| `--concurrency` | — | Override serverless concurrency limit |
| `--workers-max` | — | Max serverless workers (RunPod only) |
| `--cold-start-mode` | `fast_boot` | `fast_boot` or `no_fast_boot` |
| `--no-scale-to-zero` | off | Keep minimum 1 spot replica running |
| `--scaling-policy` | — | Path to scaling YAML (see below) |
| `--service-name` | auto | Custom service name |
| `--public` | off | Make service publicly accessible (no auth) |
| `--gcp-project` | — | Google Cloud project ID |
| `--gcp-region` | — | Google Cloud region (e.g. `us-central1`) |

Use `-v` / `--verbose` with any command for debug logging.

## Scaling Policy

All autoscaling parameters can be configured via a YAML file passed with `--scaling-policy`. If omitted, sane defaults apply.

```yaml
spot:
  min_replicas: 0        # 0 = scale to zero (default)
  max_replicas: 8
  target_qps: 10         # per-replica QPS target
  upscale_delay: 5       # seconds before adding replicas
  downscale_delay: 300   # seconds before removing replicas

serverless:
  concurrency: 32        # max concurrent requests per container
  scaledown_window: 60   # seconds idle before scaling down
  timeout: 600           # request timeout in seconds
```

**Precedence**: defaults ← YAML file ← CLI flags. For example, `--concurrency 64` overrides `serverless.concurrency` from the YAML. `--no-scale-to-zero` forces `spot.min_replicas` to at least 1 and sets `serverless.scaledown_window` to 300s.

Unknown keys in the YAML will error immediately (catches typos).

## Troubleshooting

### Endpoint not responding

```bash
# Check router and backend status
curl http://<router-ip>:8080/router/health

# Check if Modal deployment succeeded
modal app list

# Check SkyServe status
sky status --refresh
```

### Checking costs

Track spend in real time — shows actual cost, routing split, and savings vs. pure serverless or on-demand:

```bash
tandemn cost --service-name <name>
```

### High latency

Check which backend is active:

```bash
curl http://<router-ip>:8080/router/health
```

If the router is using serverless, spot instances are still booting — wait for them to come up. If spot is active but slow, check the EC2 instance type and network configuration.

## Prerequisites

- Python 3.10+
- Provider accounts as needed: Modal, RunPod, and/or Google Cloud
- AWS account (for spot instances via SkyPilot)
- SkyPilot CLI (`sky check` to verify)

## License

MIT

This project depends on [SkyPilot](https://github.com/skypilot-org/skypilot) (Apache License 2.0).
