# Tuna

Spot GPUs are 3-5x cheaper than on-demand, but they take minutes to start and can be interrupted at any time. Serverless GPUs start in seconds and never get interrupted, but you pay a premium for that convenience. What if you didn't have to choose?

Tuna is a smart router that combines both behind a single OpenAI-compatible endpoint. It serves requests from serverless while spot instances boot up, shifts traffic to spot once ready, and falls back to serverless if spot gets preempted. You only pay for the compute you actually use — spot rates for steady traffic, serverless only during cold starts and failover.

<div align="center">
<table>
<tr>
<td align="center" colspan="3"><b>Serverless</b></td>
<td align="center" colspan="1"><b>Spot</b></td>
</tr>
<tr>
<td align="center"><img src="assets/modal-logo-icon.png" height="30"><br>Modal</td>
<td align="center"><img src="assets/runpod-logo-black.svg" height="30"><br>RunPod</td>
<td align="center"><img src="assets/google-cloud-run-logo-png_seeklogo-354677.png" height="30"><br>Cloud Run</td>
<td align="center"><img src="assets/Amazon_Web_Services_Logo.svg.png" height="30"><br>AWS via SkyPilot</td>
</tr>
</table>
</div>

## Quick Start

**1. Install**

```bash
pip install tandemn-tuna
```

**2. Set up your provider credentials**

```bash
# At least one serverless provider + AWS for spot
modal token new        # if using Modal
runpodctl config       # if using RunPod
gcloud auth login      # if using Cloud Run

aws configure          # for spot instances
sky check              # verify SkyPilot can see your cloud accounts
```

**3. Deploy a model**

```bash
tuna deploy --model Qwen/Qwen3-0.6B --gpu L4
```

Tuna auto-selects the cheapest serverless provider for your GPU, launches spot instances on AWS, and gives you a single endpoint. The router handles everything — serverless covers traffic immediately while spot boots up in the background.

**4. Send requests** (OpenAI-compatible)

```bash
curl http://<router-ip>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello"}]}'
```

**5. Browse GPU pricing**

```bash
tuna show-gpus              # compare serverless pricing across providers
tuna show-gpus --spot       # include AWS spot prices
tuna show-gpus --gpu H100   # detailed pricing for a specific GPU
```

**6. Monitor costs in real time**

```bash
tuna cost --service-name <name>
```

Shows actual spend, routing split (% spot vs serverless), and savings compared to all-serverless or all-on-demand. Tracks cost per component (serverless, spot, router) so you can see exactly where your money goes.

## Architecture

```
                ┌──────────────────────┐
                │    User Traffic      │
                │ (OpenAI-compatible)  │
                └──────────┬───────────┘
                           │
                  ┌────────▼────────┐
                  │  Smart Router   │
                  │  (meta_lb)      │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
     ┌────────▼─────────┐    ┌─────────▼─────────┐
     │ Serverless        │    │ Spot GPUs          │
     │ Modal / RunPod /  │    │ AWS via SkyPilot   │
     │ Cloud Run         │    │                    │
     │                   │    │ • 3-5x cheaper     │
     │ • Fast cold start │    │ • Slower cold start│
     │ • Per-second bill │    │ • Auto-failover    │
     │ • Always ready    │    │ • Scale to zero    │
     └───────────────────┘    └────────────────────┘
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
| `--use-different-vm-for-lb` | off | Launch router on a separate VM instead of colocating on controller |
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

**Precedence**: defaults <- YAML file <- CLI flags. For example, `--concurrency 64` overrides `serverless.concurrency` from the YAML. `--no-scale-to-zero` forces `spot.min_replicas` to at least 1 and sets `serverless.scaledown_window` to 300s.

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

### High latency

Check which backend is active:

```bash
curl http://<router-ip>:8080/router/health
```

If the router is using serverless, spot instances are still booting — wait for them to come up. If spot is active but slow, check the EC2 instance type and network configuration.

## Prerequisites

- Python 3.11+
- Provider accounts as needed: Modal, RunPod, and/or Google Cloud
- AWS account (for spot instances via SkyPilot)
- SkyPilot CLI (`sky check` to verify)

## License

MIT

This project depends on [SkyPilot](https://github.com/skypilot-org/skypilot) (Apache License 2.0).
