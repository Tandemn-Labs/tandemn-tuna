
# Tuna
<div align="center">
<img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/tuna3.png" width="500" alt="Tuna">
</div>

Spot GPUs are 3-5x cheaper than on-demand, but they take minutes to start and can be interrupted at any time. Serverless GPUs start in seconds and never get interrupted, but you pay a premium for that convenience. What if you didn't have to choose?

Tuna is a smart router that combines both behind a single OpenAI-compatible endpoint. It serves requests from serverless while spot instances boot up, shifts traffic to spot once ready, and falls back to serverless if spot gets preempted. You only pay for the compute you actually use — spot rates for steady traffic, serverless only during cold starts and failover.

<div align="center">
<table>
<tr>
<td align="center" colspan="4"><b>Serverless</b></td>
<td align="center" colspan="1"><b>Spot</b></td>
</tr>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/modal-logo-icon.png" height="30"><br>Modal</td>
<td align="center"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/runpod-logo-black.svg" height="30"><br>RunPod</td>
<td align="center"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/google-cloud-run-logo-png_seeklogo-354677.png" height="30"><br>Cloud Run</td>
<td align="center"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/azure-container-vm.webp" height="30"><br>Azure Container Apps</td>
<td align="center"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/Amazon_Web_Services_Logo.svg.png" height="30"><br>AWS via SkyPilot</td>
</tr>
</table>
</div>

## Prerequisites

- Python 3.11+
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) — required for spot instances (all deployments use spot)
- At least one serverless provider account: [Modal](https://modal.com/), [RunPod](https://www.runpod.io/), [Google Cloud](https://cloud.google.com/), or [Azure](https://azure.microsoft.com/)
- For gated models (Llama, Mistral, Gemma, etc.): a [HuggingFace token](https://huggingface.co/settings/tokens) with access to the model

> **Note:** Tuna always deploys both a serverless backend and a spot backend. AWS credentials are required even if your serverless provider is Modal or RunPod, because spot instances run on AWS via [SkyPilot](https://github.com/skypilot-org/skypilot).

## Quick Start

**1. Install**

```bash
pip install tandemn-tuna[modal] --pre     # Modal as serverless provider
pip install tandemn-tuna[cloudrun] --pre  # Cloud Run as serverless provider
pip install tandemn-tuna[azure] --pre     # Azure Container Apps as serverless provider
pip install tandemn-tuna --pre            # RunPod (no extra deps needed)
pip install tandemn-tuna[all] --pre       # everything
```

> This project is under active development and experimental. For the latest version, install from source:
> ```bash
> git clone https://github.com/Tandemn-Labs/tandemn-tuna.git
> cd tandemn-tuna
> pip install -e ".[all]"
> ```

**2. Set up AWS (required for all deployments)**

```bash
aws configure          # set up AWS credentials
sky check              # verify SkyPilot can see your AWS account
```

**3. Set up your serverless provider (pick one)**

<details>
<summary><b>Modal</b></summary>

```bash
modal token new
```

</details>

<details>
<summary><b>RunPod</b></summary>

```bash
export RUNPOD_API_KEY=<your-key>  # https://www.runpod.io/console/user/settings
```

Add this to your `~/.bashrc` or `~/.zshrc` to persist it.

</details>

<details>
<summary><b>Cloud Run</b></summary>

Requires the [gcloud CLI](https://cloud.google.com/sdk/docs/install).

```bash
gcloud auth login
gcloud auth application-default login    # required for the Python SDK
gcloud config set project <YOUR_PROJECT_ID>
```

You also need billing enabled and the Cloud Run API (`run.googleapis.com`) enabled on your project.

</details>

<details>
<summary><b>Azure Container Apps</b></summary>

**Step 1: Install the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)**

```bash
# Ubuntu/Debian
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# macOS
brew install azure-cli
```

**Step 2: Log in and select your subscription**

```bash
az login                                          # opens browser for login
az account set --subscription <YOUR_SUBSCRIPTION_ID>  # select subscription
```

Don't know your subscription ID? Run `az account list -o table` to see all subscriptions.

**Step 3: Register required resource providers**

New Azure subscriptions need these services enabled (one-time step):

```bash
az provider register --namespace Microsoft.App                  # Container Apps
az provider register --namespace Microsoft.OperationalInsights  # Log Analytics (required by Container Apps)
```

Wait ~1 minute, then verify both show `Registered`:

```bash
az provider show --namespace Microsoft.App --query "registrationState" -o tsv
az provider show --namespace Microsoft.OperationalInsights --query "registrationState" -o tsv
```

**Step 4: Create a resource group**

A resource group is Azure's way of organizing resources into a folder. Create one in a GPU-supported region:

```bash
az group create --name my-tuna-rg --location eastus
export AZURE_RESOURCE_GROUP=my-tuna-rg
```

Add the export to your `~/.bashrc` or `~/.zshrc` to persist it.

**Step 5: Install the Azure SDK**

```bash
pip install tandemn-tuna[azure] --pre
# or from source:
pip install -e ".[azure]"
```

**Step 6: Verify everything**

```bash
tuna check --provider azure
```

All 7 checks should pass: az CLI, login, subscription, resource group, resource provider, SDK, GPU region.

**GPU quota (required before deploying):** Azure gives 0 GPU quota by default on Container Apps. You need to request it:

1. Go to [Azure Portal](https://portal.azure.com) → Subscriptions → your subscription → Usage + quotas
2. Search for "Container Apps GPU"
3. Request quota for `Consumption-GPU-NC8as-T4` (T4) and/or `Consumption-GPU-NC24-A100` (A100) in your region (e.g. `eastus`)
4. Approval takes **1-3 business days**

Available GPUs: T4 (16 GB, $0.26/hr) and A100 80GB ($1.90/hr). No API keys needed — authentication is handled via `az login`.

</details>

**4. (Optional) Set HuggingFace token for gated models**

```bash
export HF_TOKEN=<your-token>  # https://huggingface.co/settings/tokens
```

Required for models like Llama, Mistral, Gemma, and other gated models. Not needed for open models like Qwen.

**5. Validate your setup**

```bash
tuna check --provider modal                          # check Modal credentials
tuna check --provider runpod                         # check RunPod API key
tuna check --provider cloudrun --gcp-project <id> --gcp-region us-central1  # check Cloud Run
tuna check --provider azure                          # check Azure setup
```

**6. Deploy a model**

```bash
tuna deploy --model Qwen/Qwen3-0.6B --gpu L4 --service-name my-first-deploy
```

Tuna auto-selects the cheapest serverless provider for your GPU, launches spot instances on AWS, and gives you a single endpoint. The router handles everything — serverless covers traffic immediately while spot boots up in the background.

**7. Send requests** (OpenAI-compatible)

```bash
curl http://<router-ip>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello"}]}'
```

**8. Monitor and manage**

```bash
tuna status --service-name my-first-deploy    # check deployment status
tuna cost --service-name my-first-deploy      # real-time cost dashboard
tuna list                                     # list all deployments
tuna destroy --service-name my-first-deploy   # tear down everything
```

> **Tip:** If you don't pass `--service-name` during deploy, Tuna auto-generates a name like `tuna-a3f8c21b`. Use `tuna list` to find it.

**9. Browse GPU pricing**

```bash
tuna show-gpus                     # compare serverless pricing across providers
tuna show-gpus --spot              # include AWS spot prices
tuna show-gpus --gpu H100          # detailed pricing for a specific GPU
tuna show-gpus --provider runpod   # filter to one provider
```

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
     │ Cloud Run / Azure │    │                    │
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
| `cost` | Show cost dashboard (requires running deployment) |
| `list` | List all deployments (filter with `--status active\|destroyed\|failed`) |
| `show-gpus` | GPU pricing across providers (filter with `--provider`, `--gpu`, `--spot`) |
| `check` | Validate provider credentials and setup |

### `deploy` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace model ID (e.g. `Qwen/Qwen3-0.6B`) |
| `--gpu` | *(required)* | GPU type (e.g. `L4`, `L40S`, `A100`, `H100`) |
| `--gpu-count` | `1` | Number of GPUs |
| `--serverless-provider` | auto (cheapest for GPU) | `modal`, `runpod`, `cloudrun`, or `azure` |
| `--spots-cloud` | `aws` | Cloud provider for spot GPUs |
| `--region` | — | Cloud region for spot instances |
| `--tp-size` | `1` | Tensor parallelism degree |
| `--max-model-len` | `4096` | Maximum sequence length (context window) |
| `--concurrency` | — | Override serverless concurrency limit |
| `--workers-max` | — | Max serverless workers (RunPod only) |
| `--cold-start-mode` | `fast_boot` | `fast_boot` (uses `--enforce-eager`, faster startup but lower throughput) or `no_fast_boot` |
| `--no-scale-to-zero` | off | Keep minimum 1 spot replica running |
| `--scaling-policy` | — | Path to scaling YAML (see below) |
| `--service-name` | auto-generated | Custom service name (recommended — makes status/destroy easier) |
| `--public` | off | Make service publicly accessible (no auth) |
| `--use-different-vm-for-lb` | off | Launch router on a separate VM instead of colocating on controller |
| `--gcp-project` | — | Google Cloud project ID |
| `--gcp-region` | — | Google Cloud region (e.g. `us-central1`) |
| `--azure-subscription` | — | Azure subscription ID |
| `--azure-resource-group` | — | Azure resource group |
| `--azure-region` | `eastus` | Azure region (e.g. `eastus`, `westeurope`) |

Use `-v` / `--verbose` with any command for debug logging.

## Scaling Policy

All autoscaling parameters can be configured via a YAML file passed with `--scaling-policy`. If omitted, sane defaults apply.

```yaml
spot:
  min_replicas: 0        # 0 = scale to zero (default)
  max_replicas: 5
  target_qps: 10         # per-replica QPS target
  upscale_delay: 5       # seconds before adding replicas
  downscale_delay: 300   # seconds before removing replicas

serverless:
  concurrency: 32        # max concurrent requests per container
  scaledown_window: 60   # seconds idle before scaling down
  timeout: 600           # request timeout in seconds
  workers_min: 0         # min workers (RunPod only)
  workers_max: 1         # max workers (RunPod only)
  scaler_value: 4        # queue delay scaler threshold (RunPod only)
```

**Precedence**: defaults <- YAML file <- CLI flags. For example, `--concurrency 64` overrides `serverless.concurrency` from the YAML. `--no-scale-to-zero` forces `spot.min_replicas` to at least 1 and sets `serverless.scaledown_window` to 300s.

Unknown keys in the YAML will error immediately (catches typos).

## Troubleshooting

### Setup issues

Start with the built-in diagnostic tool:

```bash
tuna check --provider runpod
tuna check --provider modal
tuna check --provider cloudrun --gcp-project <id> --gcp-region us-central1
tuna check --provider azure
```

This validates credentials, API access, project configuration, and GPU region availability.

### Endpoint not responding

```bash
# Check your deployment status
tuna status --service-name <name>

# Check router health directly
curl http://<router-ip>:8080/router/health

# Check SkyServe status
sky status --refresh
```

### High latency

Check which backend is serving traffic:

```bash
curl http://<router-ip>:8080/router/health
```

If `skyserve_ready` is `false`, spot instances are still booting — requests are going through serverless (which is working correctly). Once spot boots, traffic shifts automatically.

### Gated model fails to load

If the deployment succeeds but the model fails to start, you likely need a HuggingFace token:

```bash
export HF_TOKEN=<your-token>
```

Then redeploy.

## Contact

- Hetarth — hetarth@tandemn.com
- Mankeerat — mankeerat@tandemn.com

## License

MIT

This project depends on [SkyPilot](https://github.com/skypilot-org/skypilot) (Apache License 2.0).
