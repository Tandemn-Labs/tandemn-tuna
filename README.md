
# Tuna
<div align="center">
<img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/tuna3.png" width="500" alt="Tuna">
</div>

Spot GPUs are 3-5x cheaper than on-demand, but they take minutes to start and can be interrupted at any time. Serverless GPUs start in seconds and never get interrupted, but you pay a premium for that convenience. What if you didn't have to choose?

Tuna is a smart router that combines both behind a single OpenAI-compatible endpoint. It serves requests from serverless while spot instances boot up, shifts traffic to spot once ready, and falls back to serverless if spot gets preempted. You only pay for the compute you actually use — spot rates for steady traffic, serverless only during cold starts and failover.

<div align="center">
<table>
<tr>
<td align="center" colspan="3"><b>Serverless</b></td>
<td align="center" colspan="1"><b>Spot</b></td>
</tr>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/modal-logo-icon.png" height="30"><br>Modal</td>
<td align="center"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/runpod-logo-black.svg" height="30"><br>RunPod</td>
<td align="center"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/google-cloud-run-logo-png_seeklogo-354677.png" height="30"><br>Cloud Run</td>
<td align="center" rowspan="2"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/Amazon_Web_Services_Logo.svg.png" height="30"><br>AWS<br><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/google-cloud-run-logo-png_seeklogo-354677.png" height="20"><br>GCP<br>via SkyPilot</td>
</tr>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/baseten.png" height="30"><br>Baseten</td>
<td align="center"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/azure-container.png" height="30"><br>Azure</td>
<td align="center"><img src="https://raw.githubusercontent.com/Tandemn-Labs/tandemn-tuna/main/assets/cerebrium.png" height="30"><br>Cerebrium</td>
</tr>
</table>
</div>

<p align="center">
  <a href="ROADMAP.md"><b>View Roadmap</b></a>
</p>

> **Note:** Not all GPU types across all providers have been end-to-end tested yet. We are actively testing more combinations. If you run into issues with a specific GPU + provider pair, please [open an issue](https://github.com/Tandemn-Labs/tandemn-tuna/issues).

## Prerequisites

- Python 3.11+
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) — required for spot instances on AWS (default)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) — required for spot instances on GCP (use `--spots-cloud gcp`)
- At least one serverless provider account: [Modal](https://modal.com/), [RunPod](https://www.runpod.io/), [Google Cloud](https://cloud.google.com/), [Baseten](https://www.baseten.co/), [Azure](https://azure.microsoft.com/), or [Cerebrium](https://www.cerebrium.ai/)
- For gated models (Llama, Mistral, Gemma, etc.): a [HuggingFace token](https://huggingface.co/settings/tokens) with access to the model

> **Note:** By default Tuna deploys both a serverless backend and a spot backend. AWS credentials are required for spot instances, which run on AWS via [SkyPilot](https://github.com/skypilot-org/skypilot). Alternatively, use `--spots-cloud gcp` for GCP spot instances. Use `--serverless-only` to skip spot + router (no cloud credentials needed for spot).

## Quick Start

**1. Install**

```bash
pip install tandemn-tuna[modal] --pre     # Modal as serverless provider
pip install tandemn-tuna[cloudrun] --pre  # Cloud Run as serverless provider
pip install tandemn-tuna[baseten] --pre   # Baseten as serverless provider
pip install tandemn-tuna[azure] --pre     # Azure Container Apps as serverless provider
pip install tandemn-tuna[cerebrium] --pre # Cerebrium as serverless provider
pip install tandemn-tuna --pre            # RunPod (no extra deps needed)
pip install tandemn-tuna[all] --pre       # everything
```

> This project is under active development and experimental. For the latest version, install from source:
> ```bash
> git clone https://github.com/Tandemn-Labs/tandemn-tuna.git
> cd tandemn-tuna
> pip install -e ".[all]"
> ```

**2. Set up spot GPU cloud (pick one)**

<details>
<summary><b>AWS (default)</b></summary>

```bash
aws configure          # set up AWS credentials
sky check aws          # verify SkyPilot can see your AWS account
```

</details>

<details>
<summary><b>GCP</b></summary>

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <YOUR_PROJECT_ID>
sky check gcp          # verify SkyPilot can see your GCP account
```

> **Note:** GCP spot instances (preemptible VMs) require GPU quota in your project.
> Check quota at: https://console.cloud.google.com/iam-admin/quotas
> Search for "Preemptible" GPU quotas in your target region.
> GCP preemptible VMs have a 24-hour maximum lifetime — SkyPilot handles automatic recovery.

</details>

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
<summary><b>Baseten</b></summary>

**Step 1: Create account** — sign up at https://app.baseten.co/signup/

**Step 2: Get API key** — go to Settings > API Keys (https://app.baseten.co/settings/api_keys), create a key, copy it immediately

**Step 3: Set the API key**

```bash
export BASETEN_API_KEY=<your-api-key>
```

Add to `~/.bashrc` or `~/.zshrc` to persist.

**Step 4: Install and authenticate the Truss CLI**

```bash
pip install --upgrade truss
truss login --api-key $BASETEN_API_KEY
```

**Step 5: (For gated models) Add HuggingFace token** — go to Settings > Secrets (https://app.baseten.co/settings/secrets), add a secret named `hf_access_token` with your HF token.

</details>

<details>
<summary><b>Azure Container Apps</b></summary>

Requires the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

**Step 1: Install Azure CLI and log in**

```bash
az login
```

**Step 2: Register required resource providers**

```bash
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

Registration can take a few minutes. Check status with `az provider show --namespace Microsoft.App --query registrationState`.

**Step 3: Create a resource group** (if you don't have one)

```bash
az group create --name tuna-rg --location eastus
```

**Step 4: Set environment variables**

```bash
export AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
export AZURE_RESOURCE_GROUP=tuna-rg
export AZURE_REGION=eastus
```

Add to `~/.bashrc` or `~/.zshrc` to persist.

**Step 5: Install the Azure SDK**

```bash
pip install tandemn-tuna[azure]
```

**Step 6: Verify setup**

```bash
tuna check --provider azure
```

**GPU availability:** Azure Container Apps supports T4 ($0.26/hr) and A100 80GB ($1.90/hr) GPUs. GPU quota must be requested via the Azure portal — search "Quotas" and request `Managed Environment Consumption T4 Gpus` or `Managed Environment Consumption NCA100 Gpus` capacity for Container Apps in your region. Note: this is separate from VM-level (Compute) GPU quota.

**Environment reuse:** The first Azure deploy creates a Container Apps environment (~30 min). Subsequent deploys reuse it (~2 min). Environments are preserved on destroy — use `--azure-cleanup-env` to remove them. An idle environment with no running apps incurs no charges.

</details>

<details>
<summary><b>Cerebrium</b></summary>

**Step 1: Create account** — sign up at https://www.cerebrium.ai/ ($30 free credits on Hobby plan)

**Step 2: Install the Cerebrium CLI**

```bash
pip install tandemn-tuna[cerebrium]
```

**Step 3: Create a service account token** — go to Dashboard > API Keys > Create Service Account > Copy the token

**Step 4: Set the API key**

```bash
export CEREBRIUM_API_KEY=<your-service-account-token>
```

Add to `~/.bashrc` or `~/.zshrc` to persist.

**Step 5: Set your project context**

The service account token contains your project ID, but the CLI needs it set explicitly:

```bash
# List your projects to find the ID
cerebrium projects list

# Set the project context
cerebrium project set <your-project-id>
```

> **Note:** Your project ID (e.g. `p-ad42316a`) can be found in the Cerebrium dashboard URL or by running `cerebrium projects list`. This step is required — without it, deploys will fail with `"no project configured"`.

> **For CI/CD / headless environments:** Set `CEREBRIUM_API_KEY` and run `cerebrium project set` before deploying. No `cerebrium login` needed.

**Step 6: Verify setup**

```bash
tuna check --provider cerebrium
```

**GPU availability:** Hobby plan ($0/mo) gives access to T4, A10, L4, L40S. A100 and H100 require the Enterprise plan.

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
tuna check --provider baseten                        # check Baseten API key + truss CLI
tuna check --provider azure                          # check Azure CLI + SDK + resource providers
tuna check --provider cerebrium                      # check Cerebrium API key + CLI
```

**6. Deploy a model**

```bash
tuna deploy --model Qwen/Qwen3-0.6B --gpu L4 --service-name my-first-deploy
```

Tuna auto-selects the cheapest serverless provider for your GPU, launches spot instances on AWS, and gives you a single endpoint. The router handles everything — serverless covers traffic immediately while spot boots up in the background.

```bash
# Deploy with GCP spot instances instead of AWS
tuna deploy --model Qwen/Qwen3-0.6B --gpu T4 --spots-cloud gcp --service-name my-gcp-deploy
```

**6a. (Alternative) Deploy serverless-only**

Skip spot + router for dev/test or low-traffic:

```bash
tuna deploy --model Qwen/Qwen3-0.6B --gpu L4 --serverless-only
```

Returns the provider's direct endpoint. No AWS credentials needed.

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
tuna destroy --service-name my-first-deploy   # tear down a specific deployment
tuna destroy --all                            # tear down all active deployments
```

> **Tip:** If you don't pass `--service-name` during deploy, Tuna auto-generates a name like `tuna-a3f8c21b`. Use `tuna list` to find it.

**9. Browse GPU pricing**

```bash
tuna show-gpus                                    # compare serverless pricing across providers
tuna show-gpus --spot                             # include AWS spot prices (default)
tuna show-gpus --spot --spots-cloud gcp           # include GCP spot prices
tuna show-gpus --gpu H100                         # detailed pricing for a specific GPU
tuna show-gpus --provider runpod                  # filter to one provider
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
     │ Modal / RunPod /  │    │ AWS / GCP via      │
     │ Cloud Run         │    │ SkyPilot           │
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
| `destroy` | Tear down a deployment (`--service-name <name>` or `--all` for all active) |
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
| `--serverless-provider` | auto (cheapest for GPU) | `modal`, `runpod`, `cloudrun`, `baseten`, `azure`, or `cerebrium` |
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
| `--serverless-only` | off | Serverless only (no spot, no router). No AWS needed. |
| `--public` | off | Make service publicly accessible (no auth) |
| `--use-different-vm-for-lb` | off | Launch router on a separate VM instead of colocating on controller |
| `--gcp-project` | — | Google Cloud project ID |
| `--gcp-region` | — | Google Cloud region (e.g. `us-central1`) |
| `--azure-subscription` | — | Azure subscription ID |
| `--azure-resource-group` | — | Azure resource group name |
| `--azure-region` | — | Azure region (e.g. `eastus`) |
| `--azure-environment` | — | Name of existing Container Apps environment to reuse |

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
tuna check --provider baseten
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
