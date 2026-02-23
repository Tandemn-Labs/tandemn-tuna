---
name: tandemn-tuna
description: Deploy and serve LLM models on GPU. Compare GPU pricing. Launch vLLM on Modal, RunPod, Cerebrium, Cloud Run, Baseten, or Azure with spot instance fallback. OpenAI-compatible inference endpoint.
version: 0.0.1
metadata:
  openclaw:
    requires:
      bins:
        - uv
      anyBins:
        - aws
        - az
      env: []
    emoji: "\U0001F41F"
    homepage: https://github.com/Tandemn-Labs/tandemn-tuna
    install:
      - kind: uv
        package: tandemn-tuna
        bins: [tuna]
---

# Tuna — Deploy and Serve LLM Models on GPU Infrastructure

Tuna is a hybrid GPU inference orchestrator. It lets you deploy, serve, and manage LLM models (Llama, Qwen, Mistral, DeepSeek, Gemma, and any HuggingFace model) on serverless GPUs from **Modal, RunPod, Cerebrium, Google Cloud Run, Baseten, or Azure Container Apps**, with optional **spot instance fallback on AWS** via SkyPilot. Every deployment gets an **OpenAI-compatible `/v1/chat/completions` endpoint**.

The key idea: serverless GPUs handle requests immediately (fast cold start, pay-per-second) while a cheaper spot GPU boots in the background. Once spot is ready, traffic shifts there. If spot gets preempted, traffic falls back to serverless automatically. This gives you **3–5x cost savings** over pure serverless with zero downtime.

## Quick Start — Deploy a Model in 3 Commands

```bash
# 1. Install tuna
uv pip install tandemn-tuna

# 2. Deploy a model (auto-picks cheapest serverless provider for the GPU)
tuna deploy --model Qwen/Qwen3-0.6B --gpu L4 --service-name my-llm

# 3. Query your endpoint (shown in deploy output)
curl http://<router-ip>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello!"}]}'
```

For serverless-only (no spot, no AWS needed):

```bash
tuna deploy --model Qwen/Qwen3-0.6B --gpu L4 --serverless-only
```

## All Commands

### `tuna deploy` — Launch a model on GPU

Deploy a model across serverless + spot infrastructure. This is the main command.

```bash
tuna deploy --model <HuggingFace-model-ID> --gpu <GPU> [options]
```

**Required arguments:**
- `--model` — HuggingFace model ID (e.g., `Qwen/Qwen3-0.6B`, `meta-llama/Llama-3-70b`)
- `--gpu` — GPU type (e.g., `T4`, `L4`, `L40S`, `A100`, `H100`, `B200`)

**Common options:**
- `--service-name` — Name for the deployment (auto-generated if omitted)
- `--serverless-provider` — Force a specific provider: `modal`, `runpod`, `cloudrun`, `baseten`, `azure`, `cerebrium` (default: cheapest available)
- `--serverless-only` — Serverless only, no spot backend or router (no AWS needed)
- `--gpu-count` — Number of GPUs (default: 1)
- `--tp-size` — Tensor parallel size (default: 1)
- `--max-model-len` — Max sequence length (default: 4096)
- `--spots-cloud` — Cloud for spot GPUs: `aws` or `azure` (default: `aws`)
- `--region` — Cloud region for spot instances
- `--concurrency` — Override serverless concurrency limit
- `--no-scale-to-zero` — Keep at least 1 spot replica running
- `--public` — Make endpoint publicly accessible (no auth)
- `--scaling-policy` — Path to YAML with scaling parameters

**Provider-specific options:**
- `--gcp-project`, `--gcp-region` — For Cloud Run
- `--azure-subscription`, `--azure-resource-group`, `--azure-region`, `--azure-environment` — For Azure

**Examples:**
```bash
# Deploy Llama 3 on Modal with hybrid spot
tuna deploy --model meta-llama/Llama-3-8b --gpu A100 --serverless-provider modal

# Deploy on RunPod, serverless-only
tuna deploy --model mistralai/Mistral-7B-Instruct-v0.3 --gpu L40S --serverless-provider runpod --serverless-only

# Deploy on Azure with an existing environment
tuna deploy --model Qwen/Qwen3-0.6B --gpu T4 --serverless-provider azure --azure-environment my-env

# Deploy a large model with tensor parallelism
tuna deploy --model meta-llama/Llama-3-70b --gpu H100 --gpu-count 4 --tp-size 4
```

### `tuna show-gpus` — Compare GPU Prices Across Providers

Show GPU pricing from all serverless providers, optionally including spot prices.

```bash
tuna show-gpus [--gpu <GPU>] [--provider <provider>] [--spot]
```

**Examples:**
```bash
# Show all GPU prices across all providers
tuna show-gpus

# Show H100 pricing specifically
tuna show-gpus --gpu H100

# Show Modal's prices only
tuna show-gpus --provider modal

# Include AWS spot prices for comparison
tuna show-gpus --spot
```

### `tuna check` — Validate Provider Setup (Preflight)

Run preflight checks to verify credentials, CLIs, and quotas for a provider before deploying.

```bash
tuna check --provider <provider> [--gpu <GPU>]
```

**Examples:**
```bash
# Check Modal setup
tuna check --provider modal

# Check Azure with specific GPU
tuna check --provider azure --gpu T4 --azure-subscription <id> --azure-resource-group <rg>
```

### `tuna status` — Check Deployment Status

```bash
tuna status --service-name <name>
```

### `tuna cost` — Show Cost Savings Dashboard

```bash
tuna cost --service-name <name>
```

### `tuna list` — List All Deployments

```bash
tuna list [--status active|destroyed|failed]
```

### `tuna destroy` — Tear Down a Deployment

```bash
# Destroy a specific deployment
tuna destroy --service-name <name>

# Destroy all deployments
tuna destroy --all
```

## Provider Setup Guide

Each serverless provider needs its own credentials. Run `tuna check --provider <name>` to verify setup.

### Modal

```bash
pip install modal  # or: uv pip install tandemn-tuna[modal]
modal token new    # opens browser to authenticate
```

No environment variables needed — token is stored in Modal's config.

### RunPod

```bash
export RUNPOD_API_KEY="your-api-key"
```

Get your API key from the [RunPod console](https://www.runpod.io/console/user/settings).

### Google Cloud Run

```bash
pip install google-cloud-run  # or: uv pip install tandemn-tuna[cloudrun]
gcloud auth login
gcloud auth application-default login
```

Optionally set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_REGION`, or pass `--gcp-project` and `--gcp-region`.

### Baseten

```bash
pip install truss  # or: uv pip install tandemn-tuna[baseten]
export BASETEN_API_KEY="your-api-key"
truss login --api-key $BASETEN_API_KEY
```

### Azure Container Apps

```bash
pip install azure-mgmt-appcontainers azure-identity  # or: uv pip install tandemn-tuna[azure]
az login
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

Pass `--azure-subscription`, `--azure-resource-group`, and `--azure-region` on deploy, or set `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_REGION` env vars. First deploy creates a GPU environment (~30 min); subsequent deploys reuse it (~2 min). Use `--azure-environment` to specify an existing environment.

### Cerebrium

```bash
pip install cerebrium  # or: uv pip install tandemn-tuna[cerebrium]
cerebrium login
export CEREBRIUM_API_KEY="your-api-key"
```

Note: Hobby plan gives T4, A10, L4, L40S. A100 and H100 require Enterprise.

### Spot GPUs (AWS via SkyPilot)

Spot is included automatically in hybrid deploys. Just configure AWS:

```bash
aws configure  # set access key, secret key, region
```

Use `--serverless-only` to skip spot if you don't have AWS set up.

## Common Scenarios

**When the user wants to deploy a model for quick testing:**
Use `--serverless-only` to skip spot setup. Pick a small GPU like L4 or T4. Example:
```bash
tuna deploy --model Qwen/Qwen3-0.6B --gpu L4 --serverless-only
```

**When the user wants the cheapest deployment:**
First run `tuna show-gpus --spot` to compare serverless and spot prices. Then deploy with hybrid mode (the default) to get spot savings. The auto provider selector already picks the cheapest serverless option for the chosen GPU.

**When the user wants to compare GPU prices:**
```bash
tuna show-gpus
tuna show-gpus --gpu A100
tuna show-gpus --spot  # includes AWS spot prices
```

**When the user asks "which providers support H100?" or a specific GPU:**
```bash
tuna show-gpus --gpu H100
```

**When the user wants to deploy on a specific provider:**
Use `--serverless-provider <name>`. Run `tuna check --provider <name>` first to verify credentials.

**When the user wants to deploy a large model (70B+):**
Use multiple GPUs with tensor parallelism:
```bash
tuna deploy --model meta-llama/Llama-3-70b --gpu H100 --gpu-count 4 --tp-size 4
```

**When the user wants to check if their setup is ready:**
```bash
tuna check --provider modal
tuna check --provider runpod
```

**When the user wants to see what's currently deployed:**
```bash
tuna list
tuna list --status active
```

**When the user wants to tear down everything:**
```bash
tuna destroy --all
```

## Supported GPUs

All GPU types that tuna supports across its providers:

| GPU | VRAM | Architecture | Available On |
|-----|------|-------------|-------------|
| T4 | 16 GB | Turing | Modal, RunPod, Baseten, Azure, Cerebrium, Spot |
| A10 | 24 GB | Ampere | Cerebrium |
| A10G | 24 GB | Ampere | Modal, Baseten, Spot |
| A4000 | 16 GB | Ampere | RunPod |
| A5000 | 24 GB | Ampere | RunPod |
| RTX 4090 | 24 GB | Ada | RunPod |
| L4 | 24 GB | Ada | Modal, RunPod, Cloud Run, Baseten, Cerebrium, Spot |
| A40 | 48 GB | Ampere | RunPod |
| A6000 | 48 GB | Ampere | RunPod |
| L40 | 48 GB | Ada | RunPod |
| L40S | 48 GB | Ada | Modal, RunPod, Cerebrium, Spot |
| A100 (40 GB) | 40 GB | Ampere | Modal, Cerebrium, Spot |
| A100 (80 GB) | 80 GB | Ampere | Modal, RunPod, Azure, Baseten, Cerebrium, Spot |
| H100 | 80 GB | Hopper | Modal, RunPod, Baseten, Cerebrium, Spot |
| H200 | 141 GB | Hopper | Spot |
| B200 | 192 GB | Blackwell | Modal, Baseten |
| RTX PRO 6000 | 32 GB | Blackwell | Cloud Run |

Use `tuna show-gpus` for current pricing across all providers.

## Error Handling

**Preflight check fails (`tuna check`):**
The output tells you exactly what's wrong — missing CLI tool, expired credentials, unregistered provider, insufficient quota. Fix the reported issue and re-run `tuna check`.

**Deploy fails:**
1. Run `tuna check --provider <provider> --gpu <gpu>` to validate the environment
2. Add `-v` for verbose logs: `tuna deploy -v ...`
3. Check `tuna status --service-name <name>` for deployment state

**Spot instance not available:**
Spot GPUs depend on cloud availability. If spot fails to launch, the serverless backend keeps serving — no downtime. Try a different region with `--region`, or use `--serverless-only`.

**"No provider supports GPU X":**
Run `tuna show-gpus --gpu <GPU>` to see which providers offer that GPU. Not all GPUs are available on all providers.

**Azure environment takes too long:**
First Azure deploy creates a GPU environment (~30 min). Subsequent deploys reuse it (~2 min). Use `--azure-environment` to specify an existing one.
