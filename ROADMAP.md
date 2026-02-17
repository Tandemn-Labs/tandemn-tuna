# Tuna Roadmap

> Hybrid GPU inference orchestrator — serverless for speed, spot for savings, one endpoint for everything.

Current state: **v0.0.1a5** | 4 serverless providers (Modal, RunPod, Cloud Run, Baseten) | 1 spot provider (AWS via SkyPilot) | ~295 tests

---

## Phase 1: Stability & Developer Experience

> Make the core rock-solid and easy to contribute to.

### Segregate Templates per Provider
Each provider gets its own `templates/<provider>/` directory with a README explaining the template variables. Makes it easy for contributors to add or modify a provider without touching others.

### Use SkyPilot SDK Instead of CLI
Replace `subprocess.run(["sky", ...])` calls with SkyPilot's Python SDK. Eliminates shell parsing issues, gives structured error handling, and removes the requirement for `sky` to be on PATH.

### `tuna destroy --all`
Convenience command to tear down every active deployment in one shot. Iterate over all `active` records in state DB and destroy each.

### Partial Deployment Cleanup
When a deployment fails midway (e.g. serverless deployed but spot didn't), `tuna destroy` should still clean up whatever was created. Track per-component status in state DB and destroy only what exists.

### Storage Cleanups After Destroy
Remove provider-side artifacts after teardown — SkyPilot bucket storage, Cloud Run container images, Baseten model artifacts, Modal volumes. Prevent orphaned resources from accumulating cost.

### `--serverless` Mode
Deploy to serverless only, no spot GPU, no router. Single endpoint, simplest path. Useful for dev/test or low-traffic models where spot savings don't justify the complexity.

```
tuna deploy meta-llama/Llama-3-8B --gpu L4 --serverless
```

---

## Phase 2: More Providers

> Broaden the GPU market. More providers = better price competition = lower costs.

### Serverless Providers
| Provider | GPU Support | Notes |
|----------|-------------|-------|
| **Replicate** | A40, A100, H100 | REST API, prediction-based billing |
| **Fal AI** | Various | Fast cold starts, queue-based |
| **BentoML** | Various | Open-source friendly, BentoCloud |
| **Azure Container Apps** | T4, A100 | GPU workload profiles, long env creation times |
| **InferX** | GPU slicing | Sub-2s cold starts, GPU fractional allocation (1/3 GPU per model), 80%+ utilization, high deployment density |

### Spot Providers (via SkyPilot)
| Cloud | Status | Notes |
|-------|--------|-------|
| **AWS** | Done | Current default |
| **GCP** | Planned | SkyPilot supports it, need catalog entries |
| **Azure** | Planned | SkyPilot supports it, need catalog entries |
| **Spheron AI** | Research | Decentralized GPU marketplace |
| **Prime Intellect** | Research | Decentralized compute |

### Heterogeneous GPU Selection
Allow different GPUs for serverless vs spot. Cheap T4 for spot (long-running, cost-optimized), faster L4/A10G for serverless (cold-start-sensitive).

```
tuna deploy model --serverless-gpu L4 --spot-gpu T4
```

---

## Phase 3: Observability

> You can't optimize what you can't measure.

### Unified Metrics Dashboard
Single `tuna metrics` command (or web UI) combining:

- **DCGM Metrics**: GPU utilization, memory usage, power draw, temperature (from spot VMs)
- **Router Metrics**: Request latency (p50/p95/p99), throughput, backend split ratio, error rates
- **Cost Metrics**: Real-time spend per backend, cumulative savings vs on-demand baseline

### Cold Start Benchmarking Suite
Automated benchmark scripts that measure cold start across providers:

```
tuna benchmark cold-start --providers modal,runpod,baseten,cloudrun --gpu T4 --model Qwen/Qwen3-0.6B
```

Output: time-to-first-token per provider, broken down by phases (container pull, weight download, model load, warmup). Run this across 5-6 providers to publish a comparison.

### Spot Preemption Prediction
Integrate spot price history and preemption rates (from SkyPilot/cloud APIs) into the cost dashboard. Show predicted uptime and expected interruption frequency per region/GPU combo.

### Log Aggregation
Stream logs from all components (serverless, spot, router) into a single `tuna logs <service-name>` command. Currently requires checking each provider's dashboard separately.

---

## Phase 4: Advanced Inference

> Beyond basic vLLM — support more engines, models, and optimizations.

### SGLang Support
Add SGLang as an alternative inference engine alongside vLLM. SGLang offers RadixAttention for efficient prefix caching and faster structured output generation. Provider templates would need SGLang-specific start commands and health checks.

### LMCache Integration
Plug in LMCache for KV cache persistence across cold starts. When a spot instance is preempted and a new one spins up, LMCache can restore the KV cache from shared storage instead of recomputing from scratch. Reduces effective cold start for returning contexts.

### Shared KV Cache Pool (Serverless <-> Spot)
The holy grail: a shared KV cache layer between serverless and spot backends. When traffic shifts from serverless to spot (or vice versa), active KV caches transfer across. Eliminates redundant prefill computation during backend transitions. Requires a shared cache store (Redis/S3) and KV-aware routing in the meta-lb.

### SAM2 / SAM2.1 Model Support
Extend beyond LLMs to vision models. SAM2 (Segment Anything Model 2) needs different serving infrastructure — no vLLM, different health checks, different API shape. This likely means a `--model-type` flag and per-type serving templates.

### Multi-Node GPU Support
For models that don't fit on a single GPU node (70B+ models), orchestrate multi-node tensor parallelism. SkyPilot supports multi-node clusters; need to wire `--num-nodes` through the deployment pipeline and configure Ray/vLLM distributed serving.

---

## Phase 5: Customization & Extensibility

> Let users bring their own stuff.

### BYOC (Bring Your Own Container)
Let users supply a custom Docker image instead of the default vLLM image. Useful for custom model architectures, fine-tuned models with custom code, or non-vLLM serving stacks.

```
tuna deploy --image ghcr.io/myorg/my-model:latest --gpu A100 --health-endpoint /ready
```

### Plugin System for Providers
Formalize the provider interface so third parties can add providers without forking. A provider is a Python package that implements `InferenceProvider` and registers via entry points.

```toml
[project.entry-points."tuna.providers"]
my_provider = "my_package:MyProvider"
```

### Config File Support
Project-level `tuna.yaml` that sets defaults (preferred providers, GPU preferences, scaling policies, team secrets). Avoids repeating CLI flags.

```yaml
defaults:
  serverless_provider: baseten
  spot_cloud: aws
  gpu: L4
  scaling:
    spot:
      min_replicas: 1
      max_replicas: 5
```

---

## Phase 6: Security & Multi-Tenancy

> Production-grade access control and isolation.

### RBAC (Role-Based Access Control)
- **Admin**: deploy, destroy, view all deployments
- **Developer**: deploy to dev/staging, view own deployments
- **Viewer**: read-only status and metrics

### API Key Management
Centralized secret store for provider API keys (Baseten, RunPod, Modal, HF tokens). Avoid scattering secrets across env vars. Support for team-level key sharing with audit logs.

### Network Isolation
- Router-level auth (already exists, needs hardening)
- mTLS between router and backends
- VPC peering for spot instances
- Rate limiting per API key at the router

### Audit Logging
Track who deployed what, when, and the cost. Useful for teams sharing a GPU budget.

---

## Phase 7: Intelligence

> Make the system smarter over time.

### Auto-Provider Selection (Beyond Price)
Current: pick cheapest provider for a GPU. Future: factor in cold start latency, reliability history, current availability, and geographic proximity to the user.

### Dynamic Pricing Updates
Replace static catalog prices with live pricing from provider APIs. RunPod, Modal, and SkyPilot already expose pricing endpoints.

### Canary Deployments
Roll out model updates gradually — deploy new version to serverless, keep old version on spot, shift traffic via router weights. If new version degrades quality/latency, auto-rollback.

### Request-Aware Routing
Route based on request characteristics: short prompts to serverless (fast TTFT), long batch jobs to spot (cost-effective). Inspect `max_tokens` or prompt length to make routing decisions.

---

## Backlog (Unscheduled)

- **Web UI**: Browser dashboard for deployments, metrics, and cost
- **Webhook Notifications**: Slack/Discord alerts on deploy, preemption, failures
- **CI/CD Hooks**: GitHub Actions integration for deploy-on-push
- **A/B Testing**: Route percentage of traffic to different model versions
- **Automatic vLLM Version Alignment**: Ensure all providers run the same vLLM version
- **Graceful Spot Drain**: When preemption signal arrives, finish in-flight requests before shutdown
- **Model Registry Integration**: Pull models from private registries (S3, GCS, custom)
- **BDN / Weights Migration** (Baseten): Revisit when Baseten BDN weight pre-mounting works reliably
- **Batch Inference Mode**: Offline batch processing on spot GPUs at lowest cost
- **LoRA Adapter Hot-Swapping**: Serve multiple LoRA adapters on a single base model deployment

---

## Contributing

Each provider lives in its own file under `tuna/providers/` with a corresponding test file in `tests/`. To add a new provider:

1. Implement the `InferenceProvider` interface (`tuna/providers/base.py`)
2. Add GPU mappings to `tuna/catalog.py`
3. Create a template in `tuna/templates/`
4. Register in `tuna/providers/registry.py`
5. Add tests in `tests/test_<provider>_provider.py`
6. Run `uv run pytest tests/ -v` to verify

See existing providers for patterns. Start with the simplest one (RunPod) as a reference.
