# Tuna Roadmap

> Hybrid GPU inference orchestrator — serverless for speed, spot for savings, one endpoint for everything.

Current state: v0.0.1a9 | 6 serverless providers (Modal, RunPod, Cloud Run, Baseten, Azure, Cerebrium) | 2 spot providers (AWS, GCP via SkyPilot) | ~572 tests

---

## Phase 1: Stability & Developer Experience

- ~~Segregate templates per provider — each provider gets its own `templates/<provider>/` directory with a README explaining template variables, so contributors can add/modify a provider without touching others~~
- ~~Use SkyPilot SDK instead of CLI — replace `subprocess.run(["sky", ...])` with the Python SDK for structured error handling and no PATH dependency~~
- ~~`tuna destroy --all` — tear down every active deployment in one shot~~
- ~~Partial deployment cleanup — when a deploy fails midway (serverless deployed but spot didn't), `tuna destroy` should clean up whatever was created~~
- Storage cleanups after destroy — remove orphaned provider-side artifacts (SkyPilot buckets, Cloud Run images, Baseten models, Modal volumes)
- ~~`--serverless` mode — deploy to serverless only, no spot, no router. Single endpoint, simplest path for dev/test or low-traffic models~~

---

## Phase 2: More Providers

Serverless:

| Provider | GPU Support | Notes |
|----------|-------------|-------|
| ~~Cerebrium~~ | ~~T4, A10, L4, L40S, A100, H100~~ | ~~Done — per-second billing, custom runtime entrypoint, $30 free credits~~ |
| Beam Cloud | T4, A10G, 4090, A100, H100 | Per-millisecond billing, H100 at $0.97/hr, open-source runtime, 10 hrs free |
| Koyeb | L4, L40S, A100, H100, H200, 8xH100 | Per-second billing, widest GPU selection, multi-GPU, $29/mo platform fee |
| BentoML | T4, L4, H100, H200, B200 | Bento format required, $10 free credits, on waitlist |
| Inferless | T4, A10, A100 | Per-second billing, cheap shared instances, no H100 |
| Novita AI | RTX 3090/4090, A100 | Per-second, explicit vLLM Docker support, 20+ regions |
| ~~Azure Container Apps~~ | ~~T4, A100~~ | ~~GPU workload profiles, long env creation times~~ |
| InferX | GPU slicing | Sub-2s cold starts, fractional GPU allocation (1/3 GPU per model), 80%+ utilization, high deployment density |
| Replicate | A40, A100, H100 | Acquired by Cloudflare (2025), pays for idle on custom models |
| Fal AI | H100, H200, A100 | Custom containers new, optimized for media generation not LLM inference |

Spot (via SkyPilot):

| Cloud | Status | Notes |
|-------|--------|-------|
| AWS | Done | Current default |
| ~~GCP~~ | ~~✅ Done~~ | ~~Docker image-based deploy, T4/L4/A100/H100 spot~~ |
| Azure | Planned | SkyPilot supports it, need catalog entries |
| Spheron AI | Research | High Quality GPU marketplace |
| Prime Intellect | Research | Decentralized compute |

- Heterogeneous GPU selection — allow different GPUs for serverless vs spot (cheap T4 for spot, faster L4 for serverless)
  ```
  tuna deploy model --serverless-gpu L4 --spot-gpu T4
  ```

---

## Phase 3: Observability

- Unified metrics dashboard — single `tuna metrics` command combining:
  - DCGM metrics: GPU utilization, memory, power, temperature (from spot VMs)
  - Router metrics: latency p50/p95/p99, throughput, backend split, error rates
  - Cost metrics: real-time spend per backend, savings vs on-demand baseline
- ~~Cold start benchmarking suite — automated scripts measuring cold start across providers, broken down by phase (container pull, weight download, model load, warmup)~~
  ```
  tuna benchmark cold-start --providers modal,runpod,baseten,cloudrun --gpu T4 --model Qwen/Qwen3-0.6B
  ```
- Spot preemption prediction — integrate spot price history and preemption rates into the cost dashboard, show predicted uptime per region/GPU
- Log aggregation — `tuna logs <service-name>` streaming logs from all components (serverless, spot, router) instead of checking each provider dashboard separately

---

## Phase 4: Advanced Inference

- SGLang support — alternative inference engine alongside vLLM, with RadixAttention for prefix caching and faster structured output
- LMCache integration — KV cache persistence across cold starts. When spot gets preempted, new instance restores KV cache from shared storage instead of recomputing
- Shared KV cache pool (serverless <-> spot) — when traffic shifts between backends, active KV caches transfer across. Requires shared cache store (Redis/S3) and KV-aware routing in the meta-lb
- SAM2/SAM2.1 model support — extend beyond LLMs to vision models. Different serving infra, health checks, API shape. Needs a `--model-type` flag and per-type templates
- Multi-node GPU support — for 70B+ models that don't fit on one node. Wire `--num-nodes` through the pipeline and configure Ray/vLLM distributed serving

---

## Phase 5: Customization & Extensibility

- BYOC (bring your own container) — supply a custom Docker image instead of default vLLM. For custom architectures, fine-tuned models, or non-vLLM stacks
  ```
  tuna deploy --image ghcr.io/myorg/my-model:latest --gpu A100 --health-endpoint /ready
  ```
- Plugin system for providers — formalize the provider interface so third parties can add providers via entry points without forking
  ```toml
  [project.entry-points."tuna.providers"]
  my_provider = "my_package:MyProvider"
  ```
- Config file support — project-level `tuna.yaml` for defaults (preferred providers, GPU preferences, scaling policies). Avoids repeating CLI flags
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

- RBAC — admin (deploy/destroy/view all), developer (deploy to dev/staging, view own), viewer (read-only)
- API key management — centralized secret store for provider keys instead of scattered env vars, with team-level sharing and audit logs
- Network isolation — harden router auth, mTLS between router and backends, VPC peering for spot, rate limiting per API key
- Audit logging — track who deployed what, when, and the cost

---

## Phase 7: Intelligence

- Auto-provider selection beyond price — factor in cold start latency, reliability history, availability, and geographic proximity
- Dynamic pricing — replace static catalog prices with live pricing from provider APIs
- Canary deployments — deploy new model version to serverless, keep old on spot, shift traffic via router weights, auto-rollback if quality degrades
- Request-aware routing — route based on request characteristics: short prompts to serverless (fast TTFT), long batch jobs to spot (cost-effective)

---

## Backlog

- Web UI for deployments, metrics, and cost
- Webhook notifications (Slack/Discord) on deploy, preemption, failures
- CI/CD hooks — GitHub Actions integration for deploy-on-push
- A/B testing — route percentage of traffic to different model versions
- Automatic vLLM version alignment across providers
- Graceful spot drain — finish in-flight requests on preemption signal
- Model registry integration (S3, GCS, custom)
- ~~BDN/weights migration (Baseten) — revisit when BDN pre-mounting works reliably~~
- Batch inference mode on spot GPUs
- LoRA adapter hot-swapping on a single base model

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide — setup, adding providers, testing patterns, and project structure.
