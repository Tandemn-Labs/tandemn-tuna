# Hybrid GPU Inference Orchestrator - MVP Architecture

## Goal

User provides a model name and config. System deploys it on **serverless** (Modal, later Cloud Run, etc.) + **spot** (SkyServe) in parallel, then returns a **single endpoint** that routes between them.

```
python -m tandemn deploy \
  --model Qwen/Qwen3-0.6B \
  --gpu L40S \
  --serverless-provider modal
```

Output:
```
Serverless (Modal):  https://xxx.modal.run   [READY in ~30s]
Spot (SkyServe):     http://x.x.x.x:30001   [STARTING ~5min]
Router:              http://localhost:8080    [READY]

All traffic → http://localhost:8080
  Currently routing to: Modal (spot warming up)
```

---

## Project Structure

```
serverless-spot/
├── tandemn/
│   ├── __init__.py
│   ├── __main__.py              # CLI entry: `python -m tandemn deploy ...`
│   ├── models.py                # All dataclasses (request, plan, result)
│   ├── orchestrator.py          # launch_hybrid() — parallel deploy + start router
│   ├── template_engine.py       # Renders templates via string replacement
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py              # ServerlessProvider ABC
│   │   ├── modal_provider.py    # Modal: plan → render → deploy → URL
│   │   └── registry.py          # Provider lookup by name
│   │
│   ├── spot/
│   │   ├── __init__.py
│   │   └── sky_launcher.py      # SkyServe: render YAML → sky serve up → URL
│   │
│   └── router/
│       ├── __init__.py
│       └── meta_lb.py           # Flask proxy (adapted from SAM load_balancer.py)
│
├── templates/
│   ├── vllm_serve_cmd.txt       # Shared vLLM command (one source of truth)
│   ├── modal_vllm_server.py.tpl # Modal app template
│   ├── skyserve_vllm.yaml.tpl   # SkyServe YAML template
│   └── router.yaml.tpl          # SkyPilot task YAML for router CPU VM
│
├── modal_vllm_server.py         # (existing file, keep as-is for now)
├── CLAUDE.md
├── ARCHITECTURE.md              # This file
│
├── modal-client/                # Reference only, DO NOT TOUCH
├── skypilot/                    # Reference only, DO NOT TOUCH
└── tandemn-profiling/           # Reference only, DO NOT TOUCH
```

~14 files to write. No more.

---

## Data Models (`tandemn/models.py`)

### What we need for MVP

```python
@dataclass
class DeployRequest:
    """What the user asks for."""
    model_name: str
    gpu: str
    gpu_count: int = 1
    tp_size: int = 1
    max_model_len: int = 4096
    serverless_provider: str = "modal"      # "modal", later "cloudrun", "runpod"
    spots_cloud: str = "aws"
    region: str | None = None
    concurrency: int = 32
    cold_start_mode: str = "fast_boot"      # "fast_boot" or "no_fast_boot"
    scale_to_zero: bool = True
    service_name: str | None = None         # auto-generated if None

@dataclass
class ProviderPlan:
    """Rendered deployment artifact, ready to execute."""
    provider: str                           # "modal", "skyserve", etc.
    rendered_script: str                    # File contents (Python or YAML)
    env: dict[str, str]                     # Env vars for subprocess
    metadata: dict[str, str]               # Provider-specific (app_name, etc.)

@dataclass
class DeploymentResult:
    """Outcome of a single backend deployment."""
    provider: str
    endpoint_url: str | None
    health_url: str | None
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

@dataclass
class HybridDeployment:
    """Combined result returned to the user."""
    serverless: DeploymentResult | None
    spot: DeploymentResult | None
    router_url: str
```

### What we skip for MVP

- `ProviderCapabilities` — not needed until multi-provider price comparison
- `CostEstimate` — not needed until `find_cheapest` feature
- `DeploymentStatus` — health checks are just HTTP GETs, no special type needed
- `vLLMSpecificConfig` — flatten into `DeployRequest` for now

---

## Provider Abstraction (`tandemn/providers/base.py`)

The key design for future extensibility (Cloud Run, RunPod, etc.):

```python
from abc import ABC, abstractmethod

class ServerlessProvider(ABC):
    """
    Base class for serverless GPU providers.

    Each provider implements: plan → deploy → status → destroy.
    The router never touches this — it only knows about URLs.
    """

    @abstractmethod
    def name(self) -> str:
        """Provider identifier: 'modal', 'cloudrun', 'runpod'."""
        ...

    @abstractmethod
    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        """
        Render the deployment artifact (script, YAML, Dockerfile, etc.)
        from the shared vllm_cmd and request config.
        Does NOT deploy anything. Pure function.
        """
        ...

    @abstractmethod
    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        """
        Execute the plan. Run `modal deploy`, `gcloud run deploy`, etc.
        Returns endpoint URL on success, error on failure.
        """
        ...

    @abstractmethod
    def destroy(self, result: DeploymentResult) -> None:
        """Tear down the deployment."""
        ...

    def health_check(self, result: DeploymentResult) -> bool:
        """
        Default: HTTP GET to health_url. Override if provider
        has a native status API (e.g., gcloud run services describe).
        """
        try:
            import requests
            resp = requests.get(result.health_url, timeout=5)
            return 200 <= resp.status_code < 300
        except Exception:
            return False
```

### Provider Registry (`tandemn/providers/registry.py`)

```python
_PROVIDERS: dict[str, type[ServerlessProvider]] = {}

def register(name: str, cls: type[ServerlessProvider]):
    _PROVIDERS[name] = cls

def get_provider(name: str) -> ServerlessProvider:
    cls = _PROVIDERS.get(name)
    if not cls:
        raise ValueError(f"Unknown provider: {name}. Available: {list(_PROVIDERS)}")
    return cls()
```

Providers register themselves at import time. Adding Cloud Run later = one new file + `register("cloudrun", CloudRunProvider)`.

---

## Shared vLLM Command (`templates/vllm_serve_cmd.txt`)

Single source of truth for the vLLM launch command. Used by **both** Modal and SkyServe.

```
vllm serve {model} \
  --host {host} \
  --port {port} \
  --max-model-len {max_model_len} \
  --served-model-name llm \
  --tensor-parallel-size {tp_size} \
  --disable-log-requests \
  --uvicorn-log-level info \
  {eager_flag}
```

The template engine replaces `{placeholders}` with values from `DeployRequest`.

---

## Modal Provider (`tandemn/providers/modal_provider.py`)

### plan()

1. Read `templates/vllm_serve_cmd.txt`, substitute placeholders → `vllm_cmd`
2. Read `templates/modal_vllm_server.py.tpl`, substitute `{vllm_cmd}`, `{gpu}`, `{app_name}`, `{concurrency}`, etc.
3. Return `ProviderPlan(provider="modal", rendered_script=<rendered .py contents>, ...)`

### deploy()

1. Write `rendered_script` to a temp file
2. Run `subprocess.run(["modal", "deploy", tmp_file], env=merged_env, check=True)`
3. Resolve URL: `modal.Function.from_name(app_name, "serve").get_web_url()`
4. Return `DeploymentResult(endpoint_url=url, health_url=url+"/health", ...)`

### destroy()

1. Run `subprocess.run(["modal", "app", "stop", app_name])`

### Template (`templates/modal_vllm_server.py.tpl`)

Based directly on the existing `modal_vllm_server.py` that already works. The template adds placeholders where the existing file has hardcoded values:

- `{app_name}` replaces `"qwen-0p6b-vllm-serverless"`
- `{gpu}` replaces `"L40s"`
- `{vllm_cmd}` replaces the hardcoded cmd list
- `{concurrency}`, `{timeout_s}`, `{scaledown_window_s}` replace literals
- `{enable_memory_snapshot}`, `{enable_gpu_snapshot}` from `cold_start_mode`

---

## SkyServe Spot Launcher (`tandemn/spot/sky_launcher.py`)

NOT a provider in the `ServerlessProvider` sense — spot is a separate backend type that always uses SkyPilot. This is intentional: serverless providers are interchangeable; spot is always SkyServe.

### plan()

1. Same `vllm_cmd` from the shared template
2. Read `templates/skyserve_vllm.yaml.tpl`, substitute `{service_name}`, `{gpu}`, `{vllm_cmd}`, `{region}`, `{min_replicas}`, `{max_replicas}`
3. Return `ProviderPlan(provider="skyserve", rendered_script=<YAML contents>, ...)`

### deploy()

1. Write rendered YAML to temp file
2. Run `sky serve up <yaml> --service-name <name>` via subprocess (or `sky.serve.up()` Python API)
3. Poll `sky serve status <name>` until endpoint is available
4. Return `DeploymentResult(endpoint_url=endpoint, health_url=endpoint+"/health", ...)`

### destroy()

1. Run `sky serve down <name>`

### Template (`templates/skyserve_vllm.yaml.tpl`)

```yaml
service:
  readiness_probe:
    path: /health
    initial_delay_seconds: 300
  replica_policy:
    min_replicas: {min_replicas}
    max_replicas: {max_replicas}
    target_qps_per_replica: 10
    upscale_delay_seconds: 5
    downscale_delay_seconds: 300

resources:
  accelerators: "{gpu}:{gpu_count}"
  use_spot: true
  disk_size: 100
  ports: 8000
  {region_block}

run: |
  {vllm_cmd}
```

---

## Router (`tandemn/router/meta_lb.py`)

Direct adaptation of the SAM `load_balancer.py`. The patterns are proven and production-tested.

### What stays the same from SAM

- Stateful proxy with `_state_lock` for thread safety
- Async health check with throttled interval (1s min)
- Async poke to trigger SkyServe scale-up (0.5s min, 0.3s timeout)
- Hop-by-hop header filtering
- Rolling window route stats
- Auth support (optional)

### What changes from SAM

| SAM | Ours |
|-----|------|
| `CLOUDRUN_BASE_URL` | `SERVERLESS_BASE_URL` (could be Modal, Cloud Run, anything) |
| `SKYSERVE_BASE_URL` | Same |
| Hardcoded env var names | Same pattern, just renamed for generality |
| Flask | Flask (keep it simple, same as SAM) |

The router is **provider-agnostic**. It gets two URLs and routes between them. It does not import Modal SDK, SkyPilot, or anything provider-specific. This is critical for extensibility — when we add Cloud Run, the router code changes zero lines.

### Routing logic (same as SAM, proven)

```
Every request:
  1. Check auth
  2. If spot is ready (async health check) → forward to spot (cheaper)
  3. Else → forward to serverless (fast) + poke spot in background
  4. Record stats
```

---

## Where the Router Lives

### MVP: SkyPilot CPU VM (Option A)

The router runs on a cheap CPU-only VM launched via `sky launch`. This is the simplest approach that keeps the router **always-on, fast, and independent** of both backends.

```
┌─────────────────────────────────────────────────────────────┐
│                     Deployment Flow                          │
│                                                              │
│  Orchestrator (your machine)                                 │
│    │                                                         │
│    ├─ 1. sky launch router.yaml  ──► CPU VM (router)         │
│    │      gets router_ip:8080         runs meta_lb.py        │
│    │                                  ~$0.02/hr              │
│    │                                                         │
│    ├─ 2. modal deploy (thread 1) ──► Modal serverless URL    │
│    │      push URL to router via       ~30s to ready         │
│    │      POST /router/config                                │
│    │                                                         │
│    └─ 3. sky serve up (thread 2) ──► SkyServe spot URL       │
│           push URL to router via       ~5min to ready        │
│           POST /router/config                                │
│                                                              │
│  User-facing endpoint: http://<router_ip>:8080               │
└─────────────────────────────────────────────────────────────┘
```

Why this works for MVP:
- Router is up in ~1 min (small CPU VM, no GPU, minimal setup)
- Survives spot preemptions and SkyServe restarts
- Survives serverless cold starts
- Independent lifecycle — `sky down router` is separate from `sky serve down`
- Cost: ~$15/month for a `m5.large` or equivalent

### Router VM SkyPilot YAML (`templates/router.yaml.tpl`)

```yaml
name: {service_name}-router

resources:
  cpus: 2+
  memory: 4+
  use_spot: false            # Router must be reliable, no spot
  ports: 8080
  {region_block}

setup: |
  pip install flask requests gunicorn

run: |
  export SERVERLESS_BASE_URL="{serverless_url}"
  export SKYSERVE_BASE_URL="{spot_url}"
  gunicorn -w 2 -k gthread --threads 8 --timeout 300 \
    --bind 0.0.0.0:8080 meta_lb:app
```

### Dynamic URL Updates

The router needs to learn the serverless and spot URLs **after** it starts, since both backends are deployed in parallel. Two approaches:

**Approach used (simple):** The router exposes a `/router/config` POST endpoint to accept URL updates at runtime. The orchestrator pushes URLs as backends come online.

```python
# In meta_lb.py — new endpoint (added on top of SAM patterns)
@app.route("/router/config", methods=["POST"])
def update_config():
    """Orchestrator pushes backend URLs here after deploy."""
    data = request.get_json()
    if "serverless_url" in data:
        set_serverless_url(data["serverless_url"])
    if "spot_url" in data:
        set_spot_url(data["spot_url"])
    return Response("ok", status=200)
```

This keeps the router stateless at startup — it launches with empty URLs and gets them pushed. No polling, no shared filesystem, no database.

### Future: SkyServe Controller VM (Option B)

When we move to Option B, the router becomes a third process on the SkyServe controller VM. The design is forward-compatible because:

1. **`meta_lb.py` is a standalone Flask app** — it doesn't import SkyPilot or know where it runs. Moving it from a separate VM to the controller VM means changing where it starts, not what it is.

2. **The `/router/config` endpoint works the same** — whether the orchestrator pushes URLs to `http://<router-vm>:8080/router/config` or `http://<controller-vm>:8080/router/config`, the router code is identical.

3. **What needs to change for Option B:**
   - Patch `sky/serve/service.py` to start `meta_lb.py` as a third `multiprocessing.Process` alongside the controller and LB
   - Add port 8080 to the controller task's `ports` list in `sky/serve/server/impl.py`
   - The orchestrator calls the controller VM IP instead of a separate router VM IP
   - Remove the `sky launch router.yaml` step from the orchestrator

4. **What stays the same:**
   - `meta_lb.py` — zero changes
   - `/router/config` push mechanism — zero changes
   - All provider code — zero changes
   - CLI — just stops launching the router VM

The migration is a deployment concern, not a code rewrite.

---

## Orchestrator (`tandemn/orchestrator.py`)

Wires everything together. Three parallel deployments: router VM, serverless, spot.

```python
def launch_hybrid(request: DeployRequest) -> HybridDeployment:
    # 1. Build shared vllm_cmd from template
    vllm_cmd = render_vllm_cmd(request)

    # 2. Launch router VM first (fastest — CPU only, ~1 min)
    router_result = launch_router_vm(request)
    router_url = f"http://{router_result.ip}:8080"

    # 3. Launch both GPU backends in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_serverless = pool.submit(_launch_serverless, request, vllm_cmd)
        fut_spot = pool.submit(_launch_spot, request, vllm_cmd)

        # 4. As each backend comes up, push its URL to the router
        serverless_result = fut_serverless.result(timeout=300)
        if serverless_result and not serverless_result.error:
            push_url_to_router(router_url, serverless_url=serverless_result.endpoint_url)

    # 5. Spot takes longer — push URL to router in background when ready
    def _wait_and_push_spot():
        try:
            spot_result = fut_spot.result(timeout=900)
            if spot_result and not spot_result.error:
                push_url_to_router(router_url, spot_url=spot_result.endpoint_url)
        except Exception:
            pass  # Spot failed — router keeps using serverless only

    threading.Thread(target=_wait_and_push_spot, daemon=True).start()

    return HybridDeployment(
        serverless=serverless_result,
        spot=None,  # Still launching; URL pushed to router when ready
        router_url=router_url,
    )


def push_url_to_router(router_url: str, **urls):
    """POST updated backend URLs to the router's /router/config endpoint."""
    import requests
    requests.post(f"{router_url}/router/config", json=urls, timeout=10)
```

Key decisions:
- **Router launches first** — it's the user-facing endpoint and it's fast (CPU only).
- **Don't block on spot.** Return the router URL as soon as serverless is up.
- **Push, don't poll.** The orchestrator pushes URLs to the router. The router doesn't poll SkyPilot or Modal for endpoint info.
- **Graceful degradation.** If spot never comes up, the router just keeps using serverless. No crash, no error for the user.

---

## CLI (`tandemn/__main__.py`)

Thin. Just parses args and calls orchestrator.

```python
# python -m tandemn deploy --model Qwen/Qwen3-0.6B --gpu L40S
# python -m tandemn destroy --service-name <name>
# python -m tandemn status --service-name <name>

Commands:
  deploy   → Build DeployRequest from args → orchestrator.launch_hybrid()
  destroy  → Tear down both backends + router
  status   → Print health of serverless, spot, router
```

Uses `argparse`. No click/typer dependency for MVP.

---

## Template Engine (`tandemn/template_engine.py`)

One function:

```python
def render_template(template_path: str, replacements: dict[str, str]) -> str:
    """Read template file, replace {key} placeholders, return rendered string."""
    content = Path(template_path).read_text()
    for key, value in replacements.items():
        content = content.replace(f"{{{key}}}", str(value))
    return content
```

That's it. No Jinja2.

---

## How adding a new provider works (e.g., Google Cloud Run)

1. Create `tandemn/providers/cloudrun_provider.py`
2. Implement `ServerlessProvider` ABC (plan, deploy, destroy)
3. Create `templates/cloudrun_vllm_service.yaml.tpl` (Cloud Run YAML)
4. Register: `register("cloudrun", CloudRunProvider)`
5. User runs: `python -m tandemn deploy --model X --serverless-provider cloudrun`

**Nothing else changes.** The orchestrator, router, shared vllm command, and CLI all work without modification. The router just gets a different URL.

This is the same pattern as SkyPilot's cloud provider plugins — each cloud implements a standard interface, the core doesn't know or care which cloud is active.

---

## Build Sequence

| Step | What | Why first |
|------|------|-----------|
| 1 | `models.py` + `template_engine.py` | Foundation — everything else depends on these |
| 2 | `templates/` (all 4 template files) | Need these to test providers |
| 3 | `providers/base.py` + `registry.py` | ABC + registry pattern |
| 4 | `providers/modal_provider.py` | Test end-to-end: render → deploy → get URL |
| 5 | `spot/sky_launcher.py` | Same flow but for SkyServe |
| 6 | `router/meta_lb.py` | Adapt from SAM load_balancer.py, add `/router/config` |
| 7 | `orchestrator.py` | Wire: launch router VM → launch backends → push URLs |
| 8 | `__main__.py` | CLI wrapper |

Each step is testable independently before wiring into the next.

---

## What's explicitly NOT in the MVP

| Feature | Why deferred |
|---------|-------------|
| Price discovery / `find_cheapest` | Hardcoded to one serverless provider per deploy |
| `ProviderCapabilities` | Not needed until multi-provider comparison |
| `CostEstimate` | Same |
| Routing policy engine | SAM's "prefer spot, fallback serverless" is enough |
| Router on SkyServe controller VM (Option B) | MVP uses separate CPU VM; migration path is documented above |
| Multi-model serving | One model per deploy |
| Web dashboard | CLI only |
| Tests | Add after core flow works end-to-end |
