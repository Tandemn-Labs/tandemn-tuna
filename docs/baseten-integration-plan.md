# Integrate Baseten as a Serverless Provider

## Context
Adding Baseten as the 4th serverless provider. It has competitive GPU pricing, OpenAI-compatible endpoints, vLLM support via Docker containers (using `vllm/vllm-openai:v0.15.1`, matching Cloud Run), and per-minute billing. This follows the existing provider pattern exactly — new file, register, add catalog prices.

## vLLM version note
Baseten will use `vllm/vllm-openai:v0.15.1` (same as Cloud Run). RunPod is stuck on v0.11.0 via `runpod/worker-v1-vllm:v2.11.3` — unifying that requires forking their worker repo. Addressed separately, not in this PR.

## Key Baseten facts from research

- **Deploy**: `truss push <dir> --publish` (CLI, like Modal's `modal deploy`)
- **Destroy**: REST API `DELETE https://api.baseten.co/v1/models/{model_id}` (no CLI destroy command)
- **Status**: REST API `GET https://api.baseten.co/v1/models/{model_id}`
- **Inference**: OpenAI-compatible at `https://model-{modelID}.api.baseten.co/environments/production/sync/v1`
- **Auth**: `BASETEN_API_KEY` env var, passed as `Authorization: Api-Key <key>` header
- **HF models**: Needs `hf_access_token` secret in Baseten workspace for gated models
- **vLLM config**: Uses `config.yaml` with `docker_server` block pointing to `vllm/vllm-openai` image
- **Cold start optimizations**: BDN (Baseten Delivery Network) caches model weights near pods, Image Streaming prefetches critical layers, Coldboost pre-warms pods
- **Autoscaling**: min/max replicas, concurrency target, scale-down delay (default 900s)
- **GPUs**: T4 ($0.63/hr), L4 ($0.85/hr), A10G ($1.21/hr), A100 ($4.00/hr), H100 ($6.50/hr), H100MIG ($3.75/hr), B200 ($9.98/hr)

## Files to create/modify

| File | Action |
|------|--------|
| `tuna/providers/baseten_provider.py` | **Create** — new provider (~180 lines) |
| `templates/baseten_config.yaml.tpl` | **Create** — Truss config template |
| `tuna/providers/registry.py` | **Modify** — add to `PROVIDER_MODULES` + `_INSTALL_HINTS` |
| `tuna/catalog.py` | **Modify** — add 6 Baseten GPU pricing entries |
| `pyproject.toml` | **Modify** — add `baseten` optional dep group |
| `tuna/__main__.py` | **Modify** — update `--serverless-provider` help text |
| `tests/test_baseten_provider.py` | **Create** — unit tests for `plan()` |

## Step 1: Create `templates/baseten_config.yaml.tpl`

Template rendered by `tuna.template_engine.render_template()` using `{key}` placeholders (single-brace). No `{{`/`}}` needed since YAML has no literal braces.

```yaml
model_metadata:
  example_model_input:
    messages:
      - role: user
        content: "Hello"
    stream: true
    max_tokens: 128

base_image:
  image: vllm/vllm-openai:v0.15.1

docker_server:
  start_command: >-
    vllm serve {model}
    --host 0.0.0.0
    --port 8000
    --max-model-len {max_model_len}
    --tensor-parallel-size {tp_size}
    --gpu-memory-utilization 0.95
    --disable-log-requests
    {eager_flag}
  readiness_endpoint: /health
  liveness_endpoint: /health
  predict_endpoint: /v1/chat/completions
  server_port: 8000

runtime:
  predict_concurrency: {concurrency}

resources:
  accelerator: {gpu}
  use_gpu: true
```

Note: `{eager_flag}` will be `--enforce-eager` for `fast_boot` mode, empty string otherwise (same pattern as `orchestrator.py:32`).

## Step 2: Create `tuna/providers/baseten_provider.py`

Follows the **exact** pattern of existing providers. Key design choices:

- **`plan()`**: Like `ModalProvider.plan()` — uses `render_template()` from `tuna.template_engine`, maps GPU via `provider_gpu_id()` from `tuna.catalog`
- **`deploy()`**: Like `ModalProvider.deploy()` — writes rendered config to temp dir, runs `truss push <dir> --publish` via subprocess, parses model_id from stdout
- **`destroy()`**: Like `RunPodProvider.destroy()` — uses `requests.delete()` to REST API `DELETE https://api.baseten.co/v1/models/{model_id}` with `Api-Key` auth header
- **`status()`**: Like `RunPodProvider.status()` — uses `requests.get()` to REST API
- **`preflight()`**: Like `CloudRunProvider.preflight()` — checks `BASETEN_API_KEY` env var + `truss` CLI installed

```python
class BasetenProvider(InferenceProvider):
    def name(self) -> str:
        return "baseten"

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        service_name = f"{request.service_name}-serverless"
        gpu_accelerator = provider_gpu_id(request.gpu, "baseten")  # raises KeyError
        eager_flag = "--enforce-eager" if request.cold_start_mode == "fast_boot" else ""
        serverless = request.scaling.serverless

        replacements = {
            "model": request.model_name,
            "max_model_len": str(request.max_model_len),
            "tp_size": str(request.tp_size),
            "gpu": gpu_accelerator,
            "concurrency": str(serverless.concurrency),
            "eager_flag": eager_flag,
        }
        rendered = render_template(str(TEMPLATES_DIR / "baseten_config.yaml.tpl"), replacements)

        metadata = {
            "service_name": service_name,
            "model_name": request.model_name,
        }
        return ProviderPlan(provider=self.name(), rendered_script=rendered, env={}, metadata=metadata)

    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        # 1. Create temp dir with config.yaml (truss expects a directory)
        # 2. subprocess.run(["truss", "push", tmpdir, "--publish"], ...)
        # 3. Parse model_id from stdout
        # 4. Construct endpoint URL + health URL
        # 5. Return DeploymentResult

    def destroy(self, result: DeploymentResult) -> None:
        # requests.delete("https://api.baseten.co/v1/models/{model_id}",
        #                 headers={"Authorization": f"Api-Key {api_key}"})

    def status(self, service_name: str) -> dict:
        # requests.get("https://api.baseten.co/v1/models/{model_id}", ...)

    def preflight(self, request: DeployRequest) -> PreflightResult:
        # Check: BASETEN_API_KEY env var
        # Check: truss CLI installed (shutil.which("truss"))
```

**Key details:**
- `TEMPLATES_DIR` = `Path(__file__).resolve().parent.parent.parent / "templates"` (same as `modal_provider.py:19`)
- `_API_BASE = "https://api.baseten.co/v1"` (same pattern as `runpod_provider.py:17`)
- `_headers()` returns `{"Authorization": f"Api-Key {api_key}"}` from `BASETEN_API_KEY` env var
- Endpoint URL: `https://model-{model_id}.api.baseten.co/environments/production/sync/v1`
- Health URL: `{endpoint_url}/health`
- `metadata` stores `model_id` + `service_name` for destroy/status
- Ends with `register("baseten", BasetenProvider)` (same as all other providers)

## Step 3: Add to registry (`tuna/providers/registry.py`)

Add to `PROVIDER_MODULES` dict:
```python
"baseten": ("tuna.providers.baseten_provider", "BasetenProvider"),
```

Add to `_INSTALL_HINTS` dict:
```python
"baseten": "pip install tandemn-tuna[baseten]",
```

## Step 4: Add GPU pricing to catalog (`tuna/catalog.py`)

Add to `_PROVIDER_GPUS` list after the Cloud Run entries:
```python
# Baseten
ProviderGpu("T4", "baseten", "T4", 0.63),
ProviderGpu("L4", "baseten", "L4", 0.85),
ProviderGpu("A10G", "baseten", "A10G", 1.21),
ProviderGpu("A100_80GB", "baseten", "A100", 4.00),
ProviderGpu("H100", "baseten", "H100", 6.50),
ProviderGpu("B200", "baseten", "B200", 9.98),
```

## Step 5: Add optional dependency (`pyproject.toml`)

```toml
baseten = ["truss>=0.9"]

# Update 'all' group:
all = [
    "google-cloud-run>=0.10.0",
    "modal>=0.73",
    "truss>=0.9",
]
```

## Step 6: Update CLI help text (`tuna/__main__.py`)

```python
help="Serverless backend: modal, runpod, cloudrun, baseten (default: cheapest for GPU)"
```

## Step 7: Tests (`tests/test_baseten_provider.py`)

Follow `test_modal_provider.py` pattern — test `plan()` renders template correctly, validates GPU mapping, checks fast_boot/no_fast_boot, and verifies YAML validity.

## Verification

1. `uv run pytest tests/test_baseten_provider.py -v` — unit tests pass
2. `uv run pytest tests/ -v` — all existing tests still pass
3. `uv run python -m tuna show-gpus` — Baseten prices appear in the table
4. `uv run python -m tuna show-gpus --gpu L4` — shows Baseten L4 pricing
5. `uv run python -m tuna deploy --help` — help text shows baseten as an option

## RunPod custom image research (for future vLLM unification)

- RunPod supports custom Docker images via `imageName` in templates (already used at `runpod_provider.py:73`)
- But RunPod serverless requires images to include the RunPod SDK handler (`runpod.serverless.start`)
- Can't use raw `vllm/vllm-openai` — need RunPod's handler wrapper
- `runpod/worker-v1-vllm:v2.11.3` Dockerfile pins `vllm==0.11.0`
- To unify: fork `runpod-workers/worker-vllm`, update vLLM pin, build custom image
- Current vLLM versions: Cloud Run=v0.15.1, RunPod=v0.11.0, Modal=latest, SkyPilot=latest
