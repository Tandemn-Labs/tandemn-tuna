# Contributing to Tuna

## Quick Start

```bash
# Clone and install
git clone https://github.com/tandemn/tuna.git
cd tuna
uv sync --all-extras

# Run tests
uv run pytest tests/ -v
```

## Adding a New Provider

Each provider lives in its own file under `tuna/providers/` (or `tuna/spot/` for spot providers). Follow these steps:

### 1. Implement the `InferenceProvider` interface

Create `tuna/providers/<name>_provider.py` and implement the base class from `tuna/providers/base.py`:

```python
from tuna.providers.base import InferenceProvider
from tuna.providers.registry import register

class MyProvider(InferenceProvider):
    def name(self) -> str:
        return "myprovider"

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        # Render deployment config from the request + vllm_cmd
        ...

    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        # Execute the plan, return endpoint URL or error
        ...

    def destroy(self, result: DeploymentResult) -> None:
        # Tear down the deployment using metadata from result
        ...

register("myprovider", MyProvider)
```

**Provider lifecycle:** `plan()` is a pure function that renders the deployment artifact (script, config, API payload). `deploy()` executes it. `destroy()` tears it down. The orchestrator calls `plan() -> deploy()` on launch and `destroy()` on teardown.

Optional overrides:
- `preflight(request)` — validate environment (API keys, CLI tools, GPU support) before deploy
- `status(service_name)` — check deployment status via provider API
- `vllm_version()` — pin the vLLM version (default: `0.15.1`)
- `auth_token()` — return auth token the router needs when proxying to this backend

### 2. Add GPU mappings to `tuna/catalog.py`

Add your provider's GPU identifier mappings to the catalog. This maps canonical GPU names (T4, L4, A100, etc.) to provider-specific IDs:

```python
"myprovider": {
    "T4": "nvidia-t4",
    "A100": "nvidia-a100-80gb",
}
```

### 3. (Optional) Create a template

If your provider needs a config file or script (like Baseten's Truss config or Modal's app script), create a template directory:

```
tuna/templates/myprovider/
├── config.yaml.tpl    # Your template
└── README.md          # Variable documentation
```

Templates use `{key}` placeholders — see `tuna/templates/README.md` for details.

If your provider is API-driven (like RunPod or Cloud Run), skip this step — build the payload directly in `plan()`.

### 4. Register in `tuna/providers/registry.py`

Add your provider to `PROVIDER_MODULES` for lazy loading:

```python
PROVIDER_MODULES = {
    ...
    "myprovider": ("tuna.providers.myprovider_provider", "MyProvider"),
}
```

If your provider requires extra pip dependencies, add an install hint to `_INSTALL_HINTS`.

### 5. Add preflight checks

Override `preflight()` to validate the user's environment before deploying. Check for:
- Required API keys / CLI tools
- Authentication (is the key valid?)
- GPU availability on this provider

Return a `PreflightResult` with individual `PreflightCheck` items, each with a `fix_command` hint.

### 6. Write tests

Create `tests/test_<name>_provider.py`. Mock all external calls (subprocess, HTTP, SDKs) with `pytest-mock`. Test:
- `plan()` renders correct output
- `deploy()` handles success and failure
- `destroy()` calls the right teardown API
- `preflight()` catches missing credentials

```bash
uv run pytest tests/test_<name>_provider.py -v
```

## Template vs API

**Use a template** when the provider expects a config file or script (Baseten's `config.yaml`, Modal's Python app, SkyPilot's YAML). Store it in `tuna/templates/<provider>/`, render it in `plan()`, and put the rendered output in `ProviderPlan.rendered_script`.

**Use direct API calls** when the provider has a REST API or SDK (RunPod, Cloud Run, Azure). Build the payload in `plan()` and store provider-specific IDs in `ProviderPlan.metadata`.

## Testing

All tests are in `tests/` and use `pytest` + `pytest-mock`.

```bash
# Run all tests
uv run pytest tests/ -v

# Run a single provider's tests
uv run pytest tests/test_modal_provider.py -v

# Run with coverage
uv run pytest tests/ --cov=tuna
```

Key patterns:
- Mock `subprocess.run` for CLI-based providers (Modal, Baseten, Cloud Run)
- Mock `requests` for API-based providers (RunPod, Baseten preflight)
- Mock SDK imports for providers with Python SDKs (Modal, SkyPilot)
- Use `tmp_path` for tests that write temp files

## Reference Providers

- **RunPod** (`tuna/providers/runpod_provider.py`) — simplest API-driven provider, no templates
- **Baseten** (`tuna/providers/baseten_provider.py`) — simplest template-driven provider

## Project Structure

```
tuna/
├── __main__.py          CLI entry point
├── orchestrator.py      Wires router + serverless + spot
├── models.py            Data models (DeployRequest, ProviderPlan, etc.)
├── catalog.py           GPU pricing catalog
├── template_engine.py   {key} placeholder renderer
├── providers/
│   ├── base.py          InferenceProvider ABC
│   ├── registry.py      Provider lookup + lazy loading
│   ├── modal_provider.py
│   ├── runpod_provider.py
│   ├── cloudrun_provider.py
│   └── baseten_provider.py
├── spot/
│   └── sky_launcher.py  SkyServe spot provider
├── templates/           See tuna/templates/README.md
│   ├── shared/
│   ├── modal/
│   ├── baseten/
│   └── skyserve/
└── router/
    └── meta_lb.py       Load balancer (runs on router VM)
```
