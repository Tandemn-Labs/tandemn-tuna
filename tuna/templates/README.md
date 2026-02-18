# Templates

Tuna uses simple text templates to generate provider-specific deployment configs and scripts. Templates live in subdirectories organized by provider.

## Directory Structure

```
tuna/templates/
├── shared/          Shared across providers
│   ├── vllm_serve_cmd.txt    vLLM command line (rendered by orchestrator)
│   └── router.yaml.tpl       SkyPilot YAML for the router VM
├── modal/           Modal serverless
│   └── vllm_server.py.tpl    Modal app script
├── baseten/         Baseten serverless
│   └── config.yaml.tpl       Truss config YAML
└── skyserve/        SkyServe spot
    └── vllm.yaml.tpl         SkyPilot task YAML
```

## How Templates Work

Templates use `{key}` placeholders that get replaced at render time by `tuna/template_engine.py`. To include a literal brace in the output (e.g. Python dicts), use `{{` and `}}` in the template — these become `{` and `}` after rendering.

Each provider's `plan()` method renders its template with the appropriate variables and returns a `ProviderPlan` containing the rendered output.

## Shared vs Provider-Specific

- **Shared templates** are used by the orchestrator (`tuna/orchestrator.py`) — the vLLM command is rendered and passed to providers (some providers like Baseten use their own inline command instead), and the router YAML is provider-independent.
- **Provider-specific templates** are used by each provider's `plan()` method.

Each subdirectory has a README documenting the template variables, their types, and which code renders them.

## Providers Without Templates

Some providers are purely API/SDK-driven and don't use templates:

- **RunPod** — REST API calls to create templates and endpoints
- **Cloud Run** — `gcloud` CLI commands
