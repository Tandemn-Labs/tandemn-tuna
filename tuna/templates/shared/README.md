# Shared Templates

Templates used by the orchestrator (`tuna/orchestrator.py`), not tied to any single provider.

## `vllm_serve_cmd.txt`

Shell command to start a vLLM server. Rendered by `build_vllm_cmd()` and passed to each provider's `plan()` method (though some providers like Baseten use their own inline command instead).

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `model` | str | HuggingFace model ID | `meta-llama/Llama-3.1-8B-Instruct` |
| `host` | str | Bind address (hardcoded `0.0.0.0`) | `0.0.0.0` |
| `port` | str | vLLM server port | `8001` |
| `max_model_len` | str | vLLM `--max-model-len` value | `4096` |
| `tp_size` | str | vLLM `--tensor-parallel-size` value | `1` |
| `eager_flag` | str | `--enforce-eager` or empty string | `--enforce-eager` |

### Rendered By

`tuna/orchestrator.py` — `build_vllm_cmd()`

## `router.yaml.tpl`

SkyPilot task YAML that launches the meta-LB router on a CPU VM (legacy separate-router-VM mode). Rendered by `_launch_router_vm()`.

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `service_name` | str | Deployment service name (used as cluster name prefix) | `my-service` |
| `serverless_url` | str | Serverless backend URL (initially empty, pushed later via `/router/config`) | `""` |
| `spot_url` | str | Spot backend URL (initially empty, pushed later via `/router/config`) | `""` |
| `meta_lb_local_path` | str | Local filesystem path to `meta_lb.py` (SkyPilot file-mounts it to the VM) | `/home/user/.../tuna/router/meta_lb.py` |
| `region_block` | str | Optional SkyPilot region constraint YAML block, or empty string | `  any_of:\n    - infra: aws/us-east-1` |

### Rendered By

`tuna/orchestrator.py` — `_launch_router_vm()`
