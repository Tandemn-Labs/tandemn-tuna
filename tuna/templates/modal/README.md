# Modal Templates

`vllm_server.py.tpl` — Modal app script that deploys a vLLM server as a serverless GPU function.

## Template Variables

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `app_name` | str | Modal app name | `my-service-serverless` |
| `gpu` | str | Modal GPU identifier | `T4`, `A100` |
| `port` | str | vLLM server port (always 8000 on Modal) | `8000` |
| `vllm_cmd` | str | Full vLLM serve command | `vllm serve meta-llama/...` |
| `vllm_version` | str | vLLM pip version | e.g., `0.15.1` |
| `max_concurrency` | str | Max concurrent inputs per container (burst limit for `@modal.concurrent`) | `32` |
| `timeout_s` | str | Max execution time per input in seconds | `3600` |
| `scaledown_window_s` | str | Max idle time (seconds) a container stays warm before scaling down | `60` |
| `startup_timeout_s` | str | Max seconds to wait for vLLM startup | `600` |
| `enable_memory_snapshot` | str | Enable memory snapshot for fast cold starts | `True` / `False` |
| `experimental_options_line` | str | Optional GPU snapshot line (empty if disabled) | `experimental_options={"enable_gpu_snapshot": True},` |

## Rendered By

`tuna/providers/modal_provider.py` — `ModalProvider.plan()`
