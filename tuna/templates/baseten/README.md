# Baseten Templates

`config.yaml.tpl` — Truss config file for deploying a vLLM model on Baseten.

## Template Variables

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `service_name` | str | Baseten model name | `my-service-serverless` |
| `model` | str | HuggingFace model ID | `meta-llama/Llama-3.1-8B-Instruct` |
| `max_model_len` | str | Max sequence length | `4096` |
| `tp_size` | str | Tensor parallel size | `1` |
| `gpu` | str | Baseten GPU accelerator ID | `A100` |
| `concurrency` | str | Max concurrent requests per replica | `32` |
| `eager_flag` | str | `--enforce-eager` or empty | `--enforce-eager` |
| `vllm_version` | str | vLLM image tag version | e.g., `0.15.1` |
| `model_cache_folder` | str | Volume folder for model weights (slashes replaced with `--`) | `meta-llama--Llama-3.1-8B-Instruct` |

## Rendered By

`tuna/providers/baseten_provider.py` — `BasetenProvider.plan()`
