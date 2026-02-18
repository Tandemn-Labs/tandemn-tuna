# SkyServe Templates

`vllm.yaml.tpl` — SkyPilot task YAML for deploying vLLM on spot GPUs via `sky serve`.

## Template Variables

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `gpu` | str | GPU accelerator name | `T4`, `L4`, `A100` |
| `gpu_count` | str | Number of GPUs per replica | `1` |
| `port` | str | vLLM server port | `8001` |
| `vllm_cmd` | str | Full vLLM serve command | `vllm serve meta-llama/...` |
| `vllm_version` | str | vLLM pip version | e.g., `0.15.1` |
| `min_replicas` | str | Minimum number of spot replicas | `1` |
| `max_replicas` | str | Maximum number of spot replicas | `3` |
| `target_qps` | str | Target QPS per replica for autoscaling | `1` |
| `upscale_delay` | str | Seconds QPS must exceed target before scaling up | `300` |
| `downscale_delay` | str | Seconds QPS must be below target before scaling down | `600` |
| `region_block` | str | Optional SkyPilot region constraint YAML block | `  any_of:\n    - infra: aws/us-east-1` |

## Rendered By

`tuna/spot/sky_launcher.py` — `SkyLauncher.plan()`
