service:
  readiness_probe:
    path: /health
    initial_delay_seconds: 1200
    timeout_seconds: 10
  replica_policy:
    min_replicas: {min_replicas}
    max_replicas: {max_replicas}
    target_qps_per_replica: {target_qps}
    upscale_delay_seconds: {upscale_delay}
    downscale_delay_seconds: {downscale_delay}

resources:
  accelerators: "{gpu}:{gpu_count}"
  use_spot: true
  disk_size: 100
  ports: {port}
{region_block}

setup: |
  pip install vllm=={vllm_version}

run: |
  {vllm_cmd}
