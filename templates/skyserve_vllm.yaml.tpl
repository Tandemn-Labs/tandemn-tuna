service:
  readiness_probe:
    path: /health
    initial_delay_seconds: 300
    timeout_seconds: 10
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
  ports: {port}
{region_block}

setup: |
  pip install vllm

run: |
  {vllm_cmd}
