model_name: {service_name}

model_metadata:
  example_model_input:
    messages:
      - role: user
        content: "Hello"
    stream: true
    max_tokens: 128

base_image:
  image: vllm/vllm-openai:v{vllm_version}

weights:
  - source: "hf://{model}@main"
    mount_location: "/models/{model_cache_folder}"
    allow_patterns: ["*.safetensors", "*.json"]

docker_server:
  start_command: >-
    vllm serve /models/{model_cache_folder}
    --served-model-name {model}
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
