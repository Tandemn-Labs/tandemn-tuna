[cerebrium.deployment]
name = "{service_name}"
python_version = "3.11"
docker_base_image_url = "nvidia/cuda:12.1.1-runtime-ubuntu22.04"

[cerebrium.hardware]
region = "{region}"
provider = "aws"
compute = "{gpu_compute}"
gpu_count = {gpu_count}
cpu = {cpu}
memory = {memory}

[cerebrium.scaling]
min_replicas = {min_replicas}
max_replicas = {max_replicas}
cooldown = {cooldown}
replica_concurrency = 1
scaling_metric = "concurrency_utilization"
scaling_target = 100
load_balancing = "min-connections"

[cerebrium.dependencies.pip]
vllm = "=={vllm_version}"

[cerebrium.runtime.custom]
entrypoint = ["vllm", "serve", "{model}", "--host", "0.0.0.0", "--port", "8080", "--max-model-len", "{max_model_len}", "--tensor-parallel-size", "{tp_size}", "--gpu-memory-utilization", "0.95", "--disable-log-requests", "--served-model-name", "{model}"{eager_flag}]
port = 8080
healthcheck_endpoint = "/health"
