import os
import subprocess
import time

import modal
import modal.experimental 

APP_NAME = "qwen-0p6b-vllm-serverless"

MODEL_NAME = os.getenv("MODEL_ID", "Qwen/Qwen3-0.6B")
GPU_TYPE = os.getenv("GPU_TYPE", "L40s")
FAST_BOOT = os.getenv("FAST_BOOT", "true").lower() == "true"
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))

VLLM_PORT = 8000

HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "build-essential",
        "libxcb1",
        "libx11-6",
        "libxext6",
        "libxrender1",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
    )
    .pip_install("vllm", "huggingface-hub")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": HF_CACHE_PATH,
        }
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=600,
    scaledown_window=60,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        VLLM_CACHE_PATH: vllm_cache_vol,
    },
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=600)
def serve():
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--served-model-name",
        "llm",
        "--disable-log-requests",
        "--uvicorn-log-level",
        "info",
    ]

    # For scale-to-zero / cold starts, eager mode avoids torch.compile + triton JIT.
    # You can turn this off for higher throughput once warm.
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # Optional: tensor parallel size if you use multi-GPU
    cmd += ["--tensor-parallel-size", "1"]

    print("Starting vLLM:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)

    _wait_ready()


def _wait_ready(timeout: int = 5 * 60) -> None:
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"http://127.0.0.1:{VLLM_PORT}/health")
            if resp.status_code == 200:
                print("vLLM is healthy")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("vLLM did not become healthy in time")


@app.local_entrypoint()
def test():
    import requests

    url = serve.get_web_url()
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "Hello from Modal!"}],
        "max_tokens": 64,
    }
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    print(r.json())