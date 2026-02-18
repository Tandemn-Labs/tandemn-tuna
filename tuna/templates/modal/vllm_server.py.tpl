import os
import subprocess
import time

import modal
import modal.experimental

APP_NAME = "{app_name}"
VLLM_PORT = {port}

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
    .pip_install("vllm=={vllm_version}", "huggingface-hub")
    .env(
        {{
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": HF_CACHE_PATH,
        }}
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="{gpu}",
    timeout={timeout_s},
    scaledown_window={scaledown_window_s},
    volumes={{
        HF_CACHE_PATH: hf_cache_vol,
        VLLM_CACHE_PATH: vllm_cache_vol,
    }},
    enable_memory_snapshot={enable_memory_snapshot},
    {experimental_options_line}
)
@modal.concurrent(max_inputs={max_concurrency})
@modal.web_server(port=VLLM_PORT, startup_timeout={startup_timeout_s})
def serve():
    cmd = """{vllm_cmd}"""
    print("Starting vLLM:", cmd)
    subprocess.Popen(cmd, shell=True)
    _wait_ready()


def _wait_ready(timeout: int = {startup_timeout_s}) -> None:
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"http://127.0.0.1:{{VLLM_PORT}}/health")
            if resp.status_code == 200:
                print("vLLM is healthy")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("vLLM did not become healthy in time")
