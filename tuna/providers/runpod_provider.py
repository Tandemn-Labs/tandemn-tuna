"""RunPod serverless provider — deploy vLLM on RunPod Serverless GPUs."""

from __future__ import annotations

import logging
import os

import requests

from tuna.catalog import provider_gpu_id, provider_gpu_map
from tuna.models import DeployRequest, DeploymentResult, PreflightCheck, PreflightResult, ProviderPlan
from tuna.providers.base import InferenceProvider
from tuna.providers.registry import register

logger = logging.getLogger(__name__)

_API_BASE = "https://rest.runpod.io/v1"
_RUNPOD_VLLM_FALLBACK = "0.11.0"


def _fetch_runpod_vllm_version() -> str:
    """Fetch vLLM version from RunPod's worker-vllm Dockerfile on GitHub.

    Falls back to hardcoded version if GitHub is unreachable or parsing fails.
    """
    url = "https://raw.githubusercontent.com/runpod-workers/worker-vllm/main/Dockerfile"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            if "vllm==" in line:
                return line.split("vllm==")[1].split()[0].rstrip('"').rstrip("'")
    except Exception:
        logger.debug("Could not fetch RunPod vLLM version from GitHub, using fallback")
    return _RUNPOD_VLLM_FALLBACK


def _headers() -> dict[str, str]:
    """Return auth headers for the RunPod REST API."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RUNPOD_API_KEY environment variable is not set. "
            "Get your API key from https://www.runpod.io/console/user/settings"
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


class RunPodProvider(InferenceProvider):
    """Deploy a vLLM server on RunPod Serverless GPUs via REST API."""

    def name(self) -> str:
        return "runpod"

    def vllm_version(self) -> str:
        return _fetch_runpod_vllm_version()

    def auth_token(self) -> str:
        return os.environ.get("RUNPOD_API_KEY", "")

    def preflight(self, request: DeployRequest) -> PreflightResult:
        result = PreflightResult(provider=self.name())

        # 1. Check RUNPOD_API_KEY is set
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            result.checks.append(PreflightCheck(
                name="api_key",
                passed=False,
                message="RUNPOD_API_KEY environment variable is not set",
                fix_command="export RUNPOD_API_KEY=<your-key>  # https://www.runpod.io/console/user/settings",
            ))
            return result

        result.checks.append(PreflightCheck(
            name="api_key",
            passed=True,
            message="RUNPOD_API_KEY is set",
        ))

        # 2. Validate the key works
        try:
            resp = requests.get(
                f"{_API_BASE}/endpoints",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 401:
                result.checks.append(PreflightCheck(
                    name="api_key_valid",
                    passed=False,
                    message="RUNPOD_API_KEY is invalid (401 Unauthorized)",
                    fix_command="export RUNPOD_API_KEY=<your-key>  # https://www.runpod.io/console/user/settings",
                ))
            else:
                resp.raise_for_status()
                result.checks.append(PreflightCheck(
                    name="api_key_valid",
                    passed=True,
                    message="RUNPOD_API_KEY is valid",
                ))
        except requests.exceptions.ConnectionError:
            result.checks.append(PreflightCheck(
                name="api_key_valid",
                passed=False,
                message="Could not reach RunPod API (connection error)",
            ))
        except Exception as e:
            result.checks.append(PreflightCheck(
                name="api_key_valid",
                passed=False,
                message=f"RunPod API check failed: {e}",
            ))

        return result

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        endpoint_name = f"{request.service_name}-serverless"

        # Map short GPU name to RunPod's full identifier
        try:
            gpu_type_id = provider_gpu_id(request.gpu, "runpod")
        except KeyError:
            raise ValueError(
                f"Unknown GPU type for RunPod: {request.gpu!r}. "
                f"Supported: {sorted(provider_gpu_map('runpod').keys())}"
            )

        serverless = request.scaling.serverless

        # Build vLLM worker environment variables
        env: dict[str, str] = {
            "MODEL_NAME": request.model_name,
            "MAX_MODEL_LEN": str(request.max_model_len),
            "TENSOR_PARALLEL_SIZE": str(request.tp_size),
            "GPU_MEMORY_UTILIZATION": "0.95",
            "MAX_CONCURRENCY": str(serverless.concurrency),
            "DISABLE_LOG_REQUESTS": "true",
        }

        if request.cold_start_mode == "fast_boot":
            env["ENFORCE_EAGER"] = "true"

        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env["HF_TOKEN"] = hf_token

        metadata = {
            "endpoint_name": endpoint_name,
            "image_name": "runpod/worker-v1-vllm:v2.11.3",
            "gpu_type_id": gpu_type_id,
            "gpu_count": str(request.gpu_count),
            "workers_min": str(serverless.workers_min),
            "workers_max": str(serverless.workers_max),
            "idle_timeout": str(serverless.scaledown_window),
            "execution_timeout_ms": str(serverless.timeout * 1000),
            "flashboot": "true" if request.cold_start_mode == "fast_boot" else "false",
            "scaler_value": str(serverless.scaler_value),
        }

        return ProviderPlan(
            provider=self.name(),
            rendered_script="",
            env=env,
            metadata=metadata,
        )

    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        endpoint_name = plan.metadata["endpoint_name"]

        try:
            headers = _headers()
        except RuntimeError as e:
            return DeploymentResult(
                provider=self.name(),
                error=str(e),
                metadata={"endpoint_name": endpoint_name},
            )

        # Step 1: Create template
        template_payload = {
            "name": endpoint_name,
            "imageName": plan.metadata["image_name"],
            "containerDiskInGb": 50,
            "env": plan.env,
            "isServerless": True,
        }

        logger.info("Creating RunPod template: %s", endpoint_name)
        try:
            resp = requests.post(
                f"{_API_BASE}/templates",
                headers=headers,
                json=template_payload,
                timeout=30,
            )
            resp.raise_for_status()
            template_data = resp.json()
            template_id = template_data["id"]
        except Exception as e:
            logger.error("RunPod template creation failed: %s", e)
            return DeploymentResult(
                provider=self.name(),
                error=f"Template creation failed: {e}",
                metadata={"endpoint_name": endpoint_name},
            )

        # Step 2: Create endpoint
        endpoint_payload = {
            "name": endpoint_name,
            "templateId": template_id,
            "gpuTypeIds": [plan.metadata["gpu_type_id"]],
            "gpuCount": int(plan.metadata["gpu_count"]),
            "workersMin": int(plan.metadata["workers_min"]),
            "workersMax": int(plan.metadata["workers_max"]),
            "idleTimeout": int(plan.metadata["idle_timeout"]),
            "executionTimeoutMs": int(plan.metadata["execution_timeout_ms"]),
            "flashboot": plan.metadata["flashboot"] == "true",
            "scalerType": "QUEUE_DELAY",
            "scalerValue": int(plan.metadata["scaler_value"]),
        }

        logger.info("Creating RunPod endpoint: %s", endpoint_name)
        try:
            resp = requests.post(
                f"{_API_BASE}/endpoints",
                headers=headers,
                json=endpoint_payload,
                timeout=30,
            )
            resp.raise_for_status()
            endpoint_data = resp.json()
            endpoint_id = endpoint_data["id"]
        except Exception as e:
            logger.error("RunPod endpoint creation failed: %s", e)
            # Clean up the template we just created
            logger.info("Cleaning up template %s after endpoint failure", template_id)
            try:
                requests.delete(
                    f"{_API_BASE}/templates/{template_id}",
                    headers=headers,
                    timeout=30,
                )
            except Exception:
                logger.warning("Failed to clean up template %s", template_id)
            return DeploymentResult(
                provider=self.name(),
                error=f"Endpoint creation failed: {e}",
                metadata={
                    "endpoint_name": endpoint_name,
                    "template_id": template_id,
                },
            )

        endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
        health_url = f"https://api.runpod.ai/v2/{endpoint_id}/health"

        logger.info("RunPod endpoint %s deployed at %s", endpoint_name, endpoint_url)
        return DeploymentResult(
            provider=self.name(),
            endpoint_url=endpoint_url,
            health_url=health_url,
            metadata={
                "endpoint_id": endpoint_id,
                "template_id": template_id,
                "endpoint_name": endpoint_name,
            },
        )

    def destroy(self, result: DeploymentResult) -> None:
        endpoint_id = result.metadata.get("endpoint_id")
        template_id = result.metadata.get("template_id")

        try:
            headers = _headers()
        except RuntimeError as e:
            logger.error("Cannot destroy RunPod resources: %s", e)
            return

        if endpoint_id:
            logger.info("Deleting RunPod endpoint %s", endpoint_id)
            try:
                requests.delete(
                    f"{_API_BASE}/endpoints/{endpoint_id}",
                    headers=headers,
                    timeout=30,
                )
            except Exception as e:
                logger.warning("Failed to delete endpoint %s: %s", endpoint_id, e)
        else:
            logger.warning("No endpoint_id in metadata, skipping endpoint deletion")

        if template_id:
            logger.info("Deleting RunPod template %s", template_id)
            try:
                requests.delete(
                    f"{_API_BASE}/templates/{template_id}",
                    headers=headers,
                    timeout=30,
                )
            except Exception as e:
                logger.warning("Failed to delete template %s: %s", template_id, e)
        else:
            logger.warning("No template_id in metadata, skipping template deletion")

    def status(self, service_name: str) -> dict:
        endpoint_name = f"{service_name}-serverless"

        try:
            headers = _headers()
        except RuntimeError:
            return {"provider": self.name(), "status": "unknown", "error": "RUNPOD_API_KEY not set"}

        # List endpoints to find by name
        try:
            resp = requests.get(
                f"{_API_BASE}/endpoints",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            endpoints = resp.json()
        except Exception as e:
            return {"provider": self.name(), "status": "unknown", "error": str(e)}

        # Find our endpoint by name (RunPod may append " -fb" for flashboot)
        endpoint_id = None
        for ep in endpoints:
            name = ep.get("name", "")
            if name == endpoint_name or name == f"{endpoint_name} -fb":
                endpoint_id = ep.get("id")
                break

        if not endpoint_id:
            return {"provider": self.name(), "endpoint_name": endpoint_name, "status": "not found"}

        # Get detailed status with worker info
        try:
            resp = requests.get(
                f"{_API_BASE}/endpoints/{endpoint_id}?includeWorkers=true",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            result = {
                "provider": self.name(),
                "endpoint_name": endpoint_name,
                "endpoint_id": endpoint_id,
                "status": "running",
                "workers": data.get("workers", {}),
            }
            if data.get("templateId"):
                result["template_id"] = data["templateId"]
            return result
        except Exception as e:
            return {
                "provider": self.name(),
                "endpoint_name": endpoint_name,
                "endpoint_id": endpoint_id,
                "status": "unknown",
                "error": str(e),
            }


register("runpod", RunPodProvider)
