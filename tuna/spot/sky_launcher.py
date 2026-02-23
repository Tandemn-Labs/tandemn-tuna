"""SkyServe spot launcher — deploy vLLM on spot GPUs via sky serve."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from sky import ClusterStatus
from sky.serve import ServiceStatus

from tuna.catalog import to_skypilot_gpu_name
from tuna.models import DeployRequest, DeploymentResult, ProviderPlan
from tuna.providers.base import InferenceProvider
from tuna.providers.registry import register
from tuna.sky_sdk import (
    cluster_status,
    serve_down,
    serve_status,
    serve_up,
    task_from_yaml_str,
)
from tuna.template_engine import render_template

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "skyserve"


class SkyLauncher(InferenceProvider):
    """Deploy a vLLM server on spot GPUs via SkyServe."""

    def name(self) -> str:
        return "skyserve"

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        service_name = f"{request.service_name}-spot"
        spot = request.scaling.spot

        # Region block for YAML — always pin the cloud, optionally with region
        cloud = request.spots_cloud.lower()
        if request.region:
            region_block = (
                f"  any_of:\n    - infra: {cloud}/{request.region}"
            )
        else:
            region_block = f"  cloud: {cloud}"

        replacements = {
            "gpu": to_skypilot_gpu_name(request.gpu),
            "gpu_count": str(request.gpu_count),
            "port": "8001",
            "vllm_cmd": vllm_cmd,
            "vllm_version": request.vllm_version,
            "min_replicas": str(spot.min_replicas),
            "max_replicas": str(spot.max_replicas),
            "target_qps": str(spot.target_qps),
            "upscale_delay": str(spot.upscale_delay),
            "downscale_delay": str(spot.downscale_delay),
            "region_block": region_block,
        }

        rendered = render_template(
            str(TEMPLATES_DIR / "vllm.yaml.tpl"), replacements
        )

        return ProviderPlan(
            provider=self.name(),
            rendered_script=rendered,
            metadata={"service_name": service_name},
        )

    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        service_name = plan.metadata["service_name"]

        try:
            logger.info("Launching SkyServe service %s", service_name)
            task = task_from_yaml_str(plan.rendered_script)
            svc_name, endpoint = serve_up(task, service_name)

            if not endpoint:
                logger.warning(
                    "sky serve up succeeded but endpoint not yet available for %s. "
                    "The router will discover it via health checks.",
                    service_name,
                )
                return DeploymentResult(
                    provider=self.name(),
                    error="Endpoint not yet available (still provisioning)",
                    metadata={"service_name": service_name},
                )

            logger.info("SkyServe %s endpoint: %s", service_name, endpoint)
            return DeploymentResult(
                provider=self.name(),
                endpoint_url=endpoint,
                health_url=f"{endpoint}/health",
                metadata={"service_name": service_name},
            )

        except Exception as e:
            logger.error("sky serve up failed: %s", e)
            return DeploymentResult(
                provider=self.name(),
                error=f"sky serve up failed: {e}",
                metadata={"service_name": service_name},
            )

    def destroy(self, result: DeploymentResult) -> None:
        service_name = result.metadata.get("service_name")
        if not service_name:
            logger.warning("No service_name in metadata, cannot destroy")
            return

        logger.info("Tearing down SkyServe service %s", service_name)

        max_attempts = 12  # 12 attempts × 15s = ~3 minutes of retries
        for attempt in range(max_attempts):
            try:
                serve_down(service_name)
            except Exception as e:
                logger.warning("sky serve down failed: %s", e)

            # Verify the service is actually gone
            if self._service_is_gone(service_name):
                return

            logger.warning(
                "Service %s still exists after sky serve down "
                "(attempt %d/%d, controller may still be starting), retrying...",
                service_name, attempt + 1, max_attempts,
            )
            time.sleep(15)

        logger.warning("Could not confirm deletion of %s after retries", service_name)

    def _service_is_gone(self, service_name: str) -> bool:
        """Check whether a SkyServe service has been fully removed."""
        try:
            statuses = serve_status(service_name)
            if not statuses:
                # Empty means gone — unless controller is still booting
                if self._controller_is_init():
                    logger.info(
                        "Controller still INIT — cannot confirm %s is gone, will retry",
                        service_name,
                    )
                    return False
                return True
            if statuses[0].get("status") == ServiceStatus.SHUTTING_DOWN:
                logger.info("Service %s still shutting down, waiting...", service_name)
                return False
            # Service still listed — not gone
            return False
        except Exception as e:
            # "No live services" means the controller has no record of
            # this service — treat the same as an empty status list.
            if "no live services" in str(e).lower():
                if self._controller_is_init():
                    return False
                return True
            # Any other error — controller probably still starting
            return False

    @staticmethod
    def _controller_is_init() -> bool:
        """Check if a SkyServe controller cluster exists in INIT state."""
        try:
            statuses = cluster_status()
            return any(
                "sky-serve-controller" in entry.name
                and entry.status == ClusterStatus.INIT
                for entry in statuses
            )
        except Exception:
            return False

    def status(self, service_name: str) -> dict:
        spot_service = f"{service_name}-spot"
        try:
            statuses = serve_status(spot_service)
            if statuses:
                svc = statuses[0]
                return {
                    "provider": self.name(),
                    "service_name": spot_service,
                    "status": svc["status"].name,
                    "endpoint": svc.get("endpoint"),
                }
            return {
                "provider": self.name(),
                "service_name": spot_service,
                "status": "NOT_FOUND",
            }
        except Exception as e:
            return {"provider": self.name(), "service_name": spot_service, "error": str(e)}


register("skyserve", SkyLauncher)
