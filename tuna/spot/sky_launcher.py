"""SkyServe spot launcher — deploy vLLM on spot GPUs via sky serve."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from tuna.models import DeployRequest, DeploymentResult, ProviderPlan
from tuna.providers.base import InferenceProvider
from tuna.providers.registry import register
from tuna.template_engine import render_template

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


class SkyLauncher(InferenceProvider):
    """Deploy a vLLM server on spot GPUs via SkyServe."""

    def name(self) -> str:
        return "skyserve"

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        service_name = f"{request.service_name}-spot"
        spot = request.scaling.spot

        # Region block for YAML — only include if region is specified
        region_block = ""
        if request.region:
            # SkyPilot uses cloud/region format under any_of
            cloud = request.spots_cloud.lower()
            region_block = (
                f"  any_of:\n    - infra: {cloud}/{request.region}"
            )

        replacements = {
            "gpu": request.gpu,
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
            str(TEMPLATES_DIR / "skyserve_vllm.yaml.tpl"), replacements
        )

        return ProviderPlan(
            provider=self.name(),
            rendered_script=rendered,
            metadata={"service_name": service_name},
        )

    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        service_name = plan.metadata["service_name"]

        # Write rendered YAML to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="tuna_sky_"
        ) as f:
            f.write(plan.rendered_script)
            yaml_path = f.name

        try:
            logger.info(
                "Launching SkyServe service %s from %s", service_name, yaml_path
            )

            result = subprocess.run(
                [
                    "sky", "serve", "up", yaml_path,
                    "--service-name", service_name,
                    "-y",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error("sky serve up failed: %s", result.stderr)
                return DeploymentResult(
                    provider=self.name(),
                    error=f"sky serve up failed: {result.stderr}",
                    metadata={"service_name": service_name},
                )

            # Poll for the endpoint URL
            endpoint = self._poll_endpoint(service_name)

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

        finally:
            Path(yaml_path).unlink(missing_ok=True)

    def destroy(self, result: DeploymentResult) -> None:
        service_name = result.metadata.get("service_name")
        if not service_name:
            logger.warning("No service_name in metadata, cannot destroy")
            return

        logger.info("Tearing down SkyServe service %s", service_name)

        for attempt in range(6):
            subprocess.run(
                ["sky", "serve", "down", service_name, "-y"],
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Verify the service is actually gone
            if self._service_is_gone(service_name):
                return

            logger.warning(
                "Service %s still exists after sky serve down "
                "(attempt %d/6, controller may still be starting), retrying...",
                service_name, attempt + 1,
            )
            time.sleep(10)

        logger.warning("Could not confirm deletion of %s after retries", service_name)

    def _service_is_gone(self, service_name: str) -> bool:
        """Check whether a SkyServe service has been fully removed."""
        try:
            check = subprocess.run(
                ["sky", "serve", "status", service_name],
                capture_output=True, text=True, timeout=30,
            )
            output = check.stdout + check.stderr
            # Gone if sky says no services or doesn't mention it at all
            if "No existing services" in output:
                return True
            if service_name not in check.stdout:
                return True
            # Still shutting down — not gone yet
            if "SHUTTING_DOWN" in check.stdout:
                logger.info("Service %s still shutting down, waiting...", service_name)
                return False
            # Service still listed — not gone
            return False
        except Exception:
            # Can't reach sky CLI — controller probably still INIT
            return False

    def status(self, service_name: str) -> dict:
        spot_service = f"{service_name}-spot"
        try:
            result = subprocess.run(
                ["sky", "serve", "status", spot_service],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {"provider": self.name(), "service_name": spot_service, "raw": result.stdout.strip()}
        except Exception as e:
            return {"provider": self.name(), "service_name": spot_service, "error": str(e)}

    def _poll_endpoint(
        self,
        service_name: str,
        max_attempts: int = 10,
        delay: float = 15.0,
    ) -> str | None:
        """Poll `sky serve status` until endpoint is available."""
        for attempt in range(max_attempts):
            try:
                result = subprocess.run(
                    ["sky", "serve", "status", service_name, "--endpoint"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                endpoint = result.stdout.strip()
                # sky serve status --endpoint returns just the URL or empty
                if endpoint and endpoint.startswith("http"):
                    return endpoint
            except Exception as e:
                logger.debug(
                    "Endpoint poll attempt %d/%d failed: %s",
                    attempt + 1,
                    max_attempts,
                    e,
                )

            if attempt < max_attempts - 1:
                time.sleep(delay)

        return None


register("skyserve", SkyLauncher)
