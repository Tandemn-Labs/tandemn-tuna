"""Cerebrium serverless provider — deploy vLLM on Cerebrium GPUs."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from tuna.catalog import provider_gpu_id, provider_gpu_map
from tuna.models import (
    DeployRequest,
    DeploymentResult,
    PreflightCheck,
    PreflightResult,
    ProviderPlan,
)
from tuna.providers.base import InferenceProvider
from tuna.providers.registry import register
from tuna.template_engine import render_template

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "cerebrium"
DEFAULT_REGION = "us-east-1"
CEREBRIUM_API_BASE = "https://rest.cerebrium.ai/v2"

# Recommended CPU/memory per GPU type (Cerebrium bundles compute resources).
_GPU_RESOURCES: dict[str, dict[str, int]] = {
    "TURING_T4": {"cpu": 4, "memory": 16},
    "ADA_L4": {"cpu": 4, "memory": 16},
    "AMPERE_A10": {"cpu": 8, "memory": 32},
    "ADA_L40": {"cpu": 8, "memory": 32},
    "AMPERE_A100_40GB": {"cpu": 8, "memory": 64},
    "AMPERE_A100_80GB": {"cpu": 12, "memory": 64},
    "HOPPER_H100": {"cpu": 12, "memory": 64},
}


def _get_project_id() -> str | None:
    """Read Cerebrium project ID from CLI config (~/.cerebrium/config.yaml)."""
    config_path = Path.home() / ".cerebrium" / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text())
        return data.get("project") or data.get("project_id") or data.get("projectId")
    except Exception:
        pass
    # Fallback: simple line parsing
    try:
        for line in config_path.read_text().splitlines():
            for key in ("project:", "project_id:", "projectId:"):
                if line.strip().startswith(key):
                    return line.split(key, 1)[1].strip().strip("'\"")
    except Exception:
        pass
    return None


class CerebriumProvider(InferenceProvider):
    """Deploy a vLLM server on Cerebrium's serverless GPUs."""

    def name(self) -> str:
        return "cerebrium"

    def vllm_version(self) -> str:
        return "0.15.1"

    def auth_token(self) -> str:
        return os.environ.get("CEREBRIUM_API_KEY", "")

    # -- Preflight checks --------------------------------------------------

    def preflight(self, request: DeployRequest) -> PreflightResult:
        result = PreflightResult(provider=self.name())

        # 1. API key set
        api_key = os.environ.get("CEREBRIUM_API_KEY", "")
        if not api_key:
            result.checks.append(PreflightCheck(
                name="api_key",
                passed=False,
                message="CEREBRIUM_API_KEY environment variable not set",
                fix_command="export CEREBRIUM_API_KEY=<your-service-account-token>",
            ))
            return result
        result.checks.append(PreflightCheck(
            name="api_key",
            passed=True,
            message="CEREBRIUM_API_KEY is set",
        ))

        # 2. CLI installed
        if not shutil.which("cerebrium"):
            result.checks.append(PreflightCheck(
                name="cli_installed",
                passed=False,
                message="cerebrium CLI not found on PATH",
                fix_command="pip install cerebrium",
            ))
            return result
        result.checks.append(PreflightCheck(
            name="cli_installed",
            passed=True,
            message="cerebrium CLI found",
        ))

        # 3. CLI authenticated (cerebrium status should succeed)
        try:
            proc = subprocess.run(
                ["cerebrium", "status"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode != 0:
                result.checks.append(PreflightCheck(
                    name="cli_authenticated",
                    passed=False,
                    message="cerebrium CLI not authenticated",
                    fix_command="cerebrium login",
                ))
                return result
        except (subprocess.TimeoutExpired, FileNotFoundError):
            result.checks.append(PreflightCheck(
                name="cli_authenticated",
                passed=False,
                message="Failed to run cerebrium status",
                fix_command="cerebrium login",
            ))
            return result
        result.checks.append(PreflightCheck(
            name="cli_authenticated",
            passed=True,
            message="cerebrium CLI authenticated",
        ))

        # 4. GPU type supported
        try:
            provider_gpu_id(request.gpu, "cerebrium")
            result.checks.append(PreflightCheck(
                name="gpu_supported",
                passed=True,
                message=f"GPU {request.gpu} is supported on Cerebrium",
            ))
        except KeyError:
            supported = sorted(provider_gpu_map("cerebrium").keys())
            result.checks.append(PreflightCheck(
                name="gpu_supported",
                passed=False,
                message=f"GPU {request.gpu!r} not supported on Cerebrium. Supported: {supported}",
            ))

        return result

    # -- Plan ---------------------------------------------------------------

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        try:
            gpu_compute = provider_gpu_id(request.gpu, "cerebrium")
        except KeyError:
            raise ValueError(
                f"Unknown GPU type for Cerebrium: {request.gpu!r}. "
                f"Supported: {sorted(provider_gpu_map('cerebrium').keys())}"
            )

        service_name = f"{request.service_name}-serverless"
        region = request.region or DEFAULT_REGION
        serverless = request.scaling.serverless

        resources = _GPU_RESOURCES.get(gpu_compute, {"cpu": 4, "memory": 16})

        eager_flag = ""
        if request.cold_start_mode == "fast_boot":
            eager_flag = ', "--enforce-eager"'

        replacements = {
            "service_name": service_name,
            "region": region,
            "gpu_compute": gpu_compute,
            "gpu_count": str(request.gpu_count),
            "cpu": str(resources["cpu"]),
            "memory": str(resources["memory"]),
            "min_replicas": str(serverless.workers_min),
            "max_replicas": str(serverless.workers_max),
            "cooldown": str(serverless.scaledown_window),
            "vllm_version": request.vllm_version,
            "model": request.model_name,
            "max_model_len": str(request.max_model_len),
            "tp_size": str(request.tp_size),
            "eager_flag": eager_flag,
        }

        rendered = render_template(
            str(TEMPLATES_DIR / "cerebrium.toml.tpl"), replacements
        )

        project_id = _get_project_id() or ""

        metadata: dict[str, str] = {
            "service_name": service_name,
            "region": region,
            "project_id": project_id,
            "gpu_compute": gpu_compute,
        }

        return ProviderPlan(
            provider=self.name(),
            rendered_script=rendered,
            env={},
            metadata=metadata,
        )

    # -- Deploy -------------------------------------------------------------

    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        api_key = os.environ.get("CEREBRIUM_API_KEY", "")
        if not api_key:
            return DeploymentResult(
                provider=self.name(),
                error="CEREBRIUM_API_KEY environment variable not set",
                metadata=dict(plan.metadata),
            )

        service_name = plan.metadata["service_name"]
        region = plan.metadata["region"]

        with tempfile.TemporaryDirectory(prefix="tuna_cerebrium_") as tmpdir:
            # Write cerebrium.toml
            toml_path = Path(tmpdir) / "cerebrium.toml"
            toml_path.write_text(plan.rendered_script)

            # Write minimal main.py placeholder (Cerebrium may require it)
            main_path = Path(tmpdir) / "main.py"
            main_path.write_text("# Placeholder — vLLM runs via custom runtime entrypoint\n")

            logger.info("Deploying Cerebrium app %s from %s", service_name, tmpdir)

            try:
                proc = subprocess.run(
                    ["cerebrium", "deploy", "-y", "--no-color"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except subprocess.TimeoutExpired:
                return DeploymentResult(
                    provider=self.name(),
                    error="cerebrium deploy timed out after 600s",
                    metadata=dict(plan.metadata),
                )

            if proc.returncode != 0:
                error_detail = proc.stderr or proc.stdout
                logger.error("cerebrium deploy failed: %s", error_detail)
                return DeploymentResult(
                    provider=self.name(),
                    error=f"cerebrium deploy failed: {error_detail[:500]}",
                    metadata=dict(plan.metadata),
                )

            deploy_output = proc.stdout

        # Resolve project_id (may have been set during deploy)
        project_id = plan.metadata.get("project_id") or _get_project_id() or ""

        # Try to parse endpoint URL from CLI output
        endpoint_url = ""
        url_match = re.search(
            r"https://api\.aws\.[^/]+\.cerebrium\.ai/v4/[^\s/]+/[^\s/]+",
            deploy_output,
        )
        if url_match:
            base_url = url_match.group(0).rstrip("/")
            # Strip trailing /{function_name} placeholder if present
            if base_url.endswith("/{function_name}"):
                base_url = base_url[: -len("/{function_name}")]
            endpoint_url = base_url
        elif project_id:
            endpoint_url = (
                f"https://api.aws.{region}.cerebrium.ai"
                f"/v4/{project_id}/{service_name}"
            )

        health_url = f"{endpoint_url}/health" if endpoint_url else ""

        final_metadata = dict(plan.metadata)
        final_metadata["project_id"] = project_id

        if not endpoint_url:
            logger.warning(
                "Could not determine Cerebrium project_id — endpoint URL unknown. "
                "Check ~/.cerebrium/config.yaml or the Cerebrium dashboard."
            )

        logger.info("Cerebrium app %s deployed at %s", service_name, endpoint_url)
        return DeploymentResult(
            provider=self.name(),
            endpoint_url=endpoint_url or None,
            health_url=health_url or None,
            metadata=final_metadata,
        )

    # -- Destroy ------------------------------------------------------------

    def destroy(self, result: DeploymentResult) -> None:
        service_name = result.metadata.get("service_name")
        if not service_name:
            logger.warning("No service_name in metadata, cannot destroy")
            return

        api_key = os.environ.get("CEREBRIUM_API_KEY", "")
        project_id = result.metadata.get("project_id", "")

        # Cerebrium app ID format: {project_id}-{service_name}
        app_id = f"{project_id}-{service_name}" if project_id else service_name

        # Try CLI first (most reliable — uses app ID)
        logger.info("Deleting Cerebrium app: %s", app_id)
        try:
            proc = subprocess.run(
                ["cerebrium", "apps", "delete", app_id, "--no-color"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0:
                logger.info("Cerebrium app %s deleted via CLI", app_id)
                return
            logger.warning("CLI delete returned %s: %s", proc.returncode, proc.stderr[:200])
        except Exception as e:
            logger.warning("CLI delete failed: %s", e)

        # Fallback to REST API
        if api_key and project_id:
            try:
                import requests

                resp = requests.delete(
                    f"{CEREBRIUM_API_BASE}/projects/{project_id}/apps/{app_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30,
                )
                if resp.status_code < 300:
                    logger.info("Cerebrium app %s deleted via REST API", app_id)
                    return
                logger.warning(
                    "REST API delete returned %s: %s", resp.status_code, resp.text[:200]
                )
            except Exception as e:
                logger.warning("REST API delete failed: %s", e)

    # -- Status -------------------------------------------------------------

    def status(self, service_name: str) -> dict:
        app_name = f"{service_name}-serverless"
        api_key = os.environ.get("CEREBRIUM_API_KEY", "")
        project_id = _get_project_id() or ""

        if api_key and project_id:
            try:
                import requests

                resp = requests.get(
                    f"{CEREBRIUM_API_BASE}/projects/{project_id}/apps/{app_name}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "provider": self.name(),
                        "service_name": service_name,
                        "status": data.get("status", "running"),
                    }
                if resp.status_code == 404:
                    return {
                        "provider": self.name(),
                        "service_name": service_name,
                        "status": "not found",
                    }
            except Exception as e:
                return {
                    "provider": self.name(),
                    "service_name": service_name,
                    "status": "unknown",
                    "error": str(e),
                }

        return {
            "provider": self.name(),
            "service_name": service_name,
            "status": "unknown",
            "error": "CEREBRIUM_API_KEY or project_id not available",
        }


register("cerebrium", CerebriumProvider)
