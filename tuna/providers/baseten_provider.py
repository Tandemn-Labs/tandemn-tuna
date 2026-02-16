"""Baseten serverless provider — deploy vLLM on Baseten GPUs."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests

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

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_API_BASE = "https://api.baseten.co/v1"


def _get_api_key() -> str | None:
    """Read BASETEN_API_KEY from environment."""
    return os.environ.get("BASETEN_API_KEY")


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Api-Key {api_key}"}


class BasetenProvider(InferenceProvider):
    """Deploy a vLLM server on Baseten's serverless GPUs."""

    def name(self) -> str:
        return "baseten"

    def vllm_version(self) -> str:
        return "0.15.1"

    def auth_token(self) -> str:
        return _get_api_key() or ""

    # -- Preflight checks ------------------------------------------------

    def preflight(self, request: DeployRequest) -> PreflightResult:
        result = PreflightResult(provider=self.name())

        # 1. BASETEN_API_KEY env var
        api_key = _get_api_key()
        if not api_key:
            result.checks.append(PreflightCheck(
                name="api_key",
                passed=False,
                message="BASETEN_API_KEY environment variable not set",
                fix_command="export BASETEN_API_KEY=<your-api-key>",
            ))
            return result

        result.checks.append(PreflightCheck(
            name="api_key",
            passed=True,
            message="BASETEN_API_KEY is set",
        ))

        # 2. Validate API key
        check = self._check_api_key(api_key)
        result.checks.append(check)
        if not check.passed:
            return result

        # 3. truss CLI installed
        check = self._check_truss_installed()
        result.checks.append(check)
        if not check.passed:
            return result

        # 4. truss CLI authenticated
        check = self._check_truss_authenticated()
        result.checks.append(check)
        if not check.passed:
            return result

        # 5. GPU type supported
        try:
            provider_gpu_id(request.gpu, "baseten")
            result.checks.append(PreflightCheck(
                name="gpu_supported",
                passed=True,
                message=f"GPU {request.gpu} is supported on Baseten",
            ))
        except KeyError:
            supported = sorted(provider_gpu_map("baseten").keys())
            result.checks.append(PreflightCheck(
                name="gpu_supported",
                passed=False,
                message=f"GPU {request.gpu!r} is not available on Baseten. Supported: {supported}",
            ))

        return result

    def _check_api_key(self, api_key: str) -> PreflightCheck:
        try:
            resp = requests.get(
                f"{_API_BASE}/models",
                headers=_headers(api_key),
                timeout=10,
            )
            if resp.status_code == 200:
                return PreflightCheck(
                    name="api_key_valid",
                    passed=True,
                    message="API key is valid",
                )
            if resp.status_code in (401, 403):
                return PreflightCheck(
                    name="api_key_valid",
                    passed=False,
                    message="API key is invalid (401/403)",
                    fix_command="Check your BASETEN_API_KEY at https://app.baseten.co/settings/api_keys",
                )
            return PreflightCheck(
                name="api_key_valid",
                passed=False,
                message=f"Unexpected response from Baseten API: {resp.status_code}",
            )
        except requests.RequestException as e:
            return PreflightCheck(
                name="api_key_valid",
                passed=False,
                message=f"Could not reach Baseten API: {e}",
            )

    def _check_truss_installed(self) -> PreflightCheck:
        if shutil.which("truss"):
            return PreflightCheck(
                name="truss_installed",
                passed=True,
                message="truss CLI found",
            )
        return PreflightCheck(
            name="truss_installed",
            passed=False,
            message="truss CLI not found",
            fix_command="pip install --upgrade truss",
        )

    def _check_truss_authenticated(self) -> PreflightCheck:
        try:
            proc = subprocess.run(
                ["truss", "whoami"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return PreflightCheck(
                    name="truss_authenticated",
                    passed=True,
                    message=f"truss authenticated as {proc.stdout.strip()}",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return PreflightCheck(
            name="truss_authenticated",
            passed=False,
            message="truss CLI is not authenticated",
            fix_command="truss login --api-key $BASETEN_API_KEY",
        )

    # -- Plan / Deploy / Destroy -----------------------------------------

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        service_name = f"{request.service_name}-serverless"

        try:
            gpu_accelerator = provider_gpu_id(request.gpu, "baseten")
        except KeyError:
            raise ValueError(
                f"Unknown GPU type for Baseten: {request.gpu!r}. "
                f"Supported: {sorted(provider_gpu_map('baseten').keys())}"
            )

        eager_flag = "--enforce-eager" if request.cold_start_mode == "fast_boot" else ""
        serverless = request.scaling.serverless

        replacements = {
            "service_name": service_name,
            "model": request.model_name,
            "max_model_len": str(request.max_model_len),
            "tp_size": str(request.tp_size),
            "gpu": gpu_accelerator,
            "concurrency": str(serverless.concurrency),
            "eager_flag": eager_flag,
            "vllm_version": request.vllm_version,
        }

        rendered = render_template(
            str(TEMPLATES_DIR / "baseten_config.yaml.tpl"), replacements
        )

        metadata = {
            "service_name": service_name,
            "model_name": request.model_name,
            "concurrency_target": str(serverless.concurrency),
            "scale_down_delay": str(serverless.scaledown_window),
        }

        return ProviderPlan(
            provider=self.name(),
            rendered_script=rendered,
            env={},
            metadata=metadata,
        )

    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        api_key = _get_api_key()
        if not api_key:
            return DeploymentResult(
                provider=self.name(),
                error="BASETEN_API_KEY not set",
                metadata=dict(plan.metadata),
            )

        with tempfile.TemporaryDirectory(prefix="tuna_baseten_") as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(plan.rendered_script)

            logger.info("Pushing Baseten model from %s", tmpdir)
            try:
                result = subprocess.run(
                    ["truss", "push", tmpdir, "--publish"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except subprocess.TimeoutExpired:
                return DeploymentResult(
                    provider=self.name(),
                    error="truss push timed out after 600s",
                    metadata=dict(plan.metadata),
                )

            if result.returncode != 0:
                # truss outputs errors to stdout, not stderr
                error_output = result.stderr.strip() or result.stdout.strip()
                logger.error("truss push failed:\nstdout: %s\nstderr: %s", result.stdout, result.stderr)
                return DeploymentResult(
                    provider=self.name(),
                    error=f"truss push failed: {error_output}",
                    metadata=dict(plan.metadata),
                )

            # Parse model_id from truss push output
            logger.info("truss push stdout:\n%s", result.stdout)
            model_id = self._parse_model_id(result.stdout)
            if not model_id:
                return DeploymentResult(
                    provider=self.name(),
                    error=f"Could not parse model_id from truss push output: {result.stdout}",
                    metadata=dict(plan.metadata),
                )

            # /production/sync passes through all paths to the container
            # so the router can call /v1/chat/completions, /health, etc.
            endpoint_url = f"https://model-{model_id}.api.baseten.co/production/sync"
            metadata = dict(plan.metadata)
            metadata["model_id"] = model_id

            self._configure_autoscaling(
                model_id,
                concurrency_target=int(plan.metadata.get("concurrency_target", "32")),
                scale_down_delay=int(plan.metadata.get("scale_down_delay", "60")),
            )

            logger.info("Baseten model deployed: %s", endpoint_url)
            return DeploymentResult(
                provider=self.name(),
                endpoint_url=endpoint_url,
                health_url=f"{endpoint_url}/health",
                metadata=metadata,
            )

    def _parse_model_id(self, stdout: str) -> str | None:
        """Extract model_id from truss push stdout.

        Handles these known output formats:
        - Dashboard URL: https://app.baseten.co/models/31d5m413/logs/31dgo51
        - Endpoint URL: https://model-{id}.api.baseten.co/...
        - Explicit: model_id: abc123
        """
        patterns = [
            r"app\.baseten\.co/models/([a-zA-Z0-9]+)",
            r"model-([a-zA-Z0-9]+)\.api\.baseten\.co",
            r"model[\s_]*id\s*:\s*([a-zA-Z0-9]+)",
        ]

        for line in stdout.strip().splitlines():
            for pattern in patterns:
                match = re.search(pattern, line.strip(), re.IGNORECASE)
                if match:
                    return match.group(1)
        return None

    def _configure_autoscaling(
        self, model_id: str, *, concurrency_target: int, scale_down_delay: int
    ) -> None:
        """Set autoscaling parameters on the production environment via REST API."""
        api_key = _get_api_key()
        if not api_key:
            return

        settings = {
            "autoscaling_settings": {
                "concurrency_target": concurrency_target,
                "scale_down_delay": scale_down_delay,
            }
        }

        try:
            resp = requests.patch(
                f"{_API_BASE}/models/{model_id}/environments/production",
                headers=_headers(api_key),
                json=settings,
                timeout=15,
            )
            if resp.status_code == 200:
                logger.info("Baseten autoscaling configured: %s", settings)
            else:
                logger.warning(
                    "Failed to configure autoscaling: %s %s",
                    resp.status_code,
                    resp.text,
                )
        except requests.RequestException as e:
            logger.warning("Could not configure autoscaling: %s", e)

    def destroy(self, result: DeploymentResult) -> None:
        model_id = result.metadata.get("model_id")
        if not model_id:
            logger.warning("No model_id in metadata, cannot destroy Baseten model")
            return

        api_key = _get_api_key()
        if not api_key:
            logger.error("BASETEN_API_KEY not set, cannot destroy model")
            return

        logger.info("Deleting Baseten model %s", model_id)
        try:
            resp = requests.delete(
                f"{_API_BASE}/models/{model_id}",
                headers=_headers(api_key),
                timeout=30,
            )
            if resp.status_code not in (200, 204):
                logger.warning("Baseten delete returned %s: %s", resp.status_code, resp.text)
        except requests.RequestException as e:
            logger.warning("Failed to delete Baseten model %s: %s", model_id, e)

    def status(self, service_name: str) -> dict:
        api_key = _get_api_key()
        if not api_key:
            return {"provider": self.name(), "status": "unknown", "error": "BASETEN_API_KEY not set"}

        # We'd need the model_id to check status; without state, do a list search
        try:
            resp = requests.get(
                f"{_API_BASE}/models",
                headers=_headers(api_key),
                timeout=10,
            )
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                for model in models:
                    if model.get("name") == f"{service_name}-serverless":
                        return {
                            "provider": self.name(),
                            "service_name": service_name,
                            "status": model.get("status", "unknown"),
                            "model_id": model.get("id", ""),
                        }
                return {"provider": self.name(), "service_name": service_name, "status": "not found"}
            return {"provider": self.name(), "status": "unknown", "error": f"API returned {resp.status_code}"}
        except requests.RequestException as e:
            return {"provider": self.name(), "status": "unknown", "error": str(e)}


register("baseten", BasetenProvider)
