"""Modal serverless provider — deploy vLLM on Modal GPUs."""

from __future__ import annotations

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


class ModalProvider(InferenceProvider):
    """Deploy a vLLM server on Modal's serverless GPUs."""

    def name(self) -> str:
        return "modal"

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        app_name = f"{request.service_name}-serverless"
        is_fast_boot = request.cold_start_mode == "fast_boot"

        # Modal uses port 8000 internally
        modal_vllm_cmd = vllm_cmd.replace("--port 8001", "--port 8000")

        serverless = request.scaling.serverless

        replacements = {
            "app_name": app_name,
            "gpu": request.gpu,
            "port": "8000",
            "vllm_cmd": modal_vllm_cmd,
            "vllm_version": request.vllm_version,
            "max_concurrency": str(serverless.concurrency),
            "timeout_s": str(serverless.timeout),
            "scaledown_window_s": str(serverless.scaledown_window),
            "startup_timeout_s": "600",
            "enable_memory_snapshot": "True" if is_fast_boot else "False",
            "experimental_options_line": (
                'experimental_options={"enable_gpu_snapshot": True},'
                if is_fast_boot
                else ""
            ),
        }

        rendered = render_template(
            str(TEMPLATES_DIR / "modal_vllm_server.py.tpl"), replacements
        )

        return ProviderPlan(
            provider=self.name(),
            rendered_script=rendered,
            env={"MODEL_ID": request.model_name},
            metadata={"app_name": app_name, "function_name": "serve"},
        )

    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        app_name = plan.metadata["app_name"]
        function_name = plan.metadata["function_name"]

        # Write rendered script to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="tuna_modal_"
        ) as f:
            f.write(plan.rendered_script)
            script_path = f.name

        try:
            logger.info("Deploying Modal app %s from %s", app_name, script_path)

            merged_env = {**os.environ, **plan.env}
            result = subprocess.run(
                ["modal", "deploy", script_path],
                env=merged_env,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error("modal deploy failed: %s", result.stderr)
                return DeploymentResult(
                    provider=self.name(),
                    error=f"modal deploy failed: {result.stderr}",
                    metadata={"app_name": app_name},
                )

            # Resolve the web URL
            url = self._resolve_web_url(app_name, function_name)

            if not url:
                return DeploymentResult(
                    provider=self.name(),
                    error="Deployed but could not resolve web URL",
                    metadata={"app_name": app_name},
                )

            logger.info("Modal app %s deployed at %s", app_name, url)
            return DeploymentResult(
                provider=self.name(),
                endpoint_url=url,
                health_url=f"{url}/health",
                metadata={"app_name": app_name, "function_name": function_name},
            )

        finally:
            Path(script_path).unlink(missing_ok=True)

    def destroy(self, result: DeploymentResult) -> None:
        app_name = result.metadata.get("app_name")
        if not app_name:
            logger.warning("No app_name in metadata, cannot destroy")
            return

        logger.info("Stopping Modal app %s", app_name)
        subprocess.run(
            ["modal", "app", "stop", app_name],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def status(self, service_name: str) -> dict:
        app_name = f"{service_name}-serverless"
        try:
            result = subprocess.run(
                ["modal", "app", "list"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if app_name in result.stdout:
                return {"provider": self.name(), "app_name": app_name, "status": "running"}
            return {"provider": self.name(), "app_name": app_name, "status": "not found"}
        except Exception as e:
            return {"provider": self.name(), "app_name": app_name, "error": str(e)}

    def _resolve_web_url(
        self, app_name: str, function_name: str, retries: int = 5, delay: float = 3.0
    ) -> str | None:
        """Resolve the web URL for a deployed Modal function.

        Retries a few times since the URL may not be immediately available
        after `modal deploy` returns.
        """
        try:
            import modal

            for attempt in range(retries):
                try:
                    fn = modal.Function.from_name(app_name, function_name)
                    url = fn.get_web_url()
                    if url:
                        return url
                except Exception as e:
                    logger.debug(
                        "URL resolve attempt %d/%d failed: %s",
                        attempt + 1,
                        retries,
                        e,
                    )
                if attempt < retries - 1:
                    time.sleep(delay)

        except ImportError:
            logger.error("modal package not installed — cannot resolve URL")

        return None


register("modal", ModalProvider)
