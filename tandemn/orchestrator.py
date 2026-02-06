"""Orchestrator — wires router, serverless, and spot deployments together."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from tandemn.models import DeployRequest, DeploymentResult, HybridDeployment, ProviderPlan
from tandemn.providers.registry import get_provider
from tandemn.spot.sky_launcher import SkyLauncher
from tandemn.template_engine import render_template

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
META_LB_PATH = Path(__file__).resolve().parent / "router" / "meta_lb.py"


def build_vllm_cmd(request: DeployRequest, port: str = "8001") -> str:
    """Render the shared vLLM command from the template."""
    eager_flag = "--enforce-eager" if request.cold_start_mode == "fast_boot" else ""

    replacements = {
        "model": request.model_name,
        "host": "0.0.0.0",
        "port": port,
        "max_model_len": str(request.max_model_len),
        "tp_size": str(request.tp_size),
        "eager_flag": eager_flag,
    }
    return render_template(
        str(TEMPLATES_DIR / "vllm_serve_cmd.txt"), replacements
    )


def _launch_router_vm(request: DeployRequest) -> DeploymentResult:
    """Launch the router on a cheap CPU VM via sky launch."""
    region_block = ""
    if request.region:
        cloud = request.spots_cloud.lower()
        region_block = f"  any_of:\n    - infra: {cloud}/{request.region}"

    replacements = {
        "service_name": request.service_name,
        "serverless_url": "",
        "spot_url": "",
        "meta_lb_local_path": str(META_LB_PATH),
        "region_block": region_block,
    }

    rendered = render_template(
        str(TEMPLATES_DIR / "router.yaml.tpl"), replacements
    )

    cluster_name = f"{request.service_name}-router"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="tandemn_router_"
    ) as f:
        f.write(rendered)
        yaml_path = f.name

    try:
        logger.info("Launching router VM: %s", cluster_name)
        result = subprocess.run(
            [
                "sky", "launch", yaml_path,
                "--cluster", cluster_name,
                "-y",
                "--down",  # Auto-teardown on failure
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            return DeploymentResult(
                provider="router",
                error=f"sky launch failed: {result.stderr}",
                metadata={"cluster_name": cluster_name},
            )

        # Get the router's public IP
        ip = _get_cluster_ip(cluster_name)
        if not ip:
            return DeploymentResult(
                provider="router",
                error="Launched but could not resolve IP",
                metadata={"cluster_name": cluster_name},
            )

        endpoint = f"http://{ip}:8080"
        logger.info("Router VM ready at %s", endpoint)
        return DeploymentResult(
            provider="router",
            endpoint_url=endpoint,
            health_url=f"{endpoint}/router/health",
            metadata={"cluster_name": cluster_name},
        )

    finally:
        Path(yaml_path).unlink(missing_ok=True)


def _get_cluster_ip(cluster_name: str) -> str | None:
    """Get the head node IP of a SkyPilot cluster."""
    try:
        result = subprocess.run(
            ["sky", "status", "--ip", cluster_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        ip = result.stdout.strip()
        if ip and not ip.startswith("No cluster"):
            return ip
    except Exception as e:
        logger.debug("Failed to get IP for %s: %s", cluster_name, e)
    return None


def push_url_to_router(
    router_url: str, serverless_url: str | None = None, spot_url: str | None = None
) -> bool:
    """POST updated backend URLs to the router's /router/config endpoint."""
    payload = {}
    if serverless_url:
        payload["serverless_url"] = serverless_url
    if spot_url:
        payload["spot_url"] = spot_url

    if not payload:
        return True

    try:
        resp = requests.post(f"{router_url}/router/config", json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        logger.error("Failed to push URL to router: %s", e)
        return False


def launch_hybrid(request: DeployRequest) -> HybridDeployment:
    """Deploy the full hybrid stack: router VM + serverless + spot — all in parallel."""
    vllm_cmd = build_vllm_cmd(request)

    router_result = None
    serverless_result = None
    router_url = None

    def _launch_serverless():
        try:
            provider = get_provider(request.serverless_provider)
            plan = provider.plan(request, vllm_cmd)
            return provider.deploy(plan)
        except Exception as e:
            logger.error("Serverless launch failed: %s", e)
            return DeploymentResult(provider=request.serverless_provider, error=str(e))

    def _launch_spot():
        try:
            launcher = SkyLauncher()
            plan = launcher.plan(request, vllm_cmd)
            return launcher.deploy(plan)
        except Exception as e:
            logger.error("Spot launch failed: %s", e)
            return DeploymentResult(provider="skyserve", error=str(e))

    logger.info("Launching router + serverless + spot in parallel")
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_router = pool.submit(_launch_router_vm, request)
        fut_serverless = pool.submit(_launch_serverless)
        fut_spot = pool.submit(_launch_spot)

        router_result = fut_router.result(timeout=600)
        if router_result.error:
            logger.error("Router launch failed: %s", router_result.error)
            return HybridDeployment(router=router_result)
        router_url = router_result.endpoint_url

        try:
            serverless_result = fut_serverless.result(timeout=600)
        except Exception as e:
            serverless_result = DeploymentResult(
                provider=request.serverless_provider, error=str(e)
            )

    if serverless_result and serverless_result.endpoint_url:
        logger.info("Pushing serverless URL to router: %s", serverless_result.endpoint_url)
        push_url_to_router(router_url, serverless_url=serverless_result.endpoint_url)

    def _wait_and_push_spot():
        try:
            spot_result = fut_spot.result(timeout=900)
            if spot_result and spot_result.endpoint_url:
                logger.info("Pushing spot URL to router: %s", spot_result.endpoint_url)
                push_url_to_router(router_url, spot_url=spot_result.endpoint_url)
            elif spot_result and spot_result.error:
                logger.warning("Spot deployment issue: %s", spot_result.error)
        except Exception as e:
            logger.error("Spot wait failed: %s", e)

    threading.Thread(target=_wait_and_push_spot, daemon=True).start()

    return HybridDeployment(
        serverless=serverless_result,
        spot=None,
        router=router_result,
        router_url=router_url,
    )


def destroy_hybrid(service_name: str) -> None:
    """Tear down all components of a hybrid deployment."""
    logger.info("Destroying hybrid deployment: %s", service_name)

    # Tear down router VM
    router_cluster = f"{service_name}-router"
    logger.info("Tearing down router: %s", router_cluster)
    subprocess.run(
        ["sky", "down", router_cluster, "-y"],
        capture_output=True, text=True, timeout=120,
    )

    # Tear down SkyServe spot
    spot_service = f"{service_name}-spot"
    logger.info("Tearing down spot service: %s", spot_service)
    subprocess.run(
        ["sky", "serve", "down", spot_service, "-y"],
        capture_output=True, text=True, timeout=120,
    )

    # Tear down Modal — need to know app_name
    modal_app = f"{service_name}-serverless"
    logger.info("Tearing down Modal app: %s", modal_app)
    subprocess.run(
        ["modal", "app", "stop", modal_app],
        capture_output=True, text=True, timeout=60,
    )

    logger.info("Destroy complete for %s", service_name)


def status_hybrid(service_name: str) -> dict:
    """Check status of all components."""
    status = {
        "service_name": service_name,
        "router": None,
        "serverless": None,
        "spot": None,
    }

    # Check router
    router_cluster = f"{service_name}-router"
    ip = _get_cluster_ip(router_cluster)
    if ip:
        router_url = f"http://{ip}:8080"
        try:
            resp = requests.get(f"{router_url}/router/health", timeout=5)
            if resp.status_code == 200:
                status["router"] = resp.json()
                status["router"]["url"] = router_url
        except Exception:
            status["router"] = {"url": router_url, "status": "unreachable"}
    else:
        status["router"] = {"status": "no cluster found"}

    # Check spot via sky serve status
    spot_service = f"{service_name}-spot"
    try:
        result = subprocess.run(
            ["sky", "serve", "status", spot_service],
            capture_output=True, text=True, timeout=30,
        )
        status["spot"] = {"raw": result.stdout.strip()}
    except Exception as e:
        status["spot"] = {"error": str(e)}

    return status
