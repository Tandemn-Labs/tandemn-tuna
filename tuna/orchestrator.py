"""Orchestrator — wires router, serverless, and spot deployments together."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from tuna.models import DeployRequest, DeploymentResult, HybridDeployment, ProviderPlan
from tuna.providers.registry import get_provider
from tuna.sky_sdk import (
    cluster_down,
    cluster_launch,
    cluster_status,
    serve_status,
    task_from_yaml_str,
)
from tuna.template_engine import render_template

if TYPE_CHECKING:
    from tuna.state import DeploymentRecord

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
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

    try:
        logger.info("Launching router VM: %s", cluster_name)
        task = task_from_yaml_str(rendered)
        job_id, handle = cluster_launch(task, cluster_name=cluster_name, down=True)

        ip = handle.head_ip if handle else None
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

    except Exception as e:
        logger.error("sky launch failed: %s", e)
        return DeploymentResult(
            provider="router",
            error=f"sky launch failed: {e}",
            metadata={"cluster_name": cluster_name},
        )


def _find_controller_cluster() -> str | None:
    """Find the SkyServe controller cluster name from ``sky status``."""
    try:
        statuses = cluster_status()
        for entry in statuses:
            if "sky-serve-controller" in entry.name:
                return entry.name
    except Exception as e:
        logger.debug("Failed to find controller cluster: %s", e)
    return None


def _get_ssh_key_path() -> str:
    """Get the SkyPilot SSH private key path."""
    from sky.utils import auth_utils
    private_key_path, _ = auth_utils.get_or_generate_keys()
    return private_key_path


def _open_port_on_cluster(cluster_name: str, port: int) -> bool:
    """Open a port on a SkyPilot cluster's security group."""
    from sky import global_user_state, provision as provision_lib
    try:
        handle = global_user_state.get_handle_from_cluster_name(cluster_name)
        if handle is None:
            return False
        config = global_user_state.get_cluster_yaml_dict(handle.cluster_yaml)
        provider_config = config["provider"]
        cloud = handle.launched_resources.cloud
        provision_lib.open_ports(
            repr(cloud),
            handle.cluster_name_on_cloud,
            [str(port)],
            provider_config,
        )
        return True
    except Exception as e:
        logger.warning("Failed to open port %d: %s", port, e)
        return False


def _get_ssh_user(cluster_name: str) -> str:
    """Get the SSH user for a SkyPilot cluster."""
    from sky import global_user_state
    try:
        handle = global_user_state.get_handle_from_cluster_name(cluster_name)
        if handle is None:
            return "ubuntu"
        config = global_user_state.get_cluster_yaml_dict(handle.cluster_yaml)
        return config.get("auth", {}).get("ssh_user", "ubuntu")
    except Exception:
        return "ubuntu"


def _launch_router_on_controller(
    request: DeployRequest,
    controller_cluster: str,
    serverless_url: str = "",
    serverless_auth_token: str = "",
    router_port: int = 8080,
) -> DeploymentResult:
    """Launch the meta_lb router on the SkyServe controller VM via SSH."""
    ip = _get_cluster_ip(controller_cluster)
    if not ip:
        return DeploymentResult(
            provider="router",
            error=f"Could not resolve IP for controller {controller_cluster}",
        )

    try:
        ssh_key = _get_ssh_key_path()
    except Exception as e:
        logger.warning("Could not get SSH key: %s", e)
        return DeploymentResult(provider="router", error=f"SSH key error: {e}")

    ssh_user = _get_ssh_user(controller_cluster)
    ssh_target = f"{ssh_user}@{ip}"
    ssh_opts = ["-i", ssh_key, "-o", "StrictHostKeyChecking=no"]

    # 1. Open port on security group
    logger.info("Opening port %d on %s", router_port, controller_cluster)
    _open_port_on_cluster(controller_cluster, router_port)

    # 2. SCP meta_lb.py to controller
    logger.info("Copying meta_lb.py to controller")
    try:
        scp_result = subprocess.run(
            ["scp", *ssh_opts, str(META_LB_PATH), f"{ssh_target}:/tmp/meta_lb.py"],
            capture_output=True, text=True, timeout=30,
        )
        if scp_result.returncode != 0:
            return DeploymentResult(
                provider="router",
                error=f"SCP failed: {scp_result.stderr}",
            )
    except subprocess.TimeoutExpired:
        return DeploymentResult(provider="router", error="SCP timed out")

    # 3. SSH: install deps, then start gunicorn in background.
    # SkyPilot controllers use conda — non-interactive SSH doesn't activate it,
    # so we source conda.sh explicitly to get pip/python on PATH.
    conda_prefix = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate base"

    # 3a. Install dependencies (can be slow on first deploy)
    logger.info("Installing dependencies on controller via SSH")
    install_cmd = f"{conda_prefix} && pip install -q flask requests gunicorn"
    try:
        subprocess.run(
            ["ssh", *ssh_opts, ssh_target, install_cmd],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        logger.warning("pip install timed out — deps may already be installed, continuing")

    # 3b. Start gunicorn — use setsid to fully detach into its own session
    # so it survives the SSH connection closing.
    start_cmd = (
        f"{conda_prefix} && "
        f"SERVERLESS_BASE_URL='{serverless_url}' "
        f"SERVERLESS_AUTH_TOKEN='{serverless_auth_token}' "
        f"SKYSERVE_BASE_URL='http://127.0.0.1:30001' "
        f"setsid gunicorn -w 1 -k gthread --threads 16 --timeout 300 "
        f"--bind 0.0.0.0:{router_port} "
        f"--chdir /tmp meta_lb:app > /tmp/meta_lb.log 2>&1 < /dev/null &"
    )
    logger.info("Starting gunicorn on controller via SSH")
    try:
        subprocess.run(
            ["ssh", *ssh_opts, ssh_target, start_cmd],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.warning("SSH start command timed out — gunicorn may still be starting")

    endpoint = f"http://{ip}:{router_port}"
    logger.info("Router colocated on controller at %s", endpoint)
    return DeploymentResult(
        provider="router",
        endpoint_url=endpoint,
        health_url=f"{endpoint}/router/health",
        metadata={
            "cluster_name": controller_cluster,
            "colocated": "true",
            "router_port": str(router_port),
        },
    )


def _get_cluster_ip(cluster_name: str) -> str | None:
    """Get the head node IP of a SkyPilot cluster."""
    try:
        statuses = cluster_status(cluster_names=[cluster_name])
        if statuses and statuses[0].handle:
            return statuses[0].handle.head_ip
    except Exception as e:
        logger.debug("Failed to get IP for %s: %s", cluster_name, e)
    return None


def push_url_to_router(
    router_url: str,
    serverless_url: str | None = None,
    serverless_auth_token: str | None = None,
    spot_url: str | None = None,
    retries: int = 5,
    delay: float = 3.0,
) -> bool:
    """POST updated backend URLs to the router's /router/config endpoint."""
    payload = {}
    if serverless_url:
        payload["serverless_url"] = serverless_url
    if serverless_auth_token:
        payload["serverless_auth_token"] = serverless_auth_token
    if spot_url:
        payload["spot_url"] = spot_url

    if not payload:
        return True

    for attempt in range(retries):
        try:
            resp = requests.post(f"{router_url}/router/config", json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            logger.warning("Push to router returned %d (attempt %d/%d)", resp.status_code, attempt + 1, retries)
        except Exception as e:
            logger.warning("Push to router failed (attempt %d/%d): %s", attempt + 1, retries, e)
        if attempt < retries - 1:
            time.sleep(delay)

    logger.error("Failed to push URLs to router after %d attempts", retries)
    return False


def launch_hybrid(request: DeployRequest, *, separate_router_vm: bool = False) -> HybridDeployment:
    """Deploy the full hybrid stack.

    Parameters
    ----------
    request : DeployRequest
        The deployment specification.
    separate_router_vm : bool
        If *True*, launch the router on a dedicated CPU VM (legacy 3-VM mode).
        If *False* (default), colocate the router on the SkyServe controller VM.
    """
    vllm_cmd = build_vllm_cmd(request)

    # Pin vLLM version to match the selected serverless provider
    serverless_prov = get_provider(request.serverless_provider)
    request.vllm_version = serverless_prov.vllm_version()
    logger.info("vLLM version: %s (from %s)", request.vllm_version, request.serverless_provider)

    # Auth token the router needs to proxy to this serverless backend
    _backend_auth_token = serverless_prov.auth_token()

    # Early preflight — fail fast before launching any VMs
    preflight = serverless_prov.preflight(request)
    if not preflight.ok:
        failures = "; ".join(c.message for c in preflight.failed)
        return HybridDeployment(
            serverless=DeploymentResult(
                provider=request.serverless_provider,
                error=f"Preflight failed: {failures}",
                metadata={"service_name": f"{request.service_name}-serverless"},
            ),
        )

    router_result = None
    serverless_result = None
    router_url = None

    # Shared dicts for capturing plan metadata before deploy() is called,
    # so error-path DeploymentResults still include provider-specific IDs.
    _serverless_meta: dict[str, str] = {}
    _spot_meta: dict[str, str] = {}

    def _launch_serverless():
        try:
            provider = get_provider(request.serverless_provider)
            plan = provider.plan(request, vllm_cmd)
            _serverless_meta.update(plan.metadata)
            return provider.deploy(plan)
        except Exception as e:
            logger.error("Serverless launch failed: %s", e)
            return DeploymentResult(
                provider=request.serverless_provider,
                error=str(e),
                metadata=dict(_serverless_meta),
            )

    def _launch_spot():
        try:
            provider = get_provider("skyserve")
            preflight = provider.preflight(request)
            if not preflight.ok:
                failures = "; ".join(c.message for c in preflight.failed)
                return DeploymentResult(
                    provider="skyserve",
                    error=f"Preflight failed: {failures}",
                    metadata={"service_name": f"{request.service_name}-spot"},
                )
            plan = provider.plan(request, vllm_cmd)
            _spot_meta.update(plan.metadata)
            return provider.deploy(plan)
        except Exception as e:
            logger.error("Spot launch failed: %s", e)
            return DeploymentResult(
                provider="skyserve",
                error=str(e),
                metadata=dict(_spot_meta),
            )

    if separate_router_vm:
        # Legacy path: 3 VMs in parallel (router + serverless + spot)
        logger.info("Launching router + serverless + spot in parallel (separate router VM)")
        spot_result = None
        pool = ThreadPoolExecutor(max_workers=3)
        try:
            fut_router = pool.submit(_launch_router_vm, request)
            fut_serverless = pool.submit(_launch_serverless)
            fut_spot = pool.submit(_launch_spot)

            try:
                router_result = fut_router.result(timeout=900)
            except Exception as e:
                logger.error("Router launch failed: %s", e)
                router_result = DeploymentResult(provider="router", error=str(e))
            if router_result.error:
                logger.error("Router launch failed: %s", router_result.error)
                # Don't return early — collect serverless/spot results for cleanup

            router_url = router_result.endpoint_url

            try:
                serverless_result = fut_serverless.result(timeout=600)
            except Exception as e:
                serverless_result = DeploymentResult(
                    provider=request.serverless_provider,
                    error=str(e),
                    metadata=dict(_serverless_meta),
                )

            if router_url and serverless_result and serverless_result.endpoint_url:
                logger.info("Pushing serverless URL to router: %s", serverless_result.endpoint_url)
                push_url_to_router(
                    router_url,
                    serverless_url=serverless_result.endpoint_url,
                    serverless_auth_token=_backend_auth_token,
                )

            try:
                spot_result = fut_spot.result(timeout=900)
            except Exception as e:
                logger.error("Spot launch failed: %s", e)
                spot_result = DeploymentResult(
                    provider="skyserve",
                    error=str(e),
                    metadata=dict(_spot_meta),
                )

            if router_url and spot_result and spot_result.endpoint_url:
                logger.info("Pushing spot URL to router: %s", spot_result.endpoint_url)
                push_url_to_router(router_url, spot_url=spot_result.endpoint_url)
            elif spot_result and spot_result.error:
                logger.warning("Spot deployment issue: %s", spot_result.error)
        finally:
            pool.shutdown(wait=False)

        return HybridDeployment(
            serverless=serverless_result,
            spot=spot_result,
            router=router_result,
            router_url=router_url,
        )

    # Default path: colocate router on controller (2 workers, then SSH)
    logger.info("Launching serverless + spot in parallel, router will colocate on controller")
    spot_result = None
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        fut_serverless = pool.submit(_launch_serverless)
        fut_spot = pool.submit(_launch_spot)

        # Wait for spot — sky serve up creates the controller
        try:
            spot_result = fut_spot.result(timeout=900)
        except Exception as e:
            logger.error("Spot launch failed: %s", e)
            spot_result = DeploymentResult(
                provider="skyserve", error=str(e), metadata=dict(_spot_meta),
            )

        # Check if serverless is done yet
        serverless_url = ""
        if fut_serverless.done():
            try:
                serverless_result = fut_serverless.result(timeout=0)
                if serverless_result and serverless_result.endpoint_url:
                    serverless_url = serverless_result.endpoint_url
            except Exception:
                pass

        # Find the controller cluster and launch router on it
        controller_cluster = _find_controller_cluster()
        if controller_cluster:
            logger.info("Controller found: %s — colocating router", controller_cluster)
            router_result = _launch_router_on_controller(
                request, controller_cluster,
                serverless_url=serverless_url,
                serverless_auth_token=_backend_auth_token,
            )
        else:
            logger.warning("Controller cluster not found, falling back to separate router VM")
            router_result = _launch_router_vm(request)

        if router_result.error:
            logger.error("Router launch failed: %s", router_result.error)
            # Try fallback if colocation failed and we haven't already fallen back
            if controller_cluster and router_result.metadata.get("colocated") != "true":
                pass  # already a fallback result
            elif controller_cluster:
                logger.warning("Colocated router failed, falling back to separate router VM")
                router_result = _launch_router_vm(request)

        router_url = router_result.endpoint_url

        # Wait for serverless if not already done
        if serverless_result is None:
            try:
                serverless_result = fut_serverless.result(timeout=600)
            except Exception as e:
                serverless_result = DeploymentResult(
                    provider=request.serverless_provider,
                    error=str(e),
                    metadata=dict(_serverless_meta),
                )

        # Push serverless URL if router is up and serverless wasn't baked in at launch
        if (router_url and serverless_result and serverless_result.endpoint_url
                and serverless_result.endpoint_url != serverless_url):
            logger.info("Pushing serverless URL to router: %s", serverless_result.endpoint_url)
            push_url_to_router(
                router_url,
                serverless_url=serverless_result.endpoint_url,
                serverless_auth_token=_backend_auth_token,
            )

        # Spot URL is localhost for colocated, but push for fallback (separate VM)
        if (router_url and spot_result and spot_result.endpoint_url
                and router_result.metadata.get("colocated") != "true"):
            logger.info("Pushing spot URL to router: %s", spot_result.endpoint_url)
            push_url_to_router(router_url, spot_url=spot_result.endpoint_url)
    finally:
        pool.shutdown(wait=False)

    return HybridDeployment(
        serverless=serverless_result,
        spot=spot_result,
        router=router_result,
        router_url=router_url,
    )


def _warmup_serverless(health_url: str, timeout: int = 300, interval: float = 5.0) -> bool:
    """Poll the health endpoint to trigger cold start and wait until ready.

    Returns True if the endpoint became healthy, False on timeout.
    """
    logger.info("Warming up serverless container: %s", health_url)
    print("Warming up container...", end="", flush=True)

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(health_url, timeout=10)
            if resp.status_code == 200:
                print(" ready!")
                logger.info("Serverless container is healthy")
                return True
        except requests.exceptions.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(interval)

    print(" timed out (container may still be starting)")
    logger.warning("Warmup timed out after %ds for %s", timeout, health_url)
    return False


def launch_serverless_only(request: DeployRequest) -> HybridDeployment:
    """Deploy only serverless, skip spot + router."""
    serverless_prov = get_provider(request.serverless_provider)

    logger.info("Deploying serverless-only via %s", request.serverless_provider)

    # Preflight
    preflight = serverless_prov.preflight(request)
    if not preflight.ok:
        failures = "; ".join(c.message for c in preflight.failed)
        return HybridDeployment(
            serverless=DeploymentResult(
                provider=request.serverless_provider,
                error=f"Preflight failed: {failures}",
                metadata={"service_name": f"{request.service_name}-serverless"},
            )
        )

    # vLLM command
    request.vllm_version = serverless_prov.vllm_version()
    logger.info("vLLM version: %s (from %s)", request.vllm_version, request.serverless_provider)
    vllm_cmd = build_vllm_cmd(request)

    # Plan + deploy (catch exceptions to preserve plan metadata for cleanup)
    _meta: dict[str, str] = {}
    try:
        plan = serverless_prov.plan(request, vllm_cmd)
        _meta.update(plan.metadata)
        serverless_result = serverless_prov.deploy(plan)
    except Exception as e:
        logger.error("Serverless deploy failed: %s", e)
        return HybridDeployment(
            serverless=DeploymentResult(
                provider=request.serverless_provider,
                error=str(e),
                metadata=dict(_meta),
            )
        )

    if serverless_result.error:
        return HybridDeployment(serverless=serverless_result)

    logger.info("Serverless endpoint: %s", serverless_result.endpoint_url)

    # Warm up the container — trigger cold start so the endpoint is ready
    health_url = serverless_result.health_url or f"{serverless_result.endpoint_url}/health"
    _warmup_serverless(health_url)

    return HybridDeployment(
        serverless=serverless_result,
        router_url=serverless_result.endpoint_url,  # Direct serverless URL
    )


def _cleanup_serve_controller() -> None:
    """Tear down the SkyServe controller VM if no services remain.

    Polls ``sky serve status`` for up to 90 seconds waiting for in-progress
    teardowns to finish before deciding whether to remove the controller.
    """
    from sky.serve import ServiceStatus

    _TERMINAL = {
        ServiceStatus.SHUTTING_DOWN,
        ServiceStatus.NO_REPLICA,
        ServiceStatus.FAILED,
        ServiceStatus.FAILED_CLEANUP,
    }

    try:
        # Wait for services to finish tearing down (up to ~90s)
        for _ in range(18):
            statuses = serve_status(None)
            if not statuses:
                break
            # If every remaining service is in a terminal state, it will
            # disappear soon — keep waiting instead of giving up.
            if all(s.get("status") in _TERMINAL for s in statuses):
                logger.debug(
                    "All remaining services in terminal state, waiting for removal..."
                )
                time.sleep(5)
                continue
            # A service is still active — leave the controller alone
            return

        # Find and tear down the controller cluster
        controller_name = _find_controller_cluster()
        if controller_name:
            logger.info("No remaining services, tearing down controller: %s", controller_name)
            cluster_down(controller_name)
    except Exception as e:
        logger.debug("Controller cleanup check failed (non-fatal): %s", e)


def destroy_hybrid(
    service_name: str,
    record: "DeploymentRecord | None" = None,
    skip_controller_cleanup: bool = False,
) -> None:
    """Tear down all components of a hybrid deployment.

    Parameters
    ----------
    service_name : str
        The deployment's service name.
    record : DeploymentRecord | None
        If provided, provider names and metadata are read from the record
        instead of using hardcoded defaults.
    skip_controller_cleanup : bool
        If True, skip the SkyServe controller cleanup at the end.  Useful
        when tearing down multiple deployments in a loop — the caller can
        run cleanup once after all teardowns finish.
    """
    from tuna.state import DeploymentRecord

    if record is None:
        logger.warning("No deployment record for %s, falling back to hardcoded providers", service_name)
        record = DeploymentRecord(service_name=service_name)

    logger.info("Destroying hybrid deployment: %s", service_name)

    # Tear down router — check if colocated or separate VM
    router_meta = record.router_metadata or {}
    if not router_meta and not record.router_endpoint and not record.spot_provider_name:
        logger.info("No router to tear down (serverless-only deployment)")
    elif router_meta.get("colocated") == "true":
        # Router is colocated on the controller — kill the gunicorn process.
        # The process also dies when the controller is torn down below.
        controller_cluster = router_meta.get("cluster_name")
        if controller_cluster:
            ip = _get_cluster_ip(controller_cluster)
            if ip:
                try:
                    ssh_key = _get_ssh_key_path()
                    ssh_user = _get_ssh_user(controller_cluster)
                    ssh_target = f"{ssh_user}@{ip}"
                    ssh_opts = ["-i", ssh_key, "-o", "StrictHostKeyChecking=no"]
                    logger.info("Killing colocated router on %s", controller_cluster)
                    subprocess.run(
                        ["ssh", *ssh_opts, ssh_target, "pkill -f 'gunicorn.*meta_lb'"],
                        capture_output=True, text=True, timeout=15,
                    )
                except Exception as e:
                    logger.debug("Failed to kill colocated router (non-fatal): %s", e)
    else:
        # Legacy path: separate router VM
        router_cluster = f"{service_name}-router"
        logger.info("Tearing down router: %s", router_cluster)
        try:
            cluster_down(router_cluster)
        except Exception as e:
            logger.debug("Router teardown failed (non-fatal): %s", e)

    # Tear down spot via provider interface (skip if spot was never launched)
    spot_name = record.spot_provider_name
    if spot_name:
        spot_meta = (record.spot_metadata or {}).copy()
        spot_meta.setdefault("service_name", f"{service_name}-spot")
        spot_provider = get_provider(spot_name)
        spot_result = DeploymentResult(
            provider=spot_provider.name(),
            metadata=spot_meta,
        )
        logger.info("Tearing down spot service via provider: %s", spot_provider.name())
        spot_provider.destroy(spot_result)
    else:
        logger.info("No spot deployment to tear down")

    # Tear down serverless via provider interface (skip if never launched)
    serverless_name = record.serverless_provider_name
    if serverless_name:
        serverless_meta = (record.serverless_metadata or {}).copy()
        svc = f"{service_name}-serverless"
        serverless_meta.setdefault("app_name", svc)       # Modal
        serverless_meta.setdefault("service_name", svc)    # CloudRun, Baseten
        serverless_provider = get_provider(serverless_name)

        # If provider needs IDs we don't have, try status() lookup
        try:
            if serverless_name == "baseten" and "model_id" not in serverless_meta:
                status = serverless_provider.status(service_name)
                if status.get("model_id"):
                    serverless_meta["model_id"] = status["model_id"]

            if serverless_name == "runpod" and "endpoint_id" not in serverless_meta:
                status = serverless_provider.status(service_name)
                if status.get("endpoint_id"):
                    serverless_meta["endpoint_id"] = status["endpoint_id"]
                if status.get("template_id"):
                    serverless_meta["template_id"] = status["template_id"]

            if serverless_name == "cloudrun":
                if "project_id" not in serverless_meta:
                    try:
                        from tuna.providers.cloudrun_provider import get_project_id
                        serverless_meta["project_id"] = get_project_id()
                    except Exception as e:
                        logger.debug("Failed to get GCP project ID for destroy fallback: %s", e)
                if "region" not in serverless_meta:
                    serverless_meta["region"] = os.environ.get(
                        "GOOGLE_CLOUD_REGION", "us-central1"
                    )
        except Exception as e:
            logger.debug("Status lookup for destroy fallback failed (non-fatal): %s", e)

        serverless_result = DeploymentResult(
            provider=serverless_provider.name(),
            metadata=serverless_meta,
        )
        logger.info("Tearing down serverless via provider: %s", serverless_provider.name())
        serverless_provider.destroy(serverless_result)
    else:
        logger.info("No serverless deployment to tear down")

    # Clean up SkyServe controller if no services remain
    if not skip_controller_cleanup:
        _cleanup_serve_controller()

    logger.info("Destroy complete for %s", service_name)


def status_hybrid(service_name: str, record: "DeploymentRecord | None" = None) -> dict:
    """Check status of all components.

    Parameters
    ----------
    service_name : str
        The deployment's service name.
    record : DeploymentRecord | None
        If provided, provider names are read from the record instead of
        using hardcoded defaults.
    """
    from tuna.state import DeploymentRecord

    if record is None:
        logger.warning("No deployment record for %s, falling back to hardcoded providers", service_name)
        record = DeploymentRecord(service_name=service_name)

    # Detect serverless-only: has serverless provider but no spot and no router
    if record.serverless_provider_name and not record.spot_provider_name and not record.router_endpoint:
        serverless_name = record.serverless_provider_name
        serverless_provider = get_provider(serverless_name)
        return {
            "service_name": service_name,
            "mode": "serverless-only",
            "router": None,
            "serverless": serverless_provider.status(service_name),
            "spot": None,
        }

    status = {
        "service_name": service_name,
        "router": None,
        "serverless": None,
        "spot": None,
    }

    # Check router — infrastructure, not an inference provider
    router_meta = record.router_metadata or {}
    if router_meta.get("colocated") == "true":
        controller_cluster = router_meta.get("cluster_name")
        router_port = router_meta.get("router_port", "8080")
        ip = _get_cluster_ip(controller_cluster) if controller_cluster else None
    else:
        ip = _get_cluster_ip(f"{service_name}-router")
        router_port = "8080"

    if ip:
        router_url = f"http://{ip}:{router_port}"
        try:
            resp = requests.get(f"{router_url}/router/health", timeout=5)
            if resp.status_code == 200:
                status["router"] = resp.json()
                status["router"]["url"] = router_url
        except Exception:
            status["router"] = {"url": router_url, "status": "unreachable"}
    else:
        status["router"] = {"status": "no cluster found"}

    spot_name = record.spot_provider_name or "skyserve"
    serverless_name = record.serverless_provider_name or "modal"

    # Check spot via provider interface
    spot_provider = get_provider(spot_name)
    status["spot"] = spot_provider.status(service_name)

    # Check serverless via provider interface
    serverless_provider = get_provider(serverless_name)
    status["serverless"] = serverless_provider.status(service_name)

    return status
