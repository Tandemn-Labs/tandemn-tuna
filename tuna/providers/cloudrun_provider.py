"""Google Cloud Run GPU provider — deploy vLLM on Cloud Run with GPUs."""

from __future__ import annotations

import json
import logging
import os
import subprocess

from pathlib import Path

from tuna.catalog import provider_gpu_id, provider_gpu_map, provider_regions
from tuna.models import DeployRequest, DeploymentResult, PreflightCheck, PreflightResult, ProviderPlan
from tuna.providers.base import InferenceProvider
from tuna.providers.registry import register

logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-central1"
DEFAULT_IMAGE = "vllm/vllm-openai:v0.15.1"
VLLM_PORT = 8000

REQUIRED_APIS = ["run.googleapis.com", "iam.googleapis.com"]


def _resolve_project_id() -> str | None:
    """Resolve Google Cloud project ID from env var or gcloud CLI.

    Returns ``None`` when the project cannot be determined.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        return project

    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        project = result.stdout.strip()
        if project and project != "(unset)":
            return project
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _get_project_id() -> str:
    """Resolve Google Cloud project ID, raising on failure."""
    project = _resolve_project_id()
    if project:
        return project
    raise RuntimeError(
        "Cannot determine Google Cloud project. "
        "Set GOOGLE_CLOUD_PROJECT env var or run 'gcloud config set project <id>'."
    )


def _require_cloudrun_sdk():
    """Import and return the Cloud Run SDK, raising ImportError on failure."""
    from google.cloud.run_v2 import ServicesClient
    return ServicesClient


class CloudRunProvider(InferenceProvider):
    """Deploy a vLLM server on Google Cloud Run with GPU support."""

    def name(self) -> str:
        return "cloudrun"

    # -- Preflight checks ------------------------------------------------

    def preflight(self, request: DeployRequest) -> PreflightResult:
        """Run all Cloud Run preflight checks in order, short-circuiting on fatal failures."""
        result = PreflightResult(provider=self.name())

        # 1. gcloud CLI
        check = self._check_gcloud_installed()
        result.checks.append(check)
        if not check.passed:
            return result  # Can't do anything without gcloud

        # 2. Application Default Credentials
        result.checks.append(self._check_adc())

        # 3. Project
        project_id = _resolve_project_id()

        if not project_id:
            result.checks.append(PreflightCheck(
                name="project",
                passed=False,
                message="No Google Cloud project configured",
                fix_command="gcloud config set project <PROJECT_ID>",
            ))
            return result  # Can't check billing/APIs without a project

        check = self._check_project(project_id)
        result.checks.append(check)
        if not check.passed:
            return result

        # 4. Billing
        result.checks.append(self._check_billing(project_id))

        # 5. APIs
        result.checks.append(self._check_and_enable_apis(project_id))

        # 6. GPU region (only if GPU is specified)
        try:
            gpu_accelerator = provider_gpu_id(request.gpu, "cloudrun")
        except KeyError:
            gpu_accelerator = None
        region = request.region or os.environ.get("GOOGLE_CLOUD_REGION", DEFAULT_REGION)
        if gpu_accelerator:
            valid_regions = provider_regions(request.gpu, "cloudrun")
            result.checks.append(self._check_gpu_region(gpu_accelerator, region, valid_regions))

        return result

    def _check_gcloud_installed(self) -> PreflightCheck:
        """Check that the gcloud CLI is installed and reachable."""
        try:
            proc = subprocess.run(
                ["gcloud", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                # Extract first line for version info
                version_line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "unknown version"
                return PreflightCheck(
                    name="gcloud_installed",
                    passed=True,
                    message=f"gcloud CLI found: {version_line}",
                )
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            pass

        return PreflightCheck(
            name="gcloud_installed",
            passed=False,
            message="gcloud CLI not found",
            fix_command="https://cloud.google.com/sdk/docs/install",
        )

    def _check_adc(self) -> PreflightCheck:
        """Check for Application Default Credentials."""
        adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        if adc_path.exists():
            return PreflightCheck(
                name="credentials",
                passed=True,
                message="Application Default Credentials found",
            )

        # Fallback: try to print an access token
        try:
            proc = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return PreflightCheck(
                    name="credentials",
                    passed=True,
                    message="Application Default Credentials found (via access token)",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return PreflightCheck(
            name="credentials",
            passed=False,
            message="No Application Default Credentials found",
            fix_command="gcloud auth application-default login",
        )

    def _check_project(self, project_id: str) -> PreflightCheck:
        """Verify the GCP project exists and is active."""
        try:
            proc = subprocess.run(
                ["gcloud", "projects", "describe", project_id, "--format=json"],
                capture_output=True, text=True, timeout=15,
            )
            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                state = data.get("lifecycleState", "UNKNOWN")
                if state == "ACTIVE":
                    return PreflightCheck(
                        name="project",
                        passed=True,
                        message=f"Project '{project_id}' is active",
                    )
                return PreflightCheck(
                    name="project",
                    passed=False,
                    message=f"Project '{project_id}' is in state: {state}",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

        return PreflightCheck(
            name="project",
            passed=False,
            message=f"Project '{project_id}' not found or not accessible",
            fix_command=f"gcloud projects describe {project_id}",
        )

    def _check_billing(self, project_id: str) -> PreflightCheck:
        """Check that billing is enabled on the project."""
        try:
            proc = subprocess.run(
                ["gcloud", "billing", "projects", "describe", project_id, "--format=json"],
                capture_output=True, text=True, timeout=15,
            )
            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                if data.get("billingEnabled"):
                    return PreflightCheck(
                        name="billing",
                        passed=True,
                        message="Billing is enabled",
                    )
                return PreflightCheck(
                    name="billing",
                    passed=False,
                    message="Billing is not enabled on this project",
                    fix_command="https://console.cloud.google.com/billing",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

        return PreflightCheck(
            name="billing",
            passed=False,
            message=f"Could not verify billing for project '{project_id}'",
            fix_command=f"gcloud billing projects describe {project_id}",
        )

    def _check_and_enable_apis(self, project_id: str) -> PreflightCheck:
        """Check required APIs, auto-enabling any that are missing."""
        try:
            proc = subprocess.run(
                ["gcloud", "services", "list", "--enabled",
                 "--project", project_id, "--format=value(config.name)"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                return PreflightCheck(
                    name="apis",
                    passed=False,
                    message=f"Failed to list enabled APIs: {proc.stderr.strip()}",
                )

            enabled = set(proc.stdout.strip().splitlines())
            missing = [api for api in REQUIRED_APIS if api not in enabled]

            if not missing:
                return PreflightCheck(
                    name="apis",
                    passed=True,
                    message=f"All required APIs enabled: {', '.join(REQUIRED_APIS)}",
                )

            # Auto-enable missing APIs
            auto_enabled = []
            failed = []
            for api in missing:
                enable_proc = subprocess.run(
                    ["gcloud", "services", "enable", api, "--project", project_id],
                    capture_output=True, text=True, timeout=60,
                )
                if enable_proc.returncode == 0:
                    auto_enabled.append(api)
                else:
                    failed.append(api)

            if failed:
                return PreflightCheck(
                    name="apis",
                    passed=False,
                    message=f"Failed to enable APIs: {', '.join(failed)}",
                    fix_command=f"gcloud services enable {' '.join(failed)} --project {project_id}",
                )

            return PreflightCheck(
                name="apis",
                passed=True,
                message=f"Auto-enabled APIs: {', '.join(auto_enabled)}",
                auto_fixed=True,
            )

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return PreflightCheck(
                name="apis",
                passed=False,
                message="Failed to check APIs (gcloud error)",
            )

    def _check_gpu_region(self, gpu_accelerator: str, region: str,
                          valid_regions: tuple[str, ...] = ()) -> PreflightCheck:
        """Verify the requested GPU type is available in the selected region."""
        if not valid_regions:
            # Unknown GPU or no region constraints — skip check rather than block
            return PreflightCheck(
                name="gpu_region",
                passed=True,
                message=f"GPU region check skipped for unknown accelerator: {gpu_accelerator}",
            )

        if region in valid_regions:
            return PreflightCheck(
                name="gpu_region",
                passed=True,
                message=f"{gpu_accelerator} available in {region}",
            )

        return PreflightCheck(
            name="gpu_region",
            passed=False,
            message=f"{gpu_accelerator} is not available in {region}. Available regions: {', '.join(valid_regions)}",
            fix_command=f"Use --region with one of: {', '.join(valid_regions)}",
        )

    # -- Plan / Deploy / Destroy -----------------------------------------

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        # Validate GPU
        try:
            gpu_accelerator = provider_gpu_id(request.gpu, "cloudrun")
        except KeyError:
            raise ValueError(
                f"Unknown GPU type for Cloud Run: {request.gpu!r}. "
                f"Supported: {sorted(provider_gpu_map('cloudrun').keys())}"
            )

        # Cloud Run supports only 1 GPU per instance
        if request.tp_size > 1 or request.gpu_count > 1:
            raise ValueError(
                "Cloud Run supports only 1 GPU per instance. "
                "Use tp_size=1 and gpu_count=1, or choose a different provider."
            )

        project_id = _get_project_id()
        region = request.region or os.environ.get("GOOGLE_CLOUD_REGION", DEFAULT_REGION)
        service_name = f"{request.service_name}-serverless"
        serverless = request.scaling.serverless

        # Build environment variables for the vLLM container
        env: dict[str, str] = {
            "MODEL_NAME": request.model_name,
            "MAX_MODEL_LEN": str(request.max_model_len),
            "GPU_MEMORY_UTILIZATION": "0.95",
            "DISABLE_LOG_REQUESTS": "true",
        }

        if request.cold_start_mode == "fast_boot":
            env["ENFORCE_EAGER"] = "true"

        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env["HF_TOKEN"] = hf_token

        # Build vLLM CLI args — always tp=1 regardless of request
        container_args = [
            "--model", request.model_name,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--max-model-len", str(request.max_model_len),
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", "0.95",
            "--disable-log-requests",
        ]
        if request.cold_start_mode == "fast_boot":
            container_args.append("--enforce-eager")

        metadata: dict[str, str] = {
            "service_name": service_name,
            "project_id": project_id,
            "region": region,
            "image": DEFAULT_IMAGE,
            "gpu_accelerator": gpu_accelerator,
            "container_port": str(VLLM_PORT),
            "container_args": json.dumps(container_args),
            "min_instance_count": str(serverless.workers_min),
            "max_instance_count": str(serverless.workers_max),
            "max_concurrency": str(serverless.concurrency),
            "timeout_seconds": str(serverless.timeout),
            "cpu": "8",
            "memory": "32Gi",
            "public_access": str(request.public).lower(),
        }

        return ProviderPlan(
            provider=self.name(),
            rendered_script="",
            env=env,
            metadata=metadata,
        )

    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        try:
            ServicesClient = _require_cloudrun_sdk()
            from google.cloud.run_v2.types import (
                Container,
                ContainerPort,
                EnvVar,
                NodeSelector,
                Probe,
                ResourceRequirements,
                RevisionScaling,
                RevisionTemplate,
                Service,
                TCPSocketAction,
            )
            from google.protobuf import duration_pb2
        except ImportError:
            return DeploymentResult(
                provider=self.name(),
                error=(
                    "google-cloud-run SDK not installed. "
                    "Install with: pip install 'tandemn-tuna[cloudrun]'"
                ),
                metadata=dict(plan.metadata),
            )

        service_name = plan.metadata["service_name"]
        project_id = plan.metadata["project_id"]
        region = plan.metadata["region"]
        parent = f"projects/{project_id}/locations/{region}"

        container_args = json.loads(plan.metadata["container_args"])
        port = int(plan.metadata["container_port"])

        # Build container definition
        env_vars = [EnvVar(name=k, value=v) for k, v in plan.env.items()]

        container = Container(
            image=plan.metadata["image"],
            args=container_args,
            ports=[ContainerPort(container_port=port)],
            resources=ResourceRequirements(
                limits={
                    "cpu": plan.metadata["cpu"],
                    "memory": plan.metadata["memory"],
                    "nvidia.com/gpu": "1",
                },
            ),
            env=env_vars,
            startup_probe=Probe(
                tcp_socket=TCPSocketAction(port=port),
                initial_delay_seconds=30,
                period_seconds=10,
                failure_threshold=30,
                timeout_seconds=5,
            ),
        )

        # Build revision template
        revision_template = RevisionTemplate(
            containers=[container],
            scaling=RevisionScaling(
                min_instance_count=int(plan.metadata["min_instance_count"]),
                max_instance_count=int(plan.metadata["max_instance_count"]),
            ),
            max_instance_request_concurrency=int(plan.metadata["max_concurrency"]),
            timeout=duration_pb2.Duration(seconds=int(plan.metadata["timeout_seconds"])),
            node_selector=NodeSelector(accelerator=plan.metadata["gpu_accelerator"]),
            # GPU zonal redundancy requires explicit quota approval from Google.
            # Most users won't have this quota, so we disable it by default.
            # TODO: expose as a CLI flag (e.g. --gpu-zonal-redundancy) once
            # users actually need multi-zone GPU deployments.
            gpu_zonal_redundancy_disabled=True,
        )

        service = Service(
            template=revision_template,
        )

        client = ServicesClient()

        # Create or update the service
        try:
            logger.info("Creating Cloud Run service: %s in %s", service_name, region)
            operation = client.create_service(
                parent=parent,
                service=service,
                service_id=service_name,
            )
            result_service = operation.result()
        except Exception as e:
            error_str = str(e)
            if "AlreadyExists" in error_str or "409" in error_str:
                logger.info("Service %s already exists, updating...", service_name)
                try:
                    service.name = f"{parent}/services/{service_name}"
                    operation = client.update_service(service=service)
                    result_service = operation.result()
                except Exception as update_err:
                    logger.error("Cloud Run service update failed: %s", update_err)
                    return DeploymentResult(
                        provider=self.name(),
                        error=f"Service update failed: {update_err}",
                        metadata=dict(plan.metadata),
                    )
            else:
                logger.error("Cloud Run service creation failed: %s", e)
                return DeploymentResult(
                    provider=self.name(),
                    error=f"Service creation failed: {e}",
                    metadata=dict(plan.metadata),
                )

        service_uri = result_service.uri

        # Only grant public access when explicitly requested via --public
        if plan.metadata.get("public_access") == "true":
            self._set_public_access(client, result_service.name)

        logger.info("Cloud Run service %s deployed at %s", service_name, service_uri)
        return DeploymentResult(
            provider=self.name(),
            endpoint_url=service_uri,
            health_url=f"{service_uri}/health",
            metadata=dict(plan.metadata),
        )

    def _set_public_access(self, client, resource_name: str) -> None:
        """Grant allUsers the roles/run.invoker role. Non-fatal on failure."""
        try:
            from google.iam.v1 import iam_policy_pb2, policy_pb2

            policy = client.get_iam_policy(
                request=iam_policy_pb2.GetIamPolicyRequest(resource=resource_name)
            )

            invoker_role = "roles/run.invoker"
            all_users_member = "allUsers"

            # Check if binding already exists
            for binding in policy.bindings:
                if binding.role == invoker_role and all_users_member in binding.members:
                    return

            policy.bindings.append(
                policy_pb2.Binding(
                    role=invoker_role,
                    members=[all_users_member],
                )
            )

            client.set_iam_policy(
                request=iam_policy_pb2.SetIamPolicyRequest(
                    resource=resource_name,
                    policy=policy,
                )
            )
            logger.info("Public access granted to %s", resource_name)
        except Exception as e:
            logger.warning(
                "Could not set public access on %s: %s (service works but requires auth)",
                resource_name,
                e,
            )

    def destroy(self, result: DeploymentResult) -> None:
        service_name = result.metadata.get("service_name")
        project_id = result.metadata.get("project_id")
        region = result.metadata.get("region")

        if not all([service_name, project_id, region]):
            logger.warning("Missing metadata for Cloud Run destroy: %s", result.metadata)
            return

        try:
            ServicesClient = _require_cloudrun_sdk()
        except ImportError:
            logger.error("google-cloud-run SDK not installed, cannot destroy service")
            return

        name = f"projects/{project_id}/locations/{region}/services/{service_name}"

        logger.info("Deleting Cloud Run service: %s", name)
        try:
            client = ServicesClient()
            client.delete_service(name=name)
        except Exception as e:
            logger.warning("Failed to delete Cloud Run service %s: %s", name, e)

    def status(self, service_name: str) -> dict:
        try:
            ServicesClient = _require_cloudrun_sdk()
        except ImportError:
            return {
                "provider": self.name(),
                "status": "unknown",
                "error": "google-cloud-run SDK not installed",
            }

        try:
            project_id = _get_project_id()
        except RuntimeError as e:
            return {"provider": self.name(), "status": "unknown", "error": str(e)}

        region = os.environ.get("GOOGLE_CLOUD_REGION", DEFAULT_REGION)
        full_name = f"projects/{project_id}/locations/{region}/services/{service_name}-serverless"

        try:
            client = ServicesClient()
            svc = client.get_service(name=full_name)
            conditions = []
            if svc.conditions:
                conditions = [
                    {"type": c.type_, "state": str(c.state), "message": c.message}
                    for c in svc.conditions
                ]
            return {
                "provider": self.name(),
                "service_name": service_name,
                "status": "running",
                "uri": svc.uri,
                "conditions": conditions,
            }
        except Exception as e:
            error_str = str(e)
            if "NotFound" in error_str or "404" in error_str:
                return {
                    "provider": self.name(),
                    "service_name": service_name,
                    "status": "not found",
                }
            return {
                "provider": self.name(),
                "service_name": service_name,
                "status": "unknown",
                "error": error_str,
            }


register("cloudrun", CloudRunProvider)
