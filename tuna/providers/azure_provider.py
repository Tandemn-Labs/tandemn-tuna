"""Azure Container Apps GPU provider — deploy vLLM on Azure Container Apps with GPUs."""

from __future__ import annotations

import json
import logging
import os
import subprocess

from tuna.catalog import provider_gpu_id, provider_gpu_map, provider_regions
from tuna.models import DeployRequest, DeploymentResult, PreflightCheck, PreflightResult, ProviderPlan
from tuna.providers.base import InferenceProvider

logger = logging.getLogger(__name__)

DEFAULT_REGION = "eastus"
DEFAULT_IMAGE = "vllm/vllm-openai:v0.15.1"
VLLM_PORT = 8000

# Resources allocated per GPU workload profile.
_GPU_PROFILE_RESOURCES: dict[str, dict[str, str]] = {
    "Consumption-GPU-NC8as-T4": {"cpu": "8", "memory": "56Gi"},
    "Consumption-GPU-NC24-A100": {"cpu": "24", "memory": "220Gi"},
}


def _resolve_subscription_id() -> str | None:
    """Resolve Azure subscription ID from env var or az CLI."""
    sub = os.environ.get("AZURE_SUBSCRIPTION_ID")
    if sub:
        return sub

    try:
        result = subprocess.run(
            ["az", "account", "show", "--query", "id", "-o", "tsv"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        sub = result.stdout.strip()
        if sub and result.returncode == 0:
            return sub
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _resolve_resource_group() -> str | None:
    """Resolve Azure resource group from env var or az CLI defaults."""
    rg = os.environ.get("AZURE_RESOURCE_GROUP")
    if rg:
        return rg

    try:
        result = subprocess.run(
            ["az", "config", "get", "defaults.group", "--query", "value", "-o", "tsv"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        rg = result.stdout.strip()
        if rg and result.returncode == 0:
            return rg
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _require_azure_sdk():
    """Import and return Azure SDK classes, raising ImportError on failure."""
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.appcontainers import ContainerAppsAPIClient
    return DefaultAzureCredential, ContainerAppsAPIClient


class AzureProvider(InferenceProvider):
    """Deploy a vLLM server on Azure Container Apps with GPU support.

    Uses environment reuse strategy: ManagedEnvironments are created once
    and reused across deploys. Only ContainerApps are created/deleted per
    deploy/destroy cycle.
    """

    def name(self) -> str:
        return "azure"

    def vllm_version(self) -> str:
        return DEFAULT_IMAGE.split(":v")[-1]

    def auth_token(self) -> str:
        # Azure Container Apps with external ingress are publicly accessible via HTTPS.
        return ""

    # -- Environment reuse -----------------------------------------------

    def _find_existing_environment(
        self, subscription_id: str, resource_group: str, gpu_profile: str,
        region: str | None = None,
    ) -> str | None:
        """Find an existing ManagedEnvironment with a matching GPU workload profile.

        1. Check AZURE_ENVIRONMENT env var (explicit override).
        2. Fall back to listing environments in the resource group via SDK.
        Only considers environments in the same region if region is specified.
        Returns the environment name, or None.
        """
        explicit = os.environ.get("AZURE_ENVIRONMENT")
        if explicit:
            logger.info("Using explicit environment from AZURE_ENVIRONMENT: %s", explicit)
            return explicit

        try:
            DefaultAzureCredential, ContainerAppsAPIClient = _require_azure_sdk()
        except ImportError:
            return None

        try:
            credential = DefaultAzureCredential()
            client = ContainerAppsAPIClient(credential, subscription_id)

            for env in client.managed_environments.list_by_resource_group(resource_group):
                if region and env.location and env.location.replace(" ", "").lower() != region.lower():
                    continue
                profiles = env.workload_profiles or []
                for wp in profiles:
                    if wp.workload_profile_type == gpu_profile:
                        logger.info(
                            "Found existing environment with %s profile: %s",
                            gpu_profile, env.name,
                        )
                        return env.name
        except Exception as e:
            logger.debug("Error listing environments: %s", e)

        return None

    # -- Preflight checks ------------------------------------------------

    def preflight(self, request: DeployRequest) -> PreflightResult:
        """Run all Azure preflight checks in order, short-circuiting on fatal failures."""
        result = PreflightResult(provider=self.name())

        # 1. az CLI installed
        check = self._check_az_installed()
        result.checks.append(check)
        if not check.passed:
            return result

        # 2. Logged in
        check = self._check_logged_in()
        result.checks.append(check)
        if not check.passed:
            return result

        # 3. Subscription ID
        subscription_id = _resolve_subscription_id()
        if not subscription_id:
            result.checks.append(PreflightCheck(
                name="subscription",
                passed=False,
                message="No Azure subscription configured",
                fix_command="az account set --subscription <SUBSCRIPTION_ID>",
            ))
            return result
        result.checks.append(PreflightCheck(
            name="subscription",
            passed=True,
            message=f"Subscription: {subscription_id}",
        ))

        # 4. Resource group
        resource_group = _resolve_resource_group()
        if not resource_group:
            result.checks.append(PreflightCheck(
                name="resource_group",
                passed=False,
                message="No Azure resource group configured",
                fix_command=(
                    "Set AZURE_RESOURCE_GROUP env var or run: "
                    "az config set defaults.group=<RESOURCE_GROUP>"
                ),
            ))
            return result
        result.checks.append(PreflightCheck(
            name="resource_group",
            passed=True,
            message=f"Resource group: {resource_group}",
        ))

        # 5. Required resource providers registered
        for namespace in ("Microsoft.App", "Microsoft.OperationalInsights"):
            check = self._check_resource_provider(namespace)
            result.checks.append(check)
            if not check.passed:
                return result

        # 6. Azure SDK importable
        check = self._check_sdk()
        result.checks.append(check)

        # 7. GPU region availability
        try:
            gpu_profile = provider_gpu_id(request.gpu, "azure")
        except KeyError:
            gpu_profile = None
        region = os.environ.get("AZURE_REGION", DEFAULT_REGION)
        if gpu_profile:
            valid_regions = provider_regions(request.gpu, "azure")
            result.checks.append(self._check_gpu_region(gpu_profile, region, valid_regions))

        # 8. Existing environment discovery
        if gpu_profile and check.passed:
            env_name = self._find_existing_environment(
                subscription_id, resource_group, gpu_profile,
                region=os.environ.get("AZURE_REGION", DEFAULT_REGION),
            )
            if env_name:
                result.checks.append(PreflightCheck(
                    name="environment",
                    passed=True,
                    message=f"Found existing environment: {env_name}",
                ))
            else:
                result.checks.append(PreflightCheck(
                    name="environment",
                    passed=True,
                    message="No environment found. First deploy will take 30+ min to create one",
                ))

        return result

    def _check_az_installed(self) -> PreflightCheck:
        """Check that the az CLI is installed and reachable."""
        try:
            proc = subprocess.run(
                ["az", "version", "-o", "tsv"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                version_line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "unknown"
                return PreflightCheck(
                    name="az_installed",
                    passed=True,
                    message=f"az CLI found: {version_line}",
                )
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            pass

        return PreflightCheck(
            name="az_installed",
            passed=False,
            message="az CLI not found",
            fix_command="https://learn.microsoft.com/en-us/cli/azure/install-azure-cli",
        )

    def _check_logged_in(self) -> PreflightCheck:
        """Check that the user is logged in to Azure."""
        try:
            proc = subprocess.run(
                ["az", "account", "show", "--query", "user.name", "-o", "tsv"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return PreflightCheck(
                    name="logged_in",
                    passed=True,
                    message=f"Logged in as: {proc.stdout.strip()}",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return PreflightCheck(
            name="logged_in",
            passed=False,
            message="Not logged in to Azure",
            fix_command="az login",
        )

    def _check_resource_provider(self, namespace: str) -> PreflightCheck:
        """Check that a given Azure resource provider is registered."""
        check_name = f"resource_provider_{namespace.split('.')[-1].lower()}"
        try:
            proc = subprocess.run(
                ["az", "provider", "show", "--namespace", namespace,
                 "--query", "registrationState", "-o", "tsv"],
                capture_output=True, text=True, timeout=15,
            )
            if proc.returncode == 0:
                state = proc.stdout.strip()
                if state == "Registered":
                    return PreflightCheck(
                        name=check_name,
                        passed=True,
                        message=f"{namespace} resource provider registered",
                    )
                return PreflightCheck(
                    name=check_name,
                    passed=False,
                    message=f"{namespace} resource provider state: {state}",
                    fix_command=f"az provider register --namespace {namespace}",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return PreflightCheck(
            name=check_name,
            passed=False,
            message=f"Could not check {namespace} resource provider",
            fix_command=f"az provider register --namespace {namespace}",
        )

    def _check_sdk(self) -> PreflightCheck:
        """Check that the Azure SDK is importable."""
        try:
            _require_azure_sdk()
            return PreflightCheck(
                name="azure_sdk",
                passed=True,
                message="Azure SDK available (azure-mgmt-appcontainers, azure-identity)",
            )
        except ImportError:
            return PreflightCheck(
                name="azure_sdk",
                passed=False,
                message="Azure SDK not installed",
                fix_command="pip install tandemn-tuna[azure]",
            )

    def _check_gpu_region(self, gpu_profile: str, region: str,
                          valid_regions: tuple[str, ...] = ()) -> PreflightCheck:
        """Verify the requested GPU type is available in the selected region."""
        if not valid_regions:
            return PreflightCheck(
                name="gpu_region",
                passed=True,
                message=f"GPU region check skipped for unknown profile: {gpu_profile}",
            )

        if region in valid_regions:
            return PreflightCheck(
                name="gpu_region",
                passed=True,
                message=f"{gpu_profile} listed for {region} (static catalog — actual availability depends on your subscription quota)",
            )

        return PreflightCheck(
            name="gpu_region",
            passed=False,
            message=f"{gpu_profile} is not listed for {region}. Listed regions: {', '.join(valid_regions)}",
            fix_command=f"Use --azure-region with one of: {', '.join(valid_regions)}",
        )

    # -- Plan / Deploy / Destroy -----------------------------------------

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        # Validate GPU
        try:
            gpu_profile = provider_gpu_id(request.gpu, "azure")
        except KeyError:
            raise ValueError(
                f"Unknown GPU type for Azure: {request.gpu!r}. "
                f"Supported: {sorted(provider_gpu_map('azure').keys())}"
            )

        # Azure supports only 1 GPU per container
        if request.tp_size > 1 or request.gpu_count > 1:
            raise ValueError(
                "Azure Container Apps supports only 1 GPU per container. "
                "Use tp_size=1 and gpu_count=1, or choose a different provider."
            )

        subscription_id = _resolve_subscription_id()
        if not subscription_id:
            raise RuntimeError(
                "Cannot determine Azure subscription. "
                "Set AZURE_SUBSCRIPTION_ID env var or run 'az account set --subscription <id>'."
            )

        resource_group = _resolve_resource_group()
        if not resource_group:
            raise RuntimeError(
                "Cannot determine Azure resource group. "
                "Set AZURE_RESOURCE_GROUP env var or run 'az config set defaults.group=<name>'."
            )

        region = os.environ.get("AZURE_REGION", DEFAULT_REGION)
        if region != DEFAULT_REGION:
            logger.info("Default region is %s — overriding to %s", DEFAULT_REGION, region)
        service_name = f"{request.service_name}-serverless"
        env_name = f"{request.service_name}-env"
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

        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env["HF_TOKEN"] = hf_token

        # Build vLLM CLI args
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
        if request.quantization:
            container_args.extend(["--quantization", request.quantization])

        resources = _GPU_PROFILE_RESOURCES.get(gpu_profile, {"cpu": "8", "memory": "56Gi"})

        metadata: dict[str, str] = {
            "service_name": service_name,
            "env_name": env_name,
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "region": region,
            "image": DEFAULT_IMAGE,
            "gpu_profile": gpu_profile,
            "container_port": str(VLLM_PORT),
            "container_args": json.dumps(container_args),
            "min_replicas": str(serverless.workers_min),
            "max_replicas": str(serverless.workers_max),
            "concurrency": str(serverless.concurrency),
            "timeout": str(serverless.timeout),
            "cpu": resources["cpu"],
            "memory": resources["memory"],
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
            DefaultAzureCredential, ContainerAppsAPIClient = _require_azure_sdk()
            from azure.mgmt.appcontainers import models as aca_models
        except ImportError:
            return DeploymentResult(
                provider=self.name(),
                error=(
                    "Azure SDK not installed. "
                    "Install with: pip install 'tandemn-tuna[azure]'"
                ),
                metadata=dict(plan.metadata),
            )

        subscription_id = plan.metadata["subscription_id"]
        resource_group = plan.metadata["resource_group"]
        region = plan.metadata["region"]
        env_name = plan.metadata["env_name"]
        service_name = plan.metadata["service_name"]
        gpu_profile = plan.metadata["gpu_profile"]

        container_args = json.loads(plan.metadata["container_args"])
        port = int(plan.metadata["container_port"])

        credential = DefaultAzureCredential()
        client = ContainerAppsAPIClient(credential, subscription_id)

        # 1. Reuse or create ManagedEnvironment
        existing_env = self._find_existing_environment(
            subscription_id, resource_group, gpu_profile, region=region,
        )

        if existing_env:
            env_name = existing_env
            logger.info("Reusing environment: %s", env_name)
        else:
            logger.info(
                "Creating Container Apps environment: %s in %s "
                "(this takes 30+ minutes)",
                env_name, region,
            )
            # Include all GPU profiles so the environment works for any GPU type.
            # Azure doesn't allow adding profiles to an existing environment.
            managed_env = aca_models.ManagedEnvironment(
                location=region,
                workload_profiles=[
                    aca_models.WorkloadProfile(
                        workload_profile_type="Consumption",
                        name="Consumption",
                    ),
                ] + [
                    aca_models.WorkloadProfile(
                        workload_profile_type=profile,
                        name=f"gpu-{profile.split('-')[-1].lower()}",
                    )
                    for profile in _GPU_PROFILE_RESOURCES
                ],
            )

            try:
                poller = client.managed_environments.begin_create_or_update(
                    resource_group, env_name, managed_env,
                )
                poller.result()
            except Exception as e:
                logger.error("Failed to create managed environment: %s", e)
                return DeploymentResult(
                    provider=self.name(),
                    error=f"Environment creation failed: {e}",
                    metadata=dict(plan.metadata),
                )

        # 2. Create Container App
        logger.info("Creating Container App: %s", service_name)

        env_vars = [
            aca_models.EnvironmentVar(name=k, value=v)
            for k, v in plan.env.items()
        ]

        container_app = aca_models.ContainerApp(
            location=region,
            managed_environment_id=(
                f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}"
                f"/providers/Microsoft.App/managedEnvironments/{env_name}"
            ),
            workload_profile_name=f"gpu-{gpu_profile.split('-')[-1].lower()}",
            configuration=aca_models.Configuration(
                ingress=aca_models.Ingress(
                    external=True,
                    target_port=port,
                    transport=aca_models.IngressTransportMethod.HTTP,
                ),
                active_revisions_mode=aca_models.ActiveRevisionsMode.SINGLE,
            ),
            template=aca_models.Template(
                containers=[
                    aca_models.Container(
                        name="vllm",
                        image=plan.metadata["image"],
                        args=container_args,
                        env=env_vars,
                        resources=aca_models.ContainerResources(
                            cpu=float(plan.metadata["cpu"]),
                            memory=plan.metadata["memory"],
                        ),
                        probes=[
                            aca_models.ContainerAppProbe(
                                type=aca_models.Type.STARTUP,
                                tcp_socket=aca_models.ContainerAppProbeTcpSocket(
                                    port=port,
                                ),
                                initial_delay_seconds=30,
                                period_seconds=10,
                                failure_threshold=30,
                                timeout_seconds=5,
                            ),
                        ],
                    ),
                ],
                scale=aca_models.Scale(
                    min_replicas=int(plan.metadata["min_replicas"]),
                    max_replicas=int(plan.metadata["max_replicas"]),
                    rules=[
                        aca_models.ScaleRule(
                            name="http-concurrency",
                            http=aca_models.HttpScaleRule(
                                metadata={"concurrentRequests": plan.metadata["concurrency"]},
                            ),
                        ),
                    ],
                ),
            ),
        )

        try:
            poller = client.container_apps.begin_create_or_update(
                resource_group, service_name, container_app,
            )
            result_app = poller.result()
        except Exception as e:
            logger.error("Failed to create container app: %s", e)
            return DeploymentResult(
                provider=self.name(),
                error=f"Container app creation failed: {e}",
                metadata=dict(plan.metadata),
            )

        # Extract FQDN
        fqdn = (
            result_app.configuration.ingress.fqdn
            if result_app.configuration and result_app.configuration.ingress
            else None
        )
        endpoint_url = f"https://{fqdn}" if fqdn else None

        if plan.metadata.get("public_access") != "true":
            logger.warning(
                "Azure Container Apps endpoints are publicly accessible by default. "
                "Use Azure VNet integration for private access."
            )

        logger.info("Container App %s deployed at %s", service_name, endpoint_url)
        # Store the actual environment name used (may differ from plan if reused)
        final_metadata = dict(plan.metadata)
        final_metadata["env_name"] = env_name
        return DeploymentResult(
            provider=self.name(),
            endpoint_url=endpoint_url,
            health_url=f"{endpoint_url}/health" if endpoint_url else None,
            metadata=final_metadata,
        )

    def destroy(self, result: DeploymentResult) -> None:
        service_name = result.metadata.get("service_name")
        resource_group = result.metadata.get("resource_group") or os.environ.get("AZURE_RESOURCE_GROUP")
        subscription_id = result.metadata.get("subscription_id") or os.environ.get("AZURE_SUBSCRIPTION_ID")

        if not all([service_name, resource_group, subscription_id]):
            logger.warning("Missing metadata for Azure destroy: %s", result.metadata)
            return

        try:
            DefaultAzureCredential, ContainerAppsAPIClient = _require_azure_sdk()
        except ImportError:
            logger.error("Azure SDK not installed, cannot destroy resources")
            return

        credential = DefaultAzureCredential()
        client = ContainerAppsAPIClient(credential, subscription_id)

        # Delete Container App only — preserve the environment for future deploys
        logger.info("Deleting Container App: %s", service_name)
        try:
            poller = client.container_apps.begin_delete(resource_group, service_name)
            poller.result()
        except Exception as e:
            logger.warning("Failed to delete Container App %s: %s", service_name, e)

        logger.info(
            "Environment preserved for future deploys. "
            "Use --azure-cleanup-env to also delete the environment."
        )

    def destroy_environment(self, result: DeploymentResult) -> None:
        """Delete the ManagedEnvironment (slow, 20+ min). Called via --azure-cleanup-env."""
        resource_group = result.metadata.get("resource_group")
        subscription_id = result.metadata.get("subscription_id")
        env_name = result.metadata.get("env_name")

        if not all([env_name, resource_group, subscription_id]):
            logger.warning("Missing metadata for environment cleanup")
            return

        try:
            DefaultAzureCredential, ContainerAppsAPIClient = _require_azure_sdk()
        except ImportError:
            logger.error("Azure SDK not installed, cannot destroy environment")
            return

        credential = DefaultAzureCredential()
        client = ContainerAppsAPIClient(credential, subscription_id)

        logger.info("Deleting Container Apps environment: %s (this takes 20+ min)", env_name)
        try:
            poller = client.managed_environments.begin_delete(resource_group, env_name)
            poller.result()
            logger.info("Environment %s deleted", env_name)
        except Exception as e:
            logger.warning(
                "Failed to delete environment %s: %s (may have other apps)",
                env_name, e,
            )

    def status(self, service_name: str) -> dict:
        try:
            DefaultAzureCredential, ContainerAppsAPIClient = _require_azure_sdk()
        except ImportError:
            return {
                "provider": self.name(),
                "status": "unknown",
                "error": "Azure SDK not installed",
            }

        subscription_id = _resolve_subscription_id()
        if not subscription_id:
            return {"provider": self.name(), "status": "unknown", "error": "No subscription configured"}

        resource_group = _resolve_resource_group()
        if not resource_group:
            return {"provider": self.name(), "status": "unknown", "error": "No resource group configured"}

        app_name = f"{service_name}-serverless"

        try:
            credential = DefaultAzureCredential()
            client = ContainerAppsAPIClient(credential, subscription_id)
            app = client.container_apps.get(resource_group, app_name)

            fqdn = None
            if app.configuration and app.configuration.ingress:
                fqdn = app.configuration.ingress.fqdn

            return {
                "provider": self.name(),
                "service_name": service_name,
                "status": app.provisioning_state or "unknown",
                "uri": f"https://{fqdn}" if fqdn else None,
            }
        except Exception as e:
            error_str = str(e)
            if "ResourceNotFound" in error_str or "404" in error_str:
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
