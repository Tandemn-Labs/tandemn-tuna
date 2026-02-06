# Hybrid GPU Inference Orchestrator (for Online Inference)

**Serverless for Cold Starts. Spot for Scale. Pay Only When You Run.**

## 🧠 Problem

Modern AI inference is expensive and wasteful:

* **Always-on GPUs** burn money during idle time
* **Spot GPUs** are cheap but slow to start and can be interrupted
* **Serverless GPUs** are fast but expensive at scale
* Teams are forced to choose between:
  * Low latency **or**
  * Low cost

👉 There is no simple way to **combine the best of both**.

## Our Idea

A **single deploy command** that:

* Finds the **cheapest available GPU provider (skypilot for serverless platforms (modal/runpod) + sky serve for spots on AWS (and other clouds))**
* Deploys **serverless GPUs for instant cold starts**
* Spins up **Spot GPUs for sustained load**
* Routes traffic between them automatically
* Scales everything to **zero when idle**

Example:

```bash
deploy llama-70b \
  --cold_start_optimized \
  --spot_scaling \
  --gpu=L40S \
  --region=us-west
```


---

## Technical Approach (ASCII Diagram)

```
                   ┌──────────────────────────┐
Client Requests ──▶│      Smart Router        │
                   │  (Latency + Cost Aware)  │
                   └───────────┬──────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
        Cold starts / low QPS         Sustained load / high QPS
                 │                           │
        ┌────────▼────────┐        ┌────────▼─────────┐
        │  Serverless GPU  │        │   Spot GPU Fleet  │
        │   (Modal/Runpod) │        │  (AWS + SkyServe) │
        │   min=0, max=5   │        │   min=0, max=5    │
        └────────┬────────┘        └────────┬─────────┘
                 │                           │
         Fast startup (~seconds)      Slow startup (~5 mins)
         Higher $/GPU-hr               Cheap $/GPU-hr
                 │                           │
                 └───────────┬───────────────┘
                             │
                     Scale-to-zero when idle
```


---

## Unit Economics (Why This Wins)

### Baselines (min=1, max=5, 24/7 idle cost) (qwen 7B model)

| Setup | Monthly Cost (1 GPU baseline) |
|-------|-------------------------------|
| AWS On-Demand L40S | \~$1,350 / month              |
| Modal L40S Always-On | \~$1,400 / month              |
| AWS Spot Always-On | \~$800 / month                |

### Hybrid (min=0 for both, bursty workload)

Assume:

* 100 GPU-hrs/month cold-start traffic
* 200 GPU-hrs/month sustained load

| Scenario | Monthly Cost |
|----------|--------------|
| Hybrid (Best) | \~$430       |
| Hybrid (Worst) | \~$510       |

✅ **Savings: 60–70% vs On-Demand** ✅ **Savings: 35–45% vs Spot Always-On** ✅ **Zero idle burn**


---

## 🧩 Competitive Landscape

| Category | Players | What They Do |
|----------|---------|--------------|
| Serverless GPUs | Modal, Runpod, Replicate | Fast startup, scale-to-zero, expensive at scale |
| Spot GPU Orchestration | SkyPilot SkyServe | Cheap GPUs, slow startup, infra-heavy |
| Inference Platforms | Baseten, Fireworks, Together | Abstract infra, opaque pricing |
| Kubernetes Gateways | Gateway API (K8s SIG) | Routing logic, not cost-aware |
| Cloud Providers | AWS, GCP | Raw primitives only |

👉 **No one combines:**

* multi-provider price discovery
* serverless + spot dual backend
* cost-aware routing
* scale-to-zero economics
* one-click deploy UX


---

## 🎯 Where to Focus for Competitive Advantage 

### 1️⃣ Price Discovery + Normalization (Huge Differentiator)

Build adapters:

```python
quote(provider="modal", gpu="L40S") # not covered in SkyPilot
quote(provider="aws_spot", gpu="L40S", region="us-west") # covered in SkyPilot
```

Normalize into:

```
effective_cost = $/GPU-hr + cold_start_penalty + egress + reliability_score
```

### 2️⃣ Routing Policy Engine 

Example policies:

* Route to serverless until Spot is warm
* Shift traffic to Spot after N seconds
* Re-route to serverless if spot interruption rate spikes
* Keep small warm pool if SLO is tight

### 3️⃣ Provider Abstraction Layer

Design once:

```python
ProviderAdapter.deploy(model, gpu, min=0, max=5)
ProviderAdapter.scale_to_zero()
ProviderAdapter.health()
```

This unlocks:

* Multi-cloud
* Multi-provider
* Arbitrage

### 4️⃣ UX / DX: One Command Deploy

Your moat is **developer experience**:

```bash
tandemn deploy llama-70b --optimize-for=cost+latency
```

No Terraform. No K8s manifests. No cloud vendor lock-in.


\
How to Implement within Orca

1 - Extend models-

A typical request should look like this - 

```python
class OnlineServingRequestOptimized(BaseModel):
    """Request to send online inference job to central server."""
    user_id: str
    description: str
    task_type: str
    task_priority: Optional[str] = None
    model_name: str
    engine: str
    quantization_bits: Optional[Literal["4", "8", "16"]] = None
    is_speculative_decode: Optional[bool] = None
    is_PD_disaggregation: Optional[bool] = None
    slo_mode: Optional[str] = "online" # the slo of online serving is always online
    placement: str

    # # online-specific config 
    vllm_specific_config: Optional[vLLMSpecificConfig] = None # if that engine has been specified
    # # resource-specific config
    gpu: str
    gpu_count: int
    region: Optional[str]
    # # scaling-specific config
    scale_to_zero: Optional[bool] = True
    concurrency: Optional[int]
    cold_start_mode: Optional[Literal["fast_boot","no_fast_boot"]] = "no_fast_boot"
    # # cloud specific config
    spots_cloud: Optional[str] = "AWS"
    serverless_cloud: Optional[str] = "Modal"
    find_cheapest: Optional[bool] = True
```

Once this sort of a contract goes to Orca, it looks up within it's serverless providers and finds the cheapest serverless. Another thread should spawn that does the same for skypilot. Launches SkyServe on one, Launches Serverless on one, and the load balancer between them. 


2 - Mini SkyPilot for Serverless

We would need this when we expand beyond Modal. Idea is to create an abstraction class iike this so we keep a track of all cloud prices, launches and all. It will be extended by other classes downstream to enable us to lazy load their SDKs and create a SkyPilot like abstraction. 

```python
class ServerlessProviders:
    def name() -> str:
        """
        Tells the name of the GPU Provider
        """
        pass

    def capabilities(OnlineServingRequestOprtimized) -> ProviderCapabilities:
        """gpu_types, timeouts, supports_volumes, supports_scale_to_zero, regions
        based on the SDK of the model providers"""
        pass
  
    def estimate_cost(OnlineServingRequestOptimized) -> CostEstimate:
        """Based on the gpu specified in the OnlineServingRequest, we check the price
        and give an estimate on what it will cost"""
        pass

    def plan(OnlineServingRequestOptimized) -> ProviderPlan:
        """
        Pre_baked_spec to look at our request and then create nice defaults and launch stuff
        Can be Pre-Baked Python Scripts or YAML files and then attaches them in the Plan Object
        """
       pass
    
    def deploy(ProviderPlan) -> DeploymentResult:
        """
        Actually runs the script (someting like modal run)
        """
        pass
        
    def status(DeploymentResult) -> DeploymentStatus:
        """
        Ping to check DeploymentResult
        (something like modal check)
        """
        pass
    
    def destroy(DeploymentResult) -> None:
        """
        teardown
        """
        pass
```


3 - Class definitions of the above - 

```python
@dataclass
class ProviderCapbilities:
    gpu_types: 
    supports_scale_to_zero: 
    supports_volumes:
    regions:
    max_timeout:
    max_concurrency:
    
@dataclass
class CostEstimate:
    provider:
    gpu:
    price_per_hour:
    notes:
    
@dataclass
class ProviderPlan:
    provider:
    pre_baked_spec: # template that helps us launch things on modal/whatever
    env: # all env variables as a list kinda
    ports:
    volumes:
    metadata:
    
@dataclass
class DeploymentResult:
    provider:
    deployment_id:
    endpoint_url:
    health_url:
    metadata:

@dataclass
class HybridDeployment:
    serverless: DeploymentResult
    spot: DeploymentResult (from skyserve)
    router_url:
    status: # data to tell us if serverless is up or spot is up
```


4 - Extend the ServerlessProvider for Modal: 

```python
class ModalProvider:
    import modal  #lazy loading
    """
    Minimal Modal implementation.

    This does NOT deploy a new app; it resolves URLs for a deployed app.
    Use Modal CLI to deploy, then use this to query and route.
    """
    
    modal = LazyImport("modal", import_error_message=_IMPORT_ERROR)

    # Price catalog 
    # in the future this will be 
    # _PRICE_PER_HOUR = Catalog.get_prices(modal)
    _PRICE_PER_HOUR = {
        "H100": 4.00,
        "A100": 2.50,
        "L40s": 1.20,
        "L4": 0.75,
        "T4": 0.35,
    }

    def name(self) -> str:
        return "modal"

    def capabilities(self, req: OnlineServingRequestOptimized) -> ProviderCapabilities:
        # Keep this minimal for MVP. Expand with actual Modal offerings per region.
        return ProviderCapabilities(
            gpu_types=list(self._PRICE_PER_HOUR.keys()),
            supports_scale_to_zero=True,
            supports_volumes=True,
            # placeholder 
            regions=["us-east", "us-west", "eu-west"],  
            # actual taken from https://modal.com/docs/guide/region-selection
            max_timeout=60 * 60,
            max_concurrency=64,
        )

    def estimate_cost(self, req: OnlineServingRequestOptimized) -> CostEstimate:
        gpu = (req.gpu or "").upper()
        price = self._PRICE_PER_HOUR.get(gpu, 0.0)
        notes = None
        if price == 0.0:
            notes = "Unknown GPU price"
        return CostEstimate(provider=self.name(), gpu=gpu, price_per_hour=price, notes=notes)

    def plan(self, req: OnlineServingRequestOptimized) -> ProviderPlan:
        """
        Create a plan that points to a pre-baked Modal script.
        The script is responsible for defining the App + web endpoint.
        """
        env = {
            "MODEL_ID": req.model_name,
            "GPU_TYPE": req.gpu,
            "MAX_MODEL_LEN": str(getattr(req.vllm_specific_config, "max_model_len", 4096) or 4096),
            "FAST_BOOT": "true" if req.cold_start_mode == "fast_boot" else "false",
        }
        ports = [8000]
        metadata = {
            # These are required for deploy() to resolve the endpoint:
            # app_name/function_name should match your deployed Modal app.
            "app_name": "example-vllm-inference",
            "function_name": "serve", 
        }
        return ProviderPlan(
            provider=self.name(),
            pre_baked_spec="serverless-spot/modal_vllm_server.py",
            env=env,
            ports=ports,
            volumes={
                "/root/.cache/huggingface": "huggingface-cache",
                "/root/.cache/vllm": "vllm-cache",
            },
            metadata=metadata,
        )

    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        """
        Resolve an already-deployed Modal endpoint URL.
        Uses modal.Function.from_name(app, fn).get_web_url()
        """
        modal = self.modal
        app_name = plan.metadata.get("app_name")
        function_name = plan.metadata.get("function_name")
        env = plan.env
        spec = plan.pre_baked_spec
        if not app_name or not function_name:
            raise ValueError("Modal plan requires metadata: app_name and function_name")
        # we will convert the spec to a modal script
        script = convert_spec_to_modal()
        #do the launching here
        # 1) Deploy via CLI (minimal, zero SDK wiring)
        subprocess.run(
            ["modal", "deploy", script], # add the envs here
            check=True,
            env=env
        )
        # 2) Resolve URL (may need a short retry)
        fn = self.modal.Function.from_name(app_name, function_name)
        url = self._get_web_url_with_retry(fn) # need to make this function
        
        return DeploymentResult(
            provider=self.name(),
            deployment_id=f"{app_name}:{function_name}",
            endpoint_url=url,
            health_url=f"{url}/health",
            metadata={"app_name": app_name, "function_name": function_name},
        )

    def status(self, deployment: DeploymentResult) -> DeploymentStatus:
        """
        Basic health check using /health. This is provider-agnostic HTTP.
        """
        try:
            import requests
            if deployment.health_url is None:
                return DeploymentStatus(state="DEGRADED", last_error="No health_url")
            resp = requests.get(deployment.health_url, timeout=5)
            if 200 <= resp.status_code < 300:
                return DeploymentStatus(state="READY")
            return DeploymentStatus(
                state="DEGRADED", last_error=f"Health status {resp.status_code}"
            )
        except Exception as e:
            return DeploymentStatus(state="FAILED", last_error=str(e))

    def destroy(self, deployment: DeploymentResult) -> None:
        """
        Modal does not currently expose a simple Python SDK call to delete a deployed app.
        For MVP, you can shell out to:
          modal app stop <app-name>
        or use the Modal web UI.
        """
        pass
```


5 - we can have this kind of a flow - 

```python
request -> build_online_serving_plan -> 
render_skyserve_from_plan -> YAML-> 
render_modal_from_plan -> Modal script
```

6 - How to integrate with the existing orca

let's create a folder called `templates/common/vllm_serve_cmd`and store `online.txt`

* ```python
  vllm serve {model} \
    --host {host} \
    --port {port} \
    -tp {tp} \
    -pp {pp} \
    {additional_params}
  ```

The existing vllm_online.yaml can import this in this way - 

```python
set -euxo pipefail
source .venv/bin/activate

~/sky_templates/ray/start_cluster

if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    echo "Starting vLLM server..."
    setsid nohup {vllm_cmd} > vllm.log 2>&1 &
    # readiness loop...
fi
```

and use the same flow for when the user wants online, BUT on normal skypilot and no fancy stuff. This will allow us to use the same vllm cmd in modal

Create a modal template `**templates/modal_vllm_web_server.py.tpl**`

```python
import os
import subprocess
import time
import modal

APP_NAME = "{app_name}"
VLLM_PORT = {port}

HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"

hf_cache_vol = modal.Volume.from_name("{hf_cache_vol}", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("{vllm_cache_vol}", create_if_missing=True)

image = (
    modal.Image.from_registry("{base_image}", add_python="{python_version}")
    .entrypoint([])
    .uv_pip_install("vllm=={vllm_version}", "huggingface-hub=={hf_hub_version}")
    .env({{"HF_XET_HIGH_PERFORMANCE": "1"}})
)

app = modal.App(APP_NAME)

@app.function(
    image=image,
    gpu="{gpu}",
    timeout={timeout_s},
    scaledown_window={scaledown_window_s},
    volumes={{
        HF_CACHE_PATH: hf_cache_vol,
        VLLM_CACHE_PATH: vllm_cache_vol,
    }},
)
@modal.concurrent(max_inputs={max_concurrency})
@modal.web_server(port=VLLM_PORT, startup_timeout={startup_timeout_s})
def serve():
    # same vLLM command as SkyServe
    cmd = """{vllm_cmd}"""
    subprocess.Popen(cmd, shell=True)
    _wait_ready()

def _wait_ready(timeout: int = {startup_timeout_s}):
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("vLLM did not become healthy in time")
```

which similar to old vllm online flow can be replaced using string search


```python
replace_cmd = {
  "model": req.model_name,
  "host": "0.0.0.0",
  "port": "8001",
  "tp": str(config.tp_size),
  "pp": str(config.pp_size),
  "additional_params": build_additional_params(req),
}
vllm_cmd = update_template("templates/common/vllm_serve_cmd", replace_cmd)
```

then for Skypilot

```python
replace_run = {"vllm_cmd": vllm_cmd}
run_string = update_template("templates/vllm_run_online", replace_run)

replace_yaml = {
  "name": config.decision_id,
  "num_nodes": config.num_nodes,
  "resources.instance_type": config.instances,
  "resources.ports": "8001",
  "run": run_string,
}
update_yaml_file("templates/vllm_online.yaml", replace_yaml, YAML_OUTPUT)
# sky.Task.from_yaml(YAML_OUTPUT) ...
```

then for Modal

```python
replace_modal = {
  "app_name": f"{config.decision_id}-serverless",
  "gpu": req.gpu,
  "port": 8000,
  "vllm_cmd": vllm_cmd.replace("--port 8001", "--port 8000"),
  "max_concurrency": req.concurrency or 32,
  "scaledown_window_s": 15,
  "timeout_s": 600,
  "startup_timeout_s": 600,
  "base_image": "nvidia/cuda:12.8.0-devel-ubuntu22.04",
  "python_version": "3.12",
  "vllm_version": "0.13.0",
  "hf_hub_version": "0.36.0",
  "hf_cache_vol": "huggingface-cache",
  "vllm_cache_vol": "vllm-cache",
}
script_contents = update_template("templates/modal_vllm_web_server.py.tpl", replace_modal)

# write script_contents to a temp file, then:
# subprocess.run(["modal", "deploy", tmp_script_path], env=merged_env, check=True)
# then resolve URL: modal.Function.from_name(app_name, "serve").get_web_url()
```


7 - Finally, all of this should run in two threads separately within the server. One thread for launching the SkyServe and one thread for launching the 

```python
def launch_hybrid(request: OnlineServingRequestOptimized, config: MagicOutput) -> Dict[str, Any]:
    result: Dict[str, Optional[str]] = {"spot_url": None, "serverless_url": None}
    errors: Dict[str, Optional[str]] = {"spot_error": None, "serverless_error": None}

    def launch_spot():
        try:
            result["spot_url"] = sp_launch_vllm_online(request, config)
        except Exception as e:
            errors["spot_error"] = str(e)

    def launch_serverless():
        try:
            modal = ModalProvider()
            plan = modal.plan(request)
            result["serverless_url"] = modal.deploy(plan)
        except Exception as e:
            errors["serverless_error"] = str(e)

    t_spot = threading.Thread(target=launch_spot, name="launch-spot")
    t_serverless = threading.Thread(target=launch_serverless, name="launch-serverless")

    t_spot.start()
    t_serverless.start()

    t_spot.join()
    t_serverless.join()

    return {
        "spot_url": result["spot_url"],
        "serverless_url": result["serverless_url"],
        "spot_error": errors["spot_error"],
        "serverless_error": errors["serverless_error"],
    }
```


8 - Now on to actually launching the load balancer. How will that look? Can we deploy the load-balancer in the CPU instance that 


A lil GPTing helped me with this - 


1. Where things run today From sky/serve/service.py and the controller template: One controller VM (SkyPilot cluster) runs a single entrypoint: python -m sky.serve.service --service-name ... That process starts two child processes on the same machine: Controller (e.g. 127.0.0.1:20001) – autoscaler, replica state, etc. Load balancer (e.g. 0.0.0.0:30001) – FastAPI, proxies to replicas. So the "LB server" is just that VM; there's no special lock-down. Anything you can run on a normal Linux box you can run there. SkyPilot doesn't "forbid" other programs; it just doesn't start any except controller + LB.
2. Does SkyPilot "allow" other programs on the LB server? Officially: There is no hook or config that says "also run this extra process." The template only runs sky.serve.service. In practice: Yes. The controller VM is a normal VM. You can: Option A – Run your meta-LB as a third process started by the same sky.serve.service process (requires a small patch/fork). Option B – After sky serve up, SSH into the controller and start your meta-LB (e.g. nohup python meta_lb.py & or systemd). No SkyPilot code change, but not managed by SkyPilot (no restart, lost if VM is recreated). Option C – Run the meta-LB on a separate small CPU instance (like your SAM router). No SkyPilot change; you pay for one small VM. So: "Can we add another LB on the same server?" — yes. "Does SkyPilot run it for us?" — only if you add it to the service process (Option A) or run it yourself (Option B/C).
3. Your desired flow (no extra instance) User → Meta-LB (single user-facing endpoint). Meta-LB (on the same controller VM): If spots are ready → forward to SkyPilot's LB (<http://127.0.0.1:30001>). If not → forward to serverless (Modal). Optionally: while routing to serverless, "poke" SkyServe (e.g. /health) so it scales up from zero. That reuses the existing controller VM; no new CPU instance.
4. Implementation options Option A – Run meta-LB alongside SkyPilot's LB (same VM, managed) Idea: In sky/serve/service.py, after the existing load_balancer_process.start(), start a third process that runs your meta-LB (same pattern as your SAM load_balancer.py: if spot ready → SkyServe LB, else → serverless + poke). Pros: Same box, no extra instance; lifecycle tied to the serve process (cleanup on down). Cons: You must patch (or fork) SkyPilot and ensure the controller cluster opens the meta-LB port (e.g. 8080). What to change (conceptually): Controller task ports In sky/serve/server/impl.py, where the controller task gets ports=\[LOAD_BALANCER_PORT_RANGE\], add the meta-LB port, e.g.: ports=\[LOAD_BALANCER_PORT_RANGE, "8080"\] so the VM firewall allows user traffic to 8080. Start meta-LB in sky/serve/service.py After starting load_balancer_process, start another multiprocessing.Process that runs your meta-LB entrypoint (e.g. run_meta_load_balancer(...)). Pass: SkyServe LB URL: <http://127.0.0.1>:{load_balancer_port} Serverless URL: from env or from a small file/DB that Orca writes when it deploys Modal. Meta-LB listen port: e.g. 8080. Meta-LB logic Same as your SAM router: If SkyServe LB is ready (e.g. GET <http://127.0.0.1:30001/health> 200) → proxy to SkyServe LB. Else → proxy to serverless URL; optionally in the background hit SkyServe /health (or your poke path) to trigger scale-up. User-facing URL Give users http://<controller_public_ip>:8080 (or the port you chose). That's the only endpoint they need. So: yes, you can put a meta-LB on the same server as SkyPilot's LB; Option A is the way to have it "sit alongside" and be managed.
5. \