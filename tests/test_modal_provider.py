"""Tests for tandemn.providers.modal_provider — plan() only, no real deploys."""

from pathlib import Path

from tandemn.models import DeployRequest
from tandemn.providers.modal_provider import ModalProvider


class TestModalProviderPlan:
    def setup_method(self):
        self.provider = ModalProvider()
        self.request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="L40S",
            service_name="test-svc",
        )
        self.vllm_cmd = (
            "vllm serve Qwen/Qwen3-0.6B "
            "--host 0.0.0.0 --port 8001 --max-model-len 4096 "
            "--served-model-name llm --tensor-parallel-size 1 "
            "--disable-log-requests --uvicorn-log-level info --enforce-eager"
        )

    def test_plan_provider_name(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.provider == "modal"

    def test_plan_app_name(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["app_name"] == "test-svc-serverless"
        assert plan.metadata["function_name"] == "serve"

    def test_plan_rendered_script_contains_app_name(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert '"test-svc-serverless"' in plan.rendered_script

    def test_plan_rendered_script_contains_gpu(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert '"L40S"' in plan.rendered_script

    def test_plan_rendered_script_contains_vllm_cmd(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert "Qwen/Qwen3-0.6B" in plan.rendered_script

    def test_plan_port_replaced_to_8000(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        # The vllm_cmd in the rendered script should use port 8000
        assert "--port 8000" in plan.rendered_script

    def test_plan_env_has_model_id(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.env["MODEL_ID"] == "Qwen/Qwen3-0.6B"

    def test_plan_fast_boot_enables_snapshots(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert "enable_memory_snapshot=True" in plan.rendered_script
        assert '"enable_gpu_snapshot": True' in plan.rendered_script

    def test_plan_no_fast_boot_disables_snapshots(self):
        self.request.cold_start_mode = "no_fast_boot"
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert "enable_memory_snapshot=False" in plan.rendered_script
        assert "enable_gpu_snapshot" not in plan.rendered_script

    def test_plan_concurrency(self):
        self.request.concurrency = 64
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert "max_inputs=64" in plan.rendered_script

    def test_plan_rendered_script_is_valid_python_syntax(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        # Should not raise SyntaxError
        compile(plan.rendered_script, "<modal_template>", "exec")
