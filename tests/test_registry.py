"""Tests for tuna.providers.registry."""

import pytest

from tuna.models import DeployRequest, DeploymentResult, ProviderPlan
from tuna.providers.base import InferenceProvider
from tuna.providers.registry import (
    PROVIDER_MODULES,
    _PROVIDERS,
    ensure_provider_registered,
    ensure_providers_for_deployment,
    get_provider,
    list_providers,
    register,
)


class _DummyProvider(InferenceProvider):
    def name(self) -> str:
        return "dummy"

    def plan(self, request, vllm_cmd):
        return ProviderPlan(provider="dummy", rendered_script="# dummy")

    def deploy(self, plan):
        return DeploymentResult(provider="dummy", endpoint_url="http://localhost")

    def destroy(self, result):
        pass


class TestRegistry:
    def setup_method(self):
        # Save and restore registry state per test
        self._backup = dict(_PROVIDERS)

    def teardown_method(self):
        _PROVIDERS.clear()
        _PROVIDERS.update(self._backup)

    def test_register_and_get(self):
        register("dummy", _DummyProvider)
        p = get_provider("dummy")
        assert isinstance(p, _DummyProvider)
        assert p.name() == "dummy"

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_list_providers(self):
        register("dummy", _DummyProvider)
        assert "dummy" in list_providers()

    def test_register_overwrites(self):
        register("dummy", _DummyProvider)
        register("dummy", _DummyProvider)
        assert list_providers().count("dummy") == 1


class TestEnsureProviderRegistered:
    def setup_method(self):
        self._backup = dict(_PROVIDERS)

    def teardown_method(self):
        _PROVIDERS.clear()
        _PROVIDERS.update(self._backup)

    def test_already_registered_is_noop(self):
        register("dummy", _DummyProvider)
        # Should not raise or import anything
        ensure_provider_registered("dummy")
        assert "dummy" in _PROVIDERS

    def test_unknown_mapping_raises(self):
        with pytest.raises(ValueError, match="No known module for provider"):
            ensure_provider_registered("totally_unknown_provider")

    def test_provider_modules_contains_known_providers(self):
        assert "modal" in PROVIDER_MODULES
        assert "skyserve" in PROVIDER_MODULES

    def test_lazy_import_registers_modal(self):
        # Remove modal if already registered to test lazy import
        _PROVIDERS.pop("modal", None)
        ensure_provider_registered("modal")
        assert "modal" in _PROVIDERS

    def test_lazy_import_registers_skyserve(self):
        _PROVIDERS.pop("skyserve", None)
        ensure_provider_registered("skyserve")
        assert "skyserve" in _PROVIDERS


class TestEnsureProvidersForDeployment:
    def setup_method(self):
        self._backup = dict(_PROVIDERS)

    def teardown_method(self):
        _PROVIDERS.clear()
        _PROVIDERS.update(self._backup)

    def test_loads_both_providers(self):
        from tuna.state import DeploymentRecord

        _PROVIDERS.pop("modal", None)
        _PROVIDERS.pop("skyserve", None)

        record = DeploymentRecord(
            service_name="test",
            serverless_provider_name="modal",
            spot_provider_name="skyserve",
        )
        ensure_providers_for_deployment(record)
        assert "modal" in _PROVIDERS
        assert "skyserve" in _PROVIDERS

    def test_handles_none_providers(self):
        from tuna.state import DeploymentRecord

        record = DeploymentRecord(
            service_name="test",
            serverless_provider_name=None,
            spot_provider_name=None,
        )
        # Should not raise
        ensure_providers_for_deployment(record)
