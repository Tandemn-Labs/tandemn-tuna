"""Tests for tandemn.providers.registry."""

import pytest

from tandemn.models import DeployRequest, DeploymentResult, ProviderPlan
from tandemn.providers.base import ServerlessProvider
from tandemn.providers.registry import (
    _PROVIDERS,
    get_provider,
    list_providers,
    register,
)


class _DummyProvider(ServerlessProvider):
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
