"""Provider registry — lookup providers by name."""

from __future__ import annotations

import importlib

from tandemn.providers.base import InferenceProvider

_PROVIDERS: dict[str, type[InferenceProvider]] = {}

# Maps provider name to the module that registers it on import.
PROVIDER_MODULES: dict[str, str] = {
    "modal": "tandemn.providers.modal_provider",
    "skyserve": "tandemn.spot.sky_launcher",
}


def register(name: str, cls: type[InferenceProvider]) -> None:
    """Register a provider class under a name."""
    _PROVIDERS[name] = cls


def get_provider(name: str) -> InferenceProvider:
    """Instantiate and return a provider by name."""
    cls = _PROVIDERS.get(name)
    if not cls:
        available = list(_PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider: {name!r}. Available: {available}"
        )
    return cls()


def list_providers() -> list[str]:
    """Return names of all registered providers."""
    return list(_PROVIDERS.keys())


def ensure_provider_registered(name: str) -> None:
    """Import the provider module if not already registered."""
    if name in _PROVIDERS:
        return
    module_path = PROVIDER_MODULES.get(name)
    if not module_path:
        raise ValueError(
            f"No known module for provider {name!r}. "
            f"Known providers: {list(PROVIDER_MODULES.keys())}"
        )
    import sys
    mod = importlib.import_module(module_path)
    # If the module was already imported, the register() call at module level
    # won't re-execute. Reload to ensure registration happens.
    if name not in _PROVIDERS:
        importlib.reload(mod)


def ensure_providers_for_deployment(record) -> None:
    """Ensure both serverless and spot providers are registered for a record."""
    if record.serverless_provider_name:
        ensure_provider_registered(record.serverless_provider_name)
    if record.spot_provider_name:
        ensure_provider_registered(record.spot_provider_name)
