"""Provider registry — lookup providers by name."""

from __future__ import annotations

from tandemn.providers.base import ServerlessProvider

_PROVIDERS: dict[str, type[ServerlessProvider]] = {}


def register(name: str, cls: type[ServerlessProvider]) -> None:
    """Register a provider class under a name."""
    _PROVIDERS[name] = cls


def get_provider(name: str) -> ServerlessProvider:
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
