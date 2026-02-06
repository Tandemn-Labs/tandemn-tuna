"""Scaling policy dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class SpotScaling:
    min_replicas: int = 0
    max_replicas: int = 5
    target_qps: int = 10
    upscale_delay: int = 5
    downscale_delay: int = 300


@dataclass
class ServerlessScaling:
    concurrency: int = 32
    scaledown_window: int = 60
    timeout: int = 600


@dataclass
class ScalingPolicy:
    spot: SpotScaling
    serverless: ServerlessScaling


_SPOT_KEYS = frozenset(f.name for f in SpotScaling.__dataclass_fields__.values())
_SERVERLESS_KEYS = frozenset(
    f.name for f in ServerlessScaling.__dataclass_fields__.values()
)


def default_scaling_policy() -> ScalingPolicy:
    return ScalingPolicy(spot=SpotScaling(), serverless=ServerlessScaling())


def load_scaling_policy(path: str) -> ScalingPolicy:
    """Load a scaling policy from a YAML file.

    Unknown keys under ``spot`` or ``serverless`` cause a ``ValueError``
    so typos are caught early.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Scaling policy YAML must be a mapping, got {type(raw).__name__}")

    allowed_sections = {"spot", "serverless"}
    unknown_sections = set(raw) - allowed_sections
    if unknown_sections:
        raise ValueError(
            f"Unknown top-level keys in scaling policy: {sorted(unknown_sections)}. "
            f"Allowed: {sorted(allowed_sections)}"
        )

    spot_raw = raw.get("spot", {}) or {}
    serverless_raw = raw.get("serverless", {}) or {}

    _validate_keys("spot", spot_raw, _SPOT_KEYS)
    _validate_keys("serverless", serverless_raw, _SERVERLESS_KEYS)

    return ScalingPolicy(
        spot=SpotScaling(**spot_raw),
        serverless=ServerlessScaling(**serverless_raw),
    )


def _validate_keys(section: str, raw: dict, allowed: frozenset[str]) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"'{section}' must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys in '{section}': {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )
