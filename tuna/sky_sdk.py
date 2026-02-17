"""Thin synchronous wrappers around SkyPilot's Python SDK.

Every public function calls the async SDK, then blocks via ``sky.get()``
so callers get plain return values instead of ``RequestId`` futures.
"""

from __future__ import annotations

import yaml

import sky
from sky import ClusterStatus
from sky.serve import ServiceStatus


def serve_up(
    task: sky.Task | sky.Dag,
    service_name: str,
) -> tuple[str, str]:
    """Launch a SkyServe service.  Returns ``(service_name, endpoint)``."""
    req = sky.serve.up(task, service_name)
    return sky.get(req)


def serve_down(
    service_names: str | list[str] | None,
    purge: bool = False,
) -> None:
    """Tear down one or more SkyServe services."""
    req = sky.serve.down(service_names, purge=purge)
    sky.get(req)


def serve_status(
    service_names: str | list[str] | None = None,
) -> list[dict]:
    """Return status dicts for SkyServe services."""
    req = sky.serve.status(service_names)
    return sky.get(req)


def cluster_launch(
    task: sky.Task | sky.Dag,
    cluster_name: str | None = None,
    *,
    down: bool = False,
) -> tuple[int | None, object | None]:
    """Launch a cluster.  Returns ``(job_id, handle)``."""
    req = sky.launch(task, cluster_name=cluster_name, down=down)
    return sky.get(req)


def cluster_status(
    cluster_names: list[str] | None = None,
) -> list:
    """Return ``StatusResponse`` objects for clusters."""
    req = sky.status(cluster_names)
    return sky.get(req)


def cluster_down(cluster_name: str, purge: bool = False) -> None:
    """Tear down a cluster."""
    req = sky.down(cluster_name, purge=purge)
    sky.get(req)


def task_from_yaml_str(yaml_str: str) -> sky.Task:
    """Build a ``sky.Task`` from a YAML string (no temp file needed)."""
    config = yaml.safe_load(yaml_str)
    return sky.Task.from_yaml_config(config)
