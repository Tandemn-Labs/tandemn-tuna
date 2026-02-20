"""SQLite persistence for deployment metadata."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class DeploymentRecord:
    """Lightweight read-side representation of a persisted deployment."""

    service_name: str
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    model_name: str = ""
    gpu: str = ""
    gpu_count: int = 1
    serverless_provider: str = ""
    spots_cloud: str = ""
    region: Optional[str] = None
    request_json: str = ""
    # Router
    router_endpoint: Optional[str] = None
    router_metadata: Optional[dict] = field(default_factory=dict)
    # Serverless
    serverless_provider_name: Optional[str] = None
    serverless_endpoint: Optional[str] = None
    serverless_metadata: Optional[dict] = field(default_factory=dict)
    # Spot
    spot_provider_name: Optional[str] = None
    spot_endpoint: Optional[str] = None
    spot_metadata: Optional[dict] = field(default_factory=dict)
    # Combined
    router_url: Optional[str] = None


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS deployments (
    service_name            TEXT PRIMARY KEY,
    status                  TEXT NOT NULL DEFAULT 'active',
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    model_name              TEXT NOT NULL,
    gpu                     TEXT NOT NULL,
    gpu_count               INTEGER NOT NULL DEFAULT 1,
    serverless_provider     TEXT NOT NULL,
    spots_cloud             TEXT NOT NULL,
    region                  TEXT,
    request_json            TEXT NOT NULL,
    router_endpoint         TEXT,
    router_metadata         TEXT,
    serverless_provider_name TEXT,
    serverless_endpoint     TEXT,
    serverless_metadata     TEXT,
    spot_provider_name      TEXT,
    spot_endpoint           TEXT,
    spot_metadata           TEXT,
    router_url              TEXT
);
"""


def _state_dir() -> Path:
    """Return the state directory, respecting TUNA_STATE_DIR env var."""
    env = os.environ.get("TUNA_STATE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".tuna"


def _db_path() -> Path:
    """Return the default database path."""
    return _state_dir() / "deployments.db"


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection, enable WAL mode, and ensure the schema exists."""
    path = Path(db_path) if db_path else _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_deployment(request, result, *, db_path=None) -> None:
    """Persist a deployment after launch_hybrid completes.

    Parameters
    ----------
    request : DeployRequest
    result : HybridDeployment
    """
    conn = _connect(db_path)
    try:
        now = _now_iso()

        # Serialize the full request (dataclass → dict → JSON)
        from dataclasses import asdict
        request_dict = asdict(request)
        request_json = json.dumps(request_dict, default=str)

        # Extract per-component data from HybridDeployment
        router_endpoint = None
        router_metadata = None
        if result.router:
            router_endpoint = result.router.endpoint_url
            router_metadata = json.dumps(result.router.metadata)

        # Always save provider names from the request so destroy can find
        # resources even when the deploy was interrupted before results arrived.
        serverless_provider_name = request.serverless_provider
        serverless_endpoint = None
        serverless_metadata = None
        if result.serverless:
            serverless_provider_name = result.serverless.provider
            serverless_endpoint = result.serverless.endpoint_url
            serverless_metadata = json.dumps(result.serverless.metadata)

        spot_provider_name = None if request.serverless_only else "skyserve"
        spot_endpoint = None
        spot_metadata = None
        if result.spot:
            spot_provider_name = result.spot.provider
            spot_endpoint = result.spot.endpoint_url
            spot_metadata = json.dumps(result.spot.metadata)

        conn.execute(
            """INSERT INTO deployments (
                service_name, status, created_at, updated_at,
                model_name, gpu, gpu_count, serverless_provider, spots_cloud, region,
                request_json,
                router_endpoint, router_metadata,
                serverless_provider_name, serverless_endpoint, serverless_metadata,
                spot_provider_name, spot_endpoint, spot_metadata,
                router_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(service_name) DO UPDATE SET
                status=excluded.status,
                updated_at=excluded.updated_at,
                model_name=excluded.model_name,
                gpu=excluded.gpu,
                gpu_count=excluded.gpu_count,
                serverless_provider=excluded.serverless_provider,
                spots_cloud=excluded.spots_cloud,
                region=excluded.region,
                request_json=excluded.request_json,
                router_endpoint=excluded.router_endpoint,
                router_metadata=excluded.router_metadata,
                serverless_provider_name=excluded.serverless_provider_name,
                serverless_endpoint=excluded.serverless_endpoint,
                serverless_metadata=excluded.serverless_metadata,
                spot_provider_name=excluded.spot_provider_name,
                spot_endpoint=excluded.spot_endpoint,
                spot_metadata=excluded.spot_metadata,
                router_url=excluded.router_url""",
            (
                request.service_name,
                "active",
                now,
                now,
                request.model_name,
                request.gpu,
                request.gpu_count,
                request.serverless_provider,
                request.spots_cloud,
                request.region,
                request_json,
                router_endpoint,
                router_metadata,
                serverless_provider_name,
                serverless_endpoint,
                serverless_metadata,
                spot_provider_name,
                spot_endpoint,
                spot_metadata,
                result.router_url,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _row_to_record(row: tuple, columns: list[str]) -> DeploymentRecord:
    """Convert a DB row to a DeploymentRecord, parsing JSON fields."""
    data = dict(zip(columns, row))

    # Parse JSON metadata fields back into dicts
    for key in ("router_metadata", "serverless_metadata", "spot_metadata"):
        val = data.get(key)
        if val and isinstance(val, str):
            data[key] = json.loads(val)
        elif val is None:
            data[key] = {}

    return DeploymentRecord(**data)


def load_deployment(service_name: str, *, db_path=None) -> DeploymentRecord | None:
    """Load a single deployment record by service name."""
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM deployments WHERE service_name = ?",
            (service_name,),
        )
        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_record(row, columns)
    finally:
        conn.close()


def update_deployment_status(service_name: str, status: str, *, db_path=None) -> None:
    """Update the status of a deployment (e.g. active -> destroyed)."""
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE deployments SET status = ?, updated_at = ? WHERE service_name = ?",
            (status, _now_iso(), service_name),
        )
        conn.commit()
    finally:
        conn.close()


def list_deployments(*, status: str | None = None, db_path=None) -> list[DeploymentRecord]:
    """List deployment records, optionally filtered by status."""
    conn = _connect(db_path)
    try:
        if status:
            cursor = conn.execute(
                "SELECT * FROM deployments WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM deployments ORDER BY created_at DESC"
            )
        columns = [desc[0] for desc in cursor.description]
        return [_row_to_record(row, columns) for row in cursor.fetchall()]
    finally:
        conn.close()
