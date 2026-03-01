"""Log watchers for providers with verified log APIs: Modal, CloudRun, Cerebrium."""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class LogPhases:
    """Absolute wall-clock epoch timestamps from provider logs."""

    container_start: Optional[float] = None
    model_load_start: Optional[float] = None
    ready: Optional[float] = None


class LogWatcher:
    """Base class: streams provider logs in a background thread."""

    _PATTERN_MODEL_LOAD = re.compile(
        r"Loading model|Starting to load model|loading model weights"
    )
    _PATTERN_READY = re.compile(
        r"Uvicorn running|Application startup complete|Started server process"
    )

    def __init__(self) -> None:
        self.phases = LogPhases()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        for ts, line in self._stream_lines():
            if self._stop_event.is_set():
                break
            self._process_line(ts, line)

    def _process_line(self, ts: float, line: str) -> None:
        if self.phases.container_start is None:
            self.phases.container_start = ts
        if self.phases.model_load_start is None and self._PATTERN_MODEL_LOAD.search(line):
            self.phases.model_load_start = ts
        if self.phases.ready is None and self._PATTERN_READY.search(line):
            self.phases.ready = ts

    def _stream_lines(self) -> Iterator[tuple[float, str]]:
        raise NotImplementedError


class ModalLogWatcher(LogWatcher):
    """Streams logs via `modal app logs <app_name>`."""

    def __init__(self, app_name: str) -> None:
        super().__init__()
        self.app_name = app_name
        self._proc: Optional[subprocess.Popen] = None

    def _stream_lines(self) -> Iterator[tuple[float, str]]:
        self._proc = subprocess.Popen(
            ["modal", "app", "logs", self.app_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            yield time.time(), line.rstrip()
        self._proc.wait()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        super().stop()


class CloudRunLogWatcher(LogWatcher):
    """Streams logs via `gcloud logging tail`."""

    def __init__(self, service_name: str, project_id: str, region: str) -> None:
        super().__init__()
        self.service_name = service_name
        self.project_id = project_id
        self.region = region
        self._proc: Optional[subprocess.Popen] = None

    def _stream_lines(self) -> Iterator[tuple[float, str]]:
        log_filter = (
            f'resource.type="cloud_run_revision" '
            f'resource.labels.service_name="{self.service_name}" '
            f'resource.labels.location="{self.region}"'
        )
        self._proc = subprocess.Popen(
            [
                "gcloud", "logging", "tail", log_filter,
                f"--project={self.project_id}",
                "--format=value(textPayload)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            yield time.time(), line.rstrip()
        self._proc.wait()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        super().stop()


class CerebriumLogWatcher(LogWatcher):
    """Streams logs via `cerebrium logs <name> --tail`."""

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name
        self._proc: Optional[subprocess.Popen] = None

    def _stream_lines(self) -> Iterator[tuple[float, str]]:
        self._proc = subprocess.Popen(
            ["cerebrium", "logs", self.service_name, "--tail"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            yield time.time(), line.rstrip()
        self._proc.wait()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        super().stop()


class BasetenLogWatcher(LogWatcher):
    """Streams logs via `truss model-logs --model-id <id> --deployment-id <id> --tail`."""

    def __init__(self, model_id: str, deployment_id: str) -> None:
        super().__init__()
        self.model_id = model_id
        self.deployment_id = deployment_id
        self._proc: Optional[subprocess.Popen] = None

    def _stream_lines(self) -> Iterator[tuple[float, str]]:
        self._proc = subprocess.Popen(
            [
                "truss", "model-logs",
                "--model-id", self.model_id,
                "--deployment-id", self.deployment_id,
                "--tail",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            yield time.time(), line.rstrip()
        self._proc.wait()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        super().stop()


def create_log_watcher(provider_name: str, metadata: dict) -> LogWatcher | None:
    """Factory: returns a log watcher for supported providers, else None."""
    if provider_name == "modal":
        app_name = metadata.get("app_name")
        if app_name:
            return ModalLogWatcher(app_name)
    elif provider_name == "cloudrun":
        svc = metadata.get("service_name")
        proj = metadata.get("project_id")
        region = metadata.get("region")
        if svc and proj and region:
            return CloudRunLogWatcher(svc, proj, region)
    elif provider_name == "cerebrium":
        svc = metadata.get("service_name")
        if svc:
            return CerebriumLogWatcher(svc)
    elif provider_name == "baseten":
        model_id = metadata.get("model_id")
        deployment_id = metadata.get("deployment_id")
        if model_id and deployment_id:
            return BasetenLogWatcher(model_id, deployment_id)
    return None
