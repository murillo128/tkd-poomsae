"""Small durable, bounded queue for local offline pipeline invocations."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from pipeline import Pipeline
from pipeline.runner import DEPENDENCIES, RunCancelled

INTERRUPTED = "service stopped before job completion; submit a new run"


class BusyProject(RuntimeError):
    """A project already has a queued or running service job."""


class QueueFull(RuntimeError):
    """The local job queue has reached its configured bound."""


def required_stages(through: str) -> set[str]:
    required: set[str] = set()

    def visit(stage: str) -> None:
        required.add(stage)
        for dependency in DEPENDENCIES[stage]:
            visit(dependency)

    visit(through)
    return required


def _save(path: Path, record: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".job-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class JobQueue:
    def __init__(self, pipeline: Pipeline, *, workers: int = 1, capacity: int = 16):
        if workers < 1 or capacity < 1:
            raise ValueError("workers and queue capacity must be positive")
        self.pipeline = pipeline
        self.directory = pipeline.store.root.namespace("runs") / "service-jobs"
        self.workers = workers
        self.capacity = capacity
        self.pending: deque[str] = deque()
        self.lock = threading.RLock()
        self.ready = threading.Condition(self.lock)
        self.stopping = False
        self.threads: list[threading.Thread] = []
        self.active: dict[str, str] = {}
        self.service_lock: Any = None

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.service_lock = (self.directory / "service.lock").open("a+b")
        try:
            fcntl.flock(self.service_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.service_lock.close()
            raise RuntimeError("another local service owns this data root") from exc
        with self.lock:
            self.stopping = False
            for path in self.directory.glob("*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["status"] in {"queued", "running"}:
                    record.update(
                        status="interrupted",
                        error=INTERRUPTED,
                    )
                    _save(path, record)
        for index in range(self.workers):
            worker = threading.Thread(
                target=self._work, name=f"analysis-{index}", daemon=True
            )
            worker.start()
            self.threads.append(worker)

    def stop(self) -> None:
        with self.ready:
            self.stopping = True
            while self.pending:
                job_id = self.pending.popleft()
                record = self._read(job_id)
                if record["status"] == "queued":
                    record.update(status="interrupted", error=INTERRUPTED)
                    _save(self._path(job_id), record)
                    self.active.pop(record["project"], None)
            self.ready.notify_all()
        for worker in self.threads:
            worker.join()
        self.threads.clear()
        if self.service_lock is not None:
            fcntl.flock(self.service_lock, fcntl.LOCK_UN)
            self.service_lock.close()
            self.service_lock = None

    def _path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("invalid run identifier")
        return self.directory / f"{job_id}.json"

    def _read(self, job_id: str) -> dict[str, Any]:
        record: dict[str, Any] = json.loads(
            self._path(job_id).read_text(encoding="utf-8")
        )
        return record

    def submit(self, project: str, through: str, rerun: str | None) -> str:
        with self.lock:
            if project in self.active:
                raise BusyProject(f"project {project} already has an active run")
            if len(self.pending) >= self.capacity:
                raise QueueFull("local analysis queue is full")
            self.pipeline._directory(project).joinpath("cancel.request").unlink(
                missing_ok=True
            )
            job_id = uuid.uuid4().hex
            _save(
                self._path(job_id),
                {
                    "id": job_id,
                    "project": project,
                    "through": through,
                    "rerun": rerun,
                    "status": "queued",
                    "error": None,
                },
            )
            self.active[project] = job_id
            self.pending.append(job_id)
            self.ready.notify()
            return job_id

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            record = self._read(job_id)
        if record["status"] == "running":
            record["stages"] = self.pipeline.status(record["project"])["stages"]
        return record

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            record = self._read(job_id)
            if record["status"] == "queued":
                self.pending.remove(job_id)
                record.update(status="cancelled", error="cancelled before execution")
                _save(self._path(job_id), record)
                self.active.pop(record["project"], None)
            elif record["status"] == "running":
                self.pipeline.cancel(record["project"])
                record["cancel_requested"] = True
                _save(self._path(job_id), record)
            return record

    def _work(self) -> None:
        while True:
            with self.ready:
                while not self.pending and not self.stopping:
                    self.ready.wait()
                if self.stopping:
                    return
                job_id = self.pending.popleft()
                record = self._read(job_id)
                record["status"] = "running"
                _save(self._path(job_id), record)
            try:
                # A shared file lease serializes device work across service processes.
                lease_path = self.directory / "analysis-device.lock"
                with lease_path.open("a+b") as lease:
                    fcntl.flock(lease, fcntl.LOCK_EX)
                    try:
                        result = self.pipeline.analyze(
                            record["project"],
                            record["through"],
                            rerun=record["rerun"],
                            reset_cancellation=False,
                        )
                    finally:
                        fcntl.flock(lease, fcntl.LOCK_UN)
                statuses = {
                    result["stages"][name]["status"]
                    for name in required_stages(record["through"])
                }
                record["status"] = (
                    "failed"
                    if "failed" in statuses
                    else "unavailable"
                    if "unavailable" in statuses
                    else "complete"
                )
                record["stages"] = result["stages"]
            except RunCancelled as exc:
                record.update(status="cancelled", error=str(exc))
            except Exception as exc:
                record.update(status="failed", error=str(exc))
            finally:
                with self.lock:
                    _save(self._path(job_id), record)
                    if self.active.get(record["project"]) == job_id:
                        del self.active[record["project"]]
