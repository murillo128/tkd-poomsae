"""Small, synchronous, resumable DAG for one offline project.

State in runs/ is a journal, not a cache. Published artifacts in derived/ are
the only evidence of completion. Each invocation rechecks source bytes and
upstream artifact identities before considering a cached result.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from contracts.models import ArtifactBase
from storage import (
    ArtifactHandle,
    ArtifactKey,
    ArtifactStore,
    MissingResource,
    hash_config,
    hash_file,
)

STAGE_ORDER = (
    "ingest",
    "sync",
    "calibration",
    "observations",
    "attachment",
    "reconstruction",
    "ground",
    "parsing",
)
STAGE_LAYERS = {
    "ingest": "source",
    "sync": "synchronization",
    "calibration": "calibration",
    "observations": "observation",
    "attachment": "observation",
    "reconstruction": "reconstruction",
    "ground": "ground",
    "parsing": "semantics",
}
DEPENDENCIES = {
    "ingest": (),
    "sync": ("ingest",),
    "calibration": ("ingest",),
    "observations": ("ingest",),
    "attachment": ("sync", "observations"),
    "reconstruction": ("calibration", "attachment"),
    "ground": ("reconstruction",),
    "parsing": ("ground",),
}


class CapabilityUnavailable(RuntimeError):
    """A required offline producer, model, or resource is unavailable."""


class RunCancelled(RuntimeError):
    """A persisted cancellation request stopped the run."""


@dataclass(frozen=True)
class StageOutput:
    artifact: ArtifactBase
    arrays: Mapping[str, np.ndarray[Any, Any]] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()
    version: str = "1"


Producer = Callable[
    [ArtifactKey, Mapping[str, ArtifactHandle], Mapping[str, Any]], StageOutput
]


@dataclass(frozen=True)
class Stage:
    name: str
    producer: Producer
    layer: str
    dependencies: tuple[str, ...]
    revision: str = "1"
    model_revision: str | None = None
    schema_version: str = "1.0.0"


def _unavailable(name: str) -> Producer:
    def produce(
        _key: ArtifactKey,
        _inputs: Mapping[str, ArtifactHandle],
        _settings: Mapping[str, Any],
    ) -> StageOutput:
        raise CapabilityUnavailable(
            f"{name} producer is not installed; configure an offline producer "
            "and provision its resources before analysis"
        )

    return produce


def default_stages() -> tuple[Stage, ...]:
    """Explicit stage slots; feature packages replace producers as they land."""
    return tuple(
        Stage(name, _unavailable(name), STAGE_LAYERS[name], DEPENDENCIES[name])
        for name in STAGE_ORDER
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


class Pipeline:
    def __init__(
        self,
        store: ArtifactStore | None = None,
        stages: tuple[Stage, ...] | None = None,
    ) -> None:
        self.store = store or ArtifactStore()
        self.stages = {stage.name: stage for stage in (stages or default_stages())}
        if tuple(self.stages) != STAGE_ORDER or any(
            stage.layer != STAGE_LAYERS[name]
            or stage.dependencies != DEPENDENCIES[name]
            or not stage.revision
            for name, stage in self.stages.items()
        ):
            raise ValueError(
                "stage registry must implement the explicit pipeline graph"
            )

    def _directory(self, project: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", project) or project in {
            ".",
            "..",
        }:
            raise ValueError("invalid project identifier")
        return self.store.root.namespace("runs") / "projects" / project

    def register(self, project: str, sources: Mapping[str, str | Path]) -> None:
        """Record source references. Source bytes stay in their existing location."""
        if len(sources) < 2 or any(not source for source in sources):
            raise ValueError("at least two named sources are required")
        directory = self._directory(project)
        if directory.exists():
            raise ValueError(f"project {project} is already registered")
        resolved = {name: str(Path(path).resolve()) for name, path in sources.items()}
        for path in resolved.values():
            if not Path(path).is_file():
                raise MissingResource(f"Missing source file: {path}")
        directory.mkdir(parents=True)
        _write_json(directory / "project.json", {"version": 1, "sources": resolved})
        _write_json(directory / "state.json", self._initial_state())

    @staticmethod
    def _initial_state() -> dict[str, Any]:
        return {
            "version": 1,
            "generation": {},
            "config": {},
            "stages": {
                name: {"status": "not-run", "progress": 0, "diagnostics": []}
                for name in STAGE_ORDER
            },
            "cancel_requested": False,
        }

    def _files(self, project: str) -> tuple[Path, Path]:
        directory = self._directory(project)
        return directory / "project.json", directory / "state.json"

    def _expected_keys(
        self, project_data: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, ArtifactKey]:
        source_hashes = {
            name: hash_file(Path(path))
            for name, path in project_data["sources"].items()
        }
        keys: dict[str, ArtifactKey] = {}
        for name in STAGE_ORDER:
            stage = self.stages[name]
            inputs = {dep: keys[dep].digest for dep in stage.dependencies}
            if name == "ingest":
                inputs.update(source_hashes)
            keys[name] = ArtifactKey(
                layer=stage.layer,
                inputs=inputs,
                schema_version=stage.schema_version,
                algorithm_revision=stage.revision,
                model_revision=stage.model_revision,
                config_digest=hash_config(
                    {
                        "settings": state["config"].get(name, {}),
                        "generation": state["generation"].get(name, 0),
                    }
                ),
            )
        return keys

    def status(self, project: str) -> dict[str, Any]:
        project_path, state_path = self._files(project)
        state = _read_json(state_path)
        state["cancel_requested"] = state["cancel_requested"] or self._cancelled(
            project
        )
        # A killed process cannot leave a truthful running status.
        keys = self._expected_keys(_read_json(project_path), state)
        for name, record in state["stages"].items():
            if record["status"] == "running":
                record["status"] = "failed"
                record["diagnostics"] = ["interrupted before artifact publication"]
            elif record.get("key") and record["key"] != keys[name].digest:
                record["status"] = "stale"
                record["progress"] = 0
                record["diagnostics"] = ["inputs or revision changed"]
        return state

    def cancel(self, project: str) -> None:
        # Cancellation is a separate file so it can be set while the run lock is held.
        self._directory(project).joinpath("cancel.request").touch()

    def report_progress(self, project: str, stage: str, fraction: float) -> None:
        """Let an active producer persist bounded progress between publications."""
        if not 0 <= fraction < 1:
            raise ValueError("running progress must be between 0 and 1")
        _, state_path = self._files(project)
        state = _read_json(state_path)
        record = state["stages"].get(stage)
        if record is None or record["status"] != "running":
            raise ValueError("stage is not running")
        record["progress"] = fraction
        _write_json(state_path, state)

    def _cancelled(self, project: str) -> bool:
        return (self._directory(project) / "cancel.request").exists()

    def analyze(
        self,
        project: str,
        through: str = "parsing",
        *,
        config: Mapping[str, Mapping[str, Any]] | None = None,
        rerun: str | None = None,
    ) -> dict[str, Any]:
        if through not in STAGE_ORDER or (
            rerun is not None and rerun not in STAGE_ORDER
        ):
            raise ValueError("unknown pipeline stage")
        active: set[str] = set()

        def require(name: str) -> None:
            active.add(name)
            for dependency in DEPENDENCIES[name]:
                require(dependency)

        require(through)
        project_path, state_path = self._files(project)
        project_data = _read_json(project_path)
        directory = state_path.parent
        with (directory / "run.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.status(project)
            if config is not None:
                if set(config) - set(STAGE_ORDER) or any(
                    not isinstance(value, Mapping) for value in config.values()
                ):
                    raise ValueError("config must map stage names to settings objects")
                state["config"] = {name: dict(value) for name, value in config.items()}
            if rerun is not None:
                state["generation"][rerun] = state["generation"].get(rerun, 0) + 1
            (directory / "cancel.request").unlink(missing_ok=True)
            state["cancel_requested"] = False
            _write_json(state_path, state)
            keys = self._expected_keys(project_data, state)
            handles: dict[str, ArtifactHandle] = {}
            for name in STAGE_ORDER:
                stage = self.stages[name]
                settings = state["config"].get(name, {})
                key = keys[name]
                record = state["stages"][name]
                if record.get("key") != key.digest and record["status"] != "not-run":
                    record.update(
                        status="stale",
                        progress=0,
                        diagnostics=["inputs or revision changed"],
                    )
                if name not in active:
                    continue
                record["key"] = key.digest
                record["schema_version"] = stage.schema_version
                record["software_revision"] = stage.revision
                record["model_revision"] = stage.model_revision
                missing = [dep for dep in stage.dependencies if dep not in handles]
                if missing:
                    record.update(
                        status="unavailable",
                        progress=0,
                        diagnostics=[
                            f"required stage unavailable: {', '.join(missing)}"
                        ],
                    )
                    _write_json(state_path, state)
                    continue
                if self._cancelled(project):
                    state["cancel_requested"] = True
                    record.update(
                        status="failed", progress=0, diagnostics=["cancelled"]
                    )
                    _write_json(state_path, state)
                    raise RunCancelled("analysis cancelled; resume to continue")
                try:
                    handle = self.store.get(key)
                except MissingResource:
                    handle = None
                if handle is not None:
                    handles[name] = handle
                    record.update(
                        status="complete",
                        progress=1,
                        diagnostics=record.get("diagnostics", []),
                        cached=True,
                        result_version=record.get("result_version", "1"),
                    )
                    _write_json(state_path, state)
                    continue
                record.update(
                    status="running", progress=0, diagnostics=[], cached=False
                )
                _write_json(state_path, state)
                output: StageOutput | None = None

                def produce() -> tuple[
                    ArtifactBase, Mapping[str, np.ndarray[Any, Any]]
                ]:
                    nonlocal output
                    if self._cancelled(project):
                        raise RunCancelled("analysis cancelled")
                    output = stage.producer(
                        key, {dep: handles[dep] for dep in stage.dependencies}, settings
                    )
                    if output.version != "1":
                        raise ValueError(
                            f"unsupported stage result version: {output.version}"
                        )
                    if self._cancelled(project):
                        raise RunCancelled("analysis cancelled")
                    return output.artifact, output.arrays

                try:
                    handles[name] = self.store.get_or_create(
                        key, produce, cancelled=lambda: self._cancelled(project)
                    )
                except (CapabilityUnavailable, MissingResource) as exc:
                    record.update(
                        status="unavailable", progress=0, diagnostics=[str(exc)]
                    )
                    _write_json(state_path, state)
                    continue
                except RunCancelled as exc:
                    record.update(status="failed", progress=0, diagnostics=[str(exc)])
                    state["cancel_requested"] = True
                    _write_json(state_path, state)
                    raise
                except Exception as exc:
                    record.update(status="failed", progress=0, diagnostics=[str(exc)])
                    state["cancel_requested"] = self._cancelled(project)
                    _write_json(state_path, state)
                    continue
                record.update(
                    status="complete",
                    progress=1,
                    diagnostics=list(output.diagnostics) if output else [],
                    cached=output is None,
                    result_version=output.version if output else "1",
                )
                _write_json(state_path, state)
            return state
