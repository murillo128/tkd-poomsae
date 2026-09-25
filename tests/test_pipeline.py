"""Synthetic producers verify the DAG and persistent state without network access."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from contracts.models import ArtifactBase, validate_artifact
from pipeline import CapabilityUnavailable, Pipeline, Stage, StageOutput
from pipeline.runner import DEPENDENCIES, STAGE_LAYERS, STAGE_ORDER, RunCancelled
from storage import ArtifactKey, ArtifactStore, StorageRoot

GOOD = {"state": "observed", "score": 1, "source_ids": []}
IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def artifact(name: str, key: ArtifactKey) -> ArtifactBase:
    kind = STAGE_LAYERS[name]
    value: dict[str, Any] = {
        "kind": kind,
        "id": name,
        "schema_version": "1.0.0",
        "provenance": {"producer": "test", "config_digest": key.config_digest},
    }
    if name == "ingest":
        value.update(
            camera_id="cam",
            width_px=10,
            height_px=10,
            time_base_num=1,
            time_base_den=30,
        )
    elif name == "sync":
        value["offsets"] = [
            {"source_id": side, "automatic_seconds": 0, "quality": GOOD}
            for side in ("left", "right")
        ]
    elif name == "calibration":
        value.update(scale="arbitrary", world_unit="arbitrary", quality=GOOD)
        value["cameras"] = [
            {
                "camera_id": side,
                "source_id": side,
                "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0},
                "world_to_camera": IDENTITY,
                "quality": GOOD,
            }
            for side in ("left", "right")
        ]
    elif name in {"observations", "attachment"}:
        value.update(
            frame={
                "source_id": "left",
                "camera_id": "left",
                "source_seconds": 0,
                "offset_seconds": 0,
                "global_seconds": 0,
            },
            landmarks=[],
        )
    elif name == "reconstruction":
        value.update(
            calibration_id="calibration",
            participant_id="person",
            scale="arbitrary",
            samples=[],
        )
    elif name == "ground":
        value.update(reconstruction_id="reconstruction", scale="arbitrary", samples=[])
    else:
        value.update(
            reconstruction_id="reconstruction",
            ground_id="ground",
            execution={"start": 0, "end": 1},
            steps=[],
            stances=[],
            actions=[],
            phases=[],
            keyframes=[],
            relations=[],
        )
    return validate_artifact(value)


def setup(
    tmp_path: Path, counts: Counter[str], behavior: dict[str, str] | None = None
) -> Pipeline:
    behavior = behavior or {}
    stages = []
    for name in STAGE_ORDER:

        def produce(
            key: ArtifactKey, _inputs: Any, _settings: Any, *, stage: str = name
        ) -> StageOutput:
            counts[stage] += 1
            if behavior.get(stage) == "interrupt":
                raise KeyboardInterrupt
            if behavior.get(stage) == "unavailable":
                raise CapabilityUnavailable(f"provision {stage} resources")
            return StageOutput(artifact(stage, key), diagnostics=(f"{stage} ready",))

        stages.append(Stage(name, produce, STAGE_LAYERS[name], DEPENDENCIES[name]))
    pipe = Pipeline(ArtifactStore(StorageRoot(tmp_path / "data")), tuple(stages))
    sources = {}
    for name in ("left", "right"):
        path = tmp_path / f"{name}.video"
        path.write_bytes(name.encode())
        sources[name] = path
    pipe.register("demo", sources)
    return pipe


def test_reuse_targeted_invalidation_and_explicit_rerun(tmp_path: Path) -> None:
    counts: Counter[str] = Counter()
    pipe = setup(tmp_path, counts)
    first = pipe.analyze("demo")
    assert all(row["status"] == "complete" for row in first["stages"].values())
    assert all(count == 1 for count in counts.values())
    assert all(row["cached"] for row in pipe.analyze("demo")["stages"].values())
    assert all(count == 1 for count in counts.values())

    pipe.analyze("demo", config={"parsing": {"threshold": 2}})
    assert counts["parsing"] == 2
    assert all(counts[name] == 1 for name in STAGE_ORDER if name != "parsing")

    pipe.analyze("demo", config={"sync": {"offset": 0.25}, "parsing": {"threshold": 2}})
    for name in ("sync", "attachment", "reconstruction", "ground", "parsing"):
        assert counts[name] == (3 if name == "parsing" else 2)
    assert counts["observations"] == 1
    assert counts["calibration"] == 2

    pipe.analyze("demo", rerun="observations")
    assert counts["observations"] == 2
    assert counts["sync"] == 2
    assert counts["calibration"] == 2
    assert counts["attachment"] == 3


def test_candidate_bytes_change_calibration_key_without_rerunning_observations(
    tmp_path: Path,
) -> None:
    counts: Counter[str] = Counter()
    pipe = setup(tmp_path, counts)
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"revision":1}')
    config = {"calibration": {"candidate": str(candidate)}}
    pipe.analyze("demo", through="observations", config=config)
    pipe.analyze("demo", through="calibration")
    assert counts["calibration"] == 1
    candidate.write_text('{"revision":2}')
    assert pipe.status("demo")["stages"]["calibration"]["status"] == "stale"
    pipe.analyze("demo", through="calibration")
    assert counts["calibration"] == 2
    assert counts["observations"] == 1


def test_resume_interruption_and_cancel(tmp_path: Path) -> None:
    counts: Counter[str] = Counter()
    behavior = {"sync": "interrupt"}
    pipe = setup(tmp_path, counts, behavior)
    with pytest.raises(KeyboardInterrupt):
        pipe.analyze("demo", "attachment")
    assert pipe.status("demo")["stages"]["sync"]["status"] == "failed"
    behavior.clear()
    state = pipe.analyze("demo", "attachment")
    assert counts["ingest"] == 1
    assert counts["sync"] == 2
    assert state["stages"]["observations"]["status"] == "complete"

    original = pipe.stages["sync"]

    def cancel_during_produce(
        key: ArtifactKey, inputs: Any, settings: Any
    ) -> StageOutput:
        pipe.cancel("demo")
        return original.producer(key, inputs, settings)

    pipe.stages["sync"] = Stage(
        "sync", cancel_during_produce, STAGE_LAYERS["sync"], DEPENDENCIES["sync"]
    )
    with pytest.raises(RunCancelled):
        pipe.analyze("demo", "attachment", rerun="sync")
    cancelled = pipe.status("demo")
    assert cancelled["stages"]["sync"]["status"] == "failed"
    assert cancelled["cancel_requested"]
    pipe.stages["sync"] = original
    assert pipe.analyze("demo", "attachment")["stages"]["sync"]["status"] == "complete"


def test_status_keeps_active_producer_running(tmp_path: Path) -> None:
    counts: Counter[str] = Counter()
    pipe = setup(tmp_path, counts)
    entered, release = Event(), Event()
    original = pipe.stages["ingest"]

    def wait_in_producer(key: ArtifactKey, inputs: Any, settings: Any) -> StageOutput:
        entered.set()
        assert release.wait(5)
        return original.producer(key, inputs, settings)

    pipe.stages["ingest"] = Stage(
        "ingest", wait_in_producer, STAGE_LAYERS["ingest"], DEPENDENCIES["ingest"]
    )
    worker = Thread(target=lambda: pipe.analyze("demo", "ingest"))
    worker.start()
    try:
        assert entered.wait(5)
        live = pipe.status("demo")["stages"]["ingest"]
        assert live["status"] == "running"
        assert live["diagnostics"] == []
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert pipe.status("demo")["stages"]["ingest"]["status"] == "complete"


def test_unavailable_calibration_preserves_native_observations(tmp_path: Path) -> None:
    counts: Counter[str] = Counter()
    pipe = setup(tmp_path, counts, {"calibration": "unavailable"})
    state = pipe.analyze("demo")
    assert state["stages"]["calibration"]["status"] == "unavailable"
    assert state["stages"]["observations"]["status"] == "complete"
    assert state["stages"]["attachment"]["status"] == "complete"
    assert state["stages"]["reconstruction"]["status"] == "unavailable"
    assert counts["observations"] == 1
    assert counts["reconstruction"] == 0


def test_observations_need_neither_sync_nor_calibration(tmp_path: Path) -> None:
    counts: Counter[str] = Counter()
    pipe = setup(
        tmp_path, counts, {"sync": "unavailable", "calibration": "unavailable"}
    )
    state = pipe.analyze("demo", "observations")
    assert state["stages"]["observations"]["status"] == "complete"
    assert counts == Counter({"ingest": 1, "observations": 1})


def test_source_edit_marks_status_stale_and_rebuilds_descendants(
    tmp_path: Path,
) -> None:
    counts: Counter[str] = Counter()
    pipe = setup(tmp_path, counts)
    pipe.analyze("demo", "observations")
    (tmp_path / "left.video").write_bytes(b"changed")
    status = pipe.status("demo")["stages"]
    assert status["ingest"]["status"] == "stale"
    assert status["observations"]["status"] == "stale"
    pipe.analyze("demo", "observations")
    assert counts == Counter({"ingest": 2, "observations": 2})
