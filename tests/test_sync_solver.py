"""Resolution-bounded alignment, ambiguity, exclusion, and manual provenance."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from contracts.models import Synchronization
from media import ingest
from pipeline import Pipeline
from storage import ArtifactKey, ArtifactStore, StorageRoot, hash_config
from sync import (
    CueSample,
    CueSet,
    CueStream,
    SyncSource,
    TimelineFailure,
    extract_cues,
    publish_offsets,
    solve_offsets,
    sources_from_manifest,
)

from .test_pipeline import setup
from .test_sync_cues import recording


def _source(
    name: str,
    offset: float,
    *,
    periodic: bool = False,
    audio: bool = True,
    motion: bool = True,
) -> SyncSource:
    step = 0.05
    times = np.arange(0, 8, step)
    events = (
        np.arange(0.7, 7.5, 0.7)
        if periodic
        else np.array([0.7, 1.5, 2.35, 3.6, 4.15, 5.4, 6.8])
    )
    strengths = (
        np.ones(len(events))
        if periodic
        else np.array([0.8, 0.5, 1, 0.65, 0.9, 0.45, 0.75])
    )
    values = sum(
        strength * np.exp(-(((times + offset - event) / 0.08) ** 2))
        for event, strength in zip(events, strengths)
    )
    values = np.clip(values, 0, 1)

    def stream(kind: Literal["audio", "motion"], available: bool) -> CueStream:
        samples = (
            tuple(
                CueSample(float(t), float(t + step), float(v), float(v))
                for t, v in zip(times, values)
            )
            if available
            else ()
        )
        return CueStream(
            kind,
            available,
            1 if available else 0,
            samples,
            (),
            (),
            () if available else ("unavailable",),
            step,
        )

    return SyncSource(
        name,
        0,
        8,
        CueSet(
            name, "0" * 64, "test", stream("audio", audio), stream("motion", motion)
        ),
    )


@pytest.mark.parametrize(
    "offsets",
    [
        {"a": 0, "b": -0.35},
        {"a": 0, "b": -0.35, "c": 0.25},
        {"a": 0, "b": -0.35, "c": 0.25, "d": -0.15},
    ],
)
def test_known_shifts_and_published_artifact(
    tmp_path: Path, offsets: dict[str, float]
) -> None:
    sources = {name: _source(name, offset) for name, offset in offsets.items()}
    key = ArtifactKey(
        "synchronization",
        {name: "0" * 64 for name in sources},
        "1.0.0",
        "solver-v1",
        hash_config({"test": 1}),
    )
    handle = publish_offsets(ArtifactStore(StorageRoot(tmp_path)), key, sources)
    artifact = handle.metadata
    assert artifact.kind == "synchronization"
    assert artifact.reference_source_id == "a"
    for row in artifact.offsets:
        assert abs(row.effective_seconds - offsets[row.source_id]) <= 0.1
        assert row.global_interval is not None
        assert row.global_interval.start == pytest.approx(row.effective_seconds)
    assert artifact.common_interval is not None


def test_periodic_ambiguity_and_missing_audio() -> None:
    with pytest.raises(TimelineFailure):
        solve_offsets(
            {
                name: _source(name, offset, periodic=True)
                for name, offset in {"a": 0, "b": -0.35}.items()
            }
        )
    result = solve_offsets(
        {"a": _source("a", 0, audio=False), "b": _source("b", -0.35, audio=False)}
    )
    assert result.offsets[1].effective_seconds == pytest.approx(-0.35, abs=0.1)


def test_no_overlap_fails_explicitly() -> None:
    first = _source("a", 0)
    second = _source("b", -0.35)
    shifted = SyncSource("b", 20, 28, second.cues)
    with pytest.raises(TimelineFailure):
        solve_offsets({"a": first, "b": shifted})


def test_inconsistent_camera_is_excluded() -> None:
    sources = {
        "a": _source("a", 0),
        "b": _source("b", -0.35),
        "bad": _source("bad", 0.2, periodic=True),
    }
    result = solve_offsets(sources, reference="a")
    assert {r.source_id for r in result.offsets if r.retained} == {"a", "b"}
    assert next(r for r in result.offsets if r.source_id == "bad").exclusion_reason


def test_manual_revision_survives_reload_and_only_stales_timed_stages(
    tmp_path: Path,
) -> None:
    counts: Counter[str] = Counter()
    pipe = setup(tmp_path, counts)
    pipe.analyze("demo")
    revision = pipe.revise_sync_offset(
        "demo",
        "right",
        -0.375,
        author="operator",
        source="visual inspection",
        reason="event mismatch",
    )
    assert revision["offset_seconds"] == -0.375
    reopened = Pipeline(pipe.store, tuple(pipe.stages.values()))
    state = reopened.status("demo")
    assert state["sync_revisions"] == [revision]
    assert state["config"]["sync"]["manual_offsets"]["right"] == revision
    assert state["stages"]["ingest"]["status"] == "complete"
    assert state["stages"]["observations"]["status"] == "complete"
    assert state["stages"]["sync"]["status"] == "stale"
    reopened.analyze("demo", "observations", config={"observations": {"quality": 1}})
    assert (
        reopened.status("demo")["config"]["sync"]["manual_offsets"]["right"] == revision
    )
    result = solve_offsets(
        {"left": _source("left", 0), "right": _source("right", -0.35)},
        reference="left",
        overrides={"right": revision},
    )
    right = next(row for row in result.offsets if row.source_id == "right")
    assert right.automatic_seconds == pytest.approx(-0.35, abs=0.1)
    assert right.effective_seconds == pytest.approx(-0.375)
    assert right.manual_author == "operator"
    assert right.manual_reason == "event mismatch"
    assert right.quality.score is not None  # automatic evidence only


def test_registered_media_manifest_and_runner_publication(tmp_path: Path) -> None:
    events = (0.8, 1.7, 2.8, 4.1, 5.3, 6.6)
    paths = {
        name: recording(
            tmp_path / f"{name}.mkv",
            video_rate=20,
            audio_rate=8000,
            pre_roll=roll,
            events=events,
        )
        for name, roll in (("left", 0.2), ("right", 0.55))
    }
    manifest = ingest(list(paths.items()))
    cues = {item.source_id: extract_cues(item) for item in manifest}
    sources = sources_from_manifest(manifest, cues)
    result = solve_offsets(sources, reference=manifest[0].source_id)
    by_id = {row.source_id: row for row in result.offsets}
    assert by_id[manifest[1].source_id].effective_seconds == pytest.approx(
        -0.35, abs=0.1
    )
    pipe = Pipeline(ArtifactStore(StorageRoot(tmp_path / "data")))
    pipe.register("media", paths)
    handle = pipe.solve_sync("media")
    assert isinstance(handle.metadata, Synchronization)
    assert pipe.status("media")["stages"]["sync"]["status"] == "complete"
    original = {row.source_id: row for row in handle.metadata.offsets}
    pipe.revise_sync_offset(
        "media",
        "right",
        -0.4,
        author="operator",
        source="inspection",
        reason="correct event",
    )
    revised = pipe.solve_sync("media")
    assert isinstance(revised.metadata, Synchronization)
    right = next(
        row
        for row in revised.metadata.offsets
        if row.source_id == manifest[1].source_id
    )
    assert right.automatic_seconds == original[right.source_id].automatic_seconds
    assert right.effective_seconds == -0.4
    assert right.manual_author == "operator"
    assert revised.path != handle.path
