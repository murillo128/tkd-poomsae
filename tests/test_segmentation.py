"""Synthetic acceptance of automatic intervals and non-footstep containers."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from contracts.models import (
    DenseArray,
    MotionFeatures,
    Provenance,
    Quality,
    Segmentation,
    Track,
)
from reconstruction.features import (
    TRACKS,
    EventCandidate,
    FeatureSeries,
    derive_features,
)
from reconstruction.segmentation import (
    SegmentationConfig,
    SegmentationResult,
    load_segmentation,
    publish_segmentation,
    segment_execution,
)
from reconstruction.temporal.core import Derivative
from storage import ArtifactKey, ArtifactStore, StorageRoot, hash_config
from tests.test_motion_features import sequence


# These are offline physical feature fixtures, never CSV or segmentation labels.
@cache
def flat_features() -> FeatureSeries:
    source, ground = sequence(flat=True)
    return derive_features(source, ground)


def features(
    rate: int = 50,
    *,
    bouts: list[tuple[float, float, list[Track]]] | None = None,
    duration: float = 7,
    offset: float = 23,
) -> FeatureSeries:
    base = flat_features()
    template = base.trajectory[:6]
    rows = []
    for i in range(round(duration * rate) + 1):
        t = i / rate
        for j, track in enumerate(TRACKS):
            row = template[j].model_copy(deep=True)
            row.motion_sample_index = i
            row.global_seconds = offset + t
            row.ground_sample_index = None
            active = any(
                a <= t <= b and track in tracks for a, b, tracks in bouts or []
            )
            row.linear = Derivative(
                velocity=(0.4 if active else 0, 0, 0),
                velocity_valid=True,
                velocity_quality=Quality(
                    state="inferred",
                    uncertainty=0.001,
                    source_ids=[base.reconstruction_id],
                ),
                support_times=[offset + t],
                unit="m",
                parent="body",
            )
            row.angular = Derivative(unit="rad")
            rows.append(row)
    return FeatureSeries(
        reconstruction_id=base.reconstruction_id,
        ground_id=base.ground_id,
        config=base.config.model_copy(deep=True),
        trajectory=rows,
        events=[],
    )


def event(
    series: FeatureSeries,
    kind: Any,
    time: float,
    start: float,
    end: float,
    track: Track = "left_leg",
) -> EventCandidate:
    times = [r.global_seconds for r in series.trajectory[::6]]
    indices = [i for i, t in enumerate(times) if start <= t <= end]
    return EventCandidate(
        id=f"{kind}:{time}",
        track=track,
        kind=kind,
        global_seconds=time,
        evidence_start_seconds=times[indices[0]],
        evidence_end_seconds=times[indices[-1]],
        duration_seconds=times[indices[-1]] - times[indices[0]],
        tolerance_seconds=0.02,
        motion_sample_indices=indices,
        quality=Quality(
            state="inferred", uncertainty=0.02, source_ids=[series.reconstruction_id]
        ),
        reasons=["synthetic_physical_event"],
    )


@pytest.mark.parametrize("rate", [25, 50, 100])
def test_pre_post_roll_stationary_arms_pause_and_terminal_hold(rate: int) -> None:
    source = features(
        rate,
        bouts=[
            (1, 1.5, ["left_arm", "right_arm"]),
            (3, 3.5, ["left_arm"]),
            (3.6, 3.9, ["right_arm"]),
        ],
    )
    result = segment_execution(source)
    assert result.execution is not None
    assert result.execution.start == pytest.approx(24, abs=1 / rate)
    assert result.execution.end == pytest.approx(27.9, abs=1 / rate)
    assert len(result.steps) == 2  # All stationary; no foot events at all.
    assert result.steps[0].interval.end == 26
    assert all(not s.action_ids for s in result.steps)
    assert all(
        set(e.active_tracks) <= {"left_arm", "right_arm"} for e in result.step_evidence
    )
    assert result.quality.state == "inferred"
    assert result.activity[0].global_seconds == 23
    assert result.activity[-1].global_seconds == 30
    assert SegmentationResult.model_validate_json(result.model_dump_json()) == result
    # Pause, overlap and asymmetric arm timing remain in original feature rows.
    dense = {i for e in result.step_evidence for i in e.motion_sample_indices}
    assert dense == set(
        range(
            result.boundaries[0].motion_sample_index,
            result.boundaries[-1].motion_sample_index + 1,
        )
    )
    assert source.trajectory[3 * rate * 6].linear.velocity == (0.4, 0, 0)
    assert source.trajectory[3 * rate * 6 + 1].linear.velocity == (0, 0, 0)


def test_kick_recovery_and_placement_stay_in_one_coarse_container() -> None:
    source = features(
        bouts=[
            (1, 1.4, ["left_leg", "body_root"]),
            (2, 2.3, ["left_leg", "body_root"]),
            (4, 4.4, ["left_arm", "right_arm"]),
        ]
    )
    source.events = [
        event(source, "lift_off", 24, 23.9, 24.1),
        event(source, "first_contact", 25.2, 25.1, 25.3),
        event(source, "stable_placement", 25.4, 25.3, 25.5),
    ]
    source = FeatureSeries.model_validate(source.model_dump())
    result = segment_execution(source)
    assert len(result.steps) == 2
    assert result.steps[0].interval.start == 24
    assert result.steps[0].interval.end == 27
    assert result.step_evidence[0].event_ids == [e.id for e in source.events]


@pytest.mark.parametrize(
    "case",
    [
        "flat",
        "head_only",
        "one_track",
        "short_flicker",
        "no_pre_roll",
        "no_post_roll",
        "missing",
        "interpolated",
        "uncertain",
        "native_gap",
    ],
)
def test_insufficient_evidence_is_indeterminate(case: str) -> None:
    source = features(bouts=[(1, 1.5, ["left_arm", "right_arm"])])
    if case in (
        "flat",
        "head_only",
        "one_track",
        "short_flicker",
        "no_pre_roll",
        "no_post_roll",
    ):
        options: dict[str, list[tuple[float, float, list[Track]]]] = {
            "flat": [],
            "head_only": [(1, 1.5, ["head"])],
            "one_track": [(1, 1.5, ["left_arm"])],
            "short_flicker": [(1, 1.02, ["left_arm", "right_arm"])],
            "no_pre_roll": [(0, 1.5, ["left_arm", "right_arm"])],
            "no_post_roll": [(6, 6.5, ["left_arm", "right_arm"])],
        }
        source = features(bouts=options[case])
    elif case == "native_gap":
        for row in source.trajectory:
            if row.global_seconds >= 25:
                row.global_seconds += 1
    else:
        for row in source.trajectory:
            if 24.2 <= row.global_seconds <= 24.5:
                if case == "missing":
                    row.linear = Derivative(unit="m")
                elif case == "uncertain":
                    row.linear.velocity_quality.uncertainty = 2
                else:
                    row.linear.velocity_quality.state = "interpolated"
    result = segment_execution(source)
    assert result.execution is None
    assert not result.steps and not result.boundaries and not result.step_evidence
    assert result.quality.state == "unknown" and result.quality.score is None
    assert result.diagnostics


def persist_features(store: ArtifactStore, source: FeatureSeries) -> Any:
    import json

    array = np.frombuffer(
        json.dumps({"features": source.model_dump(mode="json")}).encode(),
        dtype=np.uint8,
    )
    digest = hash_config(source.config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="motion_features",
        inputs={"fixture": hash_config(source.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision=source.algorithm_revision,
        config_digest=digest,
    )
    metadata = MotionFeatures(
        kind="motion_features",
        id=f"features:{key.digest}",
        schema_version="1.0.0",
        provenance=Provenance(
            producer="reconstruction.features",
            model=source.algorithm_revision,
            config_digest=digest,
        ),
        reconstruction_id=source.reconstruction_id,
        ground_id=source.ground_id,
        arrays=[
            DenseArray(
                id="motion_features_json",
                dtype="uint8",
                shape=[len(array)],
                axes=["json_byte"],
            )
        ],
    )
    return store.get_or_create(key, lambda: (metadata, {"motion_features_json": array}))


def test_persistence_cache_configuration_and_indeterminate_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reconstruction.segmentation.artifact as publisher

    store = ArtifactStore(StorageRoot(tmp_path))
    source = features(bouts=[(1, 1.5, ["left_arm", "right_arm"])])
    upstream = persist_features(store, source)
    output = publish_segmentation(store, upstream)
    assert isinstance(output.metadata, Segmentation)
    result = load_segmentation(output)
    assert result == segment_execution(source)
    changed = publish_segmentation(
        store, upstream, SegmentationConfig(terminal_hold_seconds=2)
    )
    assert changed.path != output.path
    assert load_segmentation(changed).execution is not None
    flat = publish_segmentation(store, persist_features(store, features()))
    assert load_segmentation(flat).execution is None

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must reload without rerunning segmentation or vision")

    monkeypatch.setattr(publisher, "segment_execution", forbidden)
    assert load_segmentation(output) == result
    assert publish_segmentation(store, upstream).path == output.path
    assert not output.read_array("segmentation_evidence_json").flags.writeable


def test_reload_rejects_lost_dense_links() -> None:
    result = segment_execution(features(bouts=[(1, 1.5, ["left_arm", "right_arm"])]))
    payload = result.model_dump()
    payload["step_evidence"][0]["motion_sample_indices"].pop()
    with pytest.raises(ValidationError, match="dense source links"):
        SegmentationResult.model_validate(payload)


def test_real_feature_extraction_feeds_stationary_segmentation() -> None:
    source, ground = sequence(flat=True)
    template = source.samples[0]
    source.samples = []
    for i in range(351):
        sample = template.model_copy(deep=True)
        sample.global_seconds = 23 + i / 50
        for point in sample.landmarks:
            if point.name in ("left_wrist", "right_wrist"):
                peak = 1.3 if point.name == "left_wrist" else 3.3
                pulse = max(0, 1 - abs(i / 50 - peak) / 0.3)
                assert point.xyz_world is not None
                x, _, z = point.xyz_world
                point.xyz_world = (x, 0.3 + 0.4 * pulse, z)
        source.samples.append(sample)
    ground.samples = []
    physical = derive_features(source, ground)
    result = segment_execution(physical)
    assert result.execution is not None, result.diagnostics
    assert 23.9 < result.execution.start < 24.1
    assert 27.5 < result.execution.end < 27.8
    assert len(result.steps) == 2
    assert any(e.kind == "motion_onset" for e in physical.events)


def test_spanning_event_is_retained_on_both_sides_without_clipping() -> None:
    source = features(
        bouts=[(1, 1.5, ["left_arm", "right_arm"]), (3, 3.5, ["left_arm", "right_arm"])]
    )
    # The complete evidence overlaps the coarse boundary; it is linked, not cut.
    source.events = [event(source, "extension_end", 24.4, 24.3, 26.1, "left_arm")]
    result = segment_execution(source)
    assert len(result.steps) == 2
    assert all(source.events[0].id in e.event_ids for e in result.step_evidence)
    assert source.events[0].evidence_end_seconds == 26.1


def test_missing_long_terminal_hold_is_indeterminate() -> None:
    result = segment_execution(
        features(duration=4, bouts=[(1, 1.5, ["left_arm", "right_arm"])]),
        SegmentationConfig(terminal_hold_seconds=3),
    )
    assert result.execution is None
    assert result.diagnostics == ["terminal_hold_or_post_roll_not_observed"]


def test_segmentation_bundle_requires_matching_feature_sources() -> None:
    from contracts.models import validate_bundle
    from tests.test_contracts import base, bundle

    data = bundle()
    motion = next(a for a in data if a["kind"] == "reconstruction")
    ground = next(a for a in data if a["kind"] == "ground")
    data.append(
        base("motion_features", "features")
        | {
            "reconstruction_id": motion["id"],
            "ground_id": ground["id"],
            "arrays": [],
        }
    )
    data.append(
        base("segmentation", "segmentation")
        | {
            "reconstruction_id": motion["id"],
            "ground_id": ground["id"],
            "motion_features_id": "features",
            "execution": None,
            "steps": [],
            "quality": {"state": "unknown"},
            "arrays": [],
        }
    )
    assert isinstance(validate_bundle(data)[-1], Segmentation)
    data[-1]["motion_features_id"] = motion["id"]
    with pytest.raises(ValueError, match="reference"):
        validate_bundle(data)


def test_configured_confidence_rejects_partially_known_interval() -> None:
    source = features(bouts=[(1, 1.5, ["left_arm", "right_arm"])])
    for row in source.trajectory:
        if row.track == "left_leg" and row.global_seconds == 24.3:
            row.linear = Derivative(unit="m")
    accepted = segment_execution(source)
    assert accepted.execution is not None
    assert accepted.quality.score is not None and accepted.quality.score < 1
    rejected = segment_execution(source, SegmentationConfig(min_confidence=1))
    assert rejected.execution is None
    assert rejected.diagnostics == ["confidence_below_threshold"]
