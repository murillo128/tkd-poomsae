"""Synthetic automatic arm parsing and durable evidence acceptance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from contracts.models import DenseArray, Provenance, Quality
from reconstruction.arms import (
    ArmConfig,
    ArmResult,
    load_arm_actions,
    parse_arm_actions,
    publish_arm_actions,
)
from reconstruction.detailed import DetailedSample, derive_sample
from reconstruction.detailed.core import Quantity, Relation
from reconstruction.features import FeatureSeries
from reconstruction.features.core import Position
from reconstruction.segmentation import publish_segmentation, segment_execution
from reconstruction.temporal.core import Derivative
from storage import ArtifactKey, ArtifactStore, StorageRoot, hash_config
from tests.test_motion_features import sequence
from tests.test_segmentation import event, features, persist_features


def qualified() -> Quality:
    return Quality(state="inferred", uncertainty=0.001, source_ids=["synthetic-motion"])


def inputs(
    *, centered: bool = False, rate: int = 50
) -> tuple[FeatureSeries, list[DetailedSample]]:
    physical = features(rate, bouts=[(1, 2, ["left_arm"]), (1.2, 2.2, ["right_arm"])])
    source, _ = sequence(flat=True)
    template = derive_sample(
        source.samples[0],
        reconstruction_id=physical.reconstruction_id,
        representation="regularized",
    )
    geometry = []
    for i, first in enumerate(physical.trajectory[::6]):
        t = first.global_seconds - 23
        detail = template.model_copy(deep=True)
        detail.global_seconds = first.global_seconds
        geometry.append(detail)
        for j, track in enumerate(("left_arm", "right_arm")):
            row = physical.trajectory[i * 6 + j]
            row.angular = Derivative(
                velocity=(0, 0, 0),
                velocity_valid=True,
                velocity_quality=qualified(),
                unit="rad",
            )
            if centered:
                # Mirrored inward paths over the common physical movement,
                # asymmetric onset/offset remains visible in track spans.
                x = 0.5 - 0.45 * min(1, max(0, (t - 1.2) / 0.8))
                row.position_local = Position(
                    value=(-x if j == 0 else x, 0.3, 0.5), quality=qualified()
                )
            else:
                row.position_local = Position(
                    value=(0.3 + t * 0.1, 0.4 * j, 0.5), quality=qualified()
                )
    return physical, geometry


@pytest.mark.parametrize("rate", [25, 50, 100])
def test_independent_overlapping_arms_keep_boundaries_and_unknown_role(
    rate: int,
) -> None:
    physical, geometry = inputs(rate=rate)
    result = parse_arm_actions(physical, segment_execution(physical), geometry)
    assert len(result.proposals) == 2
    left, right = result.proposals
    assert left.category == right.category == "arm"
    assert left.role == right.role == "unknown"
    assert left.interval.start == pytest.approx(24)
    assert right.interval.start == pytest.approx(24.2)
    assert left.interval.end == pytest.approx(25)
    assert right.interval.end == pytest.approx(25.2)
    assert left.interval.end > right.interval.start
    assert all(a.step_ids for a in result.proposals)
    assert ArmResult.model_validate_json(result.model_dump_json()) == result


def test_positive_automatic_centering_preserves_asymmetric_track_intervals() -> None:
    physical, geometry = inputs(centered=True)
    result = parse_arm_actions(physical, segment_execution(physical), geometry)
    assert len(result.proposals) == 1
    action = result.proposals[0]
    assert (action.category, action.role) == ("special", "special")
    assert [t.interval.start for t in action.tracks] == [24, 24.2]
    assert [t.interval.end for t in action.tracks] == [25, 25.2]
    assert "mirrored_two_arm_centering" in action.reasons


@pytest.mark.parametrize(
    "case", ["different_height", "no_centering", "unknown", "interpolated", "uncertain"]
)
def test_timing_alone_and_unqualified_geometry_never_merge(case: str) -> None:
    physical, geometry = inputs(centered=True)
    for row in physical.trajectory:
        if row.track != "right_arm":
            continue
        assert row.position_local.value is not None
        x, y, z = row.position_local.value
        if case == "different_height":
            row.position_local.value = (x, y, z + 0.3)
        elif case == "no_centering":
            row.position_local.value = (0.5, y, z)
        elif case == "unknown":
            row.position_local = Position()
        elif case == "interpolated":
            row.position_local.quality.state = "interpolated"
        else:
            row.position_local.quality.uncertainty = 0.1
    result = parse_arm_actions(physical, segment_execution(physical), geometry)
    assert len(result.proposals) == 2
    assert all(a.role == "unknown" for a in result.proposals)


def enrich(physical: FeatureSeries) -> None:
    physical.events = [
        event(physical, "preparation_candidate", 24.1, 24, 24.2, "left_arm"),
        event(physical, "extension_minimum", 24.2, 24.1, 24.3, "left_arm"),
        event(physical, "arm_crossing", 24.3, 24.2, 24.4, "left_arm"),
        event(physical, "extension_start", 24.4, 24.3, 24.5, "left_arm"),
        event(physical, "extension_maximum", 24.6, 24.5, 24.7, "left_arm"),
    ]
    for row in physical.trajectory:
        if row.track not in ("left_arm", "right_arm"):
            continue
        if 24.6 <= row.global_seconds <= 25:
            row.extension = Quantity(
                value=0.95 - (row.global_seconds - 24.6), quality=qualified()
            )
        if 24.2 <= row.global_seconds <= 24.5:
            unknown = row.global_seconds >= 24.4
            row.relations = [
                Relation(
                    subject="left_forearm",
                    object="right_forearm",
                    reference_frame="body",
                    axis="crossing",
                    value="unknown" if unknown else "crossed",
                    quality=Quality(state="unknown") if unknown else qualified(),
                    front_entity=None if unknown else "right_forearm",
                    front_quality=Quality(state="unknown") if unknown else qualified(),
                )
            ]


def test_physical_phase_candidates_crossing_depth_and_hand_evidence() -> None:
    physical, geometry = inputs()
    enrich(physical)
    result = parse_arm_actions(physical, segment_execution(physical), geometry)
    left = result.proposals[0]
    assert {p.name for p in left.phases} == {
        "preparation",
        "chamber",
        "extension",
        "retraction",
    }
    assert all(p.track == "left_arm" for p in left.phases)
    assert any(e.kind == "arm_crossing" for e in left.events)
    crossing = [r for s in left.tracks[0].snapshots for r in s.relations]
    assert any(r.front_entity == "right_forearm" for r in crossing)
    assert any(
        r.value == "unknown"
        and r.front_entity is None
        and r.front_quality.state == "unknown"
        for r in crossing
    )
    assert left.tracks[0].snapshots[0].hand == geometry[50].hands["left"]
    assert left.role == "unknown"


def persist_motion(
    store: ArtifactStore, physical: FeatureSeries, geometry: list[DetailedSample]
) -> Any:
    source, _ = sequence(flat=True)
    source.id = physical.reconstruction_id
    source.samples = []
    array = np.frombuffer(
        json.dumps(
            {
                "version": 1,
                "artifact_role": "derived_detailed_geometry",
                "samples": [g.model_dump(mode="json") for g in geometry],
            }
        ).encode(),
        dtype=np.uint8,
    )
    digest = hash_config({"fixture": "hands"})
    key = ArtifactKey(
        layer="reconstruction",
        inputs={"fixture": hash_config(physical.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision="detailed-geometry-v1",
        config_digest=digest,
    )
    source.provenance = Provenance(
        producer="reconstruction.detailed",
        model="detailed-geometry-v1",
        config_digest=digest,
    )
    source.arrays = [
        DenseArray(
            id="detailed_geometry_json",
            dtype="uint8",
            shape=[len(array)],
            axes=["json_byte"],
        )
    ]
    return store.get_or_create(key, lambda: (source, {"detailed_geometry_json": array}))


@pytest.mark.parametrize("centered", [False, True])
def test_reload_without_vision_or_parsing_and_config_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    centered: bool,
) -> None:
    import reconstruction.arms.artifact as publisher

    physical, geometry = inputs(centered=centered)
    enrich(physical)
    store = ArtifactStore(StorageRoot(tmp_path))
    upstream = persist_features(store, physical)
    coarse = publish_segmentation(store, upstream)
    motion = persist_motion(store, physical, geometry)
    handle = publish_arm_actions(store, upstream, coarse, motion)
    result = load_arm_actions(handle)
    assert result == parse_arm_actions(physical, segment_execution(physical), geometry)
    assert any(
        r.front_entity == "right_forearm"
        for a in result.proposals
        for t in a.tracks
        for s in t.snapshots
        for r in s.relations
    )
    changed = publish_arm_actions(
        store, upstream, coarse, motion, ArmConfig(center_ratio=0.05)
    )
    assert changed.path != handle.path

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not rerun automatic parsing, geometry or vision")

    monkeypatch.setattr(publisher, "parse_arm_actions", forbidden)
    assert load_arm_actions(handle) == result
    assert publish_arm_actions(store, upstream, coarse, motion).path == handle.path
    assert not handle.read_array("arm_action_evidence_json").flags.writeable


def test_mismatched_sources_and_times_fail_closed() -> None:
    physical, geometry = inputs()
    coarse = segment_execution(physical)
    coarse.reconstruction_id = "another-motion"
    with pytest.raises(ValueError, match="exact motion input"):
        parse_arm_actions(physical, coarse, geometry)
    coarse.reconstruction_id = physical.reconstruction_id
    geometry[0].global_seconds += 0.001
    with pytest.raises(ValueError, match="native time"):
        parse_arm_actions(physical, coarse, geometry)


def test_missing_measurements_split_tracks_and_no_action_for_unknown_execution() -> (
    None
):
    physical, geometry = inputs()
    coarse = segment_execution(physical)
    for row in physical.trajectory:
        if row.track == "left_arm" and 24.4 <= row.global_seconds <= 24.6:
            row.linear = Derivative(unit="m")
    result = parse_arm_actions(physical, coarse, geometry)
    left = [a for a in result.proposals if a.tracks[0].track == "left_arm"]
    assert len(left) == 2
    assert left[0].interval.end < 24.4 and left[1].interval.start > 24.6
    coarse.execution = None
    coarse.steps = []
    result = parse_arm_actions(physical, coarse, geometry)
    assert not result.proposals and "execution_indeterminate" in result.diagnostics


def test_persisted_dense_links_reject_loss() -> None:
    physical, geometry = inputs()
    result = parse_arm_actions(physical, segment_execution(physical), geometry)
    payload = result.model_dump()
    payload["proposals"][0]["tracks"][0]["motion_sample_indices"].pop()
    with pytest.raises(ValidationError, match="dense track links"):
        ArmResult.model_validate(payload)


def test_crossing_can_be_known_while_depth_and_hand_are_unknown() -> None:
    physical, geometry = inputs()
    for row in physical.trajectory:
        if row.track == "left_arm" and row.global_seconds == 24.3:
            row.relations = [
                Relation(
                    subject="left_forearm",
                    object="right_forearm",
                    reference_frame="body",
                    axis="crossing",
                    value="crossed",
                    quality=qualified(),
                    front_entity=None,
                    front_quality=Quality(state="unknown"),
                )
            ]
    geometry[65].hands = {}
    result = parse_arm_actions(physical, segment_execution(physical), geometry)
    snapshot = next(
        s for s in result.proposals[0].tracks[0].snapshots if s.global_seconds == 24.3
    )
    assert snapshot.hand is None
    assert snapshot.relations[0].value == "crossed"
    assert snapshot.relations[0].front_entity is None
    assert snapshot.relations[0].front_quality.state == "unknown"


def test_actual_geometric_features_feed_automatic_special_detector() -> None:
    from reconstruction.features import derive_features

    source, ground = sequence(flat=True)
    template = source.samples[0]
    source.samples = []
    geometry = []
    for i in range(351):
        sample = template.model_copy(deep=True)
        sample.global_seconds = 23 + i / 50
        x = 0.5 - 0.45 * min(1, max(0, (i / 50 - 1) / 1))
        for point in sample.landmarks:
            if point.name in ("left_wrist", "right_wrist"):
                point.xyz_world = (-x if point.name == "left_wrist" else x, 0.3, 1.6)
        source.samples.append(sample)
        geometry.append(
            derive_sample(
                sample, reconstruction_id=source.id, representation="regularized"
            )
        )
    physical = derive_features(source, ground, geometry=geometry)
    coarse = segment_execution(physical)
    assert coarse.execution is not None, coarse.diagnostics
    result = parse_arm_actions(physical, coarse, geometry)
    assert len(result.proposals) == 1
    assert result.proposals[0].category == "special"
    assert any(e.kind == "motion_onset" for e in result.proposals[0].events)


def test_bundle_rejects_wrong_feature_or_segmentation_identity() -> None:
    from contracts.models import ArmActions, validate_bundle
    from tests.test_contracts import base, bundle

    data = bundle()
    motion = next(a for a in data if a["kind"] == "reconstruction")
    ground = next(a for a in data if a["kind"] == "ground")
    links = {"reconstruction_id": motion["id"], "ground_id": ground["id"]}
    data.extend(
        [
            base("motion_features", "features") | links | {"arrays": []},
            base("segmentation", "coarse")
            | links
            | {
                "motion_features_id": "features",
                "execution": None,
                "steps": [],
                "quality": {"state": "unknown"},
                "arrays": [],
            },
            base("arm_actions", "arms")
            | links
            | {
                "motion_features_id": "features",
                "segmentation_id": "coarse",
                "arrays": [],
            },
        ]
    )
    assert isinstance(validate_bundle(data)[-1], ArmActions)
    data[-1]["segmentation_id"] = "features"
    with pytest.raises(ValueError, match="reference"):
        validate_bundle(data)


def test_native_gap_splits_arm_even_when_coarse_gap_policy_is_looser() -> None:
    from reconstruction.segmentation import SegmentationConfig

    physical, geometry = inputs()
    for row in physical.trajectory:
        if row.global_seconds >= 24.5:
            row.global_seconds += 1
    for sample in geometry:
        if sample.global_seconds >= 24.5:
            sample.global_seconds += 1
    coarse = segment_execution(physical, SegmentationConfig(max_gap_seconds=2))
    assert coarse.execution is not None
    result = parse_arm_actions(physical, coarse, geometry)
    left = [t for a in result.proposals for t in a.tracks if t.track == "left_arm"]
    assert len(left) == 2
    assert left[0].interval.end < 24.5 and left[1].interval.start >= 25.5


def test_coarse_step_cut_associates_without_clipping_action() -> None:
    from contracts.models import Interval, SequenceStep
    from reconstruction.segmentation.core import Boundary, StepEvidence

    physical, geometry = inputs()
    coarse = segment_execution(physical)
    assert coarse.execution is not None
    cut = 24.6
    middle = 80
    start = coarse.boundaries[0].motion_sample_index
    end = coarse.boundaries[-1].motion_sample_index
    coarse.steps = [
        SequenceStep(
            id="first",
            interval=Interval(start=coarse.execution.start, end=cut),
            action_ids=[],
        ),
        SequenceStep(
            id="second",
            interval=Interval(start=cut, end=coarse.execution.end),
            action_ids=[],
        ),
    ]
    coarse.boundaries.insert(
        1,
        Boundary(
            global_seconds=cut,
            motion_sample_index=middle,
            reason="synthetic_coarse_proposal",
            event_ids=[],
            tolerance_seconds=0.02,
        ),
    )
    coarse.step_evidence = [
        StepEvidence(
            step_id=name,
            motion_sample_indices=list(range(a, b + 1)),
            event_ids=[],
            active_tracks=["left_arm", "right_arm"],
        )
        for name, a, b in (("first", start, middle), ("second", middle, end))
    ]
    result = parse_arm_actions(physical, coarse, geometry)
    assert len(result.proposals) == 2
    assert all(a.step_ids == ["first", "second"] for a in result.proposals)
    assert result.proposals[0].interval.end == 25
    assert result.proposals[1].interval.start == 24.2
