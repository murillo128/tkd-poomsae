"""Compound motion remains navigable through final canonical semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from contracts.models import Ground, Interval, Semantics, SequenceStep
from reconstruction.arms import ArmResult, parse_arm_actions, publish_arm_actions
from reconstruction.detailed import DetailedSample, derive_sample
from reconstruction.detailed.core import Relation
from reconstruction.features import TRACKS, FeatureSeries
from reconstruction.features.core import Position
from reconstruction.lower_body import (
    LowerBodyResult,
    parse_lower_body,
    publish_lower_body,
)
from reconstruction.segmentation import (
    SegmentationResult,
    publish_segmentation,
    segment_execution,
)
from reconstruction.segmentation.core import Boundary, StepEvidence
from reconstruction.semantics import (
    AssemblyConfig,
    assemble_semantics,
    load_semantic_evidence,
    load_semantics,
    publish_semantics,
)
from reconstruction.temporal.core import Derivative
from storage import ArtifactKey, ArtifactStore, StorageRoot, hash_config, hash_file
from tests.test_arm_actions import persist_motion
from tests.test_lower_body import fixture, q
from tests.test_motion_features import sequence
from tests.test_segmentation import event


def compound(
    centered: bool = False,
) -> tuple[FeatureSeries, Ground, list[DetailedSample]]:
    physical, ground = fixture(kick=True)
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
        for j in (0, 1):
            row = physical.trajectory[i * 6 + j]
            start, end = (0.96, 1.9) if j == 0 else (1.1, 2.1)
            row.linear = Derivative(
                velocity=(0.4 if start <= t <= end else 0, 0, 0),
                velocity_valid=True,
                velocity_quality=q(),
                unit="m",
            )
            row.angular = Derivative(
                velocity=(0, 0, 0),
                velocity_valid=True,
                velocity_quality=q(),
                unit="rad",
            )
            x = 0.5 - 0.45 * min(1, max(0, (t - 1.1) / 0.8))
            row.position_local = Position(
                value=(-x if j == 0 else x, 0.3, 0.5)
                if centered
                else (0.3 + t * 0.1, 0.4 * j, 0.5),
                quality=q(),
            )
            if 24.2 <= row.global_seconds <= 24.4:
                row.relations = [
                    Relation(
                        subject="left_forearm",
                        object="right_forearm",
                        reference_frame="torso",
                        axis="crossing",
                        value="crossed",
                        quality=q(),
                        front_entity=None,
                    )
                ]
    physical.events = sorted(
        [
            event(physical, "lift_off", 23.8, 23.78, 23.84),
            event(physical, "extension_maximum", 24.3, 24.2, 24.4),
            event(physical, "first_contact", 25, 24.98, 25.02),
            event(physical, "stable_placement", 25.1, 25.04, 25.2),
            event(physical, "arm_crossing", 24.3, 24.2, 24.4, "left_arm"),
            event(physical, "direction_change", 24.42, 24.4, 24.46, "left_arm"),
        ],
        key=lambda e: (e.global_seconds, e.track, e.kind),
    )
    return FeatureSeries.model_validate(physical.model_dump()), ground, geometry


def split(
    coarse: SegmentationResult,
    start: float | None = None,
    end: float | None = None,
    cut: float = 24.2,
) -> SegmentationResult:
    assert coarse.execution is not None
    times = [a.global_seconds for a in coarse.activity]
    edges = [start or coarse.execution.start, cut, end or coarse.execution.end]
    cuts = [times.index(t) for t in edges]
    coarse.execution = Interval(start=edges[0], end=edges[-1])
    coarse.steps = [
        SequenceStep(id=f"coarse-{i}", interval=Interval(start=a, end=b), action_ids=[])
        for i, (a, b) in enumerate(zip(edges, edges[1:]))
    ]
    coarse.boundaries = [
        Boundary(
            global_seconds=times[i],
            motion_sample_index=i,
            reason="synthetic_coarse_boundary",
            event_ids=[],
            tolerance_seconds=0.02,
        )
        for i in cuts
    ]
    coarse.step_evidence = [
        StepEvidence(
            step_id=s.id,
            motion_sample_indices=list(range(a, b + 1)),
            event_ids=[],
            active_tracks=[],
        )
        for s, a, b in zip(coarse.steps, cuts, cuts[1:])
    ]
    return SegmentationResult.model_validate(coarse.model_dump())


def assembled(
    centered: bool = False,
) -> tuple[Semantics, FeatureSeries, ArmResult, LowerBodyResult]:
    physical, floor, geometry = compound(centered)
    coarse = split(segment_execution(physical))
    arms = parse_arm_actions(physical, coarse, geometry)
    lower = parse_lower_body(physical, floor, coarse)
    return assemble_semantics(physical, coarse, arms, lower), physical, arms, lower


def test_compound_kick_recovery_and_overlapping_arms_cross_coarse_boundary() -> None:
    result, physical, arms, lower = assembled()
    assert result.execution is not None
    kick = next(a for a in result.actions if a.category == "kick")
    placement = next(a for a in result.actions if a.category == "placement")
    assert placement.previous_action_id == kick.id
    assert kick.interval.end == placement.interval.start
    assert kick.interval == next(
        a.interval for a in lower.actions if a.category == "kick"
    )
    assert kick.step_id == result.steps[0].id
    assert all(kick.id in step.action_ids for step in result.steps)
    assert (
        len(
            [
                a
                for a in result.actions
                if a.source_candidate_id == kick.source_candidate_id
            ]
        )
        == 1
    )
    arm_actions = [a for a in result.actions if a.category == "arm"]
    assert len(arm_actions) == len(arms.proposals) == 2
    assert arm_actions[0].interval != arm_actions[1].interval
    assert arm_actions[0].interval.end > kick.interval.start
    assert {p.name for p in result.phases if p.action_id == kick.id} >= {
        "chamber",
        "extension",
        "retraction",
    }
    frames = [k for k in result.keyframes if k.action_id == kick.id]
    assert "extension_maximum" in {k.event for k in frames}
    times = sorted(set(k.global_seconds for k in frames))
    assert len(set(round(b - a, 6) for a, b in zip(times, times[1:]))) > 1
    assert not any("impact" in k.event or "force" in k.event for k in result.keyframes)
    source_times = [r.global_seconds for r in physical.trajectory[::6]]
    for track in TRACKS:
        links = [
            span_link
            for a in result.actions
            for span_link in a.motion_links
            if span_link.track == track
        ]
        retained = {i for span_link in links for i in span_link.motion_sample_indices}
        assert retained == {
            i
            for i, t in enumerate(source_times)
            if result.execution.start <= t <= result.execution.end
        }
    transitions = [a for a in result.actions if a.category == "transition"]
    assert transitions and all(a.quality.state == "unknown" for a in transitions)
    crossings = [r for r in result.relations if r.relation == "crossed"]
    assert crossings and all(
        r.front_entity is None and r.front_quality.state == "unknown" for r in crossings
    )
    assert Semantics.model_validate_json(result.model_dump_json()) == result


def test_coordinated_special_action_preserves_distinct_track_intervals() -> None:
    result, _, arms, _ = assembled(centered=True)
    specials = [a for a in result.actions if a.category == "special"]
    assert len(specials) == 1
    special = specials[0]
    assert special.tracks == ["left_arm", "right_arm"]
    assert special.motion_links[0].interval != special.motion_links[1].interval
    assert len(arms.proposals) == 1
    assert not any(a.category == "arm" for a in result.actions)


def test_complete_action_at_execution_edges_and_deterministic_immutable_inputs() -> (
    None
):
    physical, floor, geometry = compound()
    coarse = split(segment_execution(physical), start=24, end=24.5)
    arms = parse_arm_actions(physical, coarse, geometry)
    lower = parse_lower_body(physical, floor, coarse)
    before = [x.model_dump_json() for x in (physical, coarse, arms, lower)]
    first = assemble_semantics(physical, coarse, arms, lower)
    assert first == assemble_semantics(physical, coarse, arms, lower)
    assert before == [x.model_dump_json() for x in (physical, coarse, arms, lower)]
    assert first.execution is not None and first.execution.start == 23.8
    assert first.execution.end == 25
    assert any(a.category == "placement" for a in first.actions)
    assert first.steps[0].interval.start == 23.8
    assert first.steps[-1].interval.end == 25


@pytest.mark.parametrize("defect", ["lineage", "clock", "execution", "coarse"])
def test_mismatched_proposals_fail_closed(defect: str) -> None:
    physical, floor, geometry = compound()
    coarse = segment_execution(physical)
    arms = parse_arm_actions(physical, coarse, geometry)
    lower = parse_lower_body(physical, floor, coarse)
    if defect == "lineage":
        lower.ground_id = "wrong"
    elif defect == "clock":
        arms.motion_times[0] -= 1
    elif defect == "execution":
        lower.execution_proposal = None
    else:
        lower.sequence_steps = []
        for action in lower.actions:
            action.sequence_step_ids = []
    with pytest.raises(ValueError):
        assemble_semantics(physical, coarse, arms, lower)


def test_hierarchy_rejects_lost_cross_step_reference_and_bad_dense_links() -> None:
    result, _, _, _ = assembled()
    data = result.model_dump()
    kick = next(a for a in result.actions if a.category == "kick")
    data["steps"][1]["action_ids"].remove(kick.id)
    with pytest.raises(ValidationError, match="references disagree"):
        Semantics.model_validate(data)
    data = result.model_dump()
    data["actions"][0]["motion_links"][0]["motion_sample_indices"].pop(1)
    with pytest.raises(ValidationError, match="dense"):
        Semantics.model_validate(data)


def test_persisted_parser_only_rerun_cache_and_config_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reconstruction.arms.artifact as arm_publisher
    import reconstruction.lower_body.artifact as leg_publisher
    import reconstruction.segmentation.artifact as coarse_publisher
    import reconstruction.semantics.artifact as publisher
    from tests.test_segmentation import persist_features

    physical, floor, geometry = compound()
    store = ArtifactStore(StorageRoot(tmp_path))
    feature_handle = persist_features(store, physical)
    coarse_handle = publish_segmentation(store, feature_handle)
    # The detailed fixture retains its own reconstruction ID; align the fixture
    # source identity before using the production arm publisher.
    motion = persist_motion(store, physical, geometry)
    assert motion.metadata.id == physical.reconstruction_id
    arm_handle = publish_arm_actions(store, feature_handle, coarse_handle, motion)
    key = ArtifactKey(
        layer="ground",
        inputs={"fixture": hash_config(floor.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision="fixture",
        config_digest="0" * 64,
    )
    ground_handle = store.get_or_create(key, lambda: (floor, {}))
    lower_handle = publish_lower_body(
        store, feature_handle, ground_handle, coarse_handle
    )
    inputs = (feature_handle, coarse_handle, arm_handle, lower_handle)
    before = [
        hash_file(h.path / "manifest.json") for h in (*inputs, ground_handle, motion)
    ]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("parser-only rerun must not run upstream inference")

    monkeypatch.setattr(arm_publisher, "parse_arm_actions", forbidden)
    monkeypatch.setattr(leg_publisher, "parse_lower_body", forbidden)
    monkeypatch.setattr(coarse_publisher, "segment_execution", forbidden)
    handle = publish_semantics(store, *inputs)
    result = load_semantics(handle)
    assert result.motion_features_id == feature_handle.metadata.id
    assert result.actions and result.keyframes
    changed = publish_semantics(
        store, *inputs, config=AssemblyConfig(max_gap_seconds=0.01)
    )
    assert changed.path != handle.path
    assert load_semantics(changed).actions == result.actions
    assert before == [
        hash_file(h.path / "manifest.json") for h in (*inputs, ground_handle, motion)
    ]
    assert load_semantic_evidence(handle)["motion_times"] == [
        r.global_seconds for r in physical.trajectory[::6]
    ]
    monkeypatch.setattr(publisher, "assemble_semantics", forbidden)
    assert publish_semantics(store, *inputs).path == handle.path
    assert load_semantics(handle) == result
    assert not handle.read_array("semantic_evidence_json").flags.writeable


def test_crossed_arm_transition_retains_events_relations_and_dense_motion() -> None:
    from reconstruction.segmentation import SegmentationConfig

    physical, floor, geometry = compound()
    for row in physical.trajectory:
        if (
            row.track in ("left_arm", "right_arm")
            and 24.2 <= row.global_seconds <= 24.4
        ):
            row.linear = Derivative(unit="m")
            row.angular = Derivative(unit="rad")
    coarse = segment_execution(physical, SegmentationConfig(max_unknown_seconds=0.3))
    assert coarse.execution is not None
    arms = parse_arm_actions(physical, coarse, geometry)
    lower = parse_lower_body(physical, floor, coarse)
    result = assemble_semantics(physical, coarse, arms, lower)
    crossing = next(
        k
        for k in result.keyframes
        if k.event == "arm_crossing"
        and next(a for a in result.actions if a.id == k.action_id).category
        == "transition"
    )
    transition = next(a for a in result.actions if a.id == crossing.action_id)
    assert transition.interval.start < 24.3 < transition.interval.end
    assert (
        transition.quality.state == "unknown" and crossing.quality.state == "inferred"
    )
    assert transition.motion_links[0].motion_sample_indices == list(
        range(
            transition.motion_links[0].motion_sample_indices[0],
            transition.motion_links[0].motion_sample_indices[-1] + 1,
        )
    )
    assert any(
        r.relation == "crossed"
        and r.interval is not None
        and r.interval.start <= 24.3 <= r.interval.end
        for r in result.relations
    )


def test_missing_contact_does_not_promote_unknown_lower_body_class() -> None:
    physical, ground = fixture(kick=True, support=False)
    source, _ = sequence(flat=True)
    geometry = []
    for row in physical.trajectory[::6]:
        sample = source.samples[0].model_copy(deep=True)
        sample.global_seconds = row.global_seconds
        geometry.append(
            derive_sample(
                sample,
                reconstruction_id=physical.reconstruction_id,
                representation="regularized",
            )
        )
    coarse = segment_execution(physical)
    arms = parse_arm_actions(physical, coarse, geometry)
    lower = parse_lower_body(physical, ground, coarse)
    assert any(a.category == "unknown" for a in lower.actions)
    result = assemble_semantics(physical, coarse, arms, lower)
    assert not any(a.category in ("kick", "placement", "pivot") for a in result.actions)
    assert all(a.quality.state == "unknown" for a in result.actions)


def test_indeterminate_execution_stays_unknown_and_preserves_artifact_evidence(
    tmp_path: Path,
) -> None:
    from tests.test_segmentation import features, persist_features

    physical = features(duration=4)
    physical.reconstruction_id = "motion"
    physical.ground_id = "ground"
    _, floor = fixture()
    source, _ = sequence(flat=True)
    geometry = []
    for row in physical.trajectory[::6]:
        sample = source.samples[0].model_copy(deep=True)
        sample.global_seconds = row.global_seconds
        geometry.append(
            derive_sample(
                sample, reconstruction_id="motion", representation="regularized"
            )
        )
    store = ArtifactStore(StorageRoot(tmp_path))
    feature_handle = persist_features(store, physical)
    coarse_handle = publish_segmentation(store, feature_handle)
    motion = persist_motion(store, physical, geometry)
    arm_handle = publish_arm_actions(store, feature_handle, coarse_handle, motion)
    key = ArtifactKey(
        layer="ground",
        inputs={"fixture": hash_config(floor.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision="fixture",
        config_digest="0" * 64,
    )
    ground_handle = store.get_or_create(key, lambda: (floor, {}))
    leg_handle = publish_lower_body(store, feature_handle, ground_handle, coarse_handle)
    handle = publish_semantics(
        store, feature_handle, coarse_handle, arm_handle, leg_handle
    )
    result = load_semantics(handle)
    assert result.execution is None and result.quality.state == "unknown"
    assert not result.steps and not result.actions and not result.keyframes
    evidence = load_semantic_evidence(handle)
    assert evidence["motion_times"][0] == 23
    assert evidence["coarse"]["diagnostics"]
    assert evidence["lower_body"]["native_times"]


def test_single_sample_relation_keeps_event_time_and_unknown_front_order() -> None:
    physical, floor, geometry = compound()
    for row in physical.trajectory:
        row.relations = []
        if row.track == "left_arm" and row.global_seconds == 24.3:
            row.relations = [
                Relation(
                    subject="left_forearm",
                    object="right_forearm",
                    reference_frame="torso",
                    axis="crossing",
                    value="crossed",
                    quality=q(),
                )
            ]
    coarse = segment_execution(physical)
    arms = parse_arm_actions(physical, coarse, geometry)
    lower = parse_lower_body(physical, floor, coarse)
    result = assemble_semantics(physical, coarse, arms, lower)
    relation = next(r for r in result.relations if r.relation == "crossed")
    assert relation.global_seconds == 24.3 and relation.interval is None
    assert relation.front_entity is None and relation.front_quality.state == "unknown"


def test_bundle_checks_exact_semantic_input_identity() -> None:
    from contracts.models import validate_bundle
    from tests.test_contracts import base, bundle

    data = bundle()
    links = {"reconstruction_id": "motion", "ground_id": "ground"}
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
    semantic = next(a for a in data if a["kind"] == "semantics")
    semantic.update(
        motion_features_id="features", segmentation_id="coarse", arm_actions_id="arms"
    )
    validate_bundle(data)
    data[-1]["segmentation_id"] = "features"
    with pytest.raises(ValueError, match="reference|identity"):
        validate_bundle(data)


def test_execution_starting_during_recovery_retains_preceding_kick() -> None:
    physical, floor, geometry = compound()
    coarse = split(segment_execution(physical), start=24.8, end=25.6, cut=25)
    arms = parse_arm_actions(physical, coarse, geometry)
    lower = parse_lower_body(physical, floor, coarse)
    # Isolate the compound leg chain: another track's spanning unknown action
    # must not be necessary to discover the recovery predecessor.
    lower.actions = [a for a in lower.actions if a.track == "left_leg"]
    lower.stances = []
    kick_proposal = next(a for a in lower.actions if a.category == "kick")
    assert coarse.execution is not None
    assert kick_proposal.interval.end < coarse.execution.start
    result = assemble_semantics(physical, coarse, arms, lower)
    kick = next(a for a in result.actions if a.category == "kick")
    placement = next(a for a in result.actions if a.category == "placement")
    assert placement.previous_action_id == kick.id
    assert kick.interval == kick_proposal.interval
    assert (
        result.execution is not None and result.execution.start == kick.interval.start
    )
