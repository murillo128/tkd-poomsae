"""Physical synthetic features exercise automatic classification, not labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from contracts.models import (
    Contact,
    Ground,
    GroundSample,
    Interval,
    Pivot,
    Quality,
)
from reconstruction.features import FeatureSeries
from reconstruction.lower_body import (
    LowerBodyConfig,
    LowerBodyResult,
    load_lower_body,
    parse_lower_body,
    publish_lower_body,
)
from reconstruction.segmentation import publish_segmentation, segment_execution
from storage import ArtifactKey, ArtifactStore, StorageRoot, hash_config
from tests.test_segmentation import features, persist_features


def q() -> Quality:
    return Quality(state="inferred", uncertainty=0.001, source_ids=["motion"])


def fixture(
    *,
    kick: bool = False,
    right: bool = False,
    shift: bool = True,
    support: bool = True,
) -> tuple[FeatureSeries, Ground]:
    source = features(
        rate=50, duration=4, offset=23, bouts=[(0.8, 2, ["left_leg", "right_leg"])]
    )
    source.reconstruction_id = "motion"
    source.ground_id = "ground"
    ground = Ground(
        kind="ground",
        id="ground",
        schema_version="1.0.0",
        provenance=source_provenance(),
        reconstruction_id="motion",
        scale="metric",
        samples=[],
    )
    for i, time in enumerate([r.global_seconds for r in source.trajectory[::6]]):
        t = time - 23
        contacts = []
        for j, leg in ((2, "left_leg"), (3, "right_leg")):
            onset, landing = (0.8, 2.0) if leg == "left_leg" else (1.0, 2.2)
            airborne = onset <= t < landing and (leg == "left_leg" or right)
            state: Literal["contact", "no_contact", "unknown"] = (
                "unknown" if not support else "no_contact" if airborne else "contact"
            )
            c = Contact(
                state=state, quality=q() if support else Quality(state="unknown")
            )
            contacts.append(c)
            row = source.trajectory[i * 6 + j]
            row.ground_sample_index = i
            row.contact = c
            row.position_world.quality = q()
            x = -0.15 if leg == "left_leg" else 0.15
            progress = max(0, min(1, (t - onset) / (landing - onset)))
            row.position_world.value = (
                x,
                0.25 * progress if shift and (leg == "left_leg" or right) else 0,
                0.1,
            )
            row.extension.quality = q()
            row.extension.value = 0.7
            if kick and (leg == "left_leg" or right):
                local = t - onset
                # Chamber/flexion -> extension -> retraction -> relocation.
                row.extension.value = (
                    0.65 - 0.15 * local / 0.2
                    if 0 <= local < 0.2
                    else 0.5 + 0.48 * (local - 0.2) / 0.3
                    if 0.2 <= local < 0.5
                    else 0.98 - 0.48 * (local - 0.5) / 0.3
                    if 0.5 <= local < 0.8
                    else 0.5
                )
        left, rcontact = contacts
        support_state: Any = (
            "unknown"
            if not support
            else "both"
            if left.state == rcontact.state == "contact"
            else "neither"
            if left.state == rcontact.state == "no_contact"
            else "right"
            if left.state == "no_contact"
            else "left"
        )
        ground.samples.append(
            GroundSample(
                global_seconds=time, left=left, right=rcontact, support=support_state
            )
        )
    return FeatureSeries.model_validate(source.model_dump()), ground


def source_provenance() -> Any:
    from contracts.models import Provenance

    return Provenance(producer="fixture", config_digest="0" * 64)


def parse(source: FeatureSeries, floor: Ground) -> LowerBodyResult:
    return parse_lower_body(source, floor, segment_execution(source))


def test_kick_recovery_placement_and_resulting_stance() -> None:
    source, floor = fixture(kick=True)
    before = source.model_dump_json()
    result = parse(source, floor)
    left = [a for a in result.actions if a.track == "left_leg"]
    assert [a.category for a in left] == ["kick", "placement"]
    assert {p.name for p in left[0].phases} == {"chamber", "extension", "retraction"}
    assert left[1].previous_action_id == left[0].id
    assert left[1].interval.start == left[0].interval.end
    assert left[1].phases[0].name == "recovery"
    assert any(left[1].id in s.contributing_action_ids for s in result.stances)
    assert all(a.quality.state == "inferred" for a in left)
    assert result.native_times[0] == 23 and result.native_times[-1] == 27
    assert source.model_dump_json() == before
    assert LowerBodyResult.model_validate_json(result.model_dump_json()) == result


def test_ordinary_step_is_placement_without_kick() -> None:
    source, floor = fixture()
    result = parse(source, floor)
    assert [a.category for a in result.actions if a.track == "left_leg"] == [
        "placement"
    ]


def test_concurrent_legs_preserve_independent_intervals_and_dense_links() -> None:
    source, floor = fixture(kick=True, right=True)
    result = parse(source, floor)
    kicks = [a for a in result.actions if a.category == "kick"]
    assert len(kicks) == 2
    assert kicks[0].interval.start < kicks[1].interval.start < kicks[0].interval.end
    for a in result.actions:
        assert a.motion_sample_indices == list(
            range(a.motion_sample_indices[0], a.motion_sample_indices[-1] + 1)
        )
        assert a.interval.start == result.native_times[a.motion_sample_indices[0]]


@pytest.mark.parametrize("case", ["airborne_rotation", "missing_support", "root_only"])
def test_rotation_without_supported_foot_never_becomes_pivot(case: str) -> None:
    source, floor = fixture(shift=False, support=case != "missing_support")
    if case != "root_only":
        floor.pivots = [
            Pivot(
                id="physical-pivot",
                foot="left",
                interval=Interval(start=24, end=24.5),
                region="forefoot",
                rotation_rad=0.5,
                quality=q(),
            )
        ]
    if case == "root_only":
        from reconstruction.temporal.core import Derivative

        for row in source.trajectory:
            if row.track in ("left_leg", "right_leg"):
                row.contact = Contact(state="contact", quality=q())
                row.linear.velocity = (0, 0, 0)
            elif row.track == "body_root":
                row.angular = Derivative(
                    unit="rad",
                    velocity=(0, 0, 1),
                    velocity_valid=True,
                    velocity_quality=q(),
                    support_times=[row.global_seconds],
                )
        for sample in floor.samples:
            sample.left = sample.right = Contact(state="contact", quality=q())
            sample.support = "both"
    result = parse(source, floor)
    assert not any(a.category == "pivot" for a in result.actions)
    assert all(a.quality.state == "unknown" for a in result.actions)


def test_supported_pivot_and_stationary_stance() -> None:
    source, floor = fixture(shift=False)
    # A physical turn on the planted right foot while the left moves.
    floor.pivots = [
        Pivot(
            id="physical-pivot",
            foot="right",
            interval=Interval(start=24, end=24.5),
            region="forefoot",
            rotation_rad=0.5,
            quality=q(),
        )
    ]
    result = parse(source, floor)
    pivots = [a for a in result.actions if a.category == "pivot"]
    assert len(pivots) == 1 and pivots[0].track == "right_leg"
    assert pivots[0].ground_event_ids == ["physical-pivot"]
    assert result.stances
    assert all(not (s.interval.start < 24.2 < s.interval.end) for s in result.stances)


def test_stationary_configuration_is_state_not_action() -> None:
    source, floor = fixture(shift=False)
    for row in source.trajectory:
        if row.track in ("left_leg", "right_leg"):
            row.contact = Contact(state="contact", quality=q())
            row.linear.velocity = (0, 0, 0)
    for sample in floor.samples:
        sample.left = sample.right = Contact(state="contact", quality=q())
        sample.support = "both"
    result = parse(source, floor)
    assert not result.actions and len(result.stances) == 1
    assert result.stances[0].interval == Interval(start=23, end=27)


@pytest.mark.parametrize(
    "case", ["extension_unknown", "interpolated", "no_retraction", "gap"]
)
def test_absent_kick_evidence_does_not_create_confident_kick(case: str) -> None:
    source, floor = fixture(kick=True, shift=False)
    for row in source.trajectory:
        if row.track == "left_leg" and 24 <= row.global_seconds < 25:
            if case == "extension_unknown":
                row.extension.value = None
                row.extension.quality = Quality(state="unknown")
            elif case == "interpolated":
                row.extension.quality.state = "interpolated"
            elif case == "no_retraction" and row.global_seconds >= 24.3:
                row.extension.value = 0.98
    if case == "gap":
        for row in source.trajectory:
            if row.global_seconds >= 24.3:
                row.global_seconds += 1
        for sample in floor.samples:
            if sample.global_seconds >= 24.3:
                sample.global_seconds += 1
    result = parse(source, floor)
    assert not any(a.category == "kick" for a in result.actions)


def test_coarse_proposals_do_not_clip_actions() -> None:
    from contracts.models import SequenceStep
    from reconstruction.segmentation.core import (
        Activity,
        Boundary,
        SegmentationResult,
        StepEvidence,
    )

    source, floor = fixture(kick=True)
    # A deliberately indeterminate proposal leaves all motion classifiable.
    proposals = SegmentationResult(
        reconstruction_id="motion",
        ground_id="ground",
        config=segment_execution(source).config,
        execution=None,
        steps=[],
        quality=Quality(state="unknown"),
        diagnostics=[],
        activity=[
            Activity(
                global_seconds=t,
                motion_sample_index=i,
                known_tracks=[],
                active_tracks=[],
            )
            for i, t in enumerate([r.global_seconds for r in source.trajectory[::6]])
        ],
        boundaries=[],
        step_evidence=[],
    )
    result = parse_lower_body(source, floor, proposals)
    assert any(a.category == "kick" and not a.sequence_step_ids for a in result.actions)
    # Association is overlap-only; several proposals may intersect one action.
    proposals.steps = [
        SequenceStep(id="a", interval=Interval(start=24, end=24.3), action_ids=[]),
        SequenceStep(id="b", interval=Interval(start=24.3, end=24.5), action_ids=[]),
    ]
    proposals.execution = Interval(start=24, end=24.5)
    proposals.quality = Quality(state="inferred", source_ids=["motion", "ground"])
    cuts = [50, 65, 75]
    proposals.boundaries = [
        Boundary(
            global_seconds=proposals.activity[i].global_seconds,
            motion_sample_index=i,
            reason="coarse_proposal_fixture",
            event_ids=[],
            tolerance_seconds=0.02,
        )
        for i in cuts
    ]
    proposals.step_evidence = [
        StepEvidence(
            step_id=step.id,
            motion_sample_indices=list(range(a, b + 1)),
            event_ids=[],
            active_tracks=[],
        )
        for step, a, b in zip(proposals.steps, cuts, cuts[1:])
    ]
    proposals = SegmentationResult.model_validate(proposals.model_dump())
    result = parse_lower_body(source, floor, proposals)
    kick = next(a for a in result.actions if a.category == "kick")
    assert kick.sequence_step_ids == ["a", "b"]
    assert kick.interval.start < 24 and kick.interval.end > 24.5


def test_persistence_cache_config_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    source, floor = fixture(kick=True)
    physical = persist_features(store, source)
    proposal = publish_segmentation(store, physical)
    key = ArtifactKey(
        layer="ground",
        inputs={"fixture": hash_config(floor.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision="fixture",
        config_digest="0" * 64,
    )
    ground = store.get_or_create(key, lambda: (floor, {}))
    output = publish_lower_body(store, physical, ground, proposal)
    assert load_lower_body(output) == parse(source, floor)
    changed = publish_lower_body(
        store,
        physical,
        ground,
        proposal,
        LowerBodyConfig(extension_excursion_ratio=0.8),
    )
    assert changed.path != output.path
    assert not any(a.category == "kick" for a in load_lower_body(changed).actions)
    import reconstruction.lower_body.artifact as publisher

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("cache/reload must not rerun parser or vision")

    monkeypatch.setattr(publisher, "parse_lower_body", forbidden)
    assert publish_lower_body(store, physical, ground, proposal).path == output.path
    assert load_lower_body(output).actions
    assert not output.read_array("lower_body_evidence_json").flags.writeable


def test_rejects_lost_dense_links_and_mismatched_ground() -> None:
    source, floor = fixture(kick=True)
    result = parse(source, floor).model_dump()
    result["actions"][0]["motion_sample_indices"].pop()
    with pytest.raises(ValidationError, match="dense native links"):
        LowerBodyResult.model_validate(result)
    floor.id = "different"
    with pytest.raises(ValueError, match="lineage"):
        parse(source, floor)


def test_real_reconstructed_geometry_automatically_classifies_kick() -> None:
    import math

    from reconstruction.features import derive_features
    from tests.test_motion_features import sequence

    motion, ground = sequence(flat=True)
    for sample, physical in zip(motion.samples, ground.samples):
        t = sample.global_seconds
        ratio = 0.5 + 0.48 * max(0, 1 - abs(t - 0.55) / 0.2)
        distance = 0.9 * ratio
        points = {p.name: p for p in sample.landmarks}
        points["left_ankle"].xyz_world = (-0.15, distance, 1)
        points["left_knee"].xyz_world = (
            -0.15 + math.sqrt(0.45**2 - (distance / 2) ** 2),
            distance / 2,
            1,
        )
        physical.left = Contact(
            state="no_contact" if 0.3 <= t < 0.9 else "contact", quality=q()
        )
        physical.support = "right" if physical.left.state == "no_contact" else "both"
    physical_features = derive_features(motion, ground)
    result = parse(physical_features, ground)
    assert any(a.category == "kick" for a in result.actions)
    kick = next(a for a in result.actions if a.category == "kick")
    assert kick.feature_event_ids
    assert kick.ground_sample_indices


def test_incomplete_kick_preserves_extension_and_placement_evidence() -> None:
    source, floor = fixture(kick=True)
    for row in source.trajectory:
        if row.track == "left_leg" and 24.3 <= row.global_seconds < 25:
            row.extension.value = 0.98
    result = parse(source, floor)
    left = [a for a in result.actions if a.track == "left_leg"]
    assert [a.category for a in left] == ["unknown"]
    assert {p.name for p in left[0].phases} == {"chamber", "extension", "placement"}
    assert left[0].quality.state == "unknown"
    assert all(p.quality.state == "inferred" for p in left[0].phases)


def test_lower_body_bundle_checks_all_input_references() -> None:
    from contracts.models import LowerBodyParsing, validate_bundle
    from tests.test_contracts import base, bundle

    data = bundle()
    motion = next(a for a in data if a["kind"] == "reconstruction")
    ground = next(a for a in data if a["kind"] == "ground")
    refs = {"reconstruction_id": motion["id"], "ground_id": ground["id"]}
    data.append(base("motion_features", "features") | refs | {"arrays": []})
    data.append(
        base("segmentation", "proposal")
        | refs
        | {
            "motion_features_id": "features",
            "execution": None,
            "steps": [],
            "quality": {"state": "unknown"},
            "arrays": [],
        }
    )
    data.append(
        base("lower_body_parsing", "lower-body")
        | refs
        | {
            "motion_features_id": "features",
            "segmentation_id": "proposal",
            "arrays": [],
        }
    )
    assert isinstance(validate_bundle(data)[-1], LowerBodyParsing)
    data[-1]["segmentation_id"] = "features"
    with pytest.raises(ValueError, match="reference"):
        validate_bundle(data)


def test_next_sample_extension_peak_keeps_chamber_unknown_at_native_cadence() -> None:
    source, floor = fixture(kick=True)
    source.trajectory = [
        row for row in source.trajectory if row.motion_sample_index % 5 == 0
    ]
    floor.samples = floor.samples[::5]
    for row in source.trajectory:
        row.motion_sample_index //= 5
        if row.ground_sample_index is not None:
            row.ground_sample_index //= 5
        if row.track == "left_leg" and 23.8 <= row.global_seconds < 25:
            row.extension.value = 0.98 if row.global_seconds == 23.9 else 0.5
    source = FeatureSeries.model_validate(source.model_dump())
    floor = Ground.model_validate(floor.model_dump())
    result = parse(source, floor)
    left = [a for a in result.actions if a.track == "left_leg"]
    assert not any(a.category == "kick" for a in left)
    assert len(left) == 1 and left[0].category == "unknown"
    assert left[0].quality.state == "unknown" and left[0].quality.score is None
    assert {p.name for p in left[0].phases} == {
        "extension",
        "retraction",
        "placement",
    }
    phases = {p.name: p for p in left[0].phases}
    assert phases["extension"].interval == Interval(start=23.8, end=23.9)
    assert phases["retraction"].interval == Interval(start=23.9, end=24)
    assert phases["placement"].interval.end == 25
    assert "chamber_interval_unavailable" in left[0].reasons
    for evidence in [left[0], *left[0].phases]:
        assert evidence.motion_sample_indices == list(
            range(
                evidence.motion_sample_indices[0],
                evidence.motion_sample_indices[-1] + 1,
            )
        )
        assert evidence.ground_sample_indices == evidence.motion_sample_indices
    assert LowerBodyResult.model_validate_json(result.model_dump_json()) == result


def test_inferred_kick_requires_independent_phase_evidence_on_reload() -> None:
    source, floor = fixture(kick=True)
    payload = parse(source, floor).model_dump()
    kick = next(a for a in payload["actions"] if a["category"] == "kick")
    kick["phases"] = [p for p in kick["phases"] if p["name"] != "chamber"]
    with pytest.raises(ValidationError, match="independent kick phase evidence"):
        LowerBodyResult.model_validate(payload)
