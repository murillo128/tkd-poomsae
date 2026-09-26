"""Offline physical-placement acceptance using existing synthetic motion."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from contracts.models import Footprint, Ground, Interval, Quality, Reconstruction
from reconstruction.footprints import (
    FootprintConfig,
    derive_footprints,
    load_footprint_evidence,
    measure_placements,
    publish_footprints,
)
from reconstruction.ground import ContactConfig, derive_contacts, publish_contacts
from storage import ArtifactStore, StorageRoot
from tests.test_ground_contact import calibration, motion, persist, shape


def ground(source: Reconstruction, *, metric: bool = True) -> Ground:
    contacts = derive_contacts(
        source,
        calibration(metric),
        None if metric else shape(),
        ContactConfig(units="metric" if metric else "body_normalized"),
    )
    return Ground(
        kind="ground",
        id="contacts",
        schema_version="1.0.0",
        provenance=source.provenance,
        reconstruction_id=source.id,
        scale=source.scale,
        samples=contacts.samples,
    )


def translate(source: Reconstruction, offset: Callable[[float], float]) -> None:
    for sample in source.samples:
        for point in sample.landmarks:
            if point.name.startswith("left") and point.xyz_world:
                x, y, z = point.xyz_world
                point.xyz_world = (x + offset(sample.global_seconds), y, z)


def test_stationary_jitter_is_one_placement_per_side_with_dense_lineage() -> None:
    source = motion()
    translate(source, lambda t: 0.003 * math.sin(80 * t))
    contacts = ground(source)
    before = source.model_dump_json(), contacts.model_dump_json()
    result = derive_footprints(source, contacts, calibration())
    stable = [e for e in result.events if e.kind == "stable"]
    assert len(stable) == 2
    assert len(result.trajectory) == 2 * len(source.samples)
    assert (
        result.model_dump_json()
        == derive_footprints(source, contacts, calibration()).model_dump_json()
    )
    assert before == (source.model_dump_json(), contacts.model_dump_json())
    for event in stable:
        assert event.geometry.kind == "axis"
        assert event.geometry.polygon is None
        assert not event.geometry.approximated
        assert event.footprint.quality.state == "inferred"
        assert event.footprint.quality.uncertainty is not None
        assert "floor" in event.footprint.quality.source_ids
        for index in event.trajectory_indices:
            row = result.trajectory[index]
            assert row.event_id == event.footprint.id
            assert row.contact_sample_index == row.motion_sample_index
            assert row.landmarks == [
                p
                for p in source.samples[row.motion_sample_index].landmarks
                if p.name.startswith(event.footprint.foot)
            ]


def test_genuine_relocation_has_distinct_stable_events() -> None:
    source = motion()
    translate(source, lambda t: 0 if t < 0.5 else 0.4)
    result = derive_footprints(source, ground(source), calibration())
    left = [
        e for e in result.events if e.kind == "stable" and e.footprint.foot == "left"
    ]
    assert len(left) == 2
    assert left[0].footprint.id != left[1].footprint.id
    assert left[0].footprint.xy_ground is not None
    assert left[1].footprint.xy_ground is not None
    assert left[1].footprint.xy_ground[0] - left[0].footprint.xy_ground[
        0
    ] == pytest.approx(0.4)
    assert left[0].footprint.interval.end < left[1].footprint.interval.start


@pytest.mark.parametrize("speed", [0.2, 0.6, 2.0])
def test_sliding_contact_is_not_frozen(speed: float) -> None:
    source = motion()
    translate(source, lambda t: speed * t)
    result = derive_footprints(source, ground(source), calibration())
    left = [e for e in result.events if e.footprint.foot == "left"]
    assert len(left) == 1 and left[0].kind == "moving"
    assert left[0].footprint.xy_ground is None and left[0].footprint.yaw_rad is None
    rows = [result.trajectory[i] for i in left[0].trajectory_indices]
    assert rows[0].xy_ground != rows[-1].xy_ground
    assert all(row.contact.state == "contact" for row in rows)


def test_landing_after_slide_and_occlusion_split_without_imputation() -> None:
    source = motion(
        lambda t, side, part: None if side == "left" and 0.4 <= t <= 0.6 else 0
    )
    translate(source, lambda t: 0.5 * min(t, 0.25))
    result = derive_footprints(source, ground(source), calibration())
    left = [e for e in result.events if e.footprint.foot == "left"]
    assert [e.kind for e in left] == ["moving", "stable", "stable"]
    assert not any(
        e.footprint.interval.start < 0.5 < e.footprint.interval.end for e in left
    )
    hidden = [
        r
        for r in result.trajectory
        if r.foot == "left" and 0.4 <= r.global_seconds <= 0.6
    ]
    assert all(r.status == "unavailable" and r.event_id is None for r in hidden)
    assert all(r.xy_ground is None for r in hidden)


def test_axis_only_partial_geometry_and_missing_axis() -> None:
    source = motion(lambda t, s, p: None if p == "foot_outer" else 0)
    result = derive_footprints(source, ground(source), calibration())
    assert all(e.kind == "stable" and e.geometry.kind == "axis" for e in result.events)
    assert all(e.geometry.polygon is None for e in result.events)
    assert all(len(e.geometry.supported_points) == 2 for e in result.events)
    source = motion(lambda t, s, p: None if p == "forefoot" else 0)
    result = derive_footprints(source, ground(source), calibration())
    assert result.events == []
    assert all(
        r.geometry.kind == "points" and r.event_id is None for r in result.trajectory
    )


def footprint(name: str, xy: tuple[float, float], yaw: float) -> Footprint:
    return Footprint(
        id=name,
        foot="left",
        interval=Interval(start=1, end=2),
        xy_ground=xy,
        yaw_rad=yaw,
        quality=Quality(state="inferred", uncertainty=0.001, source_ids=[name]),
    )


@pytest.mark.parametrize("rotation", [0, 0.7, -2, math.pi])
def test_relations_coordinate_changes_and_angle_wrap(rotation: float) -> None:
    def transform(x: float, y: float) -> tuple[float, float]:
        return (
            10 + x * math.cos(rotation) - y * math.sin(rotation),
            -4 + x * math.sin(rotation) + y * math.cos(rotation),
        )

    first = footprint("a", transform(0, 0), math.pi - 0.02 + rotation)
    second = footprint("b", transform(3, 4), -math.pi + 0.03 + rotation)
    result = measure_placements(first, second, "m")
    values = {m.name: m.value for m in result.measurements}
    assert values["distance"] == pytest.approx(5)
    assert values["longitudinal_separation"] == pytest.approx(
        -3 * math.cos(0.02) + 4 * math.sin(0.02)
    )
    assert values["lateral_separation"] == pytest.approx(
        -3 * math.sin(0.02) - 4 * math.cos(0.02)
    )
    assert result.relative_foot_angle_rad == pytest.approx(0.05)
    assert result.relative_alignment_rad == pytest.approx(
        math.atan2(4, 3) - math.pi + 0.02
    )


def test_arbitrary_scale_keeps_normalized_and_angular_information() -> None:
    reference = None
    for scale in (1, 5):
        source = motion(metric=False, scale=scale)
        cal = calibration(False)
        contacts = derive_contacts(
            source, cal, shape(scale), ContactConfig(units="body_normalized")
        )
        artifact = Ground(
            kind="ground",
            id="g",
            schema_version="1.0.0",
            provenance=source.provenance,
            reconstruction_id=source.id,
            scale=source.scale,
            samples=contacts.samples,
        )
        result = derive_footprints(
            source,
            artifact,
            cal,
            shape(scale),
            FootprintConfig(threshold_units="body_normalized"),
        )
        assert len(result.events) == 2 and result.world_unit == "arbitrary"
        relation = result.relations[0]
        assert all(m.unit != "m" for m in relation.measurements)
        ratios = {
            m.name: m.value for m in relation.measurements if m.unit == "body_ratio"
        }
        assert ratios["distance"] == pytest.approx(0.4)
        assert relation.relative_foot_angle_rad == pytest.approx(0)
        assert reference is None or ratios == reference
        reference = ratios
        assert all(
            "shape-evidence" in m.quality.source_ids
            for m in relation.measurements
            if m.unit == "body_ratio"
        )
    result = derive_footprints(source, artifact, cal)
    assert len(result.events) == 2 and len(result.relations) == 1
    assert all(m.unit == "arbitrary" for m in result.relations[0].measurements)


@pytest.mark.parametrize("rate", [20, 30, 60, 100])
def test_seconds_not_frame_count(rate: int) -> None:
    source = motion(rate=rate)
    result = derive_footprints(source, ground(source), calibration())
    assert len(result.events) == 2
    assert all(e.kind == "stable" for e in result.events)
    for event in result.events:
        assert 0.08 <= event.event_seconds <= 0.08 + 2 / rate
        assert event.footprint.interval.end == 1


def test_gaps_repeated_evidence_and_high_uncertainty_prevent_stability() -> None:
    source = motion()
    contacts = ground(source)
    for sample in source.samples:
        for p in sample.landmarks:
            p.quality.source_ids = [p.name]
    result = derive_footprints(source, contacts, calibration())
    assert not any(e.kind == "stable" for e in result.events)
    source = motion()
    contacts = ground(source)
    source.samples = [
        s for s in source.samples if s.global_seconds < 0.4 or s.global_seconds > 0.7
    ]
    contacts.samples = [
        s for s in contacts.samples if s.global_seconds < 0.4 or s.global_seconds > 0.7
    ]
    result = derive_footprints(source, contacts, calibration())
    assert len([e for e in result.events if e.footprint.foot == "left"]) == 2
    source = motion()
    contacts = ground(source)
    for sample in source.samples:
        for p in sample.landmarks:
            p.quality.uncertainty = 0.03
    result = derive_footprints(source, contacts, calibration())
    assert not any(e.kind == "stable" for e in result.events)


def test_hysteresis_and_pivot_cannot_hide_in_foot_center() -> None:
    source = motion()
    translate(source, lambda t: 0 if t < 0.4 else 0.025)
    result = derive_footprints(source, ground(source), calibration())
    assert len([e for e in result.events if e.footprint.foot == "left"]) == 1
    source = motion()
    for sample in source.samples:
        angle = sample.global_seconds
        for p in sample.landmarks:
            if p.name.startswith("left") and p.xyz_world:
                x, y, z = p.xyz_world
                p.xyz_world = (
                    -0.2 + (x + 0.2) * math.cos(angle) - (y - 0.1) * math.sin(angle),
                    0.1 + (x + 0.2) * math.sin(angle) + (y - 0.1) * math.cos(angle),
                    z,
                )
    result = derive_footprints(source, ground(source), calibration())
    assert not any(
        e.kind == "stable" and e.footprint.foot == "left" for e in result.events
    )


def test_artifact_roundtrip_cache_revisions_and_immutability(tmp_path: Path) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    source, cal = persist(store, motion()), persist(store, calibration())
    contacts = publish_contacts(store, source, cal)
    assert isinstance(contacts.metadata, Ground)
    original = (
        contacts.metadata.model_dump_json(),
        contacts.read_array("contact_evidence_json").tobytes(),
    )
    output = publish_footprints(store, source, contacts, cal)
    assert isinstance(output.metadata, Ground)
    result = load_footprint_evidence(output)
    assert len(result.events) == 2 and len(output.metadata.footprints) == 2
    assert output.metadata.samples == contacts.metadata.samples
    assert original == (
        contacts.metadata.model_dump_json(),
        contacts.read_array("contact_evidence_json").tobytes(),
    )
    assert np.array_equal(
        output.read_array("contact_evidence_json"),
        contacts.read_array("contact_evidence_json"),
    )
    assert publish_footprints(store, source, contacts, cal).path == output.path
    assert (
        publish_footprints(
            store, source, contacts, cal, config=FootprintConfig(stable_seconds=0.2)
        ).path
        != output.path
    )
    with pytest.raises(ValueError, match="physical placement"):
        load_footprint_evidence(contacts)
    changed = motion()
    translate(changed, lambda t: 1)
    with pytest.raises(ValueError, match="revisions"):
        publish_footprints(store, persist(store, changed), contacts, cal)


def test_input_and_config_validation() -> None:
    source = motion()
    contacts = ground(source)
    contacts.samples.pop()
    with pytest.raises(ValueError, match="native times"):
        derive_footprints(source, contacts, calibration())
    source = motion()
    contacts = ground(source)
    source.provenance.producer = "reconstruction.triangulation"
    with pytest.raises(ValueError, match="final temporal"):
        derive_footprints(source, contacts, calibration())
    for values in (
        {"leave_distance": 0.001},
        {"stable_seconds": 0},
        {"enter_angle_rad": float("nan")},
    ):
        with pytest.raises(ValueError):
            FootprintConfig.model_validate(values)
    source = motion(metric=False)
    result = derive_footprints(
        source,
        ground(source, metric=False),
        calibration(False),
        config=FootprintConfig(threshold_units="body_normalized"),
    )
    assert not result.events
    assert any(
        "unavailable_threshold_normalization" in row.reasons
        for row in result.trajectory
    )


def test_flight_and_new_outer_sources_cannot_establish_placement() -> None:
    source = motion(lambda t, s, p: 0.2)
    result = derive_footprints(source, ground(source), calibration())
    assert result.events == []
    assert all(r.status == "no_contact" for r in result.trajectory[10:])
    source = motion()
    contacts = ground(source)
    for sample in source.samples:
        for point in sample.landmarks:
            if point.name.endswith(("_heel", "_forefoot")):
                point.quality.source_ids = [point.name]
    result = derive_footprints(source, contacts, calibration())
    assert not any(e.kind == "stable" for e in result.events)


def test_actual_temporal_to_contact_to_placement_roundtrip(tmp_path: Path) -> None:
    from reconstruction.temporal import publish_temporal_motion
    from tests.test_articulated import raw_handle

    store = ArtifactStore(StorageRoot(tmp_path))
    raw = motion()
    raw.provenance.producer = "reconstruction.triangulation"
    final = publish_temporal_motion(store, raw_handle(store, raw))
    cal = persist(store, calibration())
    contacts = publish_contacts(store, final.motion, cal, final.morphology)
    output = publish_footprints(store, final.motion, contacts, cal, final.morphology)
    result = load_footprint_evidence(output)
    assert result.reconstruction_id == final.motion.metadata.id
    assert result.morphology_id == final.morphology.metadata.id
    assert result.contact_id == contacts.metadata.id
    assert isinstance(final.motion.metadata, Reconstruction)
    assert len(result.trajectory) == 2 * len(final.motion.metadata.samples)
    assert result.trajectory[-1].status == "unavailable"
    assert result.trajectory[-1].event_id is None


def test_invalid_persisted_source_links_are_rejected() -> None:
    from reconstruction.footprints import FootprintSeries

    source = motion()
    result = derive_footprints(source, ground(source), calibration())
    # Selection links may not navigate to unrelated native evidence.
    result.events[0].trajectory_indices[0] = 1
    with pytest.raises(ValueError, match="identity"):
        FootprintSeries.model_validate(result.model_dump())
