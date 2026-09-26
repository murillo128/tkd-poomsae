"""Offline physical pivot acceptance with native synthetic source evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

from contracts.models import (
    Contact,
    Ground,
    Landmark,
    Quality,
    Quaternion,
    Reconstruction,
)
from reconstruction.footprints import derive_footprints, publish_footprints
from reconstruction.ground import publish_contacts
from reconstruction.pivots import (
    PivotConfig,
    PivotSeries,
    derive_pivots,
    load_pivot_evidence,
    publish_pivots,
)
from storage import ArtifactStore, StorageRoot
from tests.test_footprints import ground
from tests.test_ground_contact import calibration, motion, persist


def sequence(
    region: Literal["heel", "forefoot", "centre"] = "heel",
    *,
    rate: int = 50,
    sign: int = 1,
    scale: float = 1,
    initial: float = 0,
    slide: float = 0,
    rapid: bool = False,
) -> tuple[Reconstruction, Ground]:
    source = motion(rate=rate, scale=scale, metric=scale == 1)
    contacts = ground(source, metric=scale == 1)
    pivot_y = {"heel": 0, "forefoot": 0.2, "centre": 0.1}[region] * scale
    for sample in source.samples:
        t = sample.global_seconds
        duration = 0.04 if rapid else 0.3
        angle = initial + sign * min(1, max(0, (t - 0.3) / duration))
        for point in sample.landmarks:
            if point.name.startswith("left") and point.xyz_world:
                x, y, z = point.xyz_world
                dx, dy = x + 0.2 * scale, y - pivot_y
                point.xyz_world = (
                    -0.2 * scale
                    + dx * math.cos(angle)
                    - dy * math.sin(angle)
                    + slide * t * scale,
                    pivot_y + dx * math.sin(angle) + dy * math.cos(angle),
                    z,
                )
    # Evidence supplied by the contact owner; detector must never re-estimate it.
    for contact_sample in contacts.samples:
        if contact_sample.left.state == "contact":
            contact_sample.left.region = region if region != "centre" else "flat"
    return source, contacts


def result_for(
    region: Literal["heel", "forefoot", "centre"] = "heel",
    *,
    rate: int = 50,
    sign: int = 1,
    scale: float = 1,
    initial: float = 0,
    slide: float = 0,
    rapid: bool = False,
) -> PivotSeries:
    source, contacts = sequence(
        region,
        rate=rate,
        sign=sign,
        scale=scale,
        initial=initial,
        slide=slide,
        rapid=rapid,
    )
    return derive_pivots(
        derive_footprints(source, contacts, calibration(source.scale == "metric"))
    )


@pytest.mark.parametrize("region", ["heel", "forefoot"])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("rate", [20, 50, 100])
def test_supported_pivots_and_native_direction(
    region: Literal["heel", "forefoot"], sign: int, rate: int
) -> None:
    result = result_for(region=region, sign=sign, rate=rate)
    assert len(result.events) == 1
    event = result.events[0]
    assert event.pivot.region == region
    assert event.pivot.rotation_rad == pytest.approx(sign)
    assert event.region_confidence > 0
    assert event.pivot.interval.start == pytest.approx(0.3)
    assert event.pivot.interval.end == pytest.approx(0.6)
    assert event.classification == "supported_rotation"
    stationary = next(t for t in event.translations if t.landmark == region)
    assert stationary.max_excursion == pytest.approx(0)
    assert event.pivot.quality.source_ids
    assert event.placement_ids
    assert (
        result.model_dump_json()
        == PivotSeries.model_validate_json(result.model_dump_json()).model_dump_json()
    )


def test_rapid_rotation_and_wrap_remain_inspectable() -> None:
    rapid = result_for(rapid=True)
    assert len(rapid.events) == 1
    assert (
        rapid.events[0].pivot.interval.end - rapid.events[0].pivot.interval.start < 0.08
    )
    wrapped = result_for(initial=1.4)
    assert wrapped.events[0].pivot.rotation_rad == pytest.approx(1)
    assert wrapped.events[0].angular_travel_rad == pytest.approx(1)
    values = wrapped.events[0].angular_trajectory_rad
    assert max(values) > math.pi


def test_translation_unknown_region_and_normalized_scale() -> None:
    result = result_for(region="centre")
    event = result.events[0]
    assert event.pivot.region == "unknown" and event.region_confidence == 0
    assert len(event.translations) == 3
    assert event.classification == "supported_rotation"
    sliding = result_for(slide=0.8).events[0]
    assert sliding.pivot.region == "unknown"
    assert sliding.classification == "rotation_with_translation"
    assert sliding.region_confidence == 0
    reference = result_for().events[0]
    scaled = result_for(scale=7).events[0]
    assert scaled.pivot.region == reference.pivot.region
    assert [t.max_excursion_foot_lengths for t in scaled.translations] == pytest.approx(
        [t.max_excursion_foot_lengths for t in reference.translations]
    )


@pytest.mark.parametrize(
    "case", ["flight", "unknown", "other_unknown", "jitter", "slide", "root"]
)
def test_false_confident_pivots(case: str) -> None:
    source = motion()
    contacts = ground(source)
    if case in ("flight", "unknown", "other_unknown"):
        rotated, contacts = sequence()
        assert isinstance(rotated, Reconstruction)
        source = rotated
        for contact_sample in contacts.samples:
            if case == "flight":
                contact_sample.left = Contact(
                    state="no_contact", quality=Quality(state="inferred")
                )
                contact_sample.support = (
                    "unknown" if contact_sample.right.state == "unknown" else "right"
                )
            else:
                contact = Contact(state="unknown", quality=Quality(state="unknown"))
                if case == "unknown":
                    contact_sample.left = contact
                else:
                    contact_sample.right = contact
                contact_sample.support = "unknown"
    else:
        for sample in source.samples:
            if case == "root":
                sample.root_xyz_world = (sample.global_seconds, 2, 1)
                sample.root_orientation = Quaternion(
                    wxyz=(
                        math.cos(sample.global_seconds / 2),
                        0,
                        0,
                        math.sin(sample.global_seconds / 2),
                    )
                )
            for point in sample.landmarks:
                if point.name.startswith("left") and point.xyz_world:
                    x, y, z = point.xyz_world
                    angle = (
                        0.015 * math.sin(100 * sample.global_seconds)
                        if case == "jitter"
                        else 0
                    )
                    point.xyz_world = (
                        x
                        - y * math.sin(angle)
                        + (sample.global_seconds if case == "slide" else 0),
                        y * math.cos(angle),
                        z,
                    )
    result = derive_pivots(derive_footprints(source, contacts, calibration()))
    assert result.events == []
    assert len(result.trajectory) == 2 * len(source.samples)
    if case in ("flight", "unknown", "other_unknown"):
        assert any(r.unwrapped_yaw_rad is not None for r in result.trajectory)


def test_gaps_occlusion_reused_sources_and_contact_region_conflict() -> None:
    for case in ("gap", "hidden", "reuse", "region"):
        source, contacts = sequence()
        assert isinstance(source, Reconstruction)
        if case == "gap":
            source.samples = [
                s for s in source.samples if not 0.38 < s.global_seconds < 0.58
            ]
            contacts.samples = [
                s for s in contacts.samples if not 0.38 < s.global_seconds < 0.58
            ]
        elif case == "hidden":
            for sample in source.samples:
                if sample.global_seconds == 0.44:
                    for p in sample.landmarks:
                        if p.name == "left_forefoot":
                            p.xyz_world = None
                            p.quality = Quality(state="unknown")
        elif case == "reuse":
            for sample in source.samples:
                for p in sample.landmarks:
                    p.quality.source_ids = [p.name]
        else:
            for contact_sample in contacts.samples:
                if contact_sample.left.state == "contact":
                    contact_sample.left.region = "forefoot"
        result = derive_pivots(derive_footprints(source, contacts, calibration()))
        if case == "reuse":
            assert not result.events
        elif case == "region":
            assert all(e.pivot.region == "unknown" for e in result.events)
        else:
            assert not any(
                e.pivot.interval.start < 0.45 < e.pivot.interval.end
                for e in result.events
            )


def test_artifact_roundtrip_preserves_upstream_and_caches(tmp_path: Path) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    source, _ = sequence()
    assert isinstance(source, Reconstruction)
    motion_handle, cal = persist(store, source), persist(store, calibration())
    # The actual contact estimator recognizes flat synthetic contact on rotation.
    contacts = publish_contacts(store, motion_handle, cal)
    placements = publish_footprints(store, motion_handle, contacts, cal)
    original = placements.metadata.model_dump_json()
    output = publish_pivots(store, placements)
    result = load_pivot_evidence(output)
    assert len(result.events) == 1
    assert result.events[0].pivot.region == "heel"
    assert placements.metadata.model_dump_json() == original
    for name in ("contact_evidence_json", "footprint_evidence_json"):
        assert np.array_equal(output.read_array(name), placements.read_array(name))
    assert publish_pivots(store, placements).path == output.path
    assert (
        publish_pivots(store, placements, PivotConfig(min_excursion_rad=0.2)).path
        != output.path
    )
    assert load_pivot_evidence(output).model_dump_json() == result.model_dump_json()
    with pytest.raises(ValueError, match="physical pivot"):
        load_pivot_evidence(placements)
    bad = result.model_dump()
    bad["events"][0]["rotation_indices"][0] = 1
    with pytest.raises(ValueError):
        PivotSeries.model_validate(bad)


@pytest.mark.parametrize(
    "interruption",
    ["brief_pause", "long_pause", "unknown_support", "half_turn", "high_uncertainty"],
)
def test_hysteresis_reversal_and_interrupted_evidence(interruption: str) -> None:
    source = motion()
    contacts = ground(source)
    for sample in source.samples:
        time = sample.global_seconds
        if interruption == "brief_pause":
            angle = min(time, 0.4) + max(0, time - 0.44)
        elif interruption == "long_pause":
            angle = min(time, 0.3) + max(0, time - 0.6)
        elif interruption == "half_turn":
            angle = 0 if time < 0.5 else math.pi
        else:
            angle = time if time <= 0.5 else 1 - time
        for p in sample.landmarks:
            if p.name.startswith("left") and p.xyz_world:
                x, y, z = p.xyz_world
                p.xyz_world = (x - y * math.sin(angle), y * math.cos(angle), z)
                if interruption == "high_uncertainty":
                    p.quality.uncertainty = 0.06
    if interruption == "unknown_support":
        for contact_sample in contacts.samples:
            if 0.4 <= contact_sample.global_seconds <= 0.6:
                contact_sample.right = Contact(
                    state="unknown", quality=Quality(state="unknown")
                )
                contact_sample.support = "unknown"
    result = derive_pivots(derive_footprints(source, contacts, calibration()))
    if interruption in ("half_turn", "high_uncertainty"):
        assert not result.events
    elif interruption == "brief_pause":
        assert len(result.events) == 1
    elif interruption == "long_pause":
        assert len(result.events) == 2
    else:
        assert len(result.events) == 2
        assert not any(
            e.pivot.interval.start < 0.5 < e.pivot.interval.end for e in result.events
        )


def test_reversal_travel_and_corrupt_translation_links() -> None:
    source = motion()
    contacts = ground(source)
    for sample in source.samples:
        angle = (
            sample.global_seconds
            if sample.global_seconds <= 0.5
            else 1 - sample.global_seconds
        )
        for p in sample.landmarks:
            if p.name.startswith("left") and p.xyz_world:
                x, y, z = p.xyz_world
                p.xyz_world = (x - y * math.sin(angle), y * math.cos(angle), z)
    result = derive_pivots(derive_footprints(source, contacts, calibration()))
    assert len(result.events) == 1
    event = result.events[0]
    assert event.angular_travel_rad > abs(event.pivot.rotation_rad or 0) + 0.5
    bad = result.model_dump()
    bad["events"][0]["translations"][0]["positions"][0] = (999, 999)
    with pytest.raises(ValueError, match="native geometry"):
        PivotSeries.model_validate(bad)


def test_right_supporting_side_and_invalid_configuration() -> None:
    source, contacts = sequence(region="forefoot", sign=-1)
    for sample in source.samples:
        for point in sample.landmarks:
            side, part = point.name.split("_", 1)
            point.name = cast(
                Landmark, ("right" if side == "left" else "left") + "_" + part
            )
    for contact_sample in contacts.samples:
        contact_sample.left, contact_sample.right = (
            contact_sample.right,
            contact_sample.left,
        )
    result = derive_pivots(derive_footprints(source, contacts, calibration()))
    assert len(result.events) == 1
    assert result.events[0].pivot.foot == "right"
    assert result.events[0].pivot.region == "forefoot"
    assert result.events[0].pivot.rotation_rad == pytest.approx(-1)
    for values in (
        {"quiet_seconds": 0},
        {"leave_speed_rad_s": 1},
        {"min_excursion_rad": float("nan")},
    ):
        with pytest.raises(ValueError):
            PivotConfig.model_validate(values)
