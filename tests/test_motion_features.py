"""Synthetic offline feature/event acceptance on one absolute execution clock."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from pydantic import ValidationError

from contracts.models import (
    Contact,
    Ground,
    Landmark,
    Landmark3D,
    MotionFeatures,
    Quality,
    Reconstruction,
)
from reconstruction.features import (
    TRACKS,
    FeatureConfig,
    FeatureSeries,
    derive_features,
    load_feature_evidence,
    publish_features,
)
from reconstruction.footprints import derive_footprints, publish_footprints
from reconstruction.ground import publish_contacts
from reconstruction.pivots import publish_pivots
from reconstruction.temporal import publish_temporal_motion
from storage import ArtifactStore, StorageRoot
from tests.test_articulated import participant, raw_handle
from tests.test_ground_contact import calibration, motion, persist


def sequence(
    rate: int = 50, *, flat: bool = False, noisy: bool = False
) -> tuple[Reconstruction, Ground]:
    source = motion(rate=rate, duration=1.5)
    rng = np.random.default_rng(33)
    for i, sample in enumerate(source.samples):
        t = sample.global_seconds
        positions = {
            "pelvis": (0, 0, 1),
            "left_hip": (-0.15, 0, 1),
            "right_hip": (0.15, 0, 1),
            "left_shoulder": (-0.25, 0, 1.6),
            "right_shoulder": (0.25, 0, 1.6),
            "head": (0, 0, 1.85),
            "neck": (0, 0, 1.7),
            "left_ear": (-0.1, 0, 1.85),
            "right_ear": (0.1, 0, 1.85),
            "nose": (0, 0.1, 1.85),
        }
        for side, sign, peak in (("left", -1, 0.4), ("right", 1, 0.75)):
            pulse = 0 if flat else max(0, 1 - abs(t - peak) / 0.1)
            distance = 0.3 + 0.4 * pulse
            jitter = float(rng.normal(0, 0.0005)) if noisy else 0
            positions[f"{side}_wrist"] = (sign * 0.25 + jitter, distance, 1.6)
            positions[f"{side}_elbow"] = (
                sign * (0.25 + math.sqrt(0.4**2 - (distance / 2) ** 2)),
                distance / 2,
                1.6,
            )
            positions[f"{side}_knee"] = (sign * 0.15, 0.12, 0.55)
            positions[f"{side}_ankle"] = (sign * 0.15, 0, 0.1)
        for name, xyz in positions.items():
            sample.landmarks.append(
                Landmark3D(
                    name=cast(Landmark, name),
                    xyz_world=xyz,
                    quality=Quality(
                        state="observed",
                        uncertainty=1e-5,
                        source_ids=[f"observation:{i}:{name}"],
                    ),
                )
            )
    contacts = Ground(
        kind="ground",
        id="contacts",
        schema_version="1.0.0",
        provenance=source.provenance,
        reconstruction_id=source.id,
        scale=source.scale,
        samples=[],
    )
    from contracts.models import GroundSample

    for i, sample in enumerate(source.samples):
        left: Literal["contact", "no_contact"] = (
            "contact"
            if sample.global_seconds < 0.3 or sample.global_seconds >= 0.6
            else "no_contact"
        )
        contacts.samples.append(
            GroundSample(
                global_seconds=sample.global_seconds,
                left=Contact(
                    state=left,
                    quality=Quality(
                        state="inferred", uncertainty=0.001, source_ids=[source.id]
                    ),
                ),
                right=Contact(
                    state="contact",
                    quality=Quality(
                        state="inferred", uncertainty=0.001, source_ids=[source.id]
                    ),
                ),
                support="both" if left == "contact" else "right",
            )
        )
    return source, contacts


@pytest.mark.parametrize("rate", [25, 50, 100])
def test_rapid_extension_retraction_independent_tracks_and_contact(rate: int) -> None:
    source, ground = sequence(rate)
    result = derive_features(source, ground)
    assert len(result.trajectory) == len(source.samples) * 6
    assert {r.track for r in result.trajectory} == set(TRACKS)
    for track, target in (("left_arm", 0.4), ("right_arm", 0.75)):
        maxima = [
            e
            for e in result.events
            if e.track == track and e.kind == "extension_maximum"
        ]
        assert len(maxima) == 1
        assert abs(maxima[0].global_seconds - target) <= 1 / rate
        assert maxima[0].tolerance_seconds == pytest.approx(1 / rate)
        kinds = {e.kind for e in result.events if e.track == track}
        assert {
            "motion_onset",
            "extension_start",
            "extension_end",
            "direction_change",
        } <= kinds
        assert maxima[0].quality.source_ids
        assert maxima[0].duration_seconds > 0
    contacts = [e for e in result.events if e.kind in ("lift_off", "first_contact")]
    assert [e.kind for e in contacts] == ["lift_off", "first_contact"]
    assert abs(contacts[0].global_seconds - 0.3) <= 1 / rate
    assert abs(contacts[1].global_seconds - 0.6) <= 1 / rate
    assert all(e.ground_sample_indices for e in contacts)
    assert FeatureSeries.model_validate_json(result.model_dump_json()) == result
    assert all(not e.kind.endswith("strike") for e in result.events)


def test_flat_noisy_and_missing_segments_do_not_invent_events() -> None:
    for noisy in (False, True):
        source, ground = sequence(flat=True, noisy=noisy)
        ground.samples = []
        result = derive_features(source, ground)
        assert result.events == []
    source, ground = sequence()
    for sample in source.samples:
        if 0.3 <= sample.global_seconds <= 0.5:
            for point in sample.landmarks:
                if point.name.startswith("left_") and point.name not in (
                    "left_hip",
                    "left_shoulder",
                ):
                    point.xyz_world, point.quality = None, Quality(state="unknown")
    result = derive_features(source, ground)
    assert not any(e.track == "left_arm" for e in result.events)
    missing = [
        r
        for r in result.trajectory
        if r.track == "left_arm" and 0.3 <= r.global_seconds <= 0.5
    ]
    assert all(r.extension.value is None and r.linear.velocity is None for r in missing)
    # Unknown native contact cannot be converted into first contact.
    for s in ground.samples:
        if 0.56 <= s.global_seconds < 0.6:
            s.left = Contact(state="unknown", quality=Quality(state="unknown"))
            s.support = "unknown"
    assert not any(
        e.kind == "first_contact" for e in derive_features(source, ground).events
    )


def test_crossing_retains_front_order_and_requires_confirmation() -> None:
    source, ground = sequence(flat=True)
    ground.samples = []
    for sample in source.samples:
        fraction = min(1, max(0, (sample.global_seconds - 0.3) / 0.3))
        xyz = {
            "left_elbow": (-0.3, 0.1, 1.6),
            "right_elbow": (0.3, 0, 1.6),
            "left_wrist": (-0.6 + 0.9 * fraction, 0.1, 1.1),
            "right_wrist": (0.6 - 0.9 * fraction, 0, 1.1),
        }
        for p in sample.landmarks:
            if p.name in xyz:
                p.xyz_world = xyz[p.name]
    result = derive_features(source, ground)
    events = [e for e in result.events if e.kind == "arm_crossing"]
    assert len(events) == 2
    assert {e.track for e in events} == {"left_arm", "right_arm"}
    assert all(0.48 < e.global_seconds < 0.54 for e in events)
    for e in events:
        row = result.trajectory[e.motion_sample_indices[-1] * 6]
        relation = next(r for r in row.relations if r.axis == "crossing")
        assert relation.front_entity == "left_forearm"


def test_noisy_plateau_one_maximum_and_no_derivative_endpoint_events() -> None:
    source, ground = sequence()
    ground.samples = []
    # Plateau a rapid extension; tiny oscillations must not multiply the event.
    for sample in source.samples:
        if 0.38 <= sample.global_seconds <= 0.44:
            for p in sample.landmarks:
                if p.name == "left_wrist":
                    p.xyz_world = (
                        -0.25,
                        0.7 + 0.0002 * math.sin(sample.global_seconds * 900),
                        1.6,
                    )
                if p.name == "left_elbow":
                    p.xyz_world = (-0.25 - math.sqrt(0.4**2 - 0.35**2), 0.35, 1.6)
    result = derive_features(source, ground)
    maxima = [
        e
        for e in result.events
        if e.track == "left_arm" and e.kind == "extension_maximum"
    ]
    assert len(maxima) == 1
    assert 0.38 <= maxima[0].global_seconds <= 0.44
    assert not any(e.global_seconds in (0, 1.5) for e in result.events)


def test_units_absolute_timing_and_quality_gates() -> None:
    source, ground = sequence()
    original = derive_features(source, ground)
    for s in source.samples:
        s.global_seconds += 8
    for gs in ground.samples:
        gs.global_seconds += 8
    shifted = derive_features(source, ground)
    assert [e.kind for e in shifted.events] == [e.kind for e in original.events]
    assert [e.global_seconds for e in shifted.events] == pytest.approx(
        [e.global_seconds + 8 for e in original.events]
    )
    source.scale = ground.scale = "arbitrary"
    with pytest.raises(ValueError, match="threshold units"):
        derive_features(source, ground)
    assert derive_features(source, ground, FeatureConfig(world_unit="arbitrary")).events
    with pytest.raises(ValidationError):
        FeatureConfig(enter_speed=0.01)
    for sample in source.samples:
        for p in sample.landmarks:
            p.quality.uncertainty = 1
    ground.samples = []
    result = derive_features(source, ground, FeatureConfig(world_unit="arbitrary"))
    assert result.events == []
    assert all(r.position_local.value is None for r in result.trajectory)


def test_persistence_dense_time_cache_revisions_and_upstream_immutability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    source = participant()
    source.provenance.producer = "reconstruction.triangulation"
    temporal = publish_temporal_motion(store, raw_handle(store, source))
    cal = persist(store, calibration())
    contacts = publish_contacts(store, temporal.motion, cal)
    placements = publish_footprints(store, temporal.motion, contacts, cal)
    pivots = publish_pivots(store, placements)
    before = {
        h.path: (h.path / "manifest.json").read_bytes()
        for h in (temporal.motion, contacts, placements, pivots)
    }
    output = publish_features(store, temporal.motion, pivots)
    assert isinstance(output.metadata, MotionFeatures)
    result = load_feature_evidence(output)
    assert result.reconstruction_id == temporal.motion.metadata.id
    assert result.ground_id == pivots.metadata.id
    assert isinstance(temporal.motion.metadata, Reconstruction)
    assert len(result.trajectory) == 6 * len(temporal.motion.metadata.samples)
    for h in (temporal.motion, contacts, placements, pivots):
        assert (h.path / "manifest.json").read_bytes() == before[h.path]
    import reconstruction.features.artifact as adapter
    from reconstruction.features.core import derive_features as derive

    def unexpected(*args: object, **kwargs: object) -> FeatureSeries:
        raise AssertionError("cached extraction must not run")

    monkeypatch.setattr(adapter, "derive_features", unexpected)
    assert publish_features(store, temporal.motion, pivots).path == output.path
    monkeypatch.setattr(adapter, "derive_features", derive)
    assert (
        publish_features(
            store, temporal.motion, pivots, FeatureConfig(min_excursion=0.03)
        ).path
        != output.path
    )
    assert publish_features(store, temporal.motion, contacts).path != output.path
    changed_source = participant(upper=0.32)
    changed_source.provenance.producer = "reconstruction.triangulation"
    changed_motion = publish_temporal_motion(store, raw_handle(store, changed_source))
    changed_ground = publish_contacts(store, changed_motion.motion, cal)
    assert (
        publish_features(store, changed_motion.motion, changed_ground).path
        != output.path
    )


def test_stable_placement_uses_explicit_ground_evidence() -> None:
    source, ground = sequence(flat=True)
    placements = derive_footprints(source, ground, calibration())
    ground.footprints = [e.footprint for e in placements.events]
    result = derive_features(source, ground, placements=placements)
    stable = [e for e in result.events if e.kind == "stable_placement"]
    assert stable
    assert all(e.source_event_ids for e in stable)
    assert not any(
        e.kind == "stable_placement" for e in derive_features(source, ground).events
    )


def test_body_translation_head_rotation_and_leg_extension() -> None:
    from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

    source, ground = sequence(flat=True)
    ground.samples = []
    for sample in source.samples:
        t = sample.global_seconds
        translation = 0.2 * max(0, 1 - abs(t - 0.4) / 0.15)
        points = {p.name: p for p in sample.landmarks}
        # A rotation confined to the head has a distinct onset from the root.
        angle = 0.5 * min(1, max(0, (t - 0.7) / 0.15))
        yaw = Rotation.from_euler("z", angle)
        center = np.array([0, 0, 1.85])
        for name in ("left_ear", "right_ear", "nose"):
            p = points[name]
            p.xyz_world = tuple(yaw.apply(np.array(p.xyz_world) - center) + center)
        distance = 0.3 + 0.4 * max(0, 1 - abs(t - 1.1) / 0.1)
        points["left_ankle"].xyz_world = (-0.15, distance, 1)
        points["left_knee"].xyz_world = (
            -0.15 - math.sqrt(0.4**2 - (distance / 2) ** 2),
            distance / 2,
            1,
        )
        for p in sample.landmarks:
            assert p.xyz_world is not None
            x, y, z = p.xyz_world
            p.xyz_world = (x + translation, y, z)
        sample.root_xyz_world = (translation, 0, 1)
    result = derive_features(source, ground)
    root_onsets = [
        e for e in result.events if e.track == "body_root" and e.kind == "motion_onset"
    ]
    head_onsets = [
        e for e in result.events if e.track == "head" and e.kind == "motion_onset"
    ]
    assert len(root_onsets) == len(head_onsets) == 1
    assert root_onsets[0].global_seconds < 0.3
    assert 0.62 < head_onsets[0].global_seconds < 0.75
    assert any(
        e.track == "body_root"
        and e.kind == "direction_change"
        and abs(e.global_seconds - 0.4) < 0.021
        for e in result.events
    )
    leg = [
        e
        for e in result.events
        if e.track == "left_leg" and e.kind == "extension_maximum"
    ]
    assert len(leg) == 1 and leg[0].global_seconds == pytest.approx(1.1)
    assert not any(e.track in ("left_arm", "right_arm") for e in result.events)
    assert any(
        r.track == "head"
        and r.angular.velocity is not None
        and abs(r.angular.velocity[2]) > 1
        for r in result.trajectory
    )


def test_contact_flicker_and_interpolated_extension_cannot_create_boundaries() -> None:
    source, ground = sequence(flat=True)
    for s in ground.samples:
        s.left = s.right.model_copy(deep=True)
        s.support = "both"
    ground.samples[20].left = Contact(
        state="no_contact",
        quality=Quality(state="inferred", uncertainty=0.001, source_ids=[source.id]),
    )
    ground.samples[20].support = "right"
    assert derive_features(source, ground).events == []
    source, ground = sequence()
    ground.samples = []
    for sample in source.samples:
        if 0.38 <= sample.global_seconds <= 0.42:
            for p in sample.landmarks:
                if p.name == "left_wrist":
                    p.quality.state = "interpolated"
    result = derive_features(source, ground)
    assert not any(
        e.track == "left_arm" and e.kind == "extension_maximum" for e in result.events
    )


def test_pivot_boundary_candidates_reference_upstream_events() -> None:
    from tests.test_pivots import result_for

    pivots = result_for()
    from tests.test_pivots import sequence as pivot_sequence

    source, ground = pivot_sequence()
    ground.pivots = [e.pivot for e in pivots.events]
    result = derive_features(source, ground)
    boundaries = [e for e in result.events if e.kind in ("pivot_start", "pivot_end")]
    assert [e.kind for e in boundaries] == ["pivot_start", "pivot_end"]
    assert [e.global_seconds for e in boundaries] == pytest.approx([0.3, 0.6])
    assert all(e.source_event_ids == [pivots.events[0].pivot.id] for e in boundaries)


def test_preparation_minimum_precedes_extension_maximum() -> None:
    source, ground = sequence()
    ground.samples = []
    for sample in source.samples:
        t = sample.global_seconds
        if t > 0.3:
            continue
        distance = 0.3 - 0.15 * max(0, 1 - abs(t - 0.2) / 0.08)
        for p in sample.landmarks:
            if p.name == "left_wrist":
                p.xyz_world = (-0.25, distance, 1.6)
            elif p.name == "left_elbow":
                p.xyz_world = (
                    -0.25 - math.sqrt(0.4**2 - (distance / 2) ** 2),
                    distance / 2,
                    1.6,
                )
    result = derive_features(source, ground)
    preparation = [
        e
        for e in result.events
        if e.track == "left_arm" and e.kind == "preparation_candidate"
    ]
    maxima = [
        e
        for e in result.events
        if e.track == "left_arm" and e.kind == "extension_maximum"
    ]
    assert len(preparation) == len(maxima) == 1
    assert preparation[0].global_seconds == pytest.approx(0.2)
    assert preparation[0].global_seconds < maxima[0].global_seconds


def test_feature_contract_bundle_rejects_unrelated_ground() -> None:
    from contracts.models import validate_bundle
    from tests.test_contracts import base, bundle

    data = bundle()
    reconstruction = next(a for a in data if a["kind"] == "reconstruction")
    ground = next(a for a in data if a["kind"] == "ground")
    data.append(
        base("motion_features", "features")
        | {
            "reconstruction_id": reconstruction["id"],
            "ground_id": ground["id"],
            "arrays": [
                {
                    "id": "features",
                    "dtype": "uint8",
                    "shape": [1],
                    "axes": ["json_byte"],
                }
            ],
        }
    )
    assert isinstance(validate_bundle(data)[-1], MotionFeatures)
    data[-1]["ground_id"] = "nonexistent"
    with pytest.raises(ValueError, match="reference"):
        validate_bundle(data)


def rotating_head(duration: float = 0.15) -> tuple[Reconstruction, Ground]:
    from scipy.spatial.transform import Rotation

    source, ground = sequence(flat=True)
    ground.samples = []
    center = np.array([0, 0, 1.85])
    for sample in source.samples:
        yaw = Rotation.from_euler(
            "z", 0.5 * min(1, max(0, (sample.global_seconds - 0.7) / duration))
        )
        for p in sample.landmarks:
            if p.name in ("left_ear", "right_ear", "nose"):
                p.xyz_world = tuple(yaw.apply(np.array(p.xyz_world) - center) + center)
    return source, ground


@pytest.mark.parametrize("single_gap", [False, True])
def test_interpolated_head_frame_and_derivatives_cannot_create_onset(
    single_gap: bool,
) -> None:
    source, ground = rotating_head(0.04 if single_gap else 0.15)
    assert any(e.track == "head" for e in derive_features(source, ground).events)
    for sample in source.samples:
        if not single_gap or sample.global_seconds == 0.72:
            for p in sample.landmarks:
                if p.name in ("left_ear", "right_ear", "nose"):
                    p.quality.state = "interpolated"
    result = derive_features(source, ground)
    assert not any(e.track == "head" for e in result.events)
    rows = [r for r in result.trajectory if r.track == "head"]
    marked = [r for r in rows if not single_gap or r.global_seconds == 0.72]
    assert all(r.orientation.orientation is not None for r in marked)
    assert all(r.orientation.quality.state == "interpolated" for r in marked)
    assert all(r.angular.velocity_quality.state == "interpolated" for r in marked)
    assert all(r.angular.acceleration_quality.state == "interpolated" for r in marked)
    if single_gap:
        neighbor = next(r for r in rows if r.global_seconds == 0.7)
        assert neighbor.orientation.quality.state == "inferred"
        assert neighbor.angular.velocity_quality.state == "interpolated"


def test_persisted_temporal_head_interpolation_keeps_feature_quality(
    tmp_path: Path,
) -> None:
    from reconstruction.temporal.core import TemporalConfig

    source, _ = rotating_head()
    source.provenance.producer = "reconstruction.triangulation"
    for sample in source.samples:
        for p in sample.landmarks:
            if p.name in ("left_ear", "right_ear", "nose"):
                p.quality.state = "interpolated"
    store = ArtifactStore(StorageRoot(tmp_path))
    temporal = publish_temporal_motion(
        store, raw_handle(store, source), TemporalConfig(regularization=0)
    )
    assert isinstance(temporal.motion.metadata, Reconstruction)
    assert all(
        p.quality.state == "interpolated"
        for s in temporal.motion.metadata.samples
        for p in s.landmarks
        if p.name in ("left_ear", "right_ear", "nose")
    )
    _, ground = sequence(flat=True)
    ground.samples = []
    ground.reconstruction_id = temporal.motion.metadata.id
    output = publish_features(store, temporal.motion, persist(store, ground))
    result = load_feature_evidence(output)
    assert not any(e.track == "head" for e in result.events)
    head = [r for r in result.trajectory if r.track == "head"]
    assert all(r.orientation.quality.state == "interpolated" for r in head)
    assert any(r.angular.velocity is not None for r in head)
    assert all(
        r.angular.velocity_quality.state == "interpolated"
        for r in head
        if r.angular.velocity is not None
    )


@pytest.mark.parametrize(
    "part", ["body", "left_hand", "right_hand", "left_foot", "right_foot"]
)
@pytest.mark.parametrize("state", ["interpolated", "unknown"])
def test_native_frame_inputs_qualify_cached_geometry(
    part: str,
    state: Literal["interpolated", "unknown"],
) -> None:
    from reconstruction.detailed import derive_sample

    source, ground = sequence(flat=True)
    ground.samples = []
    for i, sample in enumerate(source.samples):
        points = {p.name: p for p in sample.landmarks}
        for side, sign in (("left", -1), ("right", 1)):
            wrist = points[cast(Landmark, f"{side}_wrist")]
            assert wrist.xyz_world is not None
            x, y, z = wrist.xyz_world
            for finger, delta in (("index", 0.03), ("pinky", -0.03)):
                name = cast(Landmark, f"{side}_{finger}_mcp")
                sample.landmarks.append(
                    Landmark3D(
                        name=name,
                        xyz_world=(x + delta, y + 0.06, z),
                        quality=Quality(
                            state="observed",
                            uncertainty=1e-5,
                            source_ids=[f"observation:{i}:{name}"],
                        ),
                    )
                )
            outer = points[cast(Landmark, f"{side}_foot_outer")]
            assert outer.xyz_world is not None
            x, y, z = outer.xyz_world
            outer.xyz_world = (x + sign * 0.05, y, z)
    geometry = [
        derive_sample(s, reconstruction_id=source.id, representation="regularized")
        for s in source.samples
    ]
    original_geometry = [d.model_dump_json() for d in geometry]
    for sample, detail in zip(source.samples, geometry, strict=True):
        sample.root_orientation = detail.body_frame.orientation
    names: tuple[str, ...]
    if part == "body":
        names = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
        track = "body_root"
    else:
        side, region = part.split("_")
        parts = (
            ("wrist", "index_mcp", "pinky_mcp")
            if region == "hand"
            else ("heel", "forefoot", "foot_outer")
        )
        names = tuple(f"{side}_{p}" for p in parts)
        track = f"{side}_{'arm' if region == 'hand' else 'leg'}"
    for sample in source.samples:
        for p in sample.landmarks:
            if p.name in names:
                p.quality.state = state
                if state == "unknown":
                    p.xyz_world = None
                    p.quality.uncertainty = None
    result = derive_features(
        source,
        ground,
        geometry=geometry,
        root_orientation_quality=[d.body_frame.quality for d in geometry],
        root_orientation_from_body=[True] * len(source.samples),
    )
    rows = [r for r in result.trajectory if r.track == track]
    assert all(r.orientation.quality.state == state for r in rows)
    if state == "unknown":
        assert all(
            r.orientation.orientation is None and r.angular.velocity is None
            for r in rows
        )
    else:
        assert all(r.orientation.orientation is not None for r in rows)
        assert any(r.angular.velocity is not None for r in rows)
        assert all(
            r.angular.velocity_quality.state == state
            for r in rows
            if r.angular.velocity is not None
        )
    if part == "body":
        local = [r for r in result.trajectory if r.track != "body_root"]
        assert all(r.position_local.quality.state == state for r in local)
        assert all(r.orientation.quality.state == state for r in local)
        # Interpolation in the reference frame affects relative measurements;
        # independently observed world positions keep their native quality.
        assert all(r.position_world.quality.state == "observed" for r in local)
        if state == "interpolated":
            assert all(
                next(r for r in row.relations if r.axis == "right").quality.state
                == state
                for row in local
            )
    else:
        other = [r for r in result.trajectory if r.track == "head"]
        assert all(r.orientation.quality.state == "inferred" for r in other)
    assert [d.model_dump_json() for d in geometry] == original_geometry


def test_independent_root_orientation_keeps_its_own_quality() -> None:
    from contracts.models import Quaternion

    source, ground = sequence(flat=True)
    ground.samples = []
    for sample in source.samples:
        sample.root_orientation = Quaternion(wxyz=(1, 0, 0, 0))
        for p in sample.landmarks:
            if p.name in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
                p.quality.state = "interpolated"
    result = derive_features(
        source,
        ground,
        root_orientation_quality=[
            Quality(state="observed", uncertainty=1e-4, source_ids=["root-sensor"])
            for _ in source.samples
        ],
        root_orientation_from_body=[False] * len(source.samples),
    )
    root = [r for r in result.trajectory if r.track == "body_root"]
    assert all(r.orientation.quality.state == "observed" for r in root)
    assert all(r.orientation.quality.source_ids == ["root-sensor"] for r in root)
    assert all(
        r.angular.velocity_quality.state == "inferred"
        for r in root
        if r.angular.velocity is not None
    )
