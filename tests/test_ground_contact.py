"""Known physical sequences; no dataset, model or contact-accuracy claims."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from contracts.models import (
    Artifact,
    Calibration,
    Ground,
    GroundFrame,
    Landmark,
    Landmark3D,
    Measurement,
    Morphology,
    MotionSample,
    Quality,
    Reconstruction,
    ScaleResolution,
)
from reconstruction.ground import (
    ContactConfig,
    derive_contacts,
    load_contact_evidence,
    publish_contacts,
)
from reconstruction.temporal import publish_temporal_motion
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, StorageRoot, hash_config
from tests.test_articulated import PROV, raw_handle
from tests.test_triangulation import scene


def calibration(metric: bool = True) -> Calibration:
    base = scene()[0]
    base.id = "cal"
    base.ground_status = "resolved"
    base.ground_z = 0
    base.ground_frame = GroundFrame(
        source_to_world=np.eye(4).tolist(),
        plane_normal_source=(0, 0, 1),
        plane_offset_source=0,
        inlier_count=4,
        sample_count=4,
        coverage=4,
        rms_residual=0,
        normal_uncertainty_rad=0,
        axis_uncertainty_rad=0,
        evidence_kind="target",
        evidence_ids=["floor"],
        evidence_producer="synthetic",
        source_revision="scene-v1",
    )
    if metric:
        base.scale, base.world_unit, base.scale_status = "metric", "m", "resolved"
        base.scale_evidence_ids = ["ruler"]
        base.scale_resolution = ScaleResolution(
            evidence_id="ruler",
            source_revision="scene-v1",
            kind="measured",
            measured_length=1,
            measured_unit="m",
            metres_per_source_unit=1,
            producer="synthetic",
        )
    return Calibration.model_validate(base.model_dump())


def motion(
    heights: Callable[[float, str, str], float | None] = lambda t, s, p: 0,
    *,
    rate: int = 50,
    duration: float = 1,
    scale: float = 1,
    metric: bool = True,
) -> Reconstruction:
    samples = []
    for i in range(round(duration * rate) + 1):
        time = i / rate
        points = []
        for side in ("left", "right"):
            for part in ("heel", "forefoot", "foot_outer"):
                height = heights(time, side, part)
                name = cast(Landmark, f"{side}_{part}")
                points.append(
                    Landmark3D(
                        name=name,
                        xyz_world=None
                        if height is None
                        else (
                            (-0.2 if side == "left" else 0.2) * scale,
                            (0 if part == "heel" else 0.2) * scale,
                            height * scale,
                        ),
                        quality=Quality(
                            state="unknown" if height is None else "observed",
                            uncertainty=None if height is None else 1e-5 * scale,
                            source_ids=[f"native:{i}:{name}"],
                        ),
                    )
                )
        samples.append(
            MotionSample(
                global_seconds=time,
                root_xyz_world=(0, 0, scale),
                root_orientation=None,
                landmarks=points,
                quality=Quality(state="observed"),
            )
        )
    return Reconstruction(
        kind="reconstruction",
        id="final",
        schema_version="1.0.0",
        provenance=PROV.model_copy(update={"producer": "reconstruction.temporal"}),
        calibration_id="cal",
        participant_id="participant",
        scale="metric" if metric else "arbitrary",
        samples=samples,
    )


@pytest.mark.parametrize(
    "left,right,support",
    [
        (0, 0, "both"),
        (0, 0.2, "left"),
        (0.2, 0, "right"),
        (0.2, 0.2, "neither"),
        (None, 0, "unknown"),
        (0, None, "unknown"),
    ],
)
def test_standing_flight_and_support_combinations(
    left: float | None,
    right: float | None,
    support: str,
) -> None:
    source = motion(lambda t, s, p: left if s == "left" else right)
    before = source.model_dump_json()
    result = derive_contacts(source, calibration())
    sample = result.samples[-1]
    assert sample.support == support
    assert sample.left.state == (
        "unknown" if left is None else "contact" if left == 0 else "no_contact"
    )
    if left == 0:
        assert sample.left.region == "flat"
    assert [s.global_seconds for s in result.samples] == [
        s.global_seconds for s in source.samples
    ]
    assert source.model_dump_json() == before


@pytest.mark.parametrize("region", ["heel", "forefoot"])
def test_partial_contact(region: str) -> None:
    result = derive_contacts(
        motion(lambda t, s, p: 0 if (p == "heel") == (region == "heel") else 0.15),
        calibration(),
    )
    assert result.samples[-1].left.state == "contact"
    assert result.samples[-1].left.region == region


def test_partial_geometry_does_not_fabricate_region_or_flight() -> None:
    result = derive_contacts(
        motion(lambda t, s, p: None if p == "foot_outer" else 0), calibration()
    )
    assert result.samples[-1].left.state == "contact"
    assert result.samples[-1].left.region is None
    result = derive_contacts(
        motion(lambda t, s, p: None if p == "foot_outer" else 0.2), calibration()
    )
    assert result.samples[-1].left.state == "unknown"


def lift(time: float, side: str, part: str) -> float:
    if side == "right":
        return 0
    if time < 0.3:
        return 0
    if time < 0.4:
        return (time - 0.3) * 2
    if time < 0.7:
        return 0.2
    if time < 0.8:
        return (0.8 - time) * 2
    return 0


def test_lift_off_flight_landing_timing_and_lineage() -> None:
    result = derive_contacts(motion(lift), calibration())
    by_time = {round(s.global_seconds, 2): s for s in result.samples}
    assert by_time[0.2].support == "both"
    assert by_time[0.32].support == "unknown"
    assert by_time[0.5].support == "right"
    assert by_time[0.8].support == "unknown"
    assert by_time[0.94].support == "both"
    transitions = [e for e in result.left if e.transition_bracket]
    assert len(transitions) == 3  # initial contact, lift-off, landing
    for e in transitions:
        assert e.contributing_times
        assert e.transition_bracket is not None
        assert e.transition_bracket[1] - e.transition_bracket[0] <= 0.020001
    assert "native" in result.samples[-1].left.quality.source_ids[-1]
    assert "floor" in result.samples[-1].left.quality.source_ids
    assert result.left[-1].landmarks[0].motion_interval == (0.98, 1)


@pytest.mark.parametrize("rate", [20, 30, 60, 100])
def test_equivalent_rates_within_native_temporal_resolution(rate: int) -> None:
    result = derive_contacts(motion(lift, rate=rate), calibration())
    times = [
        s.global_seconds
        for i, s in enumerate(result.samples)
        if i
        and s.left.state in {"contact", "no_contact"}
        and result.samples[i - 1].left.state != s.left.state
    ]
    baseline = derive_contacts(motion(lift, rate=100), calibration())
    expected = [
        s.global_seconds
        for i, s in enumerate(baseline.samples)
        if i
        and s.left.state in {"contact", "no_contact"}
        and baseline.samples[i - 1].left.state != s.left.state
    ]
    assert len(times) == len(expected) == 3
    assert times == pytest.approx(expected, abs=2 / rate + 1 / 100)


def test_noise_and_short_airborne_spike_do_not_chatter() -> None:
    def noisy(t: float, s: str, p: str) -> float:
        return 0.2 if 0.5 <= t < 0.52 else 0.001 * math_sin(t)

    result = derive_contacts(motion(noisy), calibration())
    assert all(s.support in {"both", "unknown"} for s in result.samples)
    # An ambiguous high-speed sample interrupts certainty and requires reacquisition;
    # it must not manufacture a flight/support-switch event from a single spike.
    assert sum(e.transition_bracket is not None for e in result.left) == 2
    assert all(
        e.candidate != "no_contact" or not e.transition_bracket for e in result.left
    )


def test_noise_in_contact_hysteresis_band_preserves_support() -> None:
    result = derive_contacts(
        motion(lambda t, s, p: min(t * 0.05, 0.035) + 0.001 * math_sin(t)),
        calibration(),
    )
    assert all(s.support == "both" for s in result.samples[6:])
    assert sum(e.transition_bracket is not None for e in result.left) == 1


def math_sin(t: float) -> float:
    return float(np.sin(30 * t))


@pytest.mark.parametrize("weakness", ["unknown", "no_sigma", "no_sources", "uncertain"])
def test_occluded_and_unreliable_geometry_is_never_airborne(weakness: str) -> None:
    source = motion()
    for sample in source.samples[20:]:
        for p in sample.landmarks:
            if p.name.startswith("left"):
                if weakness == "unknown":
                    p.xyz_world, p.quality = None, Quality(state="unknown")
                elif weakness == "no_sigma":
                    p.quality.uncertainty = None
                elif weakness == "no_sources":
                    p.quality.source_ids = []
                else:
                    p.quality.uncertainty = 0.2
    result = derive_contacts(source, calibration())
    assert result.samples[19].support == "both"
    assert all(s.left.state == "unknown" for s in result.samples[20:])
    assert all(s.support == "unknown" for s in result.samples[20:])


def test_inferred_state_alone_does_not_establish_contact() -> None:
    source = motion()
    for sample in source.samples:
        for p in sample.landmarks:
            p.quality.state, p.quality.uncertainty = "inferred", None
    result = derive_contacts(source, calibration())
    assert all(s.support == "unknown" for s in result.samples)


def test_gap_and_repeated_native_sources_cannot_confirm_contact() -> None:
    source = motion()
    first = source.samples[0]
    for sample in source.samples:
        for p, original in zip(sample.landmarks, first.landmarks, strict=True):
            p.quality.source_ids = original.quality.source_ids.copy()
            p.quality.state = "interpolated"
    result = derive_contacts(source, calibration())
    assert all(s.support == "unknown" for s in result.samples)
    assert "no_new_native_evidence" in result.left[-1].reasons
    source = motion()
    source.samples = source.samples[:15] + source.samples[35:]
    result = derive_contacts(source, calibration())
    assert result.samples[14].support == "both"
    assert result.samples[15].support == "unknown"
    assert result.samples[-1].support == "both"


def test_missing_ground_and_unresolved_units_are_explicitly_unavailable() -> None:
    cal = calibration()
    cal.ground_status, cal.ground_frame, cal.ground_z = "unresolved", None, None
    result = derive_contacts(motion(), cal)
    assert result.evidence_unit == "unavailable"
    assert all(s.support == "unknown" for s in result.samples)
    assert "unresolved_ground" in result.left[0].reasons
    result = derive_contacts(motion(metric=False), calibration(False))
    assert result.evidence_unit == "unavailable"
    assert "unresolved_metric_scale" in result.left[0].reasons


def shape(scale: float = 1, sigma: float = 1e-5) -> Morphology:
    return Morphology(
        kind="morphology",
        id="shape",
        schema_version="1.0.0",
        provenance=PROV,
        participant_id="participant",
        measurements=[
            Measurement(
                name="left_shank",
                value=scale,
                unit="arbitrary",
                quality=Quality(
                    state="inferred",
                    uncertainty=sigma * scale,
                    source_ids=["shape-evidence"],
                ),
            )
        ],
    )


def test_body_normalization_scale_invariance_and_reliability() -> None:
    config = ContactConfig(units="body_normalized")
    for scale in (0.01, 1, 100):
        result = derive_contacts(
            motion(lift, scale=scale, metric=False),
            calibration(False),
            shape(scale),
            config,
        )
        assert result.evidence_unit == "body_ratio"
        assert result.samples[25].support == "right"
        assert result.samples[-1].support == "both"
        assert result.normalization_length == scale
        assert "shape-evidence" in result.samples[-1].left.quality.source_ids
    for morphology in (None, shape(sigma=0.2)):
        result = derive_contacts(
            motion(metric=False), calibration(False), morphology, config
        )
        assert result.evidence_unit == "unavailable"
        assert result.samples[-1].support == "unknown"


def test_ground_uncertainty_and_motion_prevent_confident_contact() -> None:
    cal = calibration()
    assert cal.ground_frame is not None
    cal.ground_frame.rms_residual = 0.1
    assert derive_contacts(motion(), cal).samples[-1].support == "unknown"
    cal.ground_frame.rms_residual = 0
    cal.ground_frame.normal_uncertainty_rad = 0.2
    assert derive_contacts(motion(), cal).samples[-1].support == "unknown"
    source = motion(lambda t, s, p: 0.01 * float(np.sin(100 * t)))
    assert all(
        s.left.state != "contact"
        for s in derive_contacts(source, calibration()).samples
    )


def persist(store: ArtifactStore, model: Artifact) -> ArtifactHandle:
    key = ArtifactKey(
        layer=model.kind,
        inputs={"fixture": hash_config(model.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision="synthetic",
        config_digest=PROV.config_digest,
    )
    return store.get_or_create(key, lambda: (model, {}))


def test_independent_artifact_roundtrip_cache_lineage_and_invalid_inputs(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    source = motion(lift)
    source_handle, cal_handle = persist(store, source), persist(store, calibration())
    original = source_handle.metadata.model_dump_json()
    output = publish_contacts(store, source_handle, cal_handle)
    assert isinstance(output.metadata, Ground)
    loaded = load_contact_evidence(output)
    assert loaded == derive_contacts(source, calibration())
    assert output.metadata.samples == loaded.samples
    assert output.metadata.footprints == [] and output.metadata.pivots == []
    assert publish_contacts(store, source_handle, cal_handle).path == output.path
    assert (
        publish_contacts(
            store, source_handle, cal_handle, config=ContactConfig(contact_seconds=0.12)
        ).path
        != output.path
    )
    cal = calibration()
    assert cal.ground_frame is not None
    cal.ground_frame.rms_residual = 0.002
    assert (
        publish_contacts(store, source_handle, persist(store, cal)).path != output.path
    )
    assert source_handle.metadata.model_dump_json() == original
    with pytest.raises(ValueError, match="physical contact"):
        load_contact_evidence(source_handle)
    source.provenance.producer = "reconstruction.triangulation"
    with pytest.raises(ValueError, match="final temporal"):
        publish_contacts(store, persist(store, source), cal_handle)
    cal.ground_status, cal.ground_frame, cal.ground_z = "unresolved", None, None
    with pytest.raises(ValueError, match="unresolved ground"):
        publish_contacts(store, source_handle, persist(store, cal))


def test_temporal_publisher_to_contact_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    raw = motion()
    raw.provenance.producer = "reconstruction.triangulation"
    final = publish_temporal_motion(store, raw_handle(store, raw))
    output = publish_contacts(
        store, final.motion, persist(store, calibration()), final.morphology
    )
    result = load_contact_evidence(output)
    assert result.reconstruction_id == final.motion.metadata.id
    assert result.morphology_id == final.morphology.metadata.id
    assert result.samples[25].support == "both"
    assert result.samples[25].left.quality.state == "inferred"
    # The temporal smoother's previous window already used its final native point.
    # At that boundary there is no fresh motion evidence, so certainty is withheld.
    assert result.samples[-1].support == "unknown"
    assert "no_new_native_evidence" in result.left[-1].reasons
    first_confirmation = next(
        (sample, diag)
        for sample, diag in zip(result.samples, result.left, strict=True)
        if diag.transition_bracket
    )
    assert len(first_confirmation[0].left.quality.source_ids) > 6
    assert (
        first_confirmation[1].contributing_times[0]
        < (first_confirmation[1].contributing_times[-1])
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"leave_height": 0.01},
        {"leave_vertical_speed": 0.1},
        {"contact_seconds": 0},
        {"enter_height": float("nan")},
        {"force": True},
    ],
)
def test_invalid_configuration(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ContactConfig.model_validate(settings)


def test_mismatched_inputs_and_nonfinite_motion_rejected() -> None:
    cal = calibration()
    cal.id = "other"
    with pytest.raises(ValueError, match="calibration"):
        derive_contacts(motion(), cal)
    morphology = shape()
    morphology.participant_id = "other"
    with pytest.raises(ValueError, match="participant"):
        derive_contacts(motion(metric=False), calibration(False), morphology)
    source = motion()
    source.samples[0].global_seconds = float("nan")
    with pytest.raises(ValueError):
        derive_contacts(source, calibration())
