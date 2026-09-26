"""Conservative geometry-based contact estimates on native reconstructed times."""

from __future__ import annotations

import math
from typing import Literal, cast

from pydantic import Field, model_validator

from contracts.models import (
    Calibration,
    Contact,
    GroundSample,
    Landmark,
    Landmark3D,
    Morphology,
    Quality,
    Reconstruction,
    StrictModel,
)

REVISION: Literal["foot-contact-v1"] = "foot-contact-v1"
State = Literal["contact", "no_contact", "unknown"]
Region = Literal["heel", "forefoot", "flat"]


class ContactConfig(StrictModel):
    units: Literal["metric", "body_normalized"] = "metric"
    morphology_measurement: str = "left_shank"
    max_morphology_relative_uncertainty: float = Field(default=0.1, gt=0, le=0.5)
    enter_height: float = Field(default=0.025, gt=0)
    leave_height: float = Field(default=0.05, gt=0)
    enter_vertical_speed: float = Field(default=0.15, gt=0)
    leave_vertical_speed: float = Field(default=0.3, gt=0)
    contact_seconds: float = Field(default=0.08, gt=0)
    no_contact_seconds: float = Field(default=0.04, gt=0)
    max_gap_seconds: float = Field(default=0.15, gt=0)
    uncertainty_multiplier: float = Field(default=2, ge=1)

    @model_validator(mode="after")
    def hysteresis_order(self) -> ContactConfig:
        if self.leave_height <= self.enter_height or (
            self.leave_vertical_speed <= self.enter_vertical_speed
        ):
            raise ValueError("leave thresholds must exceed enter thresholds")
        return self


class LandmarkEvidence(StrictModel):
    landmark: Landmark
    quality: Quality
    height: float | None = None
    height_uncertainty: float | None = Field(default=None, ge=0)
    vertical_speed: float | None = None
    speed_uncertainty: float | None = Field(default=None, ge=0)
    motion_interval: tuple[float, float] | None = None
    motion_source_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class FootEvidence(StrictModel):
    candidate: State
    reasons: list[str]
    landmarks: list[LandmarkEvidence]
    # Closed sample bracket, not a fabricated continuous-contact interval.
    contributing_times: list[float] = Field(default_factory=list)
    transition_bracket: tuple[float, float] | None = None


class ContactSeries(StrictModel):
    version: Literal[1] = 1
    artifact_role: Literal["physical_contact_evidence"] = "physical_contact_evidence"
    algorithm_revision: Literal["foot-contact-v1"] = REVISION
    reconstruction_id: str
    calibration_id: str
    morphology_id: str | None
    config: ContactConfig
    evidence_unit: Literal["m", "body_ratio", "unavailable"]
    normalization_length: float | None
    normalization_quality: Quality | None
    ground_evidence_ids: list[str]
    samples: list[GroundSample]
    left: list[FootEvidence]
    right: list[FootEvidence]

    @model_validator(mode="after")
    def aligned(self) -> ContactSeries:
        if len(self.samples) != len(self.left) or len(self.samples) != len(self.right):
            raise ValueError("contact diagnostics must align with samples")
        times = [s.global_seconds for s in self.samples]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("contact samples must have increasing time")
        return self


def _supported(point: Landmark3D | None) -> bool:
    return bool(
        point is not None
        and point.xyz_world is not None
        and point.quality.state != "unknown"
        and point.quality.uncertainty is not None
        and point.quality.source_ids
    )


def _normalization(
    reconstruction: Reconstruction,
    calibration: Calibration,
    morphology: Morphology | None,
    config: ContactConfig,
) -> tuple[float | None, Quality | None, str | None]:
    if calibration.ground_status != "resolved" or calibration.ground_frame is None:
        return None, None, "unresolved_ground"
    if calibration.quality.state == "unknown":
        return None, None, "unusable_calibration"
    if config.units == "metric":
        if calibration.scale_status != "resolved" or reconstruction.scale != "metric":
            return None, None, "unresolved_metric_scale"
        return 1, None, None
    if morphology is None:
        return None, None, "missing_morphology"
    measurements = [
        m for m in morphology.measurements if m.name == config.morphology_measurement
    ]
    if len(measurements) != 1:
        return None, None, "missing_or_ambiguous_normalization_length"
    measurement = measurements[0]
    q, value = measurement.quality, measurement.value
    expected_unit = "m" if reconstruction.scale == "metric" else "arbitrary"
    if (
        value is None
        or value <= 0
        or measurement.unit != expected_unit
        or q.state == "unknown"
        or not q.source_ids
        or q.uncertainty is None
        or q.uncertainty / value > config.max_morphology_relative_uncertainty
    ):
        return None, q, "unreliable_normalization_length"
    return value, q, None


def _point_evidence(
    name: Landmark,
    point: Landmark3D | None,
    previous: Landmark3D | None,
    time: float,
    previous_time: float | None,
    calibration: Calibration,
    length: float | None,
    length_quality: Quality | None,
    unavailable: str | None,
    config: ContactConfig,
) -> LandmarkEvidence:
    q = point.quality if point else Quality(state="unknown")
    result = LandmarkEvidence(landmark=name, quality=q)
    if unavailable:
        result.reasons.append(unavailable)
        return result
    if not _supported(point):
        result.reasons.append("missing_or_unusable_landmark")
        return result
    assert point is not None and point.xyz_world is not None and length is not None
    frame = calibration.ground_frame
    assert frame is not None
    relative = float(length_quality.uncertainty or 0) / length if length_quality else 0

    def height(p: Landmark3D) -> tuple[float, float]:
        assert p.xyz_world is not None
        x, y, z = p.xyz_world
        # Ground has already been aligned to world z=0. Do not transform twice.
        sigma = (
            float(p.quality.uncertainty or 0)
            + frame.rms_residual
            + math.hypot(x, y)
            * math.sin(min(frame.normal_uncertainty_rad, math.pi / 2))
        )
        # Conservative denominator margin, including uncertain body normalization.
        return z / length, (sigma / length + abs(z / length) * relative) / (
            1 - relative
        )

    result.height, result.height_uncertainty = height(point)
    if previous_time is None or not _supported(previous):
        result.reasons.append("motion_unavailable")
    elif time - previous_time > config.max_gap_seconds + 1e-12:
        result.reasons.append("temporal_gap")
    elif not (set(q.source_ids) - set(cast(Landmark3D, previous).quality.source_ids)):
        result.reasons.append("no_new_native_evidence")
    else:
        assert previous is not None
        old_height, old_sigma = height(previous)
        dt = time - previous_time
        result.vertical_speed = (result.height - old_height) / dt
        result.speed_uncertainty = (result.height_uncertainty + old_sigma) / dt
        result.motion_interval = (previous_time, time)
        result.motion_source_ids = sorted(
            set(q.source_ids + previous.quality.source_ids)
        )
    return result


def _candidate(
    evidence: list[LandmarkEvidence], stable: State, config: ContactConfig
) -> tuple[State, Region | None, str]:
    near: set[str] = set()
    raised: set[str] = set()
    for e in evidence:
        if e.height is None or e.height_uncertainty is None:
            continue
        margin = config.uncertainty_multiplier * e.height_uncertainty
        # Deep penetration is inconsistent geometry, not evidence of contact/flight.
        if e.height + margin < -config.enter_height:
            continue
        airborne_height = (
            config.enter_height if stable == "no_contact" else config.leave_height
        )
        if e.height - margin > airborne_height:
            raised.add(e.landmark.rsplit("_", 1)[-1])
        height_limit = (
            config.leave_height if stable == "contact" else config.enter_height
        )
        speed_limit = (
            config.leave_vertical_speed
            if stable == "contact"
            else config.enter_vertical_speed
        )
        if (
            abs(e.height) + margin <= height_limit
            and e.vertical_speed is not None
            and e.speed_uncertainty is not None
            and abs(e.vertical_speed)
            + config.uncertainty_multiplier * e.speed_uncertainty
            <= speed_limit
        ):
            near.add(e.landmark.rsplit("_", 1)[-1])
    if "heel" in near or "forefoot" in near:
        region: Region | None = None
        if {"heel", "forefoot", "outer"} <= near:
            region = "flat"
        elif "heel" in near and {"forefoot", "outer"} <= raised:
            region = "heel"
        elif {"forefoot", "outer"} <= near and "heel" in raised:
            region = "forefoot"
        return "contact", region, "ground_proximity_and_low_vertical_motion"
    if len(raised) == 3:
        return "no_contact", None, "all_foot_landmarks_above_ground"
    return "unknown", None, "insufficient_or_ambiguous_foot_evidence"


def _foot(
    reconstruction: Reconstruction,
    calibration: Calibration,
    side: Literal["left", "right"],
    length: float | None,
    length_quality: Quality | None,
    unavailable: str | None,
    config: ContactConfig,
) -> tuple[list[Contact], list[FootEvidence]]:
    contacts: list[Contact] = []
    diagnostics: list[FootEvidence] = []
    stable: State = "unknown"
    pending: State = "unknown"
    pending_times: list[float] = []
    previous: dict[Landmark, Landmark3D] = {}
    previous_time: float | None = None
    names = [cast(Landmark, f"{side}_{p}") for p in ("heel", "forefoot", "foot_outer")]
    for sample in reconstruction.samples:
        time = sample.global_seconds
        points = {p.name: p for p in sample.landmarks}
        if previous_time is not None and (
            time - previous_time > config.max_gap_seconds + 1e-12
        ):
            stable, pending, pending_times = "unknown", "unknown", []
        evidence = [
            _point_evidence(
                name,
                points.get(name),
                previous.get(name),
                time,
                previous_time,
                calibration,
                length,
                length_quality,
                unavailable,
                config,
            )
            for name in names
        ]
        candidate, region, reason = _candidate(evidence, stable, config)
        reasons = sorted({reason} | {r for e in evidence for r in e.reasons})
        bracket = None
        contributing = [time]
        state = candidate
        if candidate == "unknown":
            stable, pending, pending_times = "unknown", "unknown", []
        elif candidate != stable:
            # Repeated interpolated evidence cannot satisfy elapsed-time dwell.
            if any("no_new_native_evidence" in e.reasons for e in evidence):
                pending, pending_times = "unknown", []
                state = "unknown"
            else:
                if pending != candidate:
                    pending, pending_times = candidate, []
                pending_times.append(time)
                contributing = pending_times.copy()
                dwell = (
                    config.contact_seconds
                    if candidate == "contact"
                    else config.no_contact_seconds
                )
                if time - pending_times[0] + 1e-12 >= dwell:
                    stable = candidate
                    # Transition onset is bounded by the last pre-candidate sample.
                    start_index = len(contacts) - len(pending_times) + 1
                    bracket = (
                        reconstruction.samples[max(0, start_index - 1)].global_seconds,
                        pending_times[0],
                    )
                    pending, pending_times = "unknown", []
                else:
                    state = "unknown"
                    reasons.append("temporal_confirmation_pending")
        else:
            pending, pending_times = "unknown", []
        # Keep the full confirmation window's lineage, not just its final pair.
        window = (
            [evidence]
            + [item.landmarks for item in diagnostics[-(len(contributing) - 1) :]]
            if len(contributing) > 1
            else [evidence]
        )
        sources = sorted(
            set(
                calibration.ground_frame.evidence_ids
                if calibration.ground_frame
                else []
            )
            | {
                s
                for entries in window
                for e in entries
                for s in e.quality.source_ids + e.motion_source_ids
            }
            | set(length_quality.source_ids if length_quality else [])
        )
        sigmas = [
            e.height_uncertainty
            for entries in window
            for e in entries
            if e.height_uncertainty is not None
        ]
        sigma = max(sigmas) if sigmas else None
        contacts.append(
            Contact(
                state=state,
                region=region if state == "contact" else None,
                quality=Quality(
                    state="unknown" if state == "unknown" else "inferred",
                    uncertainty=sigma,
                    source_ids=sources,
                ),
            )
        )
        diagnostics.append(
            FootEvidence(
                candidate=candidate,
                reasons=reasons,
                landmarks=evidence,
                contributing_times=contributing,
                transition_bracket=bracket,
            )
        )
        previous, previous_time = points, time
    return contacts, diagnostics


def derive_contacts(
    reconstruction: Reconstruction,
    calibration: Calibration,
    morphology: Morphology | None = None,
    config: ContactConfig | None = None,
) -> ContactSeries:
    """Estimate on supplied final samples; never resample, fill or mutate motion."""
    config = ContactConfig.model_validate((config or ContactConfig()).model_dump())
    if reconstruction.calibration_id != calibration.id or (
        reconstruction.scale != calibration.scale
    ):
        raise ValueError("reconstruction calibration identity or scale mismatch")
    if morphology and morphology.participant_id != reconstruction.participant_id:
        raise ValueError("morphology participant mismatch")
    # Revalidate mutable input models, including finite times/coordinates and ordering.
    reconstruction = Reconstruction.model_validate(reconstruction.model_dump())
    calibration = Calibration.model_validate(calibration.model_dump())
    if morphology:
        morphology = Morphology.model_validate(morphology.model_dump())
    length, length_quality, unavailable = _normalization(
        reconstruction, calibration, morphology, config
    )
    left, left_evidence = _foot(
        reconstruction, calibration, "left", length, length_quality, unavailable, config
    )
    right, right_evidence = _foot(
        reconstruction,
        calibration,
        "right",
        length,
        length_quality,
        unavailable,
        config,
    )
    support = {
        ("contact", "contact"): "both",
        ("contact", "no_contact"): "left",
        ("no_contact", "contact"): "right",
        ("no_contact", "no_contact"): "neither",
    }
    return ContactSeries(
        reconstruction_id=reconstruction.id,
        calibration_id=calibration.id,
        morphology_id=morphology.id if morphology else None,
        config=config,
        evidence_unit=(
            "unavailable"
            if unavailable
            else "m"
            if config.units == "metric"
            else "body_ratio"
        ),
        normalization_length=length,
        normalization_quality=length_quality,
        ground_evidence_ids=(
            calibration.ground_frame.evidence_ids if calibration.ground_frame else []
        ),
        samples=[
            GroundSample(
                global_seconds=s.global_seconds,
                left=left_contact,
                right=right_contact,
                support=cast(  # validated by the canonical GroundSample contract
                    Literal["both", "left", "right", "neither", "unknown"],
                    support.get((left_contact.state, right_contact.state), "unknown"),
                ),
            )
            for s, left_contact, right_contact in zip(
                reconstruction.samples, left, right, strict=True
            )
        ],
        left=left_evidence,
        right=right_evidence,
    )
