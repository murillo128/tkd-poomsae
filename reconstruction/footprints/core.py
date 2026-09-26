"""Physical placements from native motion and existing contact evidence only."""

from __future__ import annotations

import math
from typing import Literal, cast

from pydantic import Field, model_validator

from contracts.models import (
    Calibration,
    Contact,
    Footprint,
    Ground,
    Interval,
    Landmark3D,
    Measurement,
    Morphology,
    Quality,
    Reconstruction,
    StrictModel,
)
from storage import hash_config

REVISION: Literal["footprint-v1"] = "footprint-v1"
Side = Literal["left", "right"]
XY = tuple[float, float]
Status = Literal["stable", "moving", "unavailable", "no_contact"]
SIDES: tuple[Side, Side] = ("left", "right")


class FootprintConfig(StrictModel):
    threshold_units: Literal["world", "body_normalized"] = "world"
    morphology_measurement: str = "left_shank"
    max_morphology_relative_uncertainty: float = Field(default=0.1, gt=0, le=0.5)
    enter_distance: float = Field(default=0.015, gt=0)
    leave_distance: float = Field(default=0.03, gt=0)
    enter_angle_rad: float = Field(default=0.06, gt=0, lt=math.pi)
    leave_angle_rad: float = Field(default=0.12, gt=0, lt=math.pi)
    stable_seconds: float = Field(default=0.12, gt=0)
    max_gap_seconds: float = Field(default=0.15, gt=0)
    uncertainty_multiplier: float = Field(default=2, ge=1)

    @model_validator(mode="after")
    def hysteresis(self) -> FootprintConfig:
        if self.leave_distance <= self.enter_distance or (
            self.leave_angle_rad <= self.enter_angle_rad
        ):
            raise ValueError("leave thresholds must exceed enter thresholds")
        return self


class RenderingGeometry(StrictModel):
    # Landmark axes never imply a measured sole boundary or width.
    kind: Literal["axis", "points", "unavailable"]
    axis_start: XY | None = None
    axis_end: XY | None = None
    supported_points: list[XY] = Field(default_factory=list)
    polygon: list[XY] | None = None
    approximated: bool = False

    @model_validator(mode="after")
    def supported_geometry(self) -> RenderingGeometry:
        if self.kind == "axis" and (self.axis_start is None or self.axis_end is None):
            raise ValueError("axis geometry requires evidenced endpoints")
        if self.polygon is not None and not self.approximated:
            raise ValueError("landmarks do not constrain a precise sole polygon")
        return self


class TrajectorySample(StrictModel):
    foot: Side
    motion_sample_index: int
    contact_sample_index: int
    global_seconds: float
    contact: Contact
    support: Literal["both", "left", "right", "neither", "unknown"]
    landmarks: list[Landmark3D]
    xy_ground: XY | None
    yaw_rad: float | None
    geometry: RenderingGeometry
    position_uncertainty: float | None
    angle_uncertainty: float | None
    status: Status
    event_id: str | None = None
    reasons: list[str]


class PlacementEvent(StrictModel):
    footprint: Footprint
    kind: Literal["stable", "moving"]
    event_seconds: float
    confirmed_seconds: float | None
    # Indices into FootprintSeries.trajectory, carrying exact native source links.
    trajectory_indices: list[int]
    geometry: RenderingGeometry
    reasons: list[str]


class PlacementRelation(StrictModel):
    first_id: str
    second_id: str
    reference_frame: Literal["first_foot_axis"] = "first_foot_axis"
    origin_xy: XY
    longitudinal_axis_xy: XY
    lateral_axis_xy: XY
    measurements: list[Measurement]
    relative_alignment_rad: float | None
    relative_foot_angle_rad: float | None
    quality: Quality


class FootprintSeries(StrictModel):
    version: Literal[1] = 1
    artifact_role: Literal["physical_placements"] = "physical_placements"
    algorithm_revision: Literal["footprint-v1"] = REVISION
    reconstruction_id: str
    contact_id: str
    calibration_id: str
    morphology_id: str | None
    world_unit: Literal["m", "arbitrary"]
    config: FootprintConfig
    normalization: Measurement | None
    events: list[PlacementEvent]
    relations: list[PlacementRelation]
    trajectory: list[TrajectorySample]

    @model_validator(mode="after")
    def source_links(self) -> FootprintSeries:
        events = {e.footprint.id: e for e in self.events}
        if len(events) != len(self.events):
            raise ValueError("placement IDs must be unique")
        linked: set[int] = set()
        for event in self.events:
            indices = event.trajectory_indices
            if len(indices) < 2 or indices != sorted(set(indices)):
                raise ValueError("ordered nonzero placement trajectory required")
            if any(i < 0 or i >= len(self.trajectory) or i in linked for i in indices):
                raise ValueError("invalid or duplicate placement trajectory link")
            rows = [self.trajectory[i] for i in indices]
            if any(
                row.event_id != event.footprint.id
                or row.status != event.kind
                or row.foot != event.footprint.foot
                for row in rows
            ):
                raise ValueError("placement trajectory identity disagrees")
            interval = event.footprint.interval
            if (interval.start, interval.end) != (
                rows[0].global_seconds,
                rows[-1].global_seconds,
            ) or event.event_seconds != interval.start:
                raise ValueError("placement interval disagrees with native trajectory")
            if event.kind == "stable":
                if (
                    event.confirmed_seconds is None
                    or not (
                        interval.start + self.config.stable_seconds - 1e-12
                        <= event.confirmed_seconds
                        <= interval.end
                    )
                    or event.footprint.xy_ground is None
                    or event.footprint.yaw_rad is None
                ):
                    raise ValueError("stable placement requires confirmed geometry")
            elif (
                event.footprint.xy_ground is not None
                or event.footprint.yaw_rad is not None
            ):
                raise ValueError("moving contact must not freeze foot geometry")
            linked.update(indices)
        for index, row in enumerate(self.trajectory):
            if row.event_id is not None and index not in linked:
                raise ValueError("trajectory refers to an unavailable placement")
        for relation in self.relations:
            if any(
                identity not in events or events[identity].kind != "stable"
                for identity in (relation.first_id, relation.second_id)
            ):
                raise ValueError("relation requires stable placement endpoints")
            if (
                any(m.unit == "m" for m in relation.measurements)
                and self.world_unit != "m"
            ):
                raise ValueError("arbitrary scale cannot emit metric placement values")
        return self


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _length(
    source: Reconstruction, morphology: Morphology | None, config: FootprintConfig
) -> Measurement | None:
    matches = (
        [m for m in morphology.measurements if m.name == config.morphology_measurement]
        if morphology
        else []
    )
    if len(matches) != 1:
        return None
    m = matches[0]
    if (
        m.value is None
        or m.value <= 0
        or m.unit != ("m" if source.scale == "metric" else "arbitrary")
        or m.quality.state == "unknown"
        or not m.quality.source_ids
        or m.quality.uncertainty is None
        or m.quality.uncertainty / m.value > config.max_morphology_relative_uncertainty
    ):
        return None
    return m


def _sample(
    source: Reconstruction, ground: Ground, index: int, side: Side
) -> TrajectorySample:
    motion, contact_sample = source.samples[index], ground.samples[index]
    contact = contact_sample.left if side == "left" else contact_sample.right
    names = [f"{side}_{part}" for part in ("heel", "forefoot", "foot_outer")]
    landmarks = [p for p in motion.landmarks if p.name in names]
    usable: dict[str, Landmark3D] = {
        p.name: p
        for p in landmarks
        if p.xyz_world is not None
        and p.quality.state != "unknown"
        and p.quality.source_ids
        and p.quality.uncertainty is not None
    }
    heel, toe = usable.get(names[0]), usable.get(names[1])
    points = [p.xyz_world[:2] for p in usable.values() if p.xyz_world]
    center, yaw, sigma, angle_sigma = None, None, None, None
    geometry = RenderingGeometry(
        kind="points" if points else "unavailable", supported_points=points
    )
    reasons = []
    if heel and toe:
        assert heel.xyz_world and toe.xyz_world
        start, end = heel.xyz_world[:2], toe.xyz_world[:2]
        center = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        sigma = max(
            float(heel.quality.uncertainty or 0), float(toe.quality.uncertainty or 0)
        )
        length = math.dist(start, end)
        if length > 2 * sigma and length > 1e-12:
            yaw = math.atan2(end[1] - start[1], end[0] - start[0])
            angle_sigma = math.asin(min(1, 2 * sigma / length))
            geometry.kind, geometry.axis_start, geometry.axis_end = "axis", start, end
        else:
            reasons.append("degenerate_or_uncertain_axis")
    else:
        reasons.append("unavailable_foot_axis")
    if len(usable) < 3:
        reasons.append("partial_landmark_geometry")
    status: Status
    if contact.state != "contact":
        status = "no_contact" if contact.state == "no_contact" else "unavailable"
        reasons.append(contact.state)
    elif (
        yaw is None
        or contact.quality.state == "unknown"
        or (not contact.quality.source_ids or contact.quality.uncertainty is None)
    ):
        status = "unavailable"
        reasons.append("unusable_placement_evidence")
    else:
        status = "moving"  # Stability must be confirmed by elapsed native evidence.
    return TrajectorySample(
        foot=side,
        motion_sample_index=index,
        contact_sample_index=index,
        global_seconds=motion.global_seconds,
        contact=contact,
        support=contact_sample.support,
        landmarks=landmarks,
        xy_ground=center,
        yaw_rad=yaw,
        geometry=geometry,
        position_uncertainty=sigma,
        angle_uncertainty=angle_sigma,
        status=status,
        reasons=reasons,
    )


def _near(
    sample: TrajectorySample,
    anchor: TrajectorySample,
    distance: float,
    angle: float,
    divisor: float,
    config: FootprintConfig,
) -> bool:
    assert sample.yaw_rad is not None and anchor.yaw_rad is not None
    # Compare every shared landmark: a foot pivot/slide cannot hide in its center.
    a = {p.name: p for p in anchor.landmarks if p.xyz_world is not None}
    b = {p.name: p for p in sample.landmarks if p.xyz_world is not None}
    for name in a.keys() & b.keys():
        first, second = a[name], b[name]
        if first.quality.state == "unknown" or second.quality.state == "unknown":
            continue
        assert first.xyz_world and second.xyz_world
        margin = config.uncertainty_multiplier * (
            float(first.quality.uncertainty or 0)
            + float(second.quality.uncertainty or 0)
        )
        if (
            math.dist(first.xyz_world[:2], second.xyz_world[:2]) + margin
        ) / divisor > distance:
            return False
    return (
        abs(_wrap(sample.yaw_rad - anchor.yaw_rad))
        + (
            config.uncertainty_multiplier
            * (
                float(sample.angle_uncertainty or 0)
                + float(anchor.angle_uncertainty or 0)
            )
        )
        <= angle
    )


def _quality(samples: list[TrajectorySample], moving: bool = False) -> Quality:
    sources = sorted(
        {
            s
            for row in samples
            for q in [row.contact.quality] + [p.quality for p in row.landmarks]
            for s in q.source_ids
        }
    )
    uncertainty = max(float(row.position_uncertainty or 0) for row in samples)
    return Quality(
        state="inferred",
        uncertainty=None if moving else uncertainty,
        source_ids=sources,
    )


def _events(
    rows: list[TrajectorySample],
    indices: list[int],
    source_id: str,
    divisor: float | None,
    config: FootprintConfig,
) -> list[PlacementEvent]:
    events: list[PlacementEvent] = []
    candidate: list[int] = []
    stable: list[int] = []
    moving: list[int] = []
    previous: TrajectorySample | None = None
    seen_sources: set[str] = set()

    def emit(group: list[int], kind: Literal["stable", "moving"]) -> None:
        if not group:
            return
        samples = [rows[i] for i in group]
        # A lone native query does not constrain a nonzero placement interval.
        if len(samples) < 2:
            samples[0].reasons.append("isolated_sample")
            return
        first, last = samples[0], samples[-1]
        event_id = "footprint:" + hash_config(
            {
                "motion": source_id,
                "foot": first.foot,
                "kind": kind,
                "start": first.global_seconds,
                "end": last.global_seconds,
                "revision": REVISION,
                "config": config.model_dump(mode="json"),
            }
        )
        # Stable geometry is an evidenced native pose; movement keeps all poses.
        quality = _quality(samples, moving=kind == "moving")
        if kind == "stable":
            assert first.xy_ground is not None
            quality.uncertainty = max(
                math.dist(first.xy_ground, cast(XY, row.xy_ground))
                + float(row.position_uncertainty or 0)
                for row in samples
            )
        events.append(
            PlacementEvent(
                footprint=Footprint(
                    id=event_id,
                    foot=first.foot,
                    interval=Interval(
                        start=first.global_seconds, end=last.global_seconds
                    ),
                    xy_ground=first.xy_ground if kind == "stable" else None,
                    yaw_rad=first.yaw_rad if kind == "stable" else None,
                    quality=quality,
                ),
                kind=kind,
                event_seconds=first.global_seconds,
                confirmed_seconds=next(
                    (
                        row.global_seconds
                        for row in samples
                        if row.global_seconds - first.global_seconds + 1e-12
                        >= config.stable_seconds
                    ),
                    None,
                )
                if kind == "stable"
                else None,
                trajectory_indices=group.copy(),
                geometry=first.geometry
                if kind == "stable"
                else RenderingGeometry(kind="unavailable"),
                reasons=[] if kind == "stable" else ["moving_or_unconfirmed_contact"],
            )
        )
        for row in samples:
            row.status, row.event_id = kind, event_id

    def flush() -> None:
        emit(stable, "stable")
        moving.extend(candidate)
        emit(moving, "moving")
        candidate.clear()
        stable.clear()
        moving.clear()
        seen_sources.clear()

    for index in indices:
        row = rows[index]
        sources = {s for p in row.landmarks for s in p.quality.source_ids}
        if (
            previous
            and row.global_seconds - previous.global_seconds
            > config.max_gap_seconds + 1e-12
        ):
            flush()
            row.reasons.append("temporal_gap")
        if row.status != "moving" or divisor is None:
            flush()
            if divisor is None and row.status == "moving":
                row.status = "unavailable"
                row.reasons.append("unavailable_threshold_normalization")
            previous = row
            continue
        axis_points = [
            p for p in row.landmarks if p.name.endswith(("_heel", "_forefoot"))
        ]
        if any(not (set(p.quality.source_ids) - seen_sources) for p in axis_points):
            flush()
            row.status = "unavailable"
            row.reasons.append("no_new_native_evidence")
            previous = row
            continue
        seen_sources.update(sources)
        if stable:
            if _near(
                row,
                rows[stable[0]],
                config.leave_distance,
                config.leave_angle_rad,
                divisor,
                config,
            ):
                stable.append(index)
                previous = row
                continue
            emit(stable, "stable")
            stable.clear()
        if candidate and not _near(
            row,
            rows[candidate[0]],
            config.enter_distance,
            config.enter_angle_rad,
            divisor,
            config,
        ):
            moving.extend(candidate)
            candidate.clear()
        # Self-check prevents high uncertainty from ever confirming a stable pose.
        if not _near(
            row, row, config.enter_distance, config.enter_angle_rad, divisor, config
        ):
            moving.extend(candidate)
            candidate.clear()
            moving.append(index)
            row.reasons.append("uncertain_stability")
        else:
            candidate.append(index)
            if (
                row.global_seconds - rows[candidate[0]].global_seconds + 1e-12
                >= config.stable_seconds
            ):
                emit(moving, "moving")
                moving.clear()
                stable.extend(candidate)
                candidate.clear()
        previous = row
    flush()
    return events


def measure_placements(
    first: Footprint,
    second: Footprint,
    world_unit: Literal["m", "arbitrary"],
    normalization: Measurement | None = None,
) -> PlacementRelation:
    """Signed separation in the first foot's axis, angles wrapped to [-pi, pi]."""
    if first.xy_ground is None or second.xy_ground is None or first.yaw_rad is None:
        raise ValueError("placed positions and first foot axis required")
    delta = (
        second.xy_ground[0] - first.xy_ground[0],
        second.xy_ground[1] - first.xy_ground[1],
    )
    c, s = math.cos(first.yaw_rad), math.sin(first.yaw_rad)
    values = {
        "distance": math.hypot(*delta),
        "longitudinal_separation": delta[0] * c + delta[1] * s,
        "lateral_separation": -delta[0] * s + delta[1] * c,
    }
    quality = Quality(
        state="inferred",
        uncertainty=(
            float(first.quality.uncertainty or 0)
            + float(second.quality.uncertainty or 0)
        ),
        source_ids=sorted(set(first.quality.source_ids + second.quality.source_ids)),
    )
    measurements = [
        Measurement(name=name, value=value, unit=world_unit, quality=quality)
        for name, value in values.items()
    ]
    if normalization is not None:
        length, sigma = normalization.value, normalization.quality.uncertainty
        if (
            length is None
            or length <= 0
            or normalization.unit != world_unit
            or (
                normalization.quality.state == "unknown"
                or sigma is None
                or sigma >= length
                or not normalization.quality.source_ids
            )
        ):
            raise ValueError("usable matching normalization length required")
        for name, value in values.items():
            q = quality.model_copy(deep=True)
            q.source_ids = sorted(set(q.source_ids + normalization.quality.source_ids))
            q.uncertainty = (
                float(quality.uncertainty or 0) + abs(value / length) * sigma
            ) / (length - sigma)
            measurements.append(
                Measurement(
                    name=name, value=value / length, unit="body_ratio", quality=q
                )
            )
    distance = values["distance"]
    return PlacementRelation(
        first_id=first.id,
        second_id=second.id,
        origin_xy=first.xy_ground,
        longitudinal_axis_xy=(c, s),
        lateral_axis_xy=(-s, c),
        measurements=measurements,
        relative_alignment_rad=_wrap(math.atan2(delta[1], delta[0]) - first.yaw_rad)
        if distance > 1e-12
        else None,
        relative_foot_angle_rad=_wrap(second.yaw_rad - first.yaw_rad)
        if second.yaw_rad is not None
        else None,
        quality=quality,
    )


def derive_footprints(
    reconstruction: Reconstruction,
    contacts: Ground,
    calibration: Calibration,
    morphology: Morphology | None = None,
    config: FootprintConfig | None = None,
) -> FootprintSeries:
    """Read aligned artifacts; never infer new contact or semantic steps."""
    source = Reconstruction.model_validate(reconstruction.model_dump())
    ground = Ground.model_validate(contacts.model_dump())
    cal = Calibration.model_validate(calibration.model_dump())
    config = FootprintConfig.model_validate((config or FootprintConfig()).model_dump())
    morphology = (
        Morphology.model_validate(morphology.model_dump()) if morphology else None
    )
    if source.provenance.producer != "reconstruction.temporal":
        raise ValueError("final temporal reconstruction required")
    if ground.reconstruction_id != source.id or ground.scale != source.scale:
        raise ValueError("contact reconstruction identity or scale mismatch")
    if source.calibration_id != cal.id or source.scale != cal.scale:
        raise ValueError("calibration identity or scale mismatch")
    if (
        cal.ground_status != "resolved"
        or cal.ground_frame is None
        or cal.quality.state == "unknown"
    ):
        raise ValueError("resolved usable ground required")
    if morphology and morphology.participant_id != source.participant_id:
        raise ValueError("morphology participant mismatch")
    if [s.global_seconds for s in source.samples] != [
        s.global_seconds for s in ground.samples
    ]:
        raise ValueError("contact and motion native times must match")
    normalization = _length(source, morphology, config)
    divisor = (
        1.0
        if config.threshold_units == "world"
        else (
            normalization.value - float(normalization.quality.uncertainty or 0)
            if normalization and normalization.value
            else None
        )
    )
    trajectory = [
        _sample(source, ground, i, side)
        for i in range(len(source.samples))
        for side in SIDES
    ]
    events = []
    for offset in (0, 1):
        events.extend(
            _events(
                trajectory,
                list(range(offset, len(trajectory), 2)),
                source.id,
                divisor,
                config,
            )
        )
    events.sort(key=lambda e: (e.event_seconds, e.footprint.foot, e.footprint.id))
    # Relevant local relations: previous placement on this side and latest other side.
    latest: dict[Side, PlacementEvent] = {}
    relations = []
    unit: Literal["m", "arbitrary"] = "m" if source.scale == "metric" else "arbitrary"
    for event in events:
        if event.kind != "stable":
            continue
        for previous in sorted(latest.values(), key=lambda e: e.footprint.foot):
            relations.append(
                measure_placements(
                    previous.footprint, event.footprint, unit, normalization
                )
            )
        latest[event.footprint.foot] = event
    return FootprintSeries(
        reconstruction_id=source.id,
        contact_id=ground.id,
        calibration_id=cal.id,
        morphology_id=morphology.id if morphology else None,
        world_unit=unit,
        config=config,
        normalization=normalization,
        events=events,
        relations=relations,
        trajectory=trajectory,
    )
