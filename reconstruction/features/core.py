"""Offline physical measurements and heuristic candidates; no technique labels."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from pydantic import Field, model_validator
from scipy.signal import find_peaks  # type: ignore[import-untyped]

from contracts.models import (
    Contact,
    Footprint,
    Ground,
    Landmark,
    Landmark3D,
    MotionSample,
    Pivot,
    Quality,
    Quaternion,
    Reconstruction,
    StrictModel,
    Track,
)
from reconstruction.detailed import DetailedSample, GeometryConfig, derive_sample
from reconstruction.detailed.core import Quantity, Relation
from reconstruction.footprints import FootprintSeries
from reconstruction.temporal.core import (
    Derivative,
    TemporalConfig,
    Vector,
    angular_derivatives,
    derivatives,
    quaternion,
    rotation,
    vector,
)

REVISION: Literal["motion-features-v2"] = "motion-features-v2"
TRACKS: tuple[Track, ...] = (
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "body_root",
    "head",
)
EventKind = Literal[
    "motion_onset",
    "preparation_candidate",
    "extension_start",
    "extension_end",
    "extension_maximum",
    "extension_minimum",
    "direction_change",
    "arm_crossing",
    "lift_off",
    "first_contact",
    "stable_placement",
    "pivot_start",
    "pivot_end",
]


class FeatureConfig(StrictModel):
    # World thresholds use this explicit unit, never silently treated as metres.
    world_unit: Literal["m", "arbitrary"] = "m"
    max_position_uncertainty: float = Field(default=0.05, gt=0)
    max_angle_uncertainty_rad: float = Field(default=0.25, gt=0, le=1)
    enter_angular_speed_rad_s: float = Field(default=0.3, gt=0)
    leave_angular_speed_rad_s: float = Field(default=0.1, gt=0)
    min_angular_excursion_rad: float = Field(default=0.12, gt=0)
    enter_speed: float = Field(default=0.15, gt=0)
    leave_speed: float = Field(default=0.05, gt=0)
    min_excursion: float = Field(default=0.025, gt=0)
    extension_prominence_ratio: float = Field(default=0.08, gt=0, lt=1)
    min_duration_seconds: float = Field(default=0.04, gt=0)
    refractory_seconds: float = Field(default=0.08, ge=0)
    max_gap_seconds: float = Field(default=0.15, gt=0)
    derivative_window_seconds: float = Field(default=0.12, gt=0)
    uncertainty_multiplier: float = Field(default=3, ge=1)

    @model_validator(mode="after")
    def hysteresis(self) -> FeatureConfig:
        if self.enter_speed <= self.leave_speed or (
            self.enter_angular_speed_rad_s <= self.leave_angular_speed_rad_s
        ):
            raise ValueError("enter speed must exceed leave speed")
        return self


class Position(StrictModel):
    value: Vector | None = None
    quality: Quality = Field(default_factory=lambda: Quality(state="unknown"))

    @model_validator(mode="after")
    def available(self) -> Position:
        if (self.value is None) != (self.quality.state == "unknown"):
            raise ValueError("position and quality availability must agree")
        return self


class Orientation(StrictModel):
    name: str
    parent: Literal["world", "body"]
    orientation: Quaternion | None = None
    quality: Quality = Field(default_factory=lambda: Quality(state="unknown"))


class FeatureSample(StrictModel):
    track: Track
    motion_sample_index: int = Field(ge=0)
    global_seconds: float
    position_world: Position
    position_local: Position
    # Arms/legs: distal frame in torso; head: head in torso; root: world.
    orientation: Orientation
    extension: Quantity  # endpoint distance / two-segment chain length
    linear: Derivative
    angular: Derivative
    relations: list[Relation]
    relation_evidence_quality: Quality
    contact: Contact | None = None
    ground_sample_index: int | None = None


class EventCandidate(StrictModel):
    id: str
    track: Track
    kind: EventKind
    global_seconds: float
    evidence_start_seconds: float
    evidence_end_seconds: float
    duration_seconds: float = Field(ge=0)
    tolerance_seconds: float = Field(ge=0)
    motion_sample_indices: list[int]
    ground_sample_indices: list[int] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    quality: Quality
    reasons: list[str]


class FeatureSeries(StrictModel):
    version: Literal[1] = 1
    artifact_role: Literal["physical_motion_features"] = "physical_motion_features"
    algorithm_revision: Literal["motion-features-v1", "motion-features-v2"] = REVISION
    reconstruction_id: str
    ground_id: str
    config: FeatureConfig
    trajectory: list[FeatureSample]
    events: list[EventCandidate]

    @model_validator(mode="after")
    def links(self) -> FeatureSeries:
        count = len(self.trajectory) // len(TRACKS)
        if not count or len(self.trajectory) != count * len(TRACKS):
            raise ValueError("complete nonempty six-track trajectory required")
        times = [self.trajectory[i * len(TRACKS)].global_seconds for i in range(count)]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("increasing shared global time required")
        for i, row in enumerate(self.trajectory):
            if (row.track, row.motion_sample_index, row.global_seconds) != (
                TRACKS[i % len(TRACKS)],
                i // len(TRACKS),
                times[i // len(TRACKS)],
            ):
                raise ValueError("trajectory source links disagree")
        if len({e.id for e in self.events}) != len(
            self.events
        ) or self.events != sorted(
            self.events, key=lambda e: (e.global_seconds, e.track, e.kind)
        ):
            raise ValueError("unique ordered event candidates required")
        for event in self.events:
            indices = event.motion_sample_indices
            if (
                not indices
                or indices != sorted(set(indices))
                or any(i < 0 or i >= count for i in indices)
            ):
                raise ValueError("invalid event motion links")
            if (
                event.evidence_start_seconds != times[indices[0]]
                or event.evidence_end_seconds != times[indices[-1]]
                or event.global_seconds not in [times[i] for i in indices]
                or event.duration_seconds != times[indices[-1]] - times[indices[0]]
                or event.quality.state in ("unknown", "interpolated")
                or not event.quality.source_ids
            ):
                raise ValueError("event timing or evidence quality disagrees")
        return self


def usable(q: Quality, limit: float) -> bool:
    return (
        q.state != "unknown"
        and bool(q.source_ids)
        and (q.uncertainty is not None and q.uncertainty <= limit)
    )


def combine(qs: list[Quality], uncertainty: float) -> Quality:
    if any(q.state == "unknown" for q in qs):
        return Quality(
            state="unknown", source_ids=sorted({s for q in qs for s in q.source_ids})
        )
    return Quality(
        state="interpolated"
        if any(q.state == "interpolated" for q in qs)
        else "inferred",
        uncertainty=uncertainty,
        source_ids=sorted({s for q in qs for s in q.source_ids}),
    )


BODY_POINTS: tuple[Landmark, ...] = (
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
)


def _native_quality(
    derived: Quality, points: dict[Landmark, Landmark3D], names: tuple[Landmark, ...]
) -> Quality:
    """Preserve the native evidence state when a geometry helper infers a frame."""
    native = [points.get(n) for n in names]
    qs = [derived] + [p.quality for p in native if p is not None]
    if derived.uncertainty is None or any(
        p is None
        or p.xyz_world is None
        or p.quality.uncertainty is None
        or not p.quality.source_ids
        for p in native
    ):
        qs.append(Quality(state="unknown"))
    return combine(qs, float(derived.uncertainty or 0))


def _qualified_geometry(detail: DetailedSample, sample: MotionSample) -> DetailedSample:
    # Persisted detail and native inputs remain immutable. Derived frames carry
    # numeric uncertainty, but their producer can erase interpolation state.
    output = detail.model_copy(deep=True)
    points = {p.name: p for p in sample.landmarks}
    frames = [
        (output.body_frame, BODY_POINTS),
        (output.head.frame, ("left_ear", "right_ear", "nose")),
    ]
    for side in ("left", "right"):
        frames.extend(
            [
                (
                    output.hands[side].frame,
                    tuple(
                        cast(Landmark, f"{side}_{p}")
                        for p in ("wrist", "index_mcp", "pinky_mcp")
                    ),
                ),
                (
                    output.feet[side].frame,
                    tuple(
                        cast(Landmark, f"{side}_{p}")
                        for p in ("heel", "forefoot", "foot_outer")
                    ),
                ),
            ]
        )
    for frame, names in frames:
        frame.quality = _native_quality(frame.quality, points, names)
        if frame.quality.state == "unknown":
            frame.orientation = None
    output.head.orientation_quality = combine(
        [
            output.head.orientation_quality,
            output.head.frame.quality,
            output.body_frame.quality,
        ],
        float(output.head.orientation_quality.uncertainty or 0),
    )
    if output.head.orientation_quality.state == "unknown":
        output.head.orientation_in_torso = None
    for relation in output.relations:
        names = (
            tuple(
                cast(Landmark, n)
                for n in (
                    ("left_elbow", "left_wrist", "right_elbow", "right_wrist")
                    if relation.axis == "crossing"
                    else (relation.subject, relation.object)
                )
            )
            + BODY_POINTS
        )
        relation.quality = _native_quality(relation.quality, points, names)
        relation.front_quality = _native_quality(relation.front_quality, points, names)
        if relation.quality.state == "unknown":
            relation.value = "unknown"
        if relation.front_quality.state == "unknown":
            relation.front_entity = None
    return output


def _derivative_quality(
    derivative: Derivative, qualities: dict[float, Quality]
) -> None:
    # Temporal estimators retain supporting times but label their outputs inferred.
    # An interpolated sample anywhere in the estimation window remains evidence.
    qs = [qualities[t] for t in derivative.support_times]
    if derivative.velocity is not None:
        derivative.velocity_quality = combine(
            [derivative.velocity_quality, *qs],
            float(derivative.velocity_quality.uncertainty or 0),
        )
    if derivative.acceleration is not None:
        derivative.acceleration_quality = combine(
            [derivative.acceleration_quality, *qs],
            float(derivative.acceleration_quality.uncertainty or 0),
        )


def _runs(times: list[float], valid: list[bool], gap: float) -> list[list[int]]:
    runs: list[list[int]] = []
    for i, known in enumerate(valid):
        if known:
            if i == 0 or not valid[i - 1] or times[i] - times[i - 1] > gap:
                runs.append([])
            runs[-1].append(i)
    return runs


def derive_features(
    motion: Reconstruction,
    ground: Ground,
    config: FeatureConfig | None = None,
    *,
    geometry: list[DetailedSample] | None = None,
    root_position_quality: list[Quality] | None = None,
    root_orientation_quality: list[Quality] | None = None,
    root_orientation_from_body: list[bool] | None = None,
    placements: FootprintSeries | None = None,
) -> FeatureSeries:
    """Inspect the full persisted sequence; never fit thresholds or invoke vision."""
    config = config or FeatureConfig()
    unit = "m" if motion.scale == "metric" else "arbitrary"
    times = [s.global_seconds for s in motion.samples]
    if (
        not times
        or ground.reconstruction_id != motion.id
        or (ground.scale != motion.scale or config.world_unit != unit)
    ):
        raise ValueError(
            "nonempty matching motion, ground and threshold units required"
        )
    details = (
        geometry
        if geometry is not None
        else [
            derive_sample(s, reconstruction_id=motion.id, representation="regularized")
            for s in motion.samples
        ]
    )
    if len(details) != len(times) or any(
        d.global_seconds != t or d.reconstruction_id != motion.id
        for d, t in zip(details, times, strict=True)
    ):
        raise ValueError("geometry must refer to exact reconstructed samples")
    for qualities in (root_position_quality, root_orientation_quality):
        if qualities is not None and len(qualities) != len(times):
            raise ValueError("root quality must cover exact samples")
    if root_orientation_from_body is not None and len(
        root_orientation_from_body
    ) != len(times):
        raise ValueError("root orientation origin must cover exact samples")
    if placements is not None and (
        placements.reconstruction_id != motion.id
        or [e.footprint for e in placements.events] != ground.footprints
    ):
        raise ValueError("placement evidence must match canonical ground")
    ground_times = {s.global_seconds: i for i, s in enumerate(ground.samples)}
    channels: dict[Track, list[FeatureSample]] = {track: [] for track in TRACKS}
    for i, (sample, detail) in enumerate(zip(motion.samples, details, strict=True)):
        detail = _qualified_geometry(detail, sample)
        points = {p.name: p for p in sample.landmarks}
        body = detail.body_frame
        for track in TRACKS:
            world, local = Position(), Position()
            extension = Quantity(quality=Quality(state="unknown"))
            orientation = Orientation(
                name=track, parent="world" if track == "body_root" else "body"
            )
            names: list[Landmark] = []
            if track.endswith(("arm", "leg")):
                side, limb = track.split("_")
                parts = (
                    ("shoulder", "elbow", "wrist")
                    if limb == "arm"
                    else ("hip", "knee", "ankle")
                )
                names = [cast(Landmark, f"{side}_{p}") for p in parts]
                distal = points.get(names[-1])
            else:
                distal = points.get("pelvis" if track == "body_root" else "head")
            if (
                distal
                and distal.xyz_world is not None
                and usable(distal.quality, config.max_position_uncertainty)
            ):
                world = Position(value=distal.xyz_world, quality=distal.quality)
            if track == "body_root" and root_position_quality is not None:
                q = root_position_quality[i]
                world = (
                    Position(value=sample.root_xyz_world, quality=q)
                    if (
                        sample.root_xyz_world is not None
                        and usable(q, config.max_position_uncertainty)
                    )
                    else Position()
                )
            elif track == "body_root" and world.value != sample.root_xyz_world:
                world = Position()
            if track == "body_root":
                local = world.model_copy(deep=True)
                q = (
                    root_orientation_quality[i]
                    if root_orientation_quality
                    else (body.quality)
                )
                quat = (
                    sample.root_orientation
                    if root_orientation_quality
                    else body.orientation
                )
                if (
                    root_orientation_from_body is not None
                    and root_orientation_from_body[i]
                ):
                    q = _native_quality(q, points, BODY_POINTS)
                if quat and usable(q, config.max_angle_uncertainty_rad):
                    orientation = Orientation(
                        name="root", parent="world", orientation=quat, quality=q
                    )
            else:
                if (
                    world.value
                    and body.origin_world
                    and body.orientation
                    and usable(body.quality, config.max_angle_uncertainty_rad)
                ):
                    delta = np.array(world.value) - body.origin_world
                    sigma = float(world.quality.uncertainty or 0) + float(
                        np.linalg.norm(delta)
                    ) * float(body.quality.uncertainty or 0)
                    sigma += sum(
                        float(points[n].quality.uncertainty or 0)
                        for n in ("left_hip", "right_hip")
                        if n in points
                    )
                    q = combine([world.quality, body.quality], sigma)
                    if usable(q, config.max_position_uncertainty):
                        local = Position(
                            value=vector(rotation(body.orientation).inv().apply(delta)),
                            quality=q,
                        )
                if track == "head":
                    q, quat = (
                        detail.head.orientation_quality,
                        detail.head.orientation_in_torso,
                    )
                else:
                    side, limb = track.split("_")
                    distal_frame = (
                        detail.hands[side].frame
                        if limb == "arm"
                        else (detail.feet[side].frame)
                    )
                    q, quat = distal_frame.quality, None
                    if body.orientation and distal_frame.orientation:
                        quat = quaternion(
                            rotation(body.orientation).inv()
                            * rotation(distal_frame.orientation)
                        )
                        q = combine(
                            [q, body.quality],
                            float(q.uncertainty or 0)
                            + float(body.quality.uncertainty or 0),
                        )
                if quat and usable(q, config.max_angle_uncertainty_rad):
                    orientation = Orientation(
                        name=track, parent="body", orientation=quat, quality=q
                    )
                if names and all(
                    n in points
                    and points[n].xyz_world is not None
                    and usable(points[n].quality, config.max_position_uncertainty)
                    for n in names
                ):
                    a, b, c = [np.array(points[n].xyz_world) for n in names]
                    length = float(np.linalg.norm(b - a) + np.linalg.norm(c - b))
                    sigma = sum(
                        float(points[n].quality.uncertainty or 0) for n in names
                    )
                    if length > config.uncertainty_multiplier * sigma:
                        extension = Quantity(
                            value=float(np.linalg.norm(c - a)) / length,
                            quality=combine(
                                [points[n].quality for n in names], 4 * sigma / length
                            ),
                        )
            gi = ground_times.get(sample.global_seconds)
            contact = None
            if gi is not None and track in ("left_leg", "right_leg"):
                gs = ground.samples[gi]
                contact = gs.left if track == "left_leg" else gs.right
            crossing_points = [
                points.get(cast(Landmark, f"{side}_{part}"))
                for side in ("left", "right")
                for part in ("elbow", "wrist")
            ]
            relation_quality = Quality(state="unknown")
            if (
                all(
                    p is not None
                    and p.xyz_world is not None
                    and usable(p.quality, config.max_position_uncertainty)
                    for p in crossing_points
                )
                and body.orientation is not None
            ):
                qs = [p.quality for p in crossing_points if p is not None]
                relation_quality = combine(
                    qs + [body.quality], sum(float(q.uncertainty or 0) for q in qs)
                )
            channels[track].append(
                FeatureSample(
                    track=track,
                    motion_sample_index=i,
                    global_seconds=sample.global_seconds,
                    position_world=world,
                    position_local=local,
                    orientation=orientation,
                    extension=extension,
                    linear=Derivative(unit=unit),
                    angular=Derivative(unit="rad"),
                    relations=detail.relations,
                    relation_evidence_quality=relation_quality,
                    contact=contact,
                    ground_sample_index=gi,
                )
            )
    derivative_config = TemporalConfig(
        derivative_window_seconds=config.derivative_window_seconds,
        max_interval_seconds=config.max_gap_seconds,
        short_gap_seconds=0,
        max_position_uncertainty=config.max_position_uncertainty,
        geometry=GeometryConfig(
            max_angle_uncertainty=min(1, config.max_angle_uncertainty_rad)
        ),
    )
    events: list[EventCandidate] = []
    for track, rows in channels.items():
        linear = derivatives(
            times,
            [r.position_local.value for r in rows],
            [r.position_local.quality for r in rows],
            derivative_config,
            unit,
            parent="world" if track == "body_root" else "body",
        )
        angular = angular_derivatives(
            times,
            [r.orientation.orientation for r in rows],
            [r.orientation.quality for r in rows],
            derivative_config,
            "world" if track == "body_root" else "body",
        )
        for row, lin, ang in zip(rows, linear, angular, strict=True):
            row.linear, row.angular = lin, ang
        linear_qualities = {r.global_seconds: r.position_local.quality for r in rows}
        angular_qualities = {r.global_seconds: r.orientation.quality for r in rows}
        for row in rows:
            _derivative_quality(row.linear, linear_qualities)
            _derivative_quality(row.angular, angular_qualities)
        _detect(rows, config, events)
    _ground_events(channels, ground, config, events, placements)
    # Refractory per kind/track; simultaneous different physical evidence is retained.
    kept: list[EventCandidate] = []
    last: dict[tuple[Track, EventKind], float] = {}
    for event in sorted(events, key=lambda e: (e.global_seconds, e.track, e.kind)):
        key = event.track, event.kind
        if (
            event.global_seconds - last.get(key, -float("inf"))
            >= config.refractory_seconds
        ):
            event.id = f"candidate:{len(kept)}:{event.track}:{event.kind}"
            kept.append(event)
            last[key] = event.global_seconds
    return FeatureSeries(
        reconstruction_id=motion.id,
        ground_id=ground.id,
        config=config,
        trajectory=[channels[t][i] for i in range(len(times)) for t in TRACKS],
        events=kept,
    )


def _emit(
    rows: list[FeatureSample],
    kind: EventKind,
    index: int,
    indices: list[int],
    qs: list[Quality],
    events: list[EventCandidate],
    reason: str,
    source_ids: list[str] | None = None,
) -> None:
    if (
        not qs
        or any(q.state in ("unknown", "interpolated") for q in qs)
        or any(not q.source_ids or q.uncertainty is None for q in qs)
    ):
        return
    times = [r.global_seconds for r in rows]
    tolerance = max([times[b] - times[a] for a, b in zip(indices, indices[1:])] or [0])
    events.append(
        EventCandidate(
            id="pending",
            track=rows[index].track,
            kind=kind,
            global_seconds=times[index],
            evidence_start_seconds=times[indices[0]],
            evidence_end_seconds=times[indices[-1]],
            duration_seconds=times[indices[-1]] - times[indices[0]],
            tolerance_seconds=tolerance,
            motion_sample_indices=indices,
            ground_sample_indices=sorted(
                {
                    cast(int, rows[i].ground_sample_index)
                    for i in indices
                    if rows[i].ground_sample_index is not None
                }
            ),
            source_event_ids=source_ids or [],
            quality=combine(qs, tolerance),
            reasons=[reason],
        )
    )


def _bouts(
    rows: list[FeatureSample],
    config: FeatureConfig,
    events: list[EventCandidate],
    *,
    angular: bool,
) -> None:
    times = [r.global_seconds for r in rows]
    enter_speed = config.enter_angular_speed_rad_s if angular else config.enter_speed
    leave_speed = config.leave_angular_speed_rad_s if angular else config.leave_speed
    min_excursion = (
        config.min_angular_excursion_rad if angular else config.min_excursion
    )
    valid = [
        (
            r.orientation.orientation is not None
            and r.orientation.quality.state != "interpolated"
        )
        if angular
        else (
            r.position_local.value is not None
            and r.position_local.quality.state != "interpolated"
        )
        for r in rows
    ]
    for run in _runs(times, valid, config.max_gap_seconds):
        # Hysteretic bouts require observed quiet context and nontrivial displacement.
        start: int | None = None
        quiet: int | None = None
        quiet_start: int | None = None
        for i in run:
            d = rows[i].angular if angular else rows[i].linear
            if (
                d.velocity is None
                or d.velocity_quality.state == "interpolated"
                or not d.support_times
                or not (min(d.support_times) < times[i] < max(d.support_times))
            ):
                continue
            speed = float(np.linalg.norm(d.velocity))
            error = config.uncertainty_multiplier * float(
                d.velocity_quality.uncertainty or 0
            )
            if speed + error <= leave_speed:
                if quiet_start is None:
                    quiet_start = i
                if times[i] - times[quiet_start] + 1e-9 < config.min_duration_seconds:
                    continue
                if start is not None:
                    indices = list(range(start, i + 1))
                    qs = [
                        rows[j].orientation.quality
                        if angular
                        else rows[j].position_local.quality
                        for j in indices
                    ]
                    sigma = max(float(q.uncertainty or 0) for q in qs)
                    if angular:
                        base = rotation(
                            cast(Quaternion, rows[start].orientation.orientation)
                        )
                        excursion = max(
                            float(
                                (
                                    rotation(
                                        cast(
                                            Quaternion, rows[j].orientation.orientation
                                        )
                                    )
                                    * base.inv()
                                ).magnitude()
                            )
                            for j in indices
                        )
                    else:
                        values = np.array(
                            [rows[j].position_local.value for j in indices]
                        )
                        excursion = float(
                            np.max(np.linalg.norm(values - values[0], axis=1))
                        )
                    if times[i] - times[
                        start
                    ] >= config.min_duration_seconds and excursion > max(
                        min_excursion, config.uncertainty_multiplier * sigma
                    ):
                        candidate_count = len(events)
                        _emit(
                            rows,
                            "motion_onset",
                            start,
                            indices,
                            qs,
                            events,
                            "quiet-to-moving rotation bout"
                            if angular
                            else "quiet-to-moving translation bout",
                        )
                        es = [rows[j].extension.value for j in indices]
                        if (
                            not angular
                            and es[0] is not None
                            and any(
                                e is not None
                                and e - es[0] > config.extension_prominence_ratio
                                for e in es
                            )
                        ):
                            _emit(
                                rows,
                                "extension_start",
                                start,
                                indices,
                                qs,
                                events,
                                "bout includes increasing chain extension",
                            )
                        for event in events[candidate_count:]:
                            event.tolerance_seconds += (
                                config.derivative_window_seconds / 2
                            )
                            event.quality.uncertainty = event.tolerance_seconds
                    start = None
                quiet = i
            else:
                quiet_start = None
                if start is None and quiet is not None and speed - error >= enter_speed:
                    start = quiet
                    quiet = None


def _direction_changes(
    rows: list[FeatureSample], config: FeatureConfig, events: list[EventCandidate]
) -> None:
    times = [r.global_seconds for r in rows]
    valid = [
        r.position_local.value is not None
        and r.position_local.quality.state != "interpolated"
        for r in rows
    ]
    for run in _runs(times, valid, config.max_gap_seconds):
        values = np.array([rows[i].position_local.value for i in run])
        sigma = max(float(rows[i].position_local.quality.uncertainty or 0) for i in run)
        prominence = max(config.min_excursion, config.uncertainty_multiplier * sigma)
        for axis in range(3):
            for sign in (-1, 1):
                peaks, props = find_peaks(sign * values[:, axis], prominence=prominence)
                for peak, left, right in zip(
                    peaks, props["left_bases"], props["right_bases"], strict=True
                ):
                    incoming = values[peak] - values[left]
                    outgoing = values[right] - values[peak]
                    if (
                        min(
                            float(np.linalg.norm(incoming)),
                            float(np.linalg.norm(outgoing)),
                        )
                        < prominence
                    ):
                        continue
                    if float(incoming @ outgoing) >= 0:
                        continue
                    indices = run[int(left) : int(right) + 1]
                    if (
                        times[indices[-1]] - times[indices[0]]
                        >= config.min_duration_seconds
                    ):
                        _emit(
                            rows,
                            "direction_change",
                            run[int(peak)],
                            indices,
                            [rows[j].position_local.quality for j in indices],
                            events,
                            "prominent local path reversal with opposing displacement",
                        )


def _detect(
    rows: list[FeatureSample], config: FeatureConfig, events: list[EventCandidate]
) -> None:
    times = [r.global_seconds for r in rows]
    _bouts(rows, config, events, angular=False)
    _bouts(rows, config, events, angular=True)
    _direction_changes(rows, config, events)
    ext_valid = [
        r.extension.value is not None and r.extension.quality.state != "interpolated"
        for r in rows
    ]
    for run in _runs(times, ext_valid, config.max_gap_seconds):
        values = np.array([rows[i].extension.value for i in run], dtype=float)
        sigma = max(float(rows[i].extension.quality.uncertainty or 0) for i in run)
        prominence = max(
            config.extension_prominence_ratio, config.uncertainty_multiplier * sigma
        )
        for sign, kind in ((1, "extension_maximum"), (-1, "extension_minimum")):
            peaks, props = find_peaks(
                sign * values, prominence=prominence, plateau_size=True
            )
            for p, left, right in zip(
                peaks, props["left_bases"], props["right_bases"], strict=True
            ):
                indices = run[int(left) : int(right) + 1]
                if times[indices[-1]] - times[indices[0]] < config.min_duration_seconds:
                    continue
                i = run[int(p)]
                qs = [rows[j].extension.quality for j in indices]
                _emit(
                    rows,
                    cast(EventKind, kind),
                    i,
                    indices,
                    qs,
                    events,
                    "prominent interior chain-extension extremum",
                )
                _emit(
                    rows,
                    "direction_change",
                    i,
                    indices,
                    qs,
                    events,
                    "chain extension reverses across extremum",
                )
                _emit(
                    rows,
                    "extension_end" if sign == 1 else "preparation_candidate",
                    i,
                    indices,
                    qs,
                    events,
                    "physical extension boundary; no technique assignment",
                )
    if rows[0].track in ("left_arm", "right_arm"):
        crossings = [
            next((r for r in row.relations if r.axis == "crossing"), None)
            for row in rows
        ]
        previous: int | None = None
        for i, relation in enumerate(crossings):
            if i and times[i] - times[i - 1] > config.max_gap_seconds:
                previous = None
            if relation is None or rows[i].relation_evidence_quality.state in (
                "unknown",
                "interpolated",
            ):
                previous = None
                continue
            if relation.value == "not_crossed":
                previous = i
            elif relation.value == "crossed" and previous is not None:
                end = i
                while (
                    end + 1 < len(rows)
                    and crossings[end + 1] is not None
                    and (
                        cast(Relation, crossings[end + 1]).value == "crossed"
                        and times[end + 1] - times[end] <= config.max_gap_seconds
                    )
                ):
                    end += 1
                if times[end] - times[i] >= config.min_duration_seconds:
                    indices = list(range(previous, end + 1))
                    before_count = len(events)
                    _emit(
                        rows,
                        "arm_crossing",
                        i,
                        indices,
                        [rows[j].relation_evidence_quality for j in indices]
                        + [
                            cast(Relation, crossings[previous]).quality,
                            relation.quality,
                        ],
                        events,
                        "forearm crossing with bracketed ambiguous geometry",
                    )
                    if len(events) > before_count:
                        events[-1].tolerance_seconds = max(
                            events[-1].tolerance_seconds, times[i] - times[previous]
                        )
                        events[-1].quality.uncertainty = events[-1].tolerance_seconds
                previous = None


def _ground_events(
    channels: dict[Track, list[FeatureSample]],
    ground: Ground,
    config: FeatureConfig,
    events: list[EventCandidate],
    placements: FootprintSeries | None,
) -> None:
    for track in ("left_leg", "right_leg"):
        rows = channels[track]
        times = [r.global_seconds for r in rows]
        for i in range(1, len(rows)):
            a, b = rows[i - 1].contact, rows[i].contact
            if (
                a is None
                or b is None
                or "unknown" in (a.state, b.state)
                or a.state == b.state
                or (times[i] - times[i - 1] > config.max_gap_seconds)
            ):
                continue
            before = i - 1
            while (
                before > 0
                and rows[before - 1].contact is not None
                and (
                    cast(Contact, rows[before - 1].contact).state == a.state
                    and times[before] - times[before - 1] <= config.max_gap_seconds
                )
            ):
                before -= 1
            if times[i - 1] - times[before] + 1e-9 < config.min_duration_seconds:
                continue
            end = i
            while (
                end + 1 < len(rows)
                and rows[end + 1].contact is not None
                and (
                    cast(Contact, rows[end + 1].contact).state == b.state
                    and times[end + 1] - times[end] <= config.max_gap_seconds
                )
            ):
                end += 1
            if times[end] - times[i] >= config.min_duration_seconds:
                indices = list(range(before, end + 1))
                _emit(
                    rows,
                    "first_contact" if b.state == "contact" else "lift_off",
                    i,
                    indices,
                    [cast(Contact, rows[j].contact).quality for j in indices],
                    events,
                    "native ground contact transition confirmed by dwell",
                )
        lookup = {t: i for i, t in enumerate(times)}
        sources: list[Footprint | Pivot] = [*ground.footprints, *ground.pivots]
        for source in sources:
            if source.foot != track.split("_")[0] or source.quality.state == "unknown":
                continue
            start_index, end_index = (
                lookup.get(source.interval.start),
                lookup.get(source.interval.end),
            )
            if start_index is None or end_index is None:
                continue
            indices = list(range(start_index, end_index + 1))
            if any(
                times[b] - times[a] > config.max_gap_seconds
                or rows[a].contact is None
                or cast(Contact, rows[a].contact).state != "contact"
                for a, b in zip(indices, indices[1:])
            ) or (
                rows[end_index].contact is None
                or cast(Contact, rows[end_index].contact).state != "contact"
            ):
                continue
            qs = [source.quality] + [
                cast(Contact, rows[j].contact).quality for j in indices
            ]
            if isinstance(source, Pivot):
                for kind, index in (
                    ("pivot_start", start_index),
                    ("pivot_end", end_index),
                ):
                    _emit(
                        rows,
                        cast(EventKind, kind),
                        index,
                        indices,
                        qs,
                        events,
                        "upstream supported physical pivot boundary",
                        [source.id],
                    )
            elif placements is not None:
                placement = next(
                    e for e in placements.events if e.footprint.id == source.id
                )
                if (
                    placement.kind == "stable"
                    and placement.confirmed_seconds is not None
                ):
                    placement_index = lookup.get(placement.event_seconds)
                    if placement_index is not None and placement_index in indices:
                        _emit(
                            rows,
                            "stable_placement",
                            placement_index,
                            indices,
                            qs,
                            events,
                            "upstream stable placement with dwell confirmation",
                            [source.id],
                        )
