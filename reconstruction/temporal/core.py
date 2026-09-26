"""Bounded time-domain filtering and evidence-qualified derivatives."""

from __future__ import annotations

import bisect
from typing import Any, Literal, cast

import numpy as np
from pydantic import Field, model_validator
from scipy.spatial.transform import Rotation, Slerp  # type: ignore[import-untyped]

from contracts.models import (
    Landmark3D,
    MotionSample,
    Quality,
    Quaternion,
    Reconstruction,
    SegmentFrame,
    StrictModel,
)
from reconstruction.detailed import DetailedSample, GeometryConfig, derive_sample

REVISION = "temporal-motion-v1"
Vector = tuple[float, float, float]


class TemporalConfig(StrictModel):
    filter_window_seconds: float = Field(default=0.06, gt=0)
    derivative_window_seconds: float = Field(default=0.08, gt=0)
    regularization: float = Field(default=0.001, ge=0)
    max_interval_seconds: float = Field(default=0.15, gt=0)
    short_gap_seconds: float = Field(default=0.04, ge=0)
    max_position_uncertainty: float = Field(default=0.05, gt=0)
    uncertainty_floor: float = Field(default=1e-6, gt=0)
    inferred_weight: float = Field(default=0.5, gt=0, le=1)
    interpolated_weight: float = Field(default=0.25, gt=0, le=1)
    max_step_world: float | None = Field(default=None, gt=0)
    break_times: list[float] = Field(default_factory=list)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)

    @model_validator(mode="after")
    def valid_gaps(self) -> TemporalConfig:
        if self.short_gap_seconds > self.max_interval_seconds:
            raise ValueError("short gap cannot exceed maximum interval")
        if self.break_times != sorted(set(self.break_times)):
            raise ValueError("break times must be sorted and unique")
        return self


class Derivative(StrictModel):
    velocity: Vector | None = None
    acceleration: Vector | None = None
    velocity_quality: Quality = Field(default_factory=lambda: Quality(state="unknown"))
    acceleration_quality: Quality = Field(
        default_factory=lambda: Quality(state="unknown")
    )
    support_times: list[float] = Field(default_factory=list)
    parent: str = "world"
    unit: str
    time_unit: Literal["s"] = "s"
    velocity_valid: bool = False
    acceleration_valid: bool = False

    @model_validator(mode="after")
    def valid_masks(self) -> Derivative:
        if self.velocity_valid != (self.velocity is not None) or (
            self.acceleration_valid != (self.acceleration is not None)
        ):
            raise ValueError("derivative masks disagree with values")
        return self


class KinematicSample(StrictModel):
    global_seconds: float
    linear: dict[str, Derivative]
    angular: dict[str, Derivative]


def supported(q: Quality, config: TemporalConfig, *, angular: bool = False) -> bool:
    limit = (
        config.geometry.max_angle_uncertainty
        if angular
        else (config.max_position_uncertainty)
    )
    return (
        q.state != "unknown"
        and bool(q.source_ids)
        and q.uncertainty is not None
        and q.uncertainty <= limit
    )


def vector(value: Any) -> Vector:
    return float(value[0]), float(value[1]), float(value[2])


def rotation(q: Quaternion) -> Any:
    w, x, y, z = q.wxyz
    return Rotation.from_quat([x, y, z, w])


def quaternion(r: Any) -> Quaternion:
    x, y, z, w = r.as_quat()
    return Quaternion(wxyz=(float(w), float(x), float(y), float(z)))


def combined(
    qualities: list[Quality],
    sigma: float,
    state: Literal["inferred", "interpolated"] = "inferred",
) -> Quality:
    return Quality(
        state=state,
        uncertainty=sigma,
        source_ids=sorted({s for q in qualities for s in q.source_ids}),
    )


def connected(a: float, b: float, config: TemporalConfig) -> bool:
    return b - a <= config.max_interval_seconds and not any(
        a < boundary <= b for boundary in config.break_times
    )


def runs(
    times: list[float], values: list[Vector | None], config: TemporalConfig
) -> list[list[int]]:
    result: list[list[int]] = []
    for i, value in enumerate(values):
        if value is None:
            continue
        link = (
            i > 0
            and values[i - 1] is not None
            and connected(times[i - 1], times[i], config)
        )
        if link and config.max_step_world is not None:
            link = bool(
                np.linalg.norm(np.array(value) - values[i - 1]) <= config.max_step_world
            )
        if not link:
            result.append([])
        result[-1].append(i)
    return result


def time_window(
    times: list[float], run: list[int], center: int, radius: float
) -> list[int]:
    """Search timestamp bounds without rescanning the recording for each frame."""
    start = max(run[0], bisect.bisect_left(times, times[center] - radius - 1e-12))
    end = min(run[-1] + 1, bisect.bisect_right(times, times[center] + radius + 1e-12))
    return list(range(start, end))


def interpolate_positions(
    times: list[float],
    values: list[Vector | None],
    qualities: list[Quality],
    config: TemporalConfig,
) -> None:
    known = [i for i, value in enumerate(values) if value is not None]
    for a, b in zip(known, known[1:]):
        if b == a + 1 or times[b] - times[a] > config.short_gap_seconds:
            continue
        if not connected(times[a], times[b], config):
            continue
        delta = np.array(values[b]) - values[a]
        if config.max_step_world is not None and np.linalg.norm(delta) > (
            config.max_step_world
        ):
            continue
        sigma = max(float(qualities[k].uncertainty or 0) for k in (a, b)) + (
            float(np.linalg.norm(delta)) / 2
        )
        if sigma > config.max_position_uncertainty:
            continue
        for i in range(a + 1, b):
            fraction = (times[i] - times[a]) / (times[b] - times[a])
            values[i] = vector(np.array(values[a]) + fraction * delta)
            qualities[i] = combined([qualities[a], qualities[b]], sigma, "interpolated")


def polynomial(
    times: list[float],
    values: list[Vector],
    qualities: list[Quality],
    center: float,
    radius: float,
    config: TemporalConfig,
    regularization: float,
) -> tuple[Any, Any]:
    u = (np.array(times) - center) / radius
    design = np.column_stack([np.ones(len(u)), u, u**2])
    weights = np.array(
        [
            (
                config.inferred_weight
                if q.state == "inferred"
                else config.interpolated_weight
                if q.state == "interpolated"
                else 1
            )
            / max(float(q.uncertainty or 0), config.uncertainty_floor) ** 2
            for q in qualities
        ]
    )
    weights /= max(weights)
    system = design.T @ (weights[:, None] * design)
    system[2, 2] += regularization * weights.sum()
    operator = np.linalg.solve(system, design.T * weights)
    return operator @ np.array(values), operator


def regularize_positions(
    times: list[float],
    values: list[Vector | None],
    qualities: list[Quality],
    config: TemporalConfig,
) -> tuple[list[Vector | None], list[Quality]]:
    values = list(values)
    qualities = [q.model_copy(deep=True) for q in qualities]
    for i, q in enumerate(qualities):
        if values[i] is None or not supported(q, config):
            values[i], qualities[i] = (
                None,
                Quality(state="unknown", source_ids=q.source_ids),
            )
    interpolate_positions(times, values, qualities, config)
    output, evidence = list(values), list(qualities)
    radius = config.filter_window_seconds / 2
    if config.regularization == 0:
        return output, evidence
    for run in runs(times, values, config):
        for i in run:
            window = time_window(times, run, i, radius)
            if len(window) < 3:
                continue
            qs = [qualities[j] for j in window]
            coefficients, operator = polynomial(
                [times[j] for j in window],
                [cast(Vector, values[j]) for j in window],
                qs,
                times[i],
                radius,
                config,
                config.regularization,
            )
            output[i] = vector(coefficients[0])
            sigma = float(
                sum(
                    abs(w) * float(q.uncertainty or 0)
                    for w, q in zip(operator[0], qs, strict=True)
                )
            )
            displacement = float(np.linalg.norm(coefficients[0] - values[i]))
            evidence[i] = combined(
                qs, max(sigma, displacement, float(qualities[i].uncertainty or 0))
            )
            if not supported(evidence[i], config):
                output[i] = None
                evidence[i] = Quality(
                    state="unknown", source_ids=evidence[i].source_ids
                )
    return output, evidence


def derivatives(
    times: list[float],
    values: list[Vector | None],
    qualities: list[Quality],
    config: TemporalConfig,
    unit: str,
    *,
    angular: bool = False,
    parent: str = "world",
) -> list[Derivative]:
    output = [Derivative(unit=unit, parent=parent) for _ in times]
    usable = [
        v if supported(q, config, angular=angular) else None
        for v, q in zip(values, qualities, strict=True)
    ]
    radius = config.derivative_window_seconds / 2
    # Angular runs must not interpret the positional step threshold as radians.
    run_config = (
        config.model_copy(update={"max_step_world": None}) if angular else config
    )
    for run in runs(times, usable, run_config):
        for i in run:
            window = time_window(times, run, i, radius)
            # One-sided endpoints use only available points inside the same radius.
            if len(window) < 3:
                continue
            qs = [qualities[j] for j in window]
            coefficients, operator = polynomial(
                [times[j] for j in window],
                [cast(Vector, usable[j]) for j in window],
                qs,
                times[i],
                radius,
                config,
                0,
            )
            output[i] = Derivative(
                unit=unit,
                parent=parent,
                velocity=vector(coefficients[1] / radius),
                acceleration=vector(2 * coefficients[2] / radius**2),
                velocity_valid=True,
                acceleration_valid=True,
                velocity_quality=combined(
                    qs,
                    float(
                        sum(
                            abs(w) * float(q.uncertainty or 0) / radius
                            for w, q in zip(operator[1], qs, strict=True)
                        )
                    ),
                ),
                acceleration_quality=combined(
                    qs,
                    float(
                        sum(
                            2 * abs(w) * float(q.uncertainty or 0) / radius**2
                            for w, q in zip(operator[2], qs, strict=True)
                        )
                    ),
                ),
                support_times=[times[j] for j in window],
            )
    return output


def angular_derivatives(
    times: list[float],
    orientations: list[Quaternion | None],
    qualities: list[Quality],
    config: TemporalConfig,
    parent: str,
) -> list[Derivative]:
    output = [Derivative(unit="rad", parent=parent) for _ in times]
    placeholder = [
        (0.0, 0.0, 0.0)
        if q is not None and supported(quality, config, angular=True)
        else None
        for q, quality in zip(orientations, qualities, strict=True)
    ]
    for run in runs(
        times, placeholder, config.model_copy(update={"max_step_world": None})
    ):
        for i in run:
            # Recenter SO(3) log at each sample, avoiding global Euler wrap.
            window = time_window(times, run, i, config.derivative_window_seconds / 2)
            if len(window) < 3:
                continue
            base = rotation(cast(Quaternion, orientations[i]))
            relative = [
                (rotation(cast(Quaternion, orientations[j])) * base.inv()).as_rotvec()
                for j in window
            ]
            # A near-pi bracket is ambiguous; do not infer winding or spin count.
            if any(np.linalg.norm(v) >= np.pi - 1e-6 for v in relative):
                continue
            results = derivatives(
                [times[j] for j in window],
                [vector(v) for v in relative],
                [qualities[j] for j in window],
                config,
                "rad",
                angular=True,
                parent=parent,
            )
            output[i] = results[window.index(i)]
    return output


def refresh_frames(
    sample: MotionSample,
    identifier: str,
    config: TemporalConfig,
    supplied_segments: set[str],
    supplied_root: bool,
) -> DetailedSample:
    geometry = derive_sample(
        sample,
        reconstruction_id=identifier,
        representation="regularized",
        config=config.geometry,
    )
    frames = {"torso": geometry.body_frame, "head": geometry.head.frame}
    frames.update({f"{s}_hand": h.frame for s, h in geometry.hands.items()})
    frames.update({f"{s}_foot": f.frame for s, f in geometry.feet.items()})
    sample.segments = [s for s in sample.segments if s.segment in supplied_segments]
    sample.segments.extend(
        SegmentFrame(
            segment=name,
            parent="world",
            orientation=frame.orientation.model_copy(deep=True)
            if frame.orientation
            else None,
            quality=frame.quality,
        )
        for name, frame in frames.items()
        if name not in supplied_segments
    )
    if not supplied_root:
        sample.root_orientation = (
            geometry.body_frame.orientation.model_copy(deep=True)
            if geometry.body_frame.orientation
            else None
        )
    return geometry


def regularize(
    source: Reconstruction,
    config: TemporalConfig,
    identifier: str,
    root_qualities: list[Quality] | None = None,
    reference: Reconstruction | None = None,
    root_rotation_qualities: list[Quality] | None = None,
) -> tuple[list[MotionSample], list[DetailedSample], list[KinematicSample]]:
    if not source.samples:
        raise ValueError("nonempty reconstruction required")
    times = [s.global_seconds for s in source.samples]
    output = [s.model_copy(deep=True) for s in source.samples]
    names = sorted({p.name for s in source.samples for p in s.landmarks})
    maps = [{p.name: p for p in s.landmarks} for s in source.samples]
    linear: dict[str, list[Derivative]] = {}
    unit = "m" if source.scale == "metric" else "arbitrary"
    for name in names:
        points = [
            m.get(
                name,
                Landmark3D(
                    name=name,
                    xyz_world=None,
                    quality=Quality(state="unknown"),
                ),
            )
            for m in maps
        ]
        values, qualities = regularize_positions(
            times, [p.xyz_world for p in points], [p.quality for p in points], config
        )
        for i, sample in enumerate(output):
            sample.landmarks = [p for p in sample.landmarks if p.name != name]
            sample.landmarks.append(
                Landmark3D(name=name, xyz_world=values[i], quality=qualities[i])
            )
        linear[name] = derivatives(times, values, qualities, config, unit)
    root_qualities = root_qualities or [s.quality for s in source.samples]
    root_values, root_qs = regularize_positions(
        times, [s.root_xyz_world for s in source.samples], root_qualities, config
    )
    linear["root"] = derivatives(times, root_values, root_qs, config, unit)
    reference = reference or source
    detailed = []
    for i, sample in enumerate(output):
        sample.root_xyz_world = root_values[i]
        sample.quality = combined(
            [p.quality for p in sample.landmarks if p.quality.state != "unknown"],
            max(
                (float(p.quality.uncertainty or 0) for p in sample.landmarks), default=0
            ),
        )
        if all(p.xyz_world is None for p in sample.landmarks):
            sample.quality = Quality(
                state="unknown", source_ids=sample.quality.source_ids
            )
        supplied = reference.samples[i]
        detailed.append(
            refresh_frames(
                sample,
                identifier,
                config,
                {s.segment for s in supplied.segments},
                supplied.root_orientation is not None,
            )
        )
    segment_names = sorted({s.segment for sample in output for s in sample.segments})
    for sample in output:
        existing = {s.segment for s in sample.segments}
        sample.segments.extend(
            SegmentFrame(
                segment=n,
                parent="world",
                orientation=None,
                quality=Quality(state="unknown"),
            )
            for n in segment_names
            if n not in existing
        )
    angular: dict[str, list[Derivative]] = {}
    for channel in ["root"] + segment_names:
        qs = [
            (
                (
                    root_rotation_qualities[i]
                    if root_rotation_qualities
                    else source.samples[i].quality
                )
                if reference.samples[i].root_orientation is not None
                else d.body_frame.quality
            )
            if channel == "root"
            else next(s.quality for s in sample.segments if s.segment == channel)
            for i, (sample, d) in enumerate(zip(output, detailed, strict=True))
        ]
        orientations = [
            sample.root_orientation
            if channel == "root"
            else next(s.orientation for s in sample.segments if s.segment == channel)
            for sample in output
        ]
        previous: Quaternion | None = None
        for i, orientation in enumerate(orientations):
            if orientation is None:
                previous = None
                continue
            if i and not connected(times[i - 1], times[i], config):
                previous = None
            if previous and np.dot(previous.wxyz, orientation.wxyz) < 0:
                orientation.wxyz = tuple(-v for v in orientation.wxyz)  # type: ignore[assignment]
            previous = orientation
        parents = [
            "world"
            if channel == "root"
            else next(s.parent for s in sample.segments if s.segment == channel)
            for sample in output
        ]
        # Never mix derivatives across changing parent frames.
        for i in range(1, len(parents)):
            if parents[i] != parents[i - 1]:
                qs[i] = Quality(state="unknown")
        angular[channel] = angular_derivatives(
            times, orientations, qs, config, parents[0]
        )
        for i, parent in enumerate(parents):
            angular[channel][i].parent = parent
    kinematics = [
        KinematicSample(
            global_seconds=t,
            linear={name: values[i] for name, values in linear.items()},
            angular={name: values[i] for name, values in angular.items()},
        )
        for i, t in enumerate(times)
    ]
    return output, detailed, kinematics


def slerp(a: Quaternion, b: Quaternion, fraction: float) -> Quaternion:
    return quaternion(
        Slerp([0, 1], Rotation.concatenate([rotation(a), rotation(b)]))(fraction)
    )
