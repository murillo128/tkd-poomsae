"""Supported physical rotation from existing native placement evidence."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from contracts.models import Interval, Pivot, Quality, StrictModel
from reconstruction.footprints.core import XY, FootprintSeries, TrajectorySample
from storage import hash_config

REVISION: Literal["pivot-v1"] = "pivot-v1"


class PivotConfig(StrictModel):
    enter_speed_rad_s: float = Field(default=0.25, gt=0)
    leave_speed_rad_s: float = Field(default=0.1, gt=0)
    quiet_seconds: float = Field(default=0.08, gt=0)
    max_gap_seconds: float = Field(default=0.15, gt=0)
    min_excursion_rad: float = Field(default=0.12, gt=0, lt=math.pi)
    stationary_fraction: float = Field(default=0.08, gt=0, lt=0.5)
    uncertainty_multiplier: float = Field(default=2, ge=1)

    @model_validator(mode="after")
    def hysteresis(self) -> PivotConfig:
        if self.leave_speed_rad_s >= self.enter_speed_rad_s:
            raise ValueError("enter speed must exceed leave speed")
        return self


class RotationSample(StrictModel):
    # Exact link to the unmodified placement/native contact and motion evidence.
    trajectory_index: int
    unwrapped_yaw_rad: float | None
    angular_speed_rad_s: float | None
    supported: bool
    reasons: list[str]


class Translation(StrictModel):
    landmark: Literal["heel", "forefoot", "centre"]
    positions: list[XY]
    displacement: XY
    path_length: float
    max_excursion: float
    max_excursion_foot_lengths: float


class PivotEvent(StrictModel):
    pivot: Pivot
    rotation_indices: list[int]
    angular_trajectory_rad: list[float]
    angular_travel_rad: float
    translations: list[Translation]
    placement_ids: list[str]
    classification: Literal["supported_rotation", "rotation_with_translation"]
    # Confidence in the approximate region, not force or pressure measurements.
    region_confidence: float = Field(ge=0, le=1)
    reasons: list[str]


class PivotSeries(StrictModel):
    version: Literal[1] = 1
    artifact_role: Literal["physical_pivots"] = "physical_pivots"
    algorithm_revision: Literal["pivot-v1"] = REVISION
    placements: FootprintSeries
    config: PivotConfig
    trajectory: list[RotationSample]
    events: list[PivotEvent]

    @model_validator(mode="after")
    def links(self) -> PivotSeries:
        if [s.trajectory_index for s in self.trajectory] != list(
            range(len(self.placements.trajectory))
        ):
            raise ValueError("rotation trajectory must preserve placement indices")
        used: set[int] = set()
        for event in self.events:
            indices = event.rotation_indices
            if (
                len(indices) < 2
                or indices != sorted(set(indices))
                or any(b - a != 2 for a, b in zip(indices, indices[1:]))
            ):
                raise ValueError("ordered nonzero rotation interval required")
            if any(i < 0 or i >= len(self.trajectory) or i in used for i in indices):
                raise ValueError("invalid rotation source links")
            rows = [self.placements.trajectory[i] for i in indices]
            if any(r.foot != event.pivot.foot for r in rows) or any(
                not self.trajectory[i].supported
                or not _supported(self.placements.trajectory[i])
                for i in indices
            ):
                raise ValueError("pivot requires supporting side evidence")
            if (event.pivot.interval.start, event.pivot.interval.end) != (
                rows[0].global_seconds,
                rows[-1].global_seconds,
            ):
                raise ValueError("pivot interval disagrees with source")
            angles = [self.trajectory[i].unwrapped_yaw_rad for i in indices]
            if any(a is None for a in angles) or angles != event.angular_trajectory_rad:
                raise ValueError("pivot angular trajectory disagrees with source")
            if event.pivot.rotation_rad != (
                event.angular_trajectory_rad[-1] - event.angular_trajectory_rad[0]
            ):
                raise ValueError("pivot total rotation disagrees with trajectory")
            if event.placement_ids != sorted({r.event_id for r in rows if r.event_id}):
                raise ValueError("pivot placement links disagree")
            if len(event.translations) != 3 or any(
                len(t.positions) != len(indices) for t in event.translations
            ):
                raise ValueError("complete translation trajectories required")
            expected_positions = [
                [r.geometry.axis_start for r in rows],
                [r.geometry.axis_end for r in rows],
                [r.xy_ground for r in rows],
            ]
            if [t.landmark for t in event.translations] != [
                "heel",
                "forefoot",
                "centre",
            ] or any(
                t.positions != positions
                for t, positions in zip(event.translations, expected_positions)
            ):
                raise ValueError("translation disagrees with native geometry")
            if event.angular_travel_rad != sum(
                abs(b - a)
                for a, b in zip(
                    event.angular_trajectory_rad, event.angular_trajectory_rad[1:]
                )
            ):
                raise ValueError("angular travel disagrees with trajectory")
            if event.pivot.region == "unknown" and event.region_confidence != 0:
                raise ValueError("unknown region cannot carry region confidence")
            used.update(indices)
        return self


def _supported(row: TrajectorySample) -> bool:
    q = row.contact.quality
    return (
        row.contact.state == "contact"
        and row.support in ("both", row.foot)
        and q.state != "unknown"
        and bool(q.source_ids)
        and q.uncertainty is not None
    )


def _emit(indices: list[int], series: PivotSeries, source_digest: str) -> None:
    rows = [series.placements.trajectory[i] for i in indices]
    angles = [series.trajectory[i].unwrapped_yaw_rad for i in indices]
    if len(rows) < 2 or any(a is None for a in angles):
        return
    values = [float(a) for a in angles if a is not None]
    config = series.config
    margin = (
        config.uncertainty_multiplier
        * 2
        * max(float(r.angle_uncertainty or 0) for r in rows)
    )
    if max(values) - min(values) < config.min_excursion_rad + margin:
        return
    heels = [r.geometry.axis_start for r in rows]
    toes = [r.geometry.axis_end for r in rows]
    centres = [r.xy_ground for r in rows]
    if any(p is None for p in heels + toes + centres):
        return
    length = min(math.dist(h, t) for h, t in zip(heels, toes) if h and t)
    sigma = (
        config.uncertainty_multiplier
        * 2
        * max(float(r.position_uncertainty or 0) for r in rows)
    )
    translations = []
    names: tuple[Literal["heel", "forefoot", "centre"], ...] = (
        "heel",
        "forefoot",
        "centre",
    )
    for name, points in zip(names, (heels, toes, centres)):
        xy = [p for p in points if p is not None]
        excursion = max(math.dist(xy[0], p) for p in xy)
        translations.append(
            Translation(
                landmark=name,
                positions=xy,
                displacement=(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1]),
                path_length=sum(math.dist(a, b) for a, b in zip(xy, xy[1:])),
                max_excursion=excursion,
                max_excursion_foot_lengths=excursion / length,
            )
        )
    region: Literal["heel", "forefoot", "unknown"] = "unknown"
    for translation in translations[:2]:
        if (
            translation.max_excursion + sigma <= config.stationary_fraction * length
            and all(r.contact.region in (translation.landmark, "flat") for r in rows)
        ):
            region = "heel" if translation.landmark == "heel" else "forefoot"
            break
    translating = all(
        t.max_excursion + sigma > config.stationary_fraction * length
        for t in translations
    )
    sources = sorted(
        {
            s
            for r in rows
            for q in [r.contact.quality] + [p.quality for p in r.landmarks]
            for s in q.source_ids
        }
    )
    identity = "pivot:" + hash_config(
        {
            "placements": source_digest,
            "config": config.model_dump(mode="json"),
            "indices": indices,
            "revision": REVISION,
        }
    )
    series.events.append(
        PivotEvent(
            pivot=Pivot(
                id=identity,
                foot=rows[0].foot,
                interval=Interval(
                    start=rows[0].global_seconds, end=rows[-1].global_seconds
                ),
                region=region,
                rotation_rad=values[-1] - values[0],
                quality=Quality(
                    state="inferred", uncertainty=margin, source_ids=sources
                ),
            ),
            rotation_indices=indices.copy(),
            angular_trajectory_rad=values,
            angular_travel_rad=sum(abs(b - a) for a, b in zip(values, values[1:])),
            translations=translations,
            placement_ids=sorted({r.event_id for r in rows if r.event_id}),
            classification="rotation_with_translation"
            if translating
            else "supported_rotation",
            region_confidence=0 if region == "unknown" else 0.8,
            reasons=["approximate_region_not_centre_of_pressure"]
            + (["translation_without_stationary_point"] if translating else [])
            + (["indeterminate_pivot_region"] if region == "unknown" else []),
        )
    )


def derive_pivots(
    placements: FootprintSeries, config: PivotConfig | None = None
) -> PivotSeries:
    """Unwrap native axes without smoothing; never estimate contact or semantics."""
    placements = FootprintSeries.model_validate(placements.model_dump())
    config = PivotConfig.model_validate((config or PivotConfig()).model_dump())
    series = PivotSeries(
        placements=placements,
        config=config,
        trajectory=[
            RotationSample(
                trajectory_index=i,
                unwrapped_yaw_rad=None,
                angular_speed_rad_s=None,
                supported=_supported(r),
                reasons=[],
            )
            for i, r in enumerate(placements.trajectory)
        ],
        events=[],
    )
    source_digest = hash_config(placements.model_dump(mode="json"))
    for offset in (0, 1):
        previous: int | None = None
        active: list[int] = []
        last_motion: int | None = None
        seen: set[str] = set()

        def flush() -> None:
            nonlocal last_motion
            if last_motion is not None:
                _emit(active[: active.index(last_motion) + 1], series, source_digest)
            active.clear()
            last_motion = None

        for i in range(offset, len(series.trajectory), 2):
            row, out = placements.trajectory[i], series.trajectory[i]
            old = placements.trajectory[previous] if previous is not None else None
            dt = row.global_seconds - old.global_seconds if old else 0
            axis = [p for p in row.landmarks if p.name.endswith(("_heel", "_forefoot"))]
            fresh = len(axis) == 2 and all(
                set(p.quality.source_ids) - seen for p in axis
            )
            usable = row.geometry.kind == "axis" and row.yaw_rad is not None and fresh
            if not usable or (old and dt > config.max_gap_seconds + 1e-12):
                flush()
                previous = None
                seen.clear()
                out.reasons.append(
                    "unavailable_axis_or_native_evidence"
                    if not usable
                    else "temporal_gap"
                )
            if not usable:
                out.supported = False
                continue
            seen.update(s for p in axis for s in p.quality.source_ids)
            assert row.yaw_rad is not None
            out.unwrapped_yaw_rad = row.yaw_rad
            if previous is not None:
                prior = series.trajectory[previous]
                assert old and old.yaw_rad is not None
                delta = math.atan2(
                    math.sin(row.yaw_rad - old.yaw_rad),
                    math.cos(row.yaw_rad - old.yaw_rad),
                )
                if abs(delta) >= math.pi - 0.05:
                    flush()
                    out.supported = False
                    out.reasons.append("ambiguous_half_turn")
                    previous = i
                    continue
                assert prior.unwrapped_yaw_rad is not None
                out.unwrapped_yaw_rad = prior.unwrapped_yaw_rad + delta
                out.angular_speed_rad_s = delta / dt
                if out.supported and prior.supported:
                    speed = abs(delta) / dt
                    if not active and speed >= config.enter_speed_rad_s:
                        active.extend([previous, i])
                        last_motion = i
                    elif active:
                        active.append(i)
                        if speed >= config.leave_speed_rad_s:
                            last_motion = i
                        elif last_motion is not None and (
                            row.global_seconds
                            - placements.trajectory[last_motion].global_seconds
                            >= config.quiet_seconds - 1e-12
                        ):
                            flush()
                else:
                    flush()
            if not out.supported:
                out.reasons.append("unreliable_or_absent_support")
            previous = i
        flush()
    series.events.sort(key=lambda e: (e.pivot.interval.start, e.pivot.foot))
    return PivotSeries.model_validate(series.model_dump())
