"""Backend-owned ground projection and deterministic native-time inspection."""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Literal

from pydantic import Field, model_validator

from contracts.models import (
    Calibration,
    GroundSample,
    Quality,
    Reconstruction,
    StrictModel,
)
from reconstruction.footprints.core import TrajectorySample
from reconstruction.pivots.core import PivotSeries

REVISION: Literal["ground-view-v1"] = "ground-view-v1"
XY = tuple[float, float]


class GroundViewConfig(StrictModel):
    max_gap_seconds: float = Field(default=0.15, gt=0)


class RootPoint(StrictModel):
    id: str
    motion_sample_index: int
    global_seconds: float
    xy_ground: XY | None
    z_ground: float | None
    quality: Quality
    reasons: list[str]

    @model_validator(mode="after")
    def geometry(self) -> RootPoint:
        if (self.xy_ground is None) != (self.z_ground is None):
            raise ValueError("root XY and Z validity must agree")
        if (self.xy_ground is None) != (self.quality.state == "unknown"):
            raise ValueError("root validity must agree with quality")
        return self


class ContactEvent(StrictModel):
    # Native sample events, not inferred continuous contact intervals.
    id: str
    motion_sample_index: int
    sample: GroundSample
    foot_trajectory_indices: tuple[int, int]


class SceneBounds(StrictModel):
    minimum_xy: XY
    maximum_xy: XY
    minimum_z: float | None
    maximum_z: float | None


class DynamicView(StrictModel):
    requested_seconds: float
    status: Literal["native", "native_snapshot", "unavailable"]
    sampled_seconds: float | None = None
    bracket_seconds: tuple[float, float] | None = None
    root: RootPoint | None = None
    feet: list[TrajectorySample] = Field(default_factory=list)
    contact_event: ContactEvent | None = None
    placement_ids: list[str] = Field(default_factory=list)
    pivot_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class GroundViewSeries(StrictModel):
    version: Literal[1] = 1
    artifact_role: Literal["physical_ground_view"] = "physical_ground_view"
    algorithm_revision: Literal["ground-view-v1"] = REVISION
    reconstruction_id: str
    calibration_id: str
    participant_id: str
    ground_id: str | None
    ground_status: Literal["available", "unavailable"]
    world_unit: Literal["m", "arbitrary"]
    coordinate_frame: Literal["ground_aligned_world_xy_z_up"] = (
        "ground_aligned_world_xy_z_up"
    )
    ground_source_ids: list[str]
    config: GroundViewConfig
    root_trajectory: list[RootPoint]
    # Separate index runs prevent a viewer from connecting across missing geometry.
    root_path_indices: list[list[int]]
    contact_events: list[ContactEvent]
    # Preserve original geometry and separately labelled metric/body_ratio measures.
    physical: PivotSeries | None
    scene_bounds: SceneBounds | None
    reasons: list[str]

    @model_validator(mode="after")
    def links(self) -> GroundViewSeries:
        times = [p.global_seconds for p in self.root_trajectory]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("root trajectory requires increasing global times")
        if any(
            p.id != f"{self.reconstruction_id}:body_root:{i}"
            for i, p in enumerate(self.root_trajectory)
        ):
            raise ValueError("root selection IDs disagree with native source")
        if [p.motion_sample_index for p in self.root_trajectory] != list(
            range(len(times))
        ):
            raise ValueError("root trajectory must retain native sample indices")
        if self.root_path_indices != _root_runs(self.root_trajectory, self.config):
            raise ValueError("root path must preserve gaps")
        if self.ground_status == "unavailable":
            if (
                self.physical
                or self.contact_events
                or self.scene_bounds
                or any(p.xy_ground is not None for p in self.root_trajectory)
            ):
                raise ValueError("unavailable ground cannot expose projected geometry")
            return self
        if self.physical is None or self.ground_id is None:
            raise ValueError("available ground requires physical evidence")
        placements = self.physical.placements
        if (
            placements.reconstruction_id != self.reconstruction_id
            or placements.calibration_id != self.calibration_id
            or placements.world_unit != self.world_unit
            or len(placements.trajectory) != 2 * len(times)
            or len(self.contact_events) != len(times)
        ):
            raise ValueError("ground-view source identities or native times disagree")
        for i, event in enumerate(self.contact_events):
            rows = placements.trajectory[2 * i : 2 * i + 2]
            if (
                event.id != f"{placements.contact_id}:sample:{i}"
                or event.motion_sample_index != i
                or event.foot_trajectory_indices != (2 * i, 2 * i + 1)
                or event.sample.global_seconds != times[i]
                or [r.foot for r in rows] != ["left", "right"]
                or any(
                    r.global_seconds != times[i]
                    or r.motion_sample_index != i
                    or r.contact_sample_index != i
                    or r.support != event.sample.support
                    or r.contact
                    != (event.sample.left if r.foot == "left" else event.sample.right)
                    for r in rows
                )
            ):
                raise ValueError("ground-view contact links disagree with trajectory")
        if self.scene_bounds != _bounds(self.root_trajectory, self.physical):
            raise ValueError("scene bounds disagree with projected geometry")
        return self

    def query(self, global_seconds: float) -> DynamicView:
        """Select a preceding native snapshot; never infer between-sample physics."""
        if not math.isfinite(global_seconds):
            raise ValueError("finite global time required")
        answer = DynamicView(requested_seconds=global_seconds, status="unavailable")
        if self.ground_status == "unavailable":
            answer.reasons = ["ground_unavailable"]
            return answer
        times = [p.global_seconds for p in self.root_trajectory]
        if not times or global_seconds < times[0] or global_seconds > times[-1]:
            answer.reasons = ["outside_execution"]
            return answer
        i = bisect_right(times, global_seconds) - 1
        exact = times[i] == global_seconds
        if not exact:
            answer.bracket_seconds = (times[i], times[i + 1])
            if times[i + 1] - times[i] > self.config.max_gap_seconds:
                answer.reasons = ["native_time_gap"]
                return answer
        assert self.physical is not None
        answer.status = "native" if exact else "native_snapshot"
        answer.sampled_seconds = times[i]
        answer.root = self.root_trajectory[i].model_copy(deep=True)
        answer.feet = [
            r.model_copy(deep=True)
            for r in self.physical.placements.trajectory[2 * i : 2 * i + 2]
        ]
        answer.contact_event = self.contact_events[i].model_copy(deep=True)
        answer.placement_ids = sorted({r.event_id for r in answer.feet if r.event_id})
        answer.pivot_ids = [
            e.pivot.id
            for e in self.physical.events
            if 2 * i + (0 if e.pivot.foot == "left" else 1) in e.rotation_indices
        ]
        answer.reasons = [] if exact else ["display_snapshot_not_interpolated_physics"]
        return answer


def _root_runs(points: list[RootPoint], config: GroundViewConfig) -> list[list[int]]:
    runs: list[list[int]] = []
    for i, point in enumerate(points):
        if point.xy_ground is None:
            continue
        if (
            not runs
            or runs[-1][-1] != i - 1
            or point.global_seconds - points[i - 1].global_seconds
            > config.max_gap_seconds
        ):
            runs.append([])
        runs[-1].append(i)
    return runs


def _bounds(points: list[RootPoint], physical: PivotSeries) -> SceneBounds | None:
    xy = [p.xy_ground for p in points if p.xy_ground is not None]
    z = [p.z_ground for p in points if p.z_ground is not None]
    geometries = [r.geometry for r in physical.placements.trajectory] + [
        e.geometry for e in physical.placements.events
    ]
    xy.extend(
        r.xy_ground for r in physical.placements.trajectory if r.xy_ground is not None
    )
    xy.extend(
        e.footprint.xy_ground
        for e in physical.placements.events
        if e.footprint.xy_ground is not None
    )
    for geometry in geometries:
        xy.extend(geometry.supported_points)
        xy.extend(geometry.polygon or [])
        xy.extend(p for p in (geometry.axis_start, geometry.axis_end) if p is not None)
    for event in physical.events:
        for translation in event.translations:
            xy.extend(translation.positions)
    if not xy:
        return None
    return SceneBounds(
        minimum_xy=(min(p[0] for p in xy), min(p[1] for p in xy)),
        maximum_xy=(max(p[0] for p in xy), max(p[1] for p in xy)),
        minimum_z=min(z) if z else None,
        maximum_z=max(z) if z else None,
    )


def derive_ground_view(
    motion: Reconstruction,
    calibration: Calibration,
    physical: PivotSeries | None = None,
    *,
    ground_id: str | None = None,
    config: GroundViewConfig | None = None,
) -> GroundViewSeries:
    """Project already calibrated world motion; no source-frame transform or parser."""
    source = Reconstruction.model_validate(motion.model_dump())
    cal = Calibration.model_validate(calibration.model_dump())
    config = GroundViewConfig.model_validate(
        (config or GroundViewConfig()).model_dump()
    )
    if source.provenance.producer != "reconstruction.temporal":
        raise ValueError("final temporal reconstruction required")
    if source.calibration_id != cal.id or source.scale != cal.scale:
        raise ValueError("calibration identity or scale mismatch")
    available = cal.ground_status == "resolved" and cal.quality.state != "unknown"
    if available and (physical is None or ground_id is None):
        raise ValueError("resolved ground requires pivot/placement evidence and ID")
    if not available and physical is not None:
        raise ValueError(
            "unavailable ground cannot consume projected physical evidence"
        )
    physical = PivotSeries.model_validate(physical.model_dump()) if physical else None
    root = []
    for i, sample in enumerate(source.samples):
        # Pelvis is the fallback only when reconstructed root is absent, never feet.
        pelvis = next((p for p in sample.landmarks if p.name == "pelvis"), None)
        xyz, quality = sample.root_xyz_world, sample.quality
        reasons = ["root"]
        if xyz is None and pelvis is not None:
            xyz, quality = pelvis.xyz_world, pelvis.quality
            reasons = ["pelvis_fallback"]
        if not available or xyz is None or quality.state == "unknown":
            xyz = None
            quality = Quality(state="unknown", source_ids=quality.source_ids)
            reasons = ["ground_unavailable" if not available else "root_unavailable"]
        root.append(
            RootPoint(
                id=f"{source.id}:body_root:{i}",
                motion_sample_index=i,
                global_seconds=sample.global_seconds,
                xy_ground=(xyz[0], xyz[1]) if xyz else None,
                z_ground=xyz[2] if xyz else None,
                quality=quality.model_copy(deep=True),
                reasons=reasons,
            )
        )
    events = []
    if physical:
        for i, sample in enumerate(source.samples):
            rows = physical.placements.trajectory[2 * i : 2 * i + 2]
            if len(rows) != 2:
                raise ValueError("ground-view native times disagree")
            events.append(
                ContactEvent(
                    id=f"{physical.placements.contact_id}:sample:{i}",
                    motion_sample_index=i,
                    sample=GroundSample(
                        global_seconds=sample.global_seconds,
                        left=rows[0].contact,
                        right=rows[1].contact,
                        support=rows[0].support,
                    ),
                    foot_trajectory_indices=(2 * i, 2 * i + 1),
                )
            )
    return GroundViewSeries(
        reconstruction_id=source.id,
        calibration_id=cal.id,
        participant_id=source.participant_id,
        ground_id=ground_id if available else None,
        ground_status="available" if available else "unavailable",
        world_unit=cal.world_unit,
        ground_source_ids=cal.ground_frame.evidence_ids if cal.ground_frame else [],
        config=config,
        root_trajectory=root,
        root_path_indices=_root_runs(root, config),
        contact_events=events,
        physical=physical,
        scene_bounds=_bounds(root, physical) if physical else None,
        reasons=[] if available else ["ground_unavailable"],
    )
