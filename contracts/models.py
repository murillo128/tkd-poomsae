"""Canonical motion metadata. Dense values are described here, never stored here."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

VERSION = "1.0.0"
Track: TypeAlias = Literal[
    "left_arm", "right_arm", "left_leg", "right_leg", "body_root", "head"
]
EvidenceState: TypeAlias = Literal["observed", "interpolated", "inferred", "unknown"]
Landmark: TypeAlias = Literal[
    "pelvis",
    "spine",
    "neck",
    "head",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_forefoot",
    "right_forefoot",
    "left_foot_outer",
    "right_foot_outer",
    "left_thumb_cmc",
    "left_thumb_mcp",
    "left_thumb_ip",
    "left_thumb_tip",
    "left_index_mcp",
    "left_index_pip",
    "left_index_dip",
    "left_index_tip",
    "left_middle_mcp",
    "left_middle_pip",
    "left_middle_dip",
    "left_middle_tip",
    "left_ring_mcp",
    "left_ring_pip",
    "left_ring_dip",
    "left_ring_tip",
    "left_pinky_mcp",
    "left_pinky_pip",
    "left_pinky_dip",
    "left_pinky_tip",
    "right_thumb_cmc",
    "right_thumb_mcp",
    "right_thumb_ip",
    "right_thumb_tip",
    "right_index_mcp",
    "right_index_pip",
    "right_index_dip",
    "right_index_tip",
    "right_middle_mcp",
    "right_middle_pip",
    "right_middle_dip",
    "right_middle_tip",
    "right_ring_mcp",
    "right_ring_pip",
    "right_ring_dip",
    "right_ring_tip",
    "right_pinky_mcp",
    "right_pinky_pip",
    "right_pinky_dip",
    "right_pinky_tip",
]
BodyEntity: TypeAlias = (
    Landmark
    | Literal[
        "root",
        "torso",
        "left_upper_arm",
        "right_upper_arm",
        "left_forearm",
        "right_forearm",
        "left_hand",
        "right_hand",
        "left_thigh",
        "right_thigh",
        "left_shank",
        "right_shank",
        "left_foot",
        "right_foot",
    ]
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Provenance(StrictModel):
    producer: str = Field(min_length=1)
    model: str | None = None
    model_version: str | None = None
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactBase(StrictModel):
    id: str = Field(min_length=1)
    schema_version: Literal["1.0.0"]
    provenance: Provenance


class Interval(StrictModel):
    start: float
    end: float

    @model_validator(mode="after")
    def ordered(self) -> Interval:
        if self.end <= self.start:
            raise ValueError("interval end must exceed start")
        return self


class Quality(StrictModel):
    score: float | None = Field(default=None, ge=0, le=1)
    uncertainty: float | None = Field(default=None, ge=0)
    state: EvidenceState
    source_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unknown_has_no_score(self) -> Quality:
        if self.state == "unknown" and self.score is not None:
            raise ValueError("unknown quality cannot have a score")
        return self


class DenseArray(StrictModel):
    id: str = Field(min_length=1)
    dtype: Literal["float32", "float64", "int32", "int64", "uint8", "bool"]
    shape: list[int] = Field(min_length=1)
    axes: list[str]
    unit: str | None = None
    missing_mask_id: str | None = None

    @model_validator(mode="after")
    def valid_shape(self) -> DenseArray:
        if any(n <= 0 for n in self.shape) or len(self.shape) != len(self.axes):
            raise ValueError("positive shape and one axis per dimension required")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("array axes must be unique")
        return self


class Project(ArtifactBase):
    kind: Literal["project"]
    source_ids: list[str] = Field(min_length=2)
    participant_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> Project:
        if len(set(self.source_ids)) != len(self.source_ids) or len(
            set(self.participant_ids)
        ) != len(self.participant_ids):
            raise ValueError("project source and participant IDs must be unique")
        return self


class Source(ArtifactBase):
    kind: Literal["source"]
    camera_id: str = Field(min_length=1)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    time_base_num: int = Field(gt=0)
    time_base_den: int = Field(gt=0)


class SyncOffset(StrictModel):
    source_id: str
    automatic_seconds: float | None
    manual_correction_seconds: float | None = None
    manual_seconds: float | None = None
    manual_author: str | None = None
    manual_source: str | None = None
    manual_reason: str | None = None
    timing_reference: bool = False
    retained: bool = True
    exclusion_reason: str | None = None
    source_interval: Interval | None = None
    global_interval: Interval | None = None
    quality: Quality

    @property
    def effective_seconds(self) -> float:
        if self.manual_seconds is not None:
            return self.manual_seconds
        if self.timing_reference and self.automatic_seconds is None:
            return 0.0
        if self.automatic_seconds is None:
            raise ValueError("excluded source has no effective offset")
        return self.automatic_seconds + (self.manual_correction_seconds or 0.0)


class SyncPairEstimate(StrictModel):
    first: str
    second: str
    shift_seconds: float | None
    score: float
    peak_separation: float
    overlap_seconds: float
    window_scores: list[float]
    cue_kinds: list[str]
    reliable: bool
    diagnostics: list[str]


class Synchronization(ArtifactBase):
    kind: Literal["synchronization"]
    offsets: list[SyncOffset] = Field(min_length=2)
    reference_source_id: str | None = None
    common_interval: Interval | None = None
    pair_estimates: list[SyncPairEstimate] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_sources(self) -> Synchronization:
        if len({offset.source_id for offset in self.offsets}) != len(self.offsets):
            raise ValueError("synchronization source IDs must be unique")
        return self


class FrameTime(StrictModel):
    source_id: str
    camera_id: str
    frame_index: int | None = Field(default=None, ge=0)
    pts: int | None = None
    time_base_num: int | None = Field(default=None, gt=0)
    time_base_den: int | None = Field(default=None, gt=0)
    source_seconds: float
    offset_seconds: float
    global_seconds: float

    @model_validator(mode="after")
    def consistent_time(self) -> FrameTime:
        if (self.pts is None) != (self.time_base_num is None) or (
            (self.pts is None) != (self.time_base_den is None)
        ):
            raise ValueError("PTS and both time-base integers must occur together")
        if (
            self.pts is not None
            and self.time_base_num is not None
            and self.time_base_den is not None
        ):
            if (
                abs(
                    self.source_seconds
                    - self.pts * self.time_base_num / self.time_base_den
                )
                > 1e-9
            ):
                raise ValueError("source seconds disagree with native PTS")
        if abs(self.global_seconds - self.source_seconds - self.offset_seconds) > 1e-9:
            raise ValueError("global time must equal source time plus offset")
        return self


class Intrinsics(StrictModel):
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    distortion: list[float] = Field(default_factory=list)


class CameraCalibration(StrictModel):
    camera_id: str
    source_id: str
    intrinsics: Intrinsics
    # Row-major homogeneous transform: camera_xyz = world_to_camera * world_xyz.
    world_to_camera: list[list[float]]
    quality: Quality
    intrinsic_source: Literal["estimated", "imported"] | None = None
    capture_ids: list[str] = Field(default_factory=list)
    corner_count: int | None = Field(default=None, ge=4)
    rms_reprojection_px: float | None = Field(default=None, ge=0)
    pose_ambiguity_px: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def matrix_shape(self) -> CameraCalibration:
        m = self.world_to_camera
        if len(m) != 4 or any(len(row) != 4 for row in m):
            raise ValueError("world_to_camera must be 4x4")
        if m[3] != [0.0, 0.0, 0.0, 1.0]:
            raise ValueError("world_to_camera must be homogeneous")
        return self


class GroundFrame(StrictModel):
    # Coordinates of the original camera-solve gauge, before world alignment.
    source_to_world: list[list[float]]
    plane_normal_source: tuple[float, float, float]
    plane_offset_source: float
    inlier_count: int = Field(ge=3)
    sample_count: int = Field(ge=3)
    inlier_indices: list[int] = Field(default_factory=list)
    coverage: float = Field(ge=0)
    rms_residual: float = Field(ge=0)
    normal_uncertainty_rad: float = Field(ge=0)
    axis_uncertainty_rad: float = Field(ge=0)
    evidence_kind: Literal["target", "scene", "manual"]
    evidence_ids: list[str] = Field(min_length=1)
    evidence_producer: str = Field(min_length=1)
    evidence_author: str | None = None
    evidence_reason: str | None = None
    source_revision: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_frame(self) -> GroundFrame:
        if self.inlier_count > self.sample_count:
            raise ValueError("ground inliers exceed samples")
        if self.inlier_indices and (
            len(set(self.inlier_indices)) != self.inlier_count
            or min(self.inlier_indices) < 0
        ):
            raise ValueError("ground inlier indices disagree with count")
        m = self.source_to_world
        if len(m) != 4 or any(len(row) != 4 for row in m):
            raise ValueError("source_to_world must be 4x4")
        if m[3] != [0.0, 0.0, 0.0, 1.0]:
            raise ValueError("source_to_world must be homogeneous")
        basis = np.asarray(m, dtype=np.float64)[:3, :3]
        lengths = np.linalg.norm(basis, axis=1)
        if (
            np.linalg.det(basis) <= 0
            or not np.allclose(lengths, lengths[0])
            or not np.allclose(basis @ basis.T, np.eye(3) * lengths[0] ** 2)
        ):
            raise ValueError("source_to_world must preserve handedness and scale")
        if self.evidence_kind == "manual" and (
            not self.evidence_author or not self.evidence_reason
        ):
            raise ValueError("manual ground requires author and reason")
        return self


class ScaleResolution(StrictModel):
    evidence_id: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    kind: Literal["target", "measured", "manual"]
    measured_length: float = Field(gt=0)
    measured_unit: Literal["m", "cm"]
    metres_per_source_unit: float = Field(gt=0)
    producer: str = Field(min_length=1)
    author: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def manual_provenance(self) -> ScaleResolution:
        if self.kind == "manual" and (not self.author or not self.reason):
            raise ValueError("manual scale requires author and reason")
        return self


class Calibration(ArtifactBase):
    kind: Literal["calibration"]
    scale: Literal["metric", "arbitrary"]
    world_unit: Literal["m", "arbitrary"]
    cameras: list[CameraCalibration] = Field(min_length=2)
    ground_z: float | None = None
    ground_status: Literal["resolved", "unresolved"] = "unresolved"
    scale_status: Literal["resolved", "unresolved"] = "unresolved"
    ground_frame: GroundFrame | None = None
    scale_evidence_ids: list[str] = Field(default_factory=list)
    scale_resolution: ScaleResolution | None = None
    source_revision: str | None = None
    camera_status: Literal["resolved", "unresolved"] = "unresolved"
    publication_status: Literal["complete", "partial"] = "partial"
    excluded_cameras: dict[str, str] = Field(default_factory=dict)
    quality_flags: list[str] = Field(default_factory=list)
    projection_debug: list[dict[str, Any]] = Field(default_factory=list)
    evidence_links: list[str] = Field(default_factory=list)
    quality: Quality

    @model_validator(mode="after")
    def consistent_scale(self) -> Calibration:
        if set(self.excluded_cameras) & {camera.camera_id for camera in self.cameras}:
            raise ValueError("excluded cameras cannot also be retained")
        if (self.scale == "metric") != (self.world_unit == "m"):
            raise ValueError(
                "metric scale requires metres; unresolved scale is arbitrary"
            )
        if (self.scale_status == "resolved") != (self.scale == "metric"):
            raise ValueError("scale status and units disagree")
        if self.publication_status == "complete" and (
            self.camera_status != "resolved"
            or self.ground_status != "resolved"
            or self.scale_status != "resolved"
        ):
            raise ValueError("complete calibration requires camera, ground and scale")
        if self.scale == "metric" and not self.scale_evidence_ids:
            raise ValueError("metric scale requires known-size evidence")
        if self.scale == "metric" and (
            self.scale_resolution is None
            or self.scale_resolution.evidence_id not in self.scale_evidence_ids
        ):
            raise ValueError("metric scale requires measurement provenance")
        if self.scale == "arbitrary" and self.scale_resolution is not None:
            raise ValueError("unresolved scale cannot claim a measurement")
        if self.ground_status == "resolved":
            if self.ground_frame is None or self.ground_z != 0.0:
                raise ValueError("resolved ground requires a frame at z=0")
        elif self.ground_frame is not None or self.ground_z is not None:
            raise ValueError("unresolved ground cannot claim a ground frame")
        if self.source_revision is not None:
            if (
                self.ground_frame is not None
                and self.ground_frame.source_revision != self.source_revision
            ):
                raise ValueError("ground source revision mismatch")
            if (
                self.scale_resolution is not None
                and self.scale_resolution.source_revision != self.source_revision
            ):
                raise ValueError("scale source revision mismatch")
        return self


class RawScore(StrictModel):
    value: float
    range_min: float | None
    range_max: float | None
    domain: str | None = None

    @model_validator(mode="after")
    def in_range(self) -> RawScore:
        if (self.range_min is None or self.range_max is None) and not self.domain:
            raise ValueError("unbounded raw score requires its model score domain")
        if (
            (self.range_min is not None and self.value < self.range_min)
            or (self.range_max is not None and self.value > self.range_max)
            or (
                self.range_min is not None
                and self.range_max is not None
                and self.range_min >= self.range_max
            )
        ):
            raise ValueError("raw score outside declared original range")
        return self


class Landmark2D(StrictModel):
    name: Landmark
    xy_px: tuple[float, float] | None
    raw_score: RawScore | None = None
    raw_visibility: float | None = None
    quality: Quality

    @model_validator(mode="after")
    def geometry_state(self) -> Landmark2D:
        if (self.xy_px is None) != (self.quality.state == "unknown"):
            raise ValueError("unknown landmark must have null geometry, and vice versa")
        return self


class RegionOfInterest(StrictModel):
    part: Literal["left_hand", "right_hand", "left_foot", "right_foot", "head"]
    xywh_px: tuple[float, float, float, float]

    @model_validator(mode="after")
    def positive_extent(self) -> RegionOfInterest:
        if self.xywh_px[2] <= 0 or self.xywh_px[3] <= 0:
            raise ValueError("ROI width and height must be positive")
        return self


class RegionalGeometry2D(StrictModel):
    """View-space evidence, independent for each detailed foot/head region."""

    part: Literal["left_foot", "right_foot", "head"]
    availability: Literal["complete", "partial", "missing"]
    orientation_state: Literal["available", "degenerate", "unavailable"]
    provider: str = Field(min_length=1)
    supporting_landmarks: list[Landmark] = Field(default_factory=list)
    axis_start_px: tuple[float, float] | None = None
    axis_end_px: tuple[float, float] | None = None
    orientation_rad: float | None = None

    @model_validator(mode="after")
    def consistent_geometry(self) -> RegionalGeometry2D:
        if (self.axis_start_px is None) != (self.axis_end_px is None):
            raise ValueError("regional axis requires both endpoints")
        if self.availability == "missing" and (
            self.axis_start_px is not None or self.orientation_rad is not None
        ):
            raise ValueError("missing region cannot have geometry")
        if self.orientation_state != "available" and self.orientation_rad is not None:
            raise ValueError("unavailable region cannot have orientation")
        if self.orientation_state == "available" and (
            self.axis_start_px is None or self.orientation_rad is None
        ):
            raise ValueError("available region needs axis and orientation")
        if self.orientation_state in ("available", "degenerate") and (
            self.availability != "complete"
        ):
            raise ValueError("orientation assessment needs complete landmarks")
        if len(set(self.supporting_landmarks)) != len(self.supporting_landmarks):
            raise ValueError("regional supporting landmarks must be unique")
        return self


class SubjectCandidateEvidence(StrictModel):
    index: int = Field(ge=0)
    bbox_xyxy_px: tuple[float, float, float, float]
    detector_score: RawScore
    match_cost: float | None = Field(default=None, ge=0)


class SubjectSelection(StrictModel):
    state: Literal["selected", "ambiguous", "missing"]
    track_id: str | None = None
    candidate_index: int | None = Field(default=None, ge=0)
    method: Literal["initial", "temporal", "operator"] | None = None
    reasons: list[str] = Field(default_factory=list)
    candidates: list[SubjectCandidateEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_selection(self) -> SubjectSelection:
        if (self.state == "selected") != (self.candidate_index is not None):
            raise ValueError("selected subject requires exactly one candidate")
        if (self.state == "selected") != (self.method is not None):
            raise ValueError("selected subject requires a selection method")
        if self.candidate_index is not None and self.candidate_index not in {
            item.index for item in self.candidates
        }:
            raise ValueError("selected candidate must have evidence")
        if len({item.index for item in self.candidates}) != len(self.candidates):
            raise ValueError("candidate evidence indices must be unique")
        return self


class ViewRegionQuality(StrictModel):
    part: Literal["body", "left_hand", "right_hand", "left_foot", "right_foot", "head"]
    usable: bool
    reasons: list[str] = Field(default_factory=list)


class Observation(ArtifactBase):
    kind: Literal["observation"]
    frame: FrameTime
    landmarks: list[Landmark2D]
    wholebody_landmarks: list[Landmark2D] = Field(default_factory=list)
    refined_landmarks: list[Landmark2D] = Field(default_factory=list)
    regions: list[RegionOfInterest] = Field(default_factory=list)
    regional_geometry: list[RegionalGeometry2D] = Field(default_factory=list)
    source_regional_geometry: list[RegionalGeometry2D] = Field(default_factory=list)
    subject_selection: SubjectSelection | None = None
    region_quality: list[ViewRegionQuality] = Field(default_factory=list)
    arrays: list[DenseArray] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_landmarks(self) -> Observation:
        names = {landmark.name for landmark in self.landmarks}
        if len(names) != len(self.landmarks):
            raise ValueError("observation landmark names must be unique")
        for points in (self.wholebody_landmarks, self.refined_landmarks):
            if len({point.name for point in points}) != len(points):
                raise ValueError("observation source landmark names must be unique")
        if len({region.part for region in self.regional_geometry}) != len(
            self.regional_geometry
        ):
            raise ValueError("observation regional parts must be unique")
        if len({region.part for region in self.source_regional_geometry}) != len(
            self.source_regional_geometry
        ):
            raise ValueError("source regional parts must be unique")
        if any(
            not set(region.supporting_landmarks) <= names
            for region in self.regional_geometry
        ):
            raise ValueError("regional supporting landmarks must be persisted")
        if len({region.part for region in self.region_quality}) != len(
            self.region_quality
        ):
            raise ValueError("observation region quality parts must be unique")
        return self


class Alignment(ArtifactBase):
    """Derived time queries; the versioned payload retains native observations."""

    kind: Literal["alignment"]
    synchronization_id: str
    observation_digests: list[str] = Field(min_length=1)
    query_count: int = Field(gt=0)
    arrays: list[DenseArray]


class Landmark3D(StrictModel):
    name: Landmark
    xyz_world: tuple[float, float, float] | None
    quality: Quality

    @model_validator(mode="after")
    def geometry_state(self) -> Landmark3D:
        if (self.xyz_world is None) != (self.quality.state == "unknown"):
            raise ValueError("unknown landmark must have null geometry, and vice versa")
        return self


class Quaternion(StrictModel):
    # Active orientation [w, x, y, z] of a local frame relative to its parent.
    wxyz: tuple[float, float, float, float]

    @model_validator(mode="after")
    def unit_length(self) -> Quaternion:
        if abs(sum(v * v for v in self.wxyz) - 1) > 1e-5:
            raise ValueError("orientation quaternion must have unit length")
        return self


class SegmentFrame(StrictModel):
    segment: str
    parent: Literal[
        "world",
        "root",
        "torso",
        "left_forearm",
        "right_forearm",
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot",
        "head",
    ]
    orientation: Quaternion | None
    quality: Quality


class Reconstruction(ArtifactBase):
    kind: Literal["reconstruction"]
    calibration_id: str
    participant_id: str
    scale: Literal["metric", "arbitrary"]
    samples: list[MotionSample]
    arrays: list[DenseArray] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered_samples(self) -> Reconstruction:
        times = [sample.global_seconds for sample in self.samples]
        if any(next_time <= time for time, next_time in zip(times, times[1:])):
            raise ValueError("reconstruction samples must have increasing global time")
        for sample in self.samples:
            if len({landmark.name for landmark in sample.landmarks}) != len(
                sample.landmarks
            ):
                raise ValueError("reconstruction landmark names must be unique")
        return self


class MotionSample(StrictModel):
    global_seconds: float
    root_xyz_world: tuple[float, float, float] | None
    root_orientation: Quaternion | None
    landmarks: list[Landmark3D]
    segments: list[SegmentFrame] = Field(default_factory=list)
    quality: Quality


Reconstruction.model_rebuild()


class Measurement(StrictModel):
    name: str
    value: float | None
    unit: Literal["m", "arbitrary", "body_ratio", "deg", "rad"]
    quality: Quality

    @model_validator(mode="after")
    def missing_value(self) -> Measurement:
        if (self.value is None) != (self.quality.state == "unknown"):
            raise ValueError("unknown measurement must have null value, and vice versa")
        return self


class Morphology(ArtifactBase):
    kind: Literal["morphology"]
    participant_id: str
    measurements: list[Measurement]
    arrays: list[DenseArray] = Field(default_factory=list)


class Contact(StrictModel):
    state: Literal["contact", "no_contact", "unknown"]
    region: Literal["heel", "forefoot", "flat"] | None = None
    quality: Quality

    @model_validator(mode="after")
    def region_only_on_contact(self) -> Contact:
        if self.state != "contact" and self.region is not None:
            raise ValueError("contact region requires known contact")
        if (self.state == "unknown") != (self.quality.state == "unknown"):
            raise ValueError("contact and quality unknown states must agree")
        return self


class GroundSample(StrictModel):
    global_seconds: float
    left: Contact
    right: Contact
    support: Literal["both", "left", "right", "neither", "unknown"]

    @model_validator(mode="after")
    def support_matches_contact(self) -> GroundSample:
        if "unknown" in (self.left.state, self.right.state):
            if self.support != "unknown":
                raise ValueError("unknown foot contact requires unknown support")
        else:
            expected = {
                ("contact", "contact"): "both",
                ("contact", "no_contact"): "left",
                ("no_contact", "contact"): "right",
                ("no_contact", "no_contact"): "neither",
            }[(self.left.state, self.right.state)]
            if self.support != expected:
                raise ValueError("support disagrees with foot contact")
        return self


class Footprint(StrictModel):
    id: str
    foot: Literal["left", "right"]
    interval: Interval
    xy_ground: tuple[float, float] | None
    yaw_rad: float | None
    quality: Quality


class Pivot(StrictModel):
    id: str
    foot: Literal["left", "right"]
    interval: Interval
    region: Literal["heel", "forefoot", "flat", "unknown"]
    rotation_rad: float | None
    quality: Quality


class Ground(ArtifactBase):
    kind: Literal["ground"]
    reconstruction_id: str
    scale: Literal["metric", "arbitrary"]
    samples: list[GroundSample]
    footprints: list[Footprint] = Field(default_factory=list)
    pivots: list[Pivot] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    arrays: list[DenseArray] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered_samples(self) -> Ground:
        times = [sample.global_seconds for sample in self.samples]
        if any(next_time <= time for time, next_time in zip(times, times[1:])):
            raise ValueError("ground samples must have increasing global time")
        return self


class MotionFeatures(ArtifactBase):
    """Physical features/candidates, before semantic actions and grouping."""

    kind: Literal["motion_features"]
    reconstruction_id: str
    ground_id: str
    arrays: list[DenseArray]


class SequenceStep(StrictModel):
    id: str
    interval: Interval
    action_ids: list[str]


class Segmentation(ArtifactBase):
    """Automatic coarse proposals, including explicitly indeterminate output."""

    kind: Literal["segmentation"]
    reconstruction_id: str
    ground_id: str
    motion_features_id: str
    execution: Interval | None
    steps: list[SequenceStep]
    quality: Quality
    arrays: list[DenseArray]

    @model_validator(mode="after")
    def coarse_topology(self) -> Segmentation:
        if self.execution is None:
            if self.steps or self.quality.state != "unknown":
                raise ValueError("indeterminate segmentation cannot propose steps")
        else:
            if not self.steps or self.quality.state != "inferred":
                raise ValueError("resolved segmentation requires inferred steps")
            edges = [self.execution.start]
            for step in self.steps:
                if step.interval.start != edges[-1] or step.action_ids:
                    raise ValueError("coarse steps must tile execution without actions")
                edges.append(step.interval.end)
            if edges[-1] != self.execution.end or len(
                {s.id for s in self.steps}
            ) != len(self.steps):
                raise ValueError("coarse step topology disagrees with execution")
        return self


class ArmActions(ArtifactBase):
    """Automatic upper-body proposals and evidence, before final assembly."""

    kind: Literal["arm_actions"]
    reconstruction_id: str
    ground_id: str
    motion_features_id: str
    segmentation_id: str
    arrays: list[DenseArray]


class StanceState(StrictModel):
    id: str
    interval: Interval
    label: str = Field(min_length=1)
    quality: Quality


class Action(StrictModel):
    id: str
    interval: Interval
    step_id: str
    tracks: list[Track] = Field(min_length=1)
    category: Literal[
        "arm",
        "placement",
        "pivot",
        "kick",
        "stance_transition",
        "special",
        "transition",
    ]
    role: Literal["attack", "defense", "preparation", "special", "unknown"] = "unknown"


class Phase(StrictModel):
    id: str
    action_id: str
    interval: Interval
    name: str


class Keyframe(StrictModel):
    id: str
    action_id: str
    phase_id: str | None = None
    global_seconds: float
    event: str


class SpatialRelation(StrictModel):
    id: str
    subject: BodyEntity
    object: BodyEntity
    relation: Literal[
        "in_front_of", "behind", "above", "below", "left_of", "right_of", "crossed"
    ]
    interval: Interval
    front_entity: BodyEntity | None = None
    quality: Quality

    @model_validator(mode="after")
    def crossing_order(self) -> SpatialRelation:
        if self.front_entity is not None and (
            self.relation != "crossed"
            or self.front_entity not in (self.subject, self.object)
        ):
            raise ValueError("crossing front entity must be one relation endpoint")
        return self


class Semantics(ArtifactBase):
    kind: Literal["semantics"]
    reconstruction_id: str
    ground_id: str
    execution: Interval
    steps: list[SequenceStep]
    stances: list[StanceState]
    actions: list[Action]
    phases: list[Phase]
    keyframes: list[Keyframe]
    relations: list[SpatialRelation]

    @model_validator(mode="after")
    def valid_hierarchy(self) -> Semantics:
        groups: list[list[Any]] = [
            self.steps,
            self.stances,
            self.actions,
            self.phases,
            self.keyframes,
            self.relations,
        ]
        ids = [item.id for group in groups for item in group]
        if len(ids) != len(set(ids)):
            raise ValueError("semantic IDs must be unique")
        steps = {x.id: x for x in self.steps}
        actions = {x.id: x for x in self.actions}
        phases = {x.id: x for x in self.phases}

        def inside(inner: Interval, outer: Interval) -> bool:
            return outer.start <= inner.start and inner.end <= outer.end

        for step in self.steps:
            if not inside(step.interval, self.execution):
                raise ValueError("step outside execution")
            if set(step.action_ids) != {
                a.id for a in self.actions if a.step_id == step.id
            }:
                raise ValueError("step action references disagree")
        for stance in self.stances:
            if not inside(stance.interval, self.execution):
                raise ValueError("stance outside execution")
        for action in self.actions:
            if action.step_id not in steps or not inside(
                action.interval, steps[action.step_id].interval
            ):
                raise ValueError("action outside or missing step")
        for phase in self.phases:
            if phase.action_id not in actions or not inside(
                phase.interval, actions[phase.action_id].interval
            ):
                raise ValueError("phase outside or missing action")
        for keyframe in self.keyframes:
            keyframe_action = actions.get(keyframe.action_id)
            if (
                keyframe_action is None
                or not keyframe_action.interval.start
                <= keyframe.global_seconds
                <= keyframe_action.interval.end
            ):
                raise ValueError("keyframe outside or missing action")
            if keyframe.phase_id is not None:
                keyframe_phase = phases.get(keyframe.phase_id)
                if (
                    keyframe_phase is None
                    or keyframe_phase.action_id != keyframe.action_id
                    or not keyframe_phase.interval.start
                    <= keyframe.global_seconds
                    <= keyframe_phase.interval.end
                ):
                    raise ValueError("keyframe outside or missing phase")
        for relation in self.relations:
            if not inside(relation.interval, self.execution):
                raise ValueError("relation outside execution")
        return self


class ManualEdit(StrictModel):
    id: str
    target_id: str
    field_path: str = Field(pattern=r"^/[A-Za-z0-9_/-]+$")
    replacement: Any
    author: str
    reason: str | None = None

    @model_validator(mode="after")
    def json_replacement(self) -> ManualEdit:
        try:
            json.dumps(self.replacement, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("edit replacement must be finite JSON") from error
        return self


class ManualEdits(ArtifactBase):
    kind: Literal["manual_edits"]
    automatic_semantics_id: str
    edits: list[ManualEdit]


Artifact: TypeAlias = Annotated[
    Project
    | Source
    | Synchronization
    | Calibration
    | Observation
    | Alignment
    | Reconstruction
    | Morphology
    | Ground
    | MotionFeatures
    | Segmentation
    | ArmActions
    | Semantics
    | ManualEdits,
    Field(discriminator="kind"),
]
_adapter: TypeAdapter[Artifact] = TypeAdapter(Artifact)
_ArtifactT = TypeVar("_ArtifactT", bound=ArtifactBase)


def validate_artifact(data: Any) -> Artifact:
    """Reject malformed or unsupported versioned metadata."""
    return _adapter.validate_python(data)


def _qualities(value: object) -> Iterator[Quality]:
    """Find every nested quality field, including future fields in known layers."""
    if isinstance(value, Quality):
        yield value
    elif isinstance(value, BaseModel):
        for field in value.__dict__.values():
            yield from _qualities(field)
    elif isinstance(value, (list, tuple)):
        for entry in value:
            yield from _qualities(entry)
    elif isinstance(value, dict):
        for entry in value.values():
            yield from _qualities(entry)


def validate_bundle(data: list[Any]) -> list[Artifact]:
    """Validate references between artifacts in one portable interchange bundle."""
    artifacts = [validate_artifact(item) for item in data]
    by_id = {item.id: item for item in artifacts}
    if len(by_id) != len(artifacts):
        raise ValueError("artifact IDs must be unique")

    def require(ref: str, cls: type[_ArtifactT]) -> _ArtifactT:
        item = by_id.get(ref)
        if not isinstance(item, cls):
            raise ValueError(f"dangling or wrong-kind reference: {ref}")
        return item

    projects = [a for a in artifacts if isinstance(a, Project)]
    if len(projects) != 1:
        raise ValueError("bundle requires exactly one project")
    project = projects[0]
    for ref in project.source_ids:
        require(ref, Source)
    syncs = [a for a in artifacts if isinstance(a, Synchronization)]
    for sync in syncs:
        if {o.source_id for o in sync.offsets} != set(project.source_ids):
            raise ValueError("sync offsets must cover each project source once")
    offsets = {o.source_id: o for sync in syncs for o in sync.offsets}
    for item in artifacts:
        # Contributor IDs point to the immediate evidence layer, not an arbitrary
        # artifact with a matching string ID. Empty lists mean provenance is only
        # available at the enclosing artifact/frame level.
        if isinstance(item, (Synchronization, Calibration, Observation)):
            allowed_contributors: tuple[type[ArtifactBase], ...] = (Source,)
        elif isinstance(item, Reconstruction):
            allowed_contributors = (Observation,)
        elif isinstance(item, Morphology):
            allowed_contributors = (Reconstruction,)
        elif isinstance(item, Ground):
            allowed_contributors = (Reconstruction,)
        elif isinstance(item, (Semantics, MotionFeatures, Segmentation, ArmActions)):
            allowed_contributors = (Reconstruction, Ground)
        else:
            allowed_contributors = ()
        for quality in _qualities(item):
            for ref in quality.source_ids:
                if not isinstance(by_id.get(ref), allowed_contributors):
                    raise ValueError(f"dangling or wrong-kind quality source: {ref}")
        if isinstance(item, Alignment):
            require(item.synchronization_id, Synchronization)
        elif isinstance(item, Calibration):
            for camera in item.cameras:
                source = require(camera.source_id, Source)
                if source.camera_id != camera.camera_id:
                    raise ValueError("calibration camera/source mismatch")
        elif isinstance(item, Observation):
            source = require(item.frame.source_id, Source)
            if source.camera_id != item.frame.camera_id:
                raise ValueError("observation camera/source mismatch")
            offset = offsets.get(source.id)
            if (
                offset is None
                or abs(offset.effective_seconds - item.frame.offset_seconds) > 1e-9
            ):
                raise ValueError("frame offset disagrees with synchronization")
            if item.frame.pts is not None and (
                source.time_base_num,
                source.time_base_den,
            ) != (item.frame.time_base_num, item.frame.time_base_den):
                raise ValueError("frame native time base disagrees with source")
        elif isinstance(item, Reconstruction):
            calibration = require(item.calibration_id, Calibration)
            if (
                item.participant_id not in project.participant_ids
                or item.scale != calibration.scale
            ):
                raise ValueError("reconstruction participant or scale mismatch")
            if calibration.camera_status != "resolved":
                raise ValueError("reconstruction requires resolved camera geometry")
            if item.scale == "arbitrary" and any(
                array.unit in {"m", "cm"} for array in item.arrays
            ):
                raise ValueError("unresolved scale cannot emit metric arrays")
        elif isinstance(item, Morphology):
            if item.participant_id not in project.participant_ids:
                raise ValueError("morphology participant missing")
        elif isinstance(item, Ground):
            reconstruction = require(item.reconstruction_id, Reconstruction)
            calibration = require(reconstruction.calibration_id, Calibration)
            if calibration.ground_status != "resolved":
                raise ValueError(
                    "ground-dependent output requires resolved calibration ground"
                )
            if item.scale != reconstruction.scale:
                raise ValueError("ground scale mismatch")
            if item.scale == "arbitrary" and any(
                m.unit == "m" for m in item.measurements
            ):
                raise ValueError("unresolved scale cannot emit metres")
            if item.scale == "arbitrary" and any(
                array.unit in {"m", "cm"} for array in item.arrays
            ):
                raise ValueError("unresolved scale cannot emit metric arrays")
        elif isinstance(item, (Semantics, MotionFeatures, Segmentation, ArmActions)):
            require(item.reconstruction_id, Reconstruction)
            ground = require(item.ground_id, Ground)
            if ground.reconstruction_id != item.reconstruction_id:
                raise ValueError("semantic ground and motion references disagree")
            if isinstance(item, (Segmentation, ArmActions)):
                features = require(item.motion_features_id, MotionFeatures)
                if (features.reconstruction_id, features.ground_id) != (
                    item.reconstruction_id,
                    item.ground_id,
                ):
                    raise ValueError("segmentation feature references disagree")
            if isinstance(item, ArmActions):
                segmentation = require(item.segmentation_id, Segmentation)
                if (
                    segmentation.reconstruction_id,
                    segmentation.ground_id,
                    segmentation.motion_features_id,
                ) != (
                    item.reconstruction_id,
                    item.ground_id,
                    item.motion_features_id,
                ):
                    raise ValueError("arm action input references disagree")
        elif isinstance(item, ManualEdits):
            semantics = require(item.automatic_semantics_id, Semantics)
            valid_ids = {semantics.id} | {
                x.id
                for group in (
                    semantics.steps,
                    semantics.stances,
                    semantics.actions,
                    semantics.phases,
                    semantics.keyframes,
                    semantics.relations,
                )
                for x in group
            }
            if any(edit.target_id not in valid_ids for edit in item.edits):
                raise ValueError("manual edit target missing")
        arrays = getattr(item, "arrays", [])
        array_ids = {a.id for a in arrays}
        if len(array_ids) != len(arrays):
            raise ValueError("array IDs must be unique per artifact")
        for array in arrays:
            if array.missing_mask_id is not None:
                mask = next((a for a in arrays if a.id == array.missing_mask_id), None)
                if (
                    mask is None
                    or mask is array
                    or mask.dtype != "bool"
                    or mask.shape != array.shape
                    or mask.axes != array.axes
                ):
                    raise ValueError("missing mask must be a same-shape boolean array")
    return artifacts
