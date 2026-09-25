"""Canonical motion metadata. Dense values are described here, never stored here."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

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


class Calibration(ArtifactBase):
    kind: Literal["calibration"]
    scale: Literal["metric", "arbitrary"]
    world_unit: Literal["m", "arbitrary"]
    cameras: list[CameraCalibration] = Field(min_length=2)
    ground_z: float = 0.0
    quality: Quality

    @model_validator(mode="after")
    def consistent_scale(self) -> Calibration:
        if (self.scale == "metric") != (self.world_unit == "m"):
            raise ValueError(
                "metric scale requires metres; unresolved scale is arbitrary"
            )
        return self


class RawScore(StrictModel):
    value: float
    range_min: float
    range_max: float

    @model_validator(mode="after")
    def in_range(self) -> RawScore:
        if (
            self.range_min >= self.range_max
            or not self.range_min <= self.value <= self.range_max
        ):
            raise ValueError("raw score outside declared original range")
        return self


class Landmark2D(StrictModel):
    name: Landmark
    xy_px: tuple[float, float] | None
    raw_score: RawScore | None = None
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


class Observation(ArtifactBase):
    kind: Literal["observation"]
    frame: FrameTime
    landmarks: list[Landmark2D]
    regions: list[RegionOfInterest] = Field(default_factory=list)
    arrays: list[DenseArray] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_landmarks(self) -> Observation:
        if len({landmark.name for landmark in self.landmarks}) != len(self.landmarks):
            raise ValueError("observation landmark names must be unique")
        return self


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


class SequenceStep(StrictModel):
    id: str
    interval: Interval
    action_ids: list[str]


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
    | Reconstruction
    | Morphology
    | Ground
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
        elif isinstance(item, Semantics):
            allowed_contributors = (Reconstruction, Ground)
        else:
            allowed_contributors = ()
        for quality in _qualities(item):
            for ref in quality.source_ids:
                if not isinstance(by_id.get(ref), allowed_contributors):
                    raise ValueError(f"dangling or wrong-kind quality source: {ref}")
        if isinstance(item, Calibration):
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
        elif isinstance(item, Morphology):
            if item.participant_id not in project.participant_ids:
                raise ValueError("morphology participant missing")
        elif isinstance(item, Ground):
            reconstruction = require(item.reconstruction_id, Reconstruction)
            if item.scale != reconstruction.scale:
                raise ValueError("ground scale mismatch")
            if item.scale == "arbitrary" and any(
                m.unit == "m" for m in item.measurements
            ):
                raise ValueError("unresolved scale cannot emit metres")
        elif isinstance(item, Semantics):
            require(item.reconstruction_id, Reconstruction)
            ground = require(item.ground_id, Ground)
            if ground.reconstruction_id != item.reconstruction_id:
                raise ValueError("semantic ground and motion references disagree")
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
