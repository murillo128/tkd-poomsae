"""Resolve one auditable world frame from target or classified scene evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from calibration.cameras import CameraModel
from calibration.natural import SceneCandidate
from calibration.quality import QualityThresholds, SceneAssessment, assess_scene
from contracts.models import (
    Calibration,
    CameraCalibration,
    GroundFrame,
    Intrinsics,
    Provenance,
    Quality,
    ScaleResolution,
)
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config


def scene_revision(candidate: SceneCandidate) -> str:
    payload = json.dumps(
        candidate.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GroundEvidence:
    id: str
    source_revision: str
    floor_indices: tuple[int, ...]
    above_indices: tuple[int, ...]
    axis_indices: tuple[int, int]
    vertical_indices: tuple[int, int]
    vertical_reference_id: str
    kind: Literal["scene", "manual"] = "scene"
    producer: str = ""
    author: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"scene", "manual"}:
            raise ValueError("unsupported ground evidence kind")
        if not self.id or not self.source_revision or not self.vertical_reference_id:
            raise ValueError("ground evidence requires identity and source revision")
        if len(set(self.floor_indices)) < 6 or not self.above_indices:
            raise ValueError("floor samples and above-floor sign evidence required")
        if (
            len(self.vertical_indices) != 2
            or self.vertical_indices[0] == self.vertical_indices[1]
        ):
            raise ValueError("vertical reference requires distinct ordered endpoints")
        if self.kind == "manual":
            if not self.author or not self.reason:
                raise ValueError("manual ground recovery requires author and reason")
        elif not self.producer:
            raise ValueError("automatic scene classification requires producer")


@dataclass(frozen=True)
class SizeEvidence:
    id: str
    source_revision: str
    point_indices: tuple[int, int]
    length: float
    unit: Literal["m", "cm"]
    kind: Literal["measured", "manual"] = "measured"
    producer: str = ""
    author: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.unit not in {"m", "cm"}:
            raise ValueError("measurement unit must be m or cm")
        if self.kind not in {"measured", "manual"}:
            raise ValueError("unsupported size evidence kind")
        if not self.id or not self.source_revision:
            raise ValueError("size evidence requires identity and source revision")
        if not np.isfinite(self.length) or self.length <= 0:
            raise ValueError("measured dimension must be positive and finite")
        if self.point_indices[0] == self.point_indices[1]:
            raise ValueError("measured endpoints must differ")
        if self.kind == "manual":
            if not self.author or not self.reason:
                raise ValueError("manual scale recovery requires author and reason")
        elif not self.producer:
            raise ValueError("measurement requires documented producer")


def _points(candidate: SceneCandidate, indices: tuple[int, ...]) -> NDArray[np.float64]:
    if any(not 0 <= i < len(candidate.static_points) for i in indices):
        raise ValueError("evidence refers to absent scene point")
    result = np.asarray(
        [candidate.static_points[i]["xyz"] for i in indices], dtype=np.float64
    )
    if result.shape != (len(indices), 3) or not np.isfinite(result).all():
        raise ValueError("scene points must be finite xyz")
    return result


def _require_retained_support(
    candidate: SceneCandidate,
    indices: tuple[int, ...],
    assessment: SceneAssessment,
) -> None:
    for index in indices:
        if not 0 <= index < len(candidate.static_points):
            raise ValueError("evidence refers to absent scene point")
        if index not in assessment.shared_point_indices:
            raise ValueError("ground or size evidence lacks coherent retained support")
        observations = candidate.static_points[index].get("observations", {})
        if (
            not isinstance(observations, dict)
            or len(set(observations) & set(assessment.retained)) < 2
        ):
            raise ValueError("ground or size evidence lacks retained multiview support")


def _vertical_direction(
    candidate: SceneCandidate,
    evidence: GroundEvidence,
    retained: tuple[str, ...],
) -> NDArray[np.float64]:
    indices = evidence.vertical_indices
    endpoints = _points(candidate, indices)
    if evidence.kind == "scene":
        references = candidate.evidence.get("vertical_references", [])
        matches = [
            item
            for item in references
            if isinstance(item, dict)
            and item.get("id") == evidence.vertical_reference_id
        ]
        if len(matches) != 1 or (
            matches[0].get("kind") != "known_upright"
            or matches[0].get("point_indices") != list(indices)
            or matches[0].get("source_kind") != "upright_target"
            or not matches[0].get("source_id")
            or not matches[0].get("producer")
            or matches[0].get("producer") == evidence.producer
        ):
            raise ValueError("scene ground requires a matching independent upright cue")
        # A named cue alone is insufficient: both endpoints must be actual
        # triangulated scene tracks with consistent observations in two views.
        for index in indices:
            point = candidate.static_points[index]
            observations = point.get("observations", {})
            if (
                not isinstance(observations, dict)
                or len(set(observations) & set(retained)) < 2
            ):
                raise ValueError("upright cue lacks multiview point observations")
            for camera_id, pixel in observations.items():
                if camera_id not in retained:
                    continue
                record = candidate.cameras.get(camera_id)
                if record is None:
                    raise ValueError("upright cue refers to absent camera")
                camera = CameraModel(
                    Intrinsics.model_validate(record["intrinsics"]),
                    np.asarray(record["world_to_camera"], dtype=np.float64),
                )
                projected = camera.project(_points(candidate, (index,)))[0]
                observed = np.asarray(pixel, dtype=np.float64)
                if (
                    observed.shape != (2,)
                    or not np.isfinite(observed).all()
                    or np.linalg.norm(projected - observed) > 3.0
                ):
                    raise ValueError("upright cue has inconsistent multiview geometry")
    direction = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        raise ValueError("upright cue is degenerate")
    return np.asarray(direction / length, dtype=np.float64)


def _ground_frame(
    candidate: SceneCandidate,
    evidence: GroundEvidence,
    revision: str,
    scale: float,
    retained: tuple[str, ...],
) -> tuple[GroundFrame, NDArray[np.float64], NDArray[np.float64]]:
    samples = _points(candidate, evidence.floor_indices)
    extent = float(np.linalg.norm(np.ptp(samples, axis=0)))
    if extent <= 1e-8:
        raise ValueError("floor evidence has no spatial coverage")
    threshold = max(1e-6, 0.01 * extent)
    rng = np.random.default_rng(17)
    best = np.zeros(len(samples), dtype=bool)
    for _ in range(256):
        a, b, c = samples[rng.choice(len(samples), 3, replace=False)]
        trial = np.cross(b - a, c - a)
        length = np.linalg.norm(trial)
        if length <= 1e-10:
            continue
        mask = np.abs((samples - a) @ (trial / length)) <= threshold
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < max(6, int(np.ceil(0.7 * len(samples)))):
        raise ValueError("floor plane has insufficient robust inliers")
    inliers = samples[best]
    origin = inliers.mean(axis=0)
    _, singular, axes = np.linalg.svd(inliers - origin, full_matrices=False)
    if singular[1] < 0.08 * singular[0]:
        raise ValueError("floor samples do not cover a plane")
    normal = axes[2]
    residuals = (inliers - origin) @ normal
    rms = float(np.sqrt(np.mean(residuals**2)))
    if rms > threshold:
        raise ValueError("floor residual exceeds supported tolerance")
    above = _points(candidate, evidence.above_indices)
    signed = (above - origin) @ normal
    if np.any(np.abs(signed) < 3 * threshold) or np.any(signed > 0) != np.all(
        signed > 0
    ):
        raise ValueError("vertical sign evidence is ambiguous")
    if np.all(signed < 0):
        normal = -normal
    up = _vertical_direction(candidate, evidence, retained)
    if float(normal @ up) < float(np.cos(np.deg2rad(15))):
        raise ValueError("floor plane conflicts with independent vertical cue")
    axis = _points(candidate, evidence.axis_indices)
    x = axis[1] - axis[0]
    x -= np.dot(x, normal) * normal
    axis_length = float(np.linalg.norm(x))
    if axis_length < 3 * threshold:
        raise ValueError("ground axis cue is degenerate")
    x /= axis_length
    y = np.cross(normal, x)
    rotation = np.stack((x, y, normal))
    planar = (inliers - origin) @ rotation[:2].T
    coverage = float(np.prod(np.ptp(planar, axis=0)))
    if coverage < 0.02 * extent**2:
        raise ValueError("floor samples have poor planar coverage")
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = -scale * rotation @ origin
    frame = GroundFrame(
        source_to_world=transform.tolist(),
        plane_normal_source=(float(normal[0]), float(normal[1]), float(normal[2])),
        plane_offset_source=-float(normal @ origin),
        inlier_count=int(best.sum()),
        sample_count=len(samples),
        inlier_indices=[evidence.floor_indices[i] for i in np.flatnonzero(best)],
        coverage=coverage * scale**2,
        rms_residual=rms * scale,
        normal_uncertainty_rad=float(
            np.arctan2(rms, singular[1] / np.sqrt(len(inliers)))
        ),
        axis_uncertainty_rad=float(np.arctan2(rms, axis_length)),
        evidence_kind=evidence.kind,
        evidence_ids=[evidence.id, evidence.vertical_reference_id],
        evidence_producer=evidence.producer or "operator",
        evidence_author=evidence.author or None,
        evidence_reason=evidence.reason or None,
        source_revision=revision,
    )
    return frame, rotation, origin


def resolve_scene(
    candidate: SceneCandidate,
    ground: GroundEvidence | None = None,
    size: SizeEvidence | None = None,
    thresholds: QualityThresholds = QualityThresholds(),
) -> Calibration:
    """Keep unsupported capabilities unavailable; never infer floor or metres."""
    assessment: SceneAssessment = assess_scene(candidate, thresholds)
    revision = scene_revision(candidate)
    if any(e.source_revision != revision for e in (ground, size) if e is not None):
        raise ValueError("evidence belongs to incompatible scene revision")
    if ground and size and ground.id == size.id:
        raise ValueError("ground and size evidence IDs must differ")
    scale = 1.0
    if size is not None:
        _require_retained_support(candidate, size.point_indices, assessment)
        endpoints = _points(candidate, size.point_indices)
        distance = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        if distance <= 1e-8:
            raise ValueError("measured scene segment is degenerate")
        scale = size.length * (0.01 if size.unit == "cm" else 1.0) / distance
    frame = None
    rotation = np.eye(3)
    origin = np.zeros(3)
    if ground is not None:
        _require_retained_support(
            candidate,
            ground.floor_indices
            + ground.above_indices
            + ground.axis_indices
            + ground.vertical_indices,
            assessment,
        )
        frame, rotation, origin = _ground_frame(
            candidate,
            ground,
            revision,
            scale,
            assessment.retained,
        )
    cameras = []
    for camera_id in assessment.retained:
        record = candidate.cameras[camera_id]
        pose = np.asarray(record["world_to_camera"], dtype=np.float64)
        intrinsics = Intrinsics.model_validate(record["intrinsics"])
        CameraModel(intrinsics, pose)
        world_pose = np.eye(4)
        world_pose[:3, :3] = pose[:3, :3] @ rotation.T
        world_pose[:3, 3] = scale * (pose[:3, :3] @ origin + pose[:3, 3])
        cameras.append(
            CameraCalibration(
                camera_id=camera_id,
                source_id=record["source_id"],
                intrinsics=intrinsics,
                world_to_camera=world_pose.tolist(),
                quality=Quality(state="observed", source_ids=[record["source_id"]]),
                intrinsic_source="imported",
            )
        )
    if len({camera.source_id for camera in cameras}) != len(cameras):
        raise ValueError("scene cameras must have distinct sources")
    config = {
        "scene_revision": revision,
        "ground": vars(ground) if ground else None,
        "size": vars(size) if size else None,
        "thresholds": vars(thresholds),
    }
    digest = hash_config(config)
    return Calibration(
        id=f"calibration-{digest[:24]}",
        kind="calibration",
        schema_version="1.0.0",
        provenance=Provenance(producer="scene_ground_v1", config_digest=digest),
        cameras=cameras,
        scale="metric" if size else "arbitrary",
        world_unit="m" if size else "arbitrary",
        scale_status="resolved" if size else "unresolved",
        scale_evidence_ids=[size.id] if size else [],
        scale_resolution=ScaleResolution(
            evidence_id=size.id,
            source_revision=revision,
            kind=size.kind,
            measured_length=size.length,
            measured_unit=size.unit,
            metres_per_source_unit=scale,
            producer=size.producer or "operator",
            author=size.author or None,
            reason=size.reason or None,
        )
        if size
        else None,
        ground_status="resolved" if frame else "unresolved",
        ground_z=0.0 if frame else None,
        ground_frame=frame,
        source_revision=revision,
        camera_status="resolved",
        publication_status="complete" if frame and size else "partial",
        excluded_cameras=assessment.excluded,
        quality_flags=list(assessment.flags)
        + (["ground unresolved"] if not frame else [])
        + (["metric scale unresolved"] if not size else []),
        projection_debug=[assessment.diagnostics],
        evidence_links=[
            f"scene-candidate:sha256:{revision}",
            f"synchronization:{candidate.evidence['synchronization']['artifact_id']}",
        ]
        + ([f"ground:{ground.id}"] if ground else [])
        + ([f"size:{size.id}"] if size else []),
        quality=Quality(
            state="observed" if frame else "unknown",
            uncertainty=frame.normal_uncertainty_rad if frame else None,
            source_ids=sorted({c.source_id for c in cameras}),
        ),
    )


def persist_scene_calibration(
    store: ArtifactStore,
    candidate: SceneCandidate,
    calibration: Calibration,
) -> ArtifactHandle:
    revision = scene_revision(candidate)
    if calibration.source_revision != revision:
        raise ValueError("calibration and scene revisions disagree")
    key = ArtifactKey(
        layer="calibration",
        inputs={"scene_candidate": revision},
        schema_version="1.0.0",
        algorithm_revision="scene-ground-v1",
        config_digest=calibration.provenance.config_digest,
    )
    return store.get_or_create(key, lambda: (calibration, {}))
