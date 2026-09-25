"""ChArUco observations, intrinsic estimation, board pose and immutable candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from calibration.cameras import (
    CameraModel,
    IntrinsicProfile,
    _matrix,
    validate_intrinsics,
)
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


class CalibrationFailure(ValueError):
    """Evidence cannot support the requested camera calibration."""


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    dictionary_id: int = cv2.aruco.DICT_4X4_50
    legacy_pattern: bool = False

    def __post_init__(self) -> None:
        if (
            self.squares_x < 3 or self.squares_y < 3
            or not 0 < self.marker_length_m < self.square_length_m
        ):
            raise ValueError(
                "board needs >=3 squares per axis and valid metric lengths"
            )
        cv2.aruco.getPredefinedDictionary(self.dictionary_id)

    def board(self) -> Any:
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_length_m,
            self.marker_length_m,
            cv2.aruco.getPredefinedDictionary(self.dictionary_id),
        )
        board.setLegacyPattern(self.legacy_pattern)
        return board

    def object_corners(self) -> NDArray[np.float64]:
        return np.asarray(self.board().getChessboardCorners(), dtype=np.float64)


@dataclass(frozen=True)
class Capture:
    """One decoded frame, including exact source identity and optional board placement.

    board_to_world is a 4x4 metric transform for a known placement, commonly
    the shared stationary board on the floor. Unplaced frames still help infer
    intrinsics, but cannot define a camera's world pose.
    """

    id: str
    source_id: str
    camera_id: str
    sha256: str
    image_size: tuple[int, int]
    crop_xywh: tuple[int, int, int, int]
    rotation_cw: int = 0
    board_to_world: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size", tuple(self.image_size))
        object.__setattr__(self, "crop_xywh", tuple(self.crop_xywh))
        if len(self.image_size) != 2 or len(self.crop_xywh) != 4:
            raise ValueError("capture image_size/crop_xywh have invalid lengths")
        if self.board_to_world is not None:
            object.__setattr__(self, "board_to_world", tuple(
                tuple(row) for row in self.board_to_world
            ))
        if not self.id or not self.source_id or not self.camera_id:
            raise ValueError("capture IDs must be nonempty")
        if len(self.sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.sha256
        ):
            raise ValueError("capture sha256 must be lowercase hex")
        x, y, width, height = self.crop_xywh
        if (
            min(*self.image_size, width, height) <= 0 or x < 0 or y < 0
            or x + width > self.image_size[0]
            or y + height > self.image_size[1]
            or self.rotation_cw not in (0, 90, 180, 270)
        ):
            raise ValueError("invalid capture image geometry")
        if self.board_to_world is not None:
            pose = _matrix(self.board_to_world, (4, 4))
            if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9):
                raise ValueError("board_to_world must be homogeneous")
            if not np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-6):
                raise ValueError("board_to_world rotation must be orthonormal")
            if abs(np.linalg.det(pose[:3, :3]) - 1) > 1e-6:
                raise ValueError("board_to_world rotation must be right-handed")
            if (
                not np.allclose(pose[:3, 2], [0, 0, -1], atol=1e-6)
                or abs(pose[2, 3]) > 1e-6
            ):
                raise ValueError("placed board must lie face-up on world ground z=0")

    @property
    def decoded_size(self) -> tuple[int, int]:
        width, height = self.crop_xywh[2:]
        return (height, width) if self.rotation_cw in (90, 270) else (width, height)


@dataclass(frozen=True)
class Detection:
    capture: Capture
    corner_ids: NDArray[np.int32]
    pixels: NDArray[np.float64]

    def __post_init__(self) -> None:
        ids = np.asarray(self.corner_ids, dtype=np.int32).reshape(-1)
        pixels = np.asarray(self.pixels, dtype=np.float64)
        if (
            pixels.shape != (len(ids), 2) or not np.isfinite(pixels).all()
            or len(set(ids.tolist())) != len(ids) or np.any(ids < 0)
        ):
            raise ValueError("invalid detected corner IDs/pixels")
        object.__setattr__(self, "corner_ids", ids.copy())
        object.__setattr__(self, "pixels", pixels.copy())


def observe_capture(path: Path, capture: Capture, board: BoardSpec) -> Detection:
    """Verify image bytes, crop/rotate, then detect board corners automatically."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != capture.sha256:
        raise CalibrationFailure(f"capture {capture.id}: image hash mismatch")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or (image.shape[1], image.shape[0]) != capture.image_size:
        raise CalibrationFailure(f"capture {capture.id}: decoded image size mismatch")
    x, y, width, height = capture.crop_xywh
    image = image[y:y + height, x:x + width]
    if capture.rotation_cw:
        image = cv2.rotate(image, {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[capture.rotation_cw])
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board.board()).detectBoard(image)
    if corners is None or ids is None:
        return Detection(capture, np.empty(0, np.int32), np.empty((0, 2)))
    return Detection(capture, np.asarray(ids, np.int32),
                     np.asarray(corners, np.float64).reshape(-1, 2))


def _points(board: BoardSpec, detection: Detection) -> NDArray[np.float64]:
    corners = board.object_corners()
    if np.any(detection.corner_ids >= len(corners)):
        raise CalibrationFailure(
            f"capture {detection.capture.id}: corner ID outside board"
        )
    return corners[detection.corner_ids]


def _intrinsics(
    board: BoardSpec, detections: list[Detection], image_size: tuple[int, int],
) -> tuple[Intrinsics, float]:
    if len(detections) < 3:
        raise CalibrationFailure(
            "intrinsic estimation requires at least three board poses"
        )
    if any(len(d.corner_ids) < 8 for d in detections):
        raise CalibrationFailure("intrinsic estimation requires eight corners per pose")
    centers = np.asarray([d.pixels.mean(axis=0) for d in detections])
    spans = np.asarray([np.ptp(d.pixels, axis=0) for d in detections])
    relative_motion = np.max(np.ptp(centers, axis=0) / image_size)
    relative_scale = np.ptp(np.linalg.norm(spans, axis=1)) / max(image_size)
    if relative_motion < 0.04 and relative_scale < 0.05:
        raise CalibrationFailure(
            "board poses have insufficient image coverage/diversity"
        )
    objects = [_points(board, d).astype(np.float32) for d in detections]
    images = [d.pixels.astype(np.float32) for d in detections]
    try:
        rms, matrix, distortion, rotation_vectors, _ = cv2.calibrateCamera(
            objects, images, image_size, None, None,
            flags=cv2.CALIB_FIX_K3,
        )
    except cv2.error as exc:
        raise CalibrationFailure("intrinsic solver failed") from exc
    normals = np.asarray([
        cv2.Rodrigues(vector)[0][:, 2] for vector in rotation_vectors
    ])
    if np.max(np.linalg.norm(normals[:, None, :] - normals[None, :, :], axis=2)) < 0.1:
        raise CalibrationFailure("board poses have insufficient angular diversity")
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise CalibrationFailure("intrinsic solver returned non-finite parameters")
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    width, height = image_size
    if (
        not 0.2 * max(image_size) < fx < 8 * max(image_size)
        or not 0.2 * max(image_size) < fy < 8 * max(image_size)
        or not 0 <= cx < width or not 0 <= cy < height
        or not np.isfinite(rms) or rms > 2.0
    ):
        raise CalibrationFailure("intrinsic solution is implausible or high-error")
    coefficients = distortion.reshape(-1)[:4].tolist()
    intrinsics = Intrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy, distortion=coefficients,
    )
    try:
        validate_intrinsics(intrinsics, image_size)
    except ValueError as exc:
        raise CalibrationFailure(f"intrinsic solution is invalid: {exc}") from exc
    return intrinsics, float(rms)


def _pose(
    board: BoardSpec, detection: Detection, intrinsics: Intrinsics,
) -> tuple[NDArray[np.float64], float, float]:
    points = _points(board, detection)
    if len(points) < 8:
        raise CalibrationFailure(
            f"capture {detection.capture.id}: insufficient corners"
        )
    model = CameraModel(intrinsics, np.eye(4))
    results = cv2.solvePnPGeneric(
        points, detection.pixels, model.intrinsic_matrix, model.distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not results[0]:
        raise CalibrationFailure(f"capture {detection.capture.id}: no pose solution")
    candidates: list[tuple[float, NDArray[np.float64]]] = []
    for rvec, tvec in zip(results[1], results[2], strict=True):
        rotation, _ = cv2.Rodrigues(rvec)
        depths = (points @ rotation.T + np.asarray(tvec).reshape(3))[:, 2]
        if np.any(depths <= 0):
            continue
        projected, _ = cv2.projectPoints(
            points, rvec, tvec, model.intrinsic_matrix, model.distortion,
        )
        error = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(-1, 2) - detection.pixels) ** 2, axis=1,
        ))))
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(tvec).reshape(3)
        candidates.append((error, transform))
    candidates.sort(key=lambda candidate: candidate[0])
    if not candidates:
        raise CalibrationFailure(
            f"capture {detection.capture.id}: points behind camera"
        )
    best_error = candidates[0][0]
    ambiguity = candidates[1][0] - best_error if len(candidates) > 1 else float("inf")
    if best_error > 2.0:
        raise CalibrationFailure(f"capture {detection.capture.id}: high pose error")
    if ambiguity < 0.2:
        raise CalibrationFailure(
            f"capture {detection.capture.id}: ambiguous mirrored planar pose"
        )
    return candidates[0][1], best_error, ambiguity


def estimate_calibration(
    board: BoardSpec, detections: list[Detection],
    profiles: dict[str, IntrinsicProfile] | None = None,
) -> Calibration:
    """Solve every camera from target observations; reject weak camera sets."""
    profiles = profiles or {}
    grouped: dict[str, list[Detection]] = {}
    if len({d.capture.id for d in detections}) != len(detections):
        raise CalibrationFailure("duplicate capture ID")
    for detection in detections:
        grouped.setdefault(detection.capture.camera_id, []).append(detection)
    if len(grouped) < 2:
        raise CalibrationFailure("at least two cameras are required")
    if set(profiles) - set(grouped):
        raise CalibrationFailure("intrinsic profile for absent camera")
    cameras: list[CameraCalibration] = []
    for camera_id, views in sorted(grouped.items()):
        sources = {d.capture.source_id for d in views}
        geometries = {
            (d.capture.image_size, d.capture.crop_xywh, d.capture.rotation_cw)
            for d in views
        }
        if len(sources) != 1 or len(geometries) != 1:
            raise CalibrationFailure(
                f"camera {camera_id}: inconsistent source or image geometry"
            )
        source_id = next(iter(sources))
        image_size = views[0].capture.decoded_size
        good = [d for d in views if len(d.corner_ids) >= 8]
        profile = profiles.get(camera_id)
        if profile is None:
            intrinsics, intrinsic_rms = _intrinsics(board, good, image_size)
            intrinsic_source: Literal["estimated", "imported"] = "estimated"
        else:
            for d in views:
                c = d.capture
                profile.require_compatible(c.camera_id, c.image_size,
                                           c.crop_xywh, c.rotation_cw)
            intrinsics, intrinsic_rms = profile.intrinsics, 0.0
            intrinsic_source = "imported"
        placed = [d for d in good if d.capture.board_to_world is not None]
        if not placed:
            raise CalibrationFailure(f"camera {camera_id}: no known board placement")
        poses: list[tuple[NDArray[np.float64], float, float, Detection]] = []
        for detection in placed:
            board_to_camera, error, ambiguity = _pose(board, detection, intrinsics)
            placement = _matrix(detection.capture.board_to_world, (4, 4))
            poses.append((board_to_camera @ np.linalg.inv(placement),
                          error, ambiguity, detection))
        poses.sort(key=lambda item: item[1])
        world_to_camera, pose_rms, ambiguity, anchor = poses[0]
        for other, error, _, _ in poses[1:]:
            centers = -world_to_camera[:3, :3].T @ world_to_camera[:3, 3]
            other_center = -other[:3, :3].T @ other[:3, 3]
            if np.linalg.norm(centers - other_center) > 0.1 or (
                np.trace(world_to_camera[:3, :3] @ other[:3, :3].T) < 2.98
            ):
                raise CalibrationFailure(
                    f"camera {camera_id}: inconsistent placed board poses"
                )
            pose_rms = max(pose_rms, error)
        score = 1 / (1 + max(intrinsic_rms, pose_rms))
        cameras.append(CameraCalibration(
            camera_id=camera_id, source_id=source_id, intrinsics=intrinsics,
            world_to_camera=world_to_camera.tolist(),
            quality=Quality(score=score, uncertainty=max(intrinsic_rms, pose_rms),
                            state="observed", source_ids=[source_id]),
            intrinsic_source=intrinsic_source,
            capture_ids=[d.capture.id for d in good],
            corner_count=sum(len(d.corner_ids) for d in good),
            rms_reprojection_px=max(intrinsic_rms, pose_rms),
            pose_ambiguity_px=ambiguity if np.isfinite(ambiguity) else None,
        ))
    config = {
        "board": vars(board),
        "captures": [vars(d.capture) for d in detections],
        "profiles": {
            key: {**vars(profile), "intrinsics": profile.intrinsics.model_dump()}
            for key, profile in profiles.items()
        },
    }
    digest = hash_config(config)
    source_ids = [camera.source_id for camera in cameras]
    if len(set(source_ids)) != len(source_ids):
        raise CalibrationFailure("different cameras cannot share a source ID")
    producer = ("charuco_target_with_imported_intrinsics" if profiles
                else "charuco_target_auto_intrinsics")
    return Calibration(
        kind="calibration", id=f"calibration-{digest[:24]}", schema_version="1.0.0",
        provenance=Provenance(producer=producer, config_digest=digest),
        scale="metric", world_unit="m", cameras=cameras, ground_z=0.0,
        ground_status="resolved", scale_status="resolved",
        scale_evidence_ids=[d.capture.id for d in detections
                            if d.capture.board_to_world is not None],
        scale_resolution=ScaleResolution(
            evidence_id=next(d.capture.id for d in detections
                             if d.capture.board_to_world is not None),
            source_revision=digest, kind="target",
            measured_length=board.square_length_m, measured_unit="m",
            metres_per_source_unit=1.0, producer="charuco-board-spec",
        ),
        source_revision=digest,
        ground_frame=GroundFrame(
            source_to_world=np.eye(4).tolist(),
            plane_normal_source=(0.0, 0.0, 1.0), plane_offset_source=0.0,
            inlier_count=sum(len(d.corner_ids) for d in detections
                             if d.capture.board_to_world is not None),
            sample_count=sum(len(d.corner_ids) for d in detections
                             if d.capture.board_to_world is not None),
            coverage=float(board.squares_x * board.squares_y)
                     * board.square_length_m**2,
            rms_residual=0.0, normal_uncertainty_rad=0.0,
            axis_uncertainty_rad=0.0,
            evidence_kind="target",
            evidence_ids=[d.capture.id for d in detections
                          if d.capture.board_to_world is not None],
            evidence_producer="placed-charuco-board",
            source_revision=digest,
        ),
        quality=Quality(score=min(c.quality.score or 0 for c in cameras),
                        uncertainty=max(c.quality.uncertainty or 0 for c in cameras),
                        state="observed", source_ids=source_ids),
    )


def persist_calibration(
    store: ArtifactStore, candidate: Calibration, detections: list[Detection],
) -> ArtifactHandle:
    """Store a content-addressed, immutable calibration using source frame hashes."""
    if {c.source_id for c in candidate.cameras} != {
        d.capture.source_id for d in detections
    }:
        raise ValueError("candidate and capture sources disagree")
    inputs = {d.capture.id: d.capture.sha256 for d in detections}
    if len(inputs) != len(detections):
        raise ValueError("duplicate capture ID")
    key = ArtifactKey(
        layer="calibration", inputs=inputs, schema_version="1.0.0",
        algorithm_revision="charuco-target-v1",
        config_digest=candidate.provenance.config_digest,
    )
    return store.get_or_create(key, lambda: (candidate, {}))
