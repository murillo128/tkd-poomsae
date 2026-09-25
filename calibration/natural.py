"""Conservative static-scene camera candidates from synchronized native images.

The result is deliberately not a Calibration artifact: image-only geometry has
neither a measured ground frame nor metric scale. A later quality gate must
resolve those before accepting it for downstream reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]
from scipy.sparse import lil_matrix  # type: ignore[import-untyped]

from calibration.cameras import CameraModel, validate_intrinsics
from contracts.models import Intrinsics, Synchronization

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SceneView:
    camera_id: str
    source_id: str
    frames: tuple[NDArray[np.uint8], ...]
    intrinsics: Intrinsics | None = None
    source_sha256: str | None = None
    frame_seconds: tuple[float, ...] | None = None
    frame_native_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.camera_id or not self.source_id or not self.frames:
            raise ValueError("view needs camera, source and native frames")
        shape = self.frames[0].shape
        if len(shape) not in (2, 3) or (len(shape) == 3 and shape[2] != 3):
            raise ValueError("frames must be grayscale or RGB images")
        if any(
            frame.dtype != np.uint8 or frame.shape != shape for frame in self.frames
        ):
            raise ValueError("frames must have one uint8 image geometry")
        if self.intrinsics is not None:
            validate_intrinsics(self.intrinsics, (shape[1], shape[0]))
        if self.frame_seconds is not None and (
            len(self.frame_seconds) != len(self.frames)
            or not np.isfinite(self.frame_seconds).all()
        ):
            raise ValueError("frame timestamps must be finite and match frames")
        if self.frame_native_ids is not None and (
            len(self.frame_native_ids) != len(self.frames)
            or not all(self.frame_native_ids)
        ):
            raise ValueError("native frame IDs must match frames")


@dataclass
class SceneCandidate:
    status: Literal["candidate", "weak", "unavailable"]
    reasons: list[str]
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)
    static_points: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "route": "natural_scene_v1",
            "status": self.status,
            "reasons": self.reasons,
            "scale": "unresolved",
            "ground": "unresolved",
            "cameras": self.cameras,
            "static_points": self.static_points,
            "evidence": self.evidence,
        }


@dataclass
class _Features:
    pixels: FloatArray
    descriptors: NDArray[np.float32]
    stable_fraction: float


@dataclass
class _Edge:
    a: str
    b: str
    ids_a: NDArray[np.int32]
    ids_b: NDArray[np.int32]
    fundamental_inliers: int
    homography_inliers: int
    raw_matches: int = 0
    rotation: FloatArray | None = None
    translation: FloatArray | None = None
    cheirality: float | None = None
    median_angle_deg: float | None = None
    reason: str | None = None


def _features(view: SceneView, max_features: int) -> _Features:
    gray = [
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        for frame in view.frames
    ]
    stack = np.stack(gray).astype(np.float32)
    median = np.median(stack, axis=0).astype(np.uint8)
    if len(gray) >= 3:
        deviation = np.median(np.abs(stack - median), axis=0)
        stable = (deviation <= 12).astype(np.uint8) * 255
        stable = cv2.erode(stable, np.ones((5, 5), np.uint8))
    else:
        stable = np.full(median.shape, 255, np.uint8)
    detector = cv2.SIFT_create(nfeatures=max_features)  # type: ignore[attr-defined]
    keypoints, descriptors = detector.detectAndCompute(median, stable)
    if descriptors is None:
        descriptors = np.empty((0, 128), np.float32)
    return _Features(
        np.asarray([point.pt for point in keypoints], dtype=np.float64).reshape(-1, 2),
        descriptors,
        float(np.count_nonzero(stable) / stable.size),
    )


def _matches(a: _Features, b: _Features) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    if min(len(a.descriptors), len(b.descriptors)) < 2:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a.descriptors, b.descriptors, k=2)
    backward = matcher.knnMatch(b.descriptors, a.descriptors, k=2)
    reverse = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in backward
        if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
    }
    pairs = [
        (pair[0].queryIdx, pair[0].trainIdx)
        for pair in forward
        if len(pair) == 2
        and pair[0].distance < 0.72 * pair[1].distance
        and reverse.get(pair[0].trainIdx) == pair[0].queryIdx
    ]
    return (
        np.asarray([p[0] for p in pairs], np.int32),
        np.asarray([p[1] for p in pairs], np.int32),
    )


def _normalized(pixels: FloatArray, intrinsics: Intrinsics) -> FloatArray:
    camera = CameraModel(intrinsics, np.eye(4))
    return np.asarray(
        cv2.undistortPoints(
            pixels.reshape(-1, 1, 2),
            camera.intrinsic_matrix,
            camera.distortion,
        ),
        dtype=np.float64,
    ).reshape(-1, 2)


def _triangle_angles(points: FloatArray, center: FloatArray) -> FloatArray:
    first = points / np.linalg.norm(points, axis=1, keepdims=True)
    second = (points - center) / np.linalg.norm(points - center, axis=1, keepdims=True)
    cosine = np.sum(first * second, axis=1).clip(-1, 1)
    return np.asarray(np.degrees(np.arccos(cosine)), dtype=np.float64)


def _triangulate(
    a: FloatArray,
    b: FloatArray,
    transform_a: FloatArray,
    transform_b: FloatArray,
    intrinsics_a: Intrinsics,
    intrinsics_b: Intrinsics,
) -> FloatArray:
    rays_a = _normalized(a, intrinsics_a).T
    rays_b = _normalized(b, intrinsics_b).T
    points = cv2.triangulatePoints(transform_a[:3], transform_b[:3], rays_a, rays_b)
    return np.asarray((points[:3] / points[3]).T, dtype=np.float64)


def _edge(
    a: str,
    b: str,
    fa: _Features,
    fb: _Features,
    ka: Intrinsics | None,
    kb: Intrinsics | None,
) -> _Edge:
    ia, ib = _matches(fa, fb)
    edge = _Edge(a, b, ia, ib, 0, 0, len(ia))
    if len(ia) < 24:
        edge.reason = "insufficient matched texture"
        return edge
    pa, pb = fa.pixels[ia], fb.pixels[ib]
    _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5, 0.999)
    if mask is None:
        edge.reason = "epipolar estimation failed"
        return edge
    good = mask.ravel().astype(bool)
    edge.ids_a, edge.ids_b = ia[good], ib[good]
    edge.fundamental_inliers = int(good.sum())
    if edge.fundamental_inliers < 24:
        edge.reason = "insufficient epipolar inliers"
        return edge
    pa, pb = fa.pixels[edge.ids_a], fb.pixels[edge.ids_b]
    _, hmask = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
    edge.homography_inliers = int(hmask.sum()) if hmask is not None else 0
    if edge.homography_inliers >= 0.9 * edge.fundamental_inliers:
        edge.reason = "planar-only or pure-rotation ambiguity"
        return edge
    if ka is None or kb is None:
        edge.reason = "intrinsics unavailable; Euclidean pose unobservable"
        return edge
    na, nb = _normalized(pa, ka), _normalized(pb, kb)
    essential, emask = cv2.findEssentialMat(
        na,
        nb,
        np.eye(3),
        cv2.RANSAC,
        0.999,
        0.003,
    )
    if essential is None or emask is None:
        edge.reason = "essential-matrix estimation failed"
        return edge
    # OpenCV may return several essential matrices; score each by cheirality.
    best: tuple[int, FloatArray, FloatArray, NDArray[np.uint8]] | None = None
    for candidate in np.asarray(essential).reshape(-1, 3, 3):
        count, rotation, translation, pose_mask = cv2.recoverPose(
            candidate,
            na,
            nb,
            np.eye(3),
            mask=emask.copy(),
        )
        if best is None or count > best[0]:
            best = (
                count,
                np.asarray(rotation, dtype=np.float64),
                np.asarray(translation, dtype=np.float64).reshape(3),
                np.asarray(pose_mask, dtype=np.uint8),
            )
    assert best is not None
    count, rotation, translation, pose_mask = best
    edge.cheirality = count / len(pa)
    if edge.cheirality < 0.7:
        edge.reason = "reflected or behind-camera solution"
        return edge
    in_pose = pose_mask.ravel() != 0
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    points = _triangulate(pa[in_pose], pb[in_pose], np.eye(4), transform, ka, kb)
    center = -rotation.T @ translation
    angles = _triangle_angles(points, center)
    edge.median_angle_deg = float(np.median(angles))
    if edge.median_angle_deg < 1.5:
        edge.reason = "near-zero baseline or triangulation angle"
        return edge
    edge.ids_a = edge.ids_a[in_pose]
    edge.ids_b = edge.ids_b[in_pose]
    edge.rotation, edge.translation = rotation, translation
    return edge


def _tracks(
    edges: list[_Edge], features: dict[str, _Features]
) -> list[dict[str, FloatArray]]:
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def root(item: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = root(parent[item])
        return parent[item]

    for edge in edges:
        if edge.reason is not None:
            continue
        for ia, ib in zip(edge.ids_a, edge.ids_b, strict=True):
            left, right = root((edge.a, int(ia))), root((edge.b, int(ib)))
            parent[right] = left
    groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for item in parent:
        groups.setdefault(root(item), []).append(item)
    tracks = []
    for group in groups.values():
        if len(group) < 2 or len({camera for camera, _ in group}) != len(group):
            continue
        tracks.append(
            {camera: features[camera].pixels[index] for camera, index in group}
        )
    return tracks


def _project(
    points: FloatArray, transform: FloatArray, intrinsics: Intrinsics
) -> FloatArray:
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    camera = CameraModel(intrinsics, transform)
    pixels, _ = cv2.projectPoints(
        points, rvec, transform[:3, 3], camera.intrinsic_matrix, camera.distortion
    )
    return np.asarray(pixels, np.float64).reshape(-1, 2)


def _bundle(
    poses: dict[str, FloatArray],
    points: dict[int, FloatArray],
    tracks: list[dict[str, FloatArray]],
    intrinsics: dict[str, Intrinsics],
    root: str,
    second: str,
) -> tuple[dict[str, FloatArray], dict[int, FloatArray], FloatArray, float]:
    cameras = [camera for camera in poses if camera != root]
    point_ids = sorted(points)
    observations = [
        (camera, pid, pixel)
        for pid in point_ids
        for camera, pixel in tracks[pid].items()
        if camera in poses
    ]
    offsets: dict[str, int] = {}
    parameters: list[float] = []
    for camera in cameras:
        offsets[camera] = len(parameters)
        rvec, _ = cv2.Rodrigues(poses[camera][:3, :3])
        parameters.extend(rvec.ravel().tolist())
        if camera != second:
            parameters.extend(poses[camera][:3, 3].tolist())
    points_offset = len(parameters)
    parameters.extend(np.concatenate([points[pid] for pid in point_ids]).tolist())
    point_index = {pid: index for index, pid in enumerate(point_ids)}
    sparsity = lil_matrix((2 * len(observations), len(parameters)), dtype=int)
    for index, (camera, pid, _) in enumerate(observations):
        if camera != root:
            width = 3 if camera == second else 6
            sparsity[
                2 * index : 2 * index + 2, offsets[camera] : offsets[camera] + width
            ] = 1
        column = points_offset + 3 * point_index[pid]
        sparsity[2 * index : 2 * index + 2, column : column + 3] = 1

    def unpack(
        values: FloatArray,
    ) -> tuple[dict[str, FloatArray], dict[int, FloatArray]]:
        result = {root: poses[root]}
        for camera in cameras:
            start = offsets[camera]
            rotation, _ = cv2.Rodrigues(values[start : start + 3])
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = (
                poses[second][:3, 3]
                if camera == second
                else values[start + 3 : start + 6]
            )
            result[camera] = transform
        locations = {
            pid: values[points_offset + 3 * index : points_offset + 3 * index + 3]
            for pid, index in point_index.items()
        }
        return result, locations

    def residual(values: FloatArray) -> FloatArray:
        candidate_poses, locations = unpack(values)
        projected = {
            camera: _project(
                np.asarray([locations[pid] for pid in point_ids]),
                candidate_poses[camera],
                intrinsics[camera],
            )
            for camera in candidate_poses
        }
        return np.concatenate(
            [
                projected[camera][point_index[pid]] - pixel
                for camera, pid, pixel in observations
            ]
        )

    result = least_squares(
        residual,
        np.asarray(parameters),
        jac_sparsity=sparsity.tocsr(),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=50,
    )
    refined_poses, refined_points = unpack(result.x)
    return (
        refined_poses,
        refined_points,
        residual(result.x).reshape(-1, 2),
        float(result.cost),
    )


def estimate_scene(
    views: list[SceneView],
    synchronization: Synchronization | None = None,
    *,
    max_features: int = 1600,
    max_bundle_points: int = 180,
) -> SceneCandidate:
    """Estimate only evidence-supported relative camera geometry.

    A source-bound Synchronization artifact and exact native PTS are required.
    Supplied intrinsics are treated as fixed pinhole plus OpenCV distortion.
    Missing intrinsics are never guessed from image size. Their constrained
    model/priors and observability are recorded as unavailable.
    """
    if len(views) < 2 or len({v.camera_id for v in views}) != len(views):
        raise ValueError("at least two distinct camera IDs are required")
    if len({v.source_id for v in views}) != len(views):
        raise ValueError("camera source IDs must be distinct")
    if max_features < 100 or max_bundle_points < 24:
        raise ValueError("feature and bundle limits are too small")
    evidence: dict[str, Any] = {
        "intrinsic_model": "fixed supplied pinhole with OpenCV distortion",
        "missing_intrinsics": {
            view.camera_id: {
                "model": (
                    "constrained pinhole; principal point, focal length and "
                    "distortion unknown"
                ),
                "priors": "none supplied; image resolution is not a focal prior",
                "observability": (
                    "Euclidean pose unavailable without independent calibration"
                ),
            }
            for view in views
            if view.intrinsics is None
        },
        "views": {
            view.camera_id: {
                "source_id": view.source_id,
                "source_sha256": view.source_sha256,
                "frames": len(view.frames),
                "frame_seconds": view.frame_seconds,
                "frame_native_ids": view.frame_native_ids,
            }
            for view in views
        },
    }
    if synchronization is None:
        return SceneCandidate(
            "unavailable",
            ["verified synchronization artifact is required"],
            evidence=evidence,
        )
    by_source = {row.source_id: row for row in synchronization.offsets}
    if {view.source_id for view in views} - set(by_source):
        return SceneCandidate(
            "unavailable",
            ["synchronization does not cover exact source IDs"],
            evidence=evidence,
        )
    evidence["synchronization"] = {
        "artifact_id": synchronization.id,
        "producer": synchronization.provenance.producer,
        "config_digest": synchronization.provenance.config_digest,
        "offsets": {},
    }
    if synchronization.common_interval is None:
        return SceneCandidate(
            "unavailable",
            ["synchronization lacks a common interval"],
            evidence=evidence,
        )
    sample_counts = {len(view.frames) for view in views}
    if len(sample_counts) != 1:
        raise ValueError("synchronized views need equal sample counts")
    global_times = []
    for view in views:
        row = by_source[view.source_id]
        if (
            view.source_sha256 is not None
            and view.source_id != f"source:{view.source_sha256}"
        ):
            return SceneCandidate(
                "unavailable",
                [f"source hash identity mismatch: {view.camera_id}"],
                evidence=evidence,
            )
        if (
            not row.retained
            or row.source_interval is None
            or row.global_interval is None
            or not (
                row.quality.state == "observed"
                or (
                    row.manual_seconds is not None
                    and row.manual_author
                    and row.manual_source
                    and row.manual_reason
                )
                or (
                    row.timing_reference
                    and row.source_id == synchronization.reference_source_id
                )
            )
        ):
            return SceneCandidate(
                "unavailable",
                [f"unverified timing for {view.camera_id}"],
                evidence=evidence,
            )
        if view.frame_seconds is None or view.frame_native_ids is None:
            return SceneCandidate(
                "unavailable",
                [f"native frame timing missing for {view.camera_id}"],
                evidence=evidence,
            )
        try:
            for seconds, native_id in zip(
                view.frame_seconds, view.frame_native_ids, strict=True
            ):
                pts_text, rational = native_id.split(":", 1)
                numerator_text, denominator_text = rational.split("/", 1)
                native_seconds = (
                    int(pts_text) * int(numerator_text) / int(denominator_text)
                )
                if abs(native_seconds - seconds) > 1e-9:
                    raise ValueError("native frame time disagrees with PTS")
        except (ValueError, ZeroDivisionError) as exc:
            return SceneCandidate(
                "unavailable",
                [f"invalid native PTS for {view.camera_id}: {exc}"],
                evidence=evidence,
            )
        offset = row.effective_seconds
        if (
            abs(row.global_interval.start - row.source_interval.start - offset) > 1e-9
            or abs(row.global_interval.end - row.source_interval.end - offset) > 1e-9
        ):
            return SceneCandidate(
                "unavailable",
                [f"inconsistent sync interval: {view.camera_id}"],
                evidence=evidence,
            )
        evidence["synchronization"]["offsets"][view.source_id] = offset
        global_sample_times = np.asarray(view.frame_seconds) + offset
        if any(
            source_time < row.source_interval.start - 1e-9
            or source_time > row.source_interval.end + 1e-9
            for source_time in view.frame_seconds
        ) or any(
            global_time
            < max(row.global_interval.start, synchronization.common_interval.start)
            - 1e-9
            or global_time
            > min(row.global_interval.end, synchronization.common_interval.end) + 1e-9
            for global_time in global_sample_times
        ):
            return SceneCandidate(
                "unavailable",
                [f"frames outside verified timeline: {view.camera_id}"],
                evidence=evidence,
            )
        global_times.append(global_sample_times)
    max_spread = float(np.max(np.ptp(np.asarray(global_times), axis=0)))
    evidence["frame_alignment"] = {
        "max_global_time_spread_seconds": max_spread,
        "state": "aligned" if max_spread <= 0.05 else "unverified",
    }
    if max_spread > 0.05:
        return SceneCandidate(
            "unavailable",
            ["frames are not aligned in verified global time"],
            evidence=evidence,
        )
    feature_map = {view.camera_id: _features(view, max_features) for view in views}
    intrinsics = {view.camera_id: view.intrinsics for view in views}
    for view in views:
        evidence["views"][view.camera_id].update(
            {
                "features": len(feature_map[view.camera_id].pixels),
                "stable_pixel_fraction": feature_map[view.camera_id].stable_fraction,
            }
        )
    edges = [
        _edge(
            a.camera_id,
            b.camera_id,
            feature_map[a.camera_id],
            feature_map[b.camera_id],
            a.intrinsics,
            b.intrinsics,
        )
        for index, a in enumerate(views)
        for b in views[index + 1 :]
    ]
    evidence["overlap_edges"] = [
        {
            "cameras": [edge.a, edge.b],
            "raw_matches": edge.raw_matches,
            "fundamental_inliers": edge.fundamental_inliers,
            "homography_inliers": edge.homography_inliers,
            "cheirality_fraction": edge.cheirality,
            "median_triangulation_angle_deg": edge.median_angle_deg,
            "rejection": edge.reason,
        }
        for edge in edges
    ]
    good = sorted(
        (edge for edge in edges if edge.reason is None),
        key=lambda edge: edge.fundamental_inliers,
        reverse=True,
    )
    if not good:
        reasons = sorted({edge.reason or "no usable overlap" for edge in edges})
        status: Literal["weak", "unavailable"] = (
            "unavailable" if evidence["missing_intrinsics"] else "weak"
        )
        return SceneCandidate(status, reasons, evidence=evidence)
    root, second = good[0].a, good[0].b
    transform = np.eye(4)
    assert good[0].rotation is not None and good[0].translation is not None
    transform[:3, :3] = good[0].rotation
    transform[:3, 3] = good[0].translation
    poses: dict[str, FloatArray] = {root: np.eye(4), second: transform}
    tracks = _tracks(good, feature_map)
    points: dict[int, FloatArray] = {}
    root_intrinsics = intrinsics[root]
    second_intrinsics = intrinsics[second]
    assert root_intrinsics is not None and second_intrinsics is not None
    for pid, track in enumerate(tracks):
        if root not in track or second not in track:
            continue
        xyz = _triangulate(
            track[root][None],
            track[second][None],
            poses[root],
            poses[second],
            root_intrinsics,
            second_intrinsics,
        )[0]
        if (
            np.isfinite(xyz).all()
            and xyz[2] > 0
            and (transform[:3, :3] @ xyz + transform[:3, 3])[2] > 0
        ):
            points[pid] = xyz
    pending = {view.camera_id for view in views} - set(poses)
    while pending:
        progressed = False
        for camera in sorted(pending):
            camera_intrinsics = intrinsics[camera]
            if camera_intrinsics is None:
                continue
            matches = [
                (pid, tracks[pid][camera]) for pid in points if camera in tracks[pid]
            ]
            if len(matches) < 16:
                continue
            model = CameraModel(camera_intrinsics, np.eye(4))
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.asarray([points[pid] for pid, _ in matches]),
                np.asarray([pixel for _, pixel in matches]),
                model.intrinsic_matrix,
                model.distortion,
                iterationsCount=1000,
                reprojectionError=3.0,
                confidence=0.999,
            )
            if not success or inliers is None or len(inliers) < 16:
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            candidate = np.eye(4)
            candidate[:3, :3], candidate[:3, 3] = rotation, tvec.ravel()
            visible = np.asarray(
                [points[matches[int(index)][0]] for index in inliers.ravel()]
            )
            if np.mean((visible @ rotation.T + tvec.ravel())[:, 2] > 0) < 0.9:
                continue
            poses[camera] = candidate
            pending.remove(camera)
            progressed = True
            for pid, track in enumerate(tracks):
                if pid in points or camera not in track:
                    continue
                for existing in poses:
                    if existing == camera or existing not in track:
                        continue
                    existing_intrinsics = intrinsics[existing]
                    assert existing_intrinsics is not None
                    xyz = _triangulate(
                        track[existing][None],
                        track[camera][None],
                        poses[existing],
                        candidate,
                        existing_intrinsics,
                        camera_intrinsics,
                    )[0]
                    if np.isfinite(xyz).all() and all(
                        (pose[:3, :3] @ xyz + pose[:3, 3])[2] > 0
                        for pose in (poses[existing], candidate)
                    ):
                        points[pid] = xyz
                    break
        if not progressed:
            break
    evidence["connected_cameras"] = sorted(poses)
    if pending:
        return SceneCandidate(
            "weak",
            ["disconnected or unscaled camera overlap: " + ", ".join(sorted(pending))],
            evidence=evidence,
        )
    if len(points) < 24:
        return SceneCandidate(
            "weak", ["insufficient triangulated static points"], evidence=evidence
        )
    # Bound BA cost while retaining spatial coverage with deterministic sampling.
    point_ids = sorted(points)
    if len(point_ids) > max_bundle_points:
        selected = np.linspace(0, len(point_ids) - 1, max_bundle_points, dtype=int)
        points = {
            point_ids[int(index)]: points[point_ids[int(index)]] for index in selected
        }
    fixed_intrinsics = {
        key: value for key, value in intrinsics.items() if value is not None
    }
    poses, points, residuals, cost = _bundle(
        poses,
        points,
        tracks,
        fixed_intrinsics,
        root,
        second,
    )
    norms = np.linalg.norm(residuals, axis=1)
    locations = np.asarray(list(points.values()))
    singular = np.linalg.svd(locations - locations.mean(axis=0), compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    evidence["bundle"] = {
        "robust_cost": cost,
        "observations": len(norms),
        "residual_px": {
            "median": float(np.median(norms)),
            "p90": float(np.percentile(norms, 90)),
            "max": float(np.max(norms)),
        },
        "point_condition_number": condition,
        "conditioning_scope": (
            "centered static-point covariance; BA normal not estimated"
        ),
        "scale_gauge": "first camera fixed; second translation fixed to unit length",
        "metric_scale": "unresolved",
    }
    camera_centers = {
        camera: -pose[:3, :3].T @ pose[:3, 3] for camera, pose in poses.items()
    }
    angles = []
    for pid, point in points.items():
        visible_cameras = [camera for camera in tracks[pid] if camera in poses]
        for index, camera in enumerate(visible_cameras):
            first_ray = point - camera_centers[camera]
            for other in visible_cameras[index + 1 :]:
                second_ray = point - camera_centers[other]
                cosine = np.dot(first_ray, second_ray) / (
                    np.linalg.norm(first_ray) * np.linalg.norm(second_ray)
                )
                angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    evidence["bundle"]["triangulation_angle_deg"] = {
        "p10": float(np.percentile(angles, 10)),
        "median": float(np.median(angles)),
        "p90": float(np.percentile(angles, 90)),
    }
    reasons = []
    if condition > 100:
        reasons.append("planar or poorly conditioned point cloud")
    if float(np.percentile(norms, 90)) > 3:
        reasons.append("large bundle residuals")
    if float(np.percentile(angles, 10)) < 1.0:
        reasons.append("some points have near-zero triangulation angle")
    behind = sum(
        (pose[:3, :3] @ point + pose[:3, 3])[2] <= 0
        for pid, point in points.items()
        for camera, pose in poses.items()
        if camera in tracks[pid]
    )
    evidence["bundle"]["behind_camera_observations"] = int(behind)
    if behind:
        reasons.append("bundle solution includes points behind a camera")
    camera_output = {}
    for view in views:
        pose = poses[view.camera_id]
        pixels = np.asarray(
            [
                tracks[pid][view.camera_id]
                for pid in points
                if view.camera_id in tracks[pid]
            ]
        )
        height, width = view.frames[0].shape[:2]
        hull_area = (
            float(cv2.contourArea(cv2.convexHull(pixels.astype(np.float32))))
            if len(pixels) >= 3
            else 0.0
        )
        grid = {
            (min(3, max(0, int(x * 4 / width))), min(3, max(0, int(y * 4 / height))))
            for x, y in pixels
        }
        camera_output[view.camera_id] = {
            "source_id": view.source_id,
            "intrinsics": view.intrinsics.model_dump() if view.intrinsics else None,
            "intrinsic_source": "supplied",
            "world_to_camera": pose.tolist(),
            "spatial_coverage": {
                "image_hull_fraction": hull_area / (width * height),
                "occupied_grid_cells_4x4": len(grid),
                "observed_static_points": len(pixels),
            },
        }
        if hull_area / (width * height) < 0.03 or len(grid) < 4:
            reasons.append(f"camera {view.camera_id}: poor spatial coverage")
    point_output = [
        {
            "xyz": point.tolist(),
            "observations": {
                camera: pixel.tolist() for camera, pixel in tracks[pid].items()
            },
        }
        for pid, point in points.items()
    ]
    return SceneCandidate(
        "weak" if reasons else "candidate",
        reasons,
        camera_output,
        point_output,
        evidence,
    )


def persist_scene_candidate(candidate: SceneCandidate, directory: Path) -> Path:
    """Atomically retain a deterministic candidate and its complete diagnostics."""
    payload = json.dumps(
        candidate.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"natural-scene-{digest}.json"
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError("candidate hash collision or modified artifact")
        return target
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=".natural-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, target)
    except FileExistsError:
        if target.read_bytes() != payload:
            raise ValueError("concurrent candidate differs") from None
    finally:
        temporary.unlink()
    return target
