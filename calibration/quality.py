"""Publication gate for synchronized, geometrically supported scene cameras."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from calibration.cameras import CameraModel
from calibration.natural import SceneCandidate
from contracts.models import Intrinsics, Synchronization


@dataclass(frozen=True)
class QualityThresholds:
    """Conservative limits in pixels, degrees, and fractions of an image."""

    min_shared_points: int = 24
    min_camera_points: int = 24
    min_hull_fraction: float = 0.03
    min_grid_cells: int = 4
    max_camera_p90_px: float = 3.0
    max_bundle_p90_px: float = 3.0
    min_cheirality: float = 0.95
    min_angle_deg: float = 1.0
    max_condition: float = 100.0

    def __post_init__(self) -> None:
        if not all(
            np.isfinite(value)
            for value in (
                self.min_hull_fraction,
                self.max_camera_p90_px,
                self.max_bundle_p90_px,
                self.min_cheirality,
                self.min_angle_deg,
                self.max_condition,
            )
        ):
            raise ValueError("quality thresholds must be finite")
        if (
            self.min_shared_points < 24
            or self.min_camera_points < 16
            or self.min_hull_fraction < 0.03
            or self.min_grid_cells < 4
            or not 0.9 <= self.min_cheirality <= 1
            or self.min_angle_deg < 1
            or self.max_condition > 100
            or not 0 < self.max_camera_p90_px <= 3
            or not 0 < self.max_bundle_p90_px <= 3
        ):
            raise ValueError(
                "quality thresholds may tighten but not weaken safety floors"
            )


@dataclass(frozen=True)
class SceneAssessment:
    retained: tuple[str, ...]
    excluded: dict[str, str]
    flags: tuple[str, ...]
    shared_point_indices: tuple[int, ...]
    diagnostics: dict[str, Any]


def _view_source_ids(views: dict[str, Any]) -> list[str]:
    if any(
        not isinstance(view, dict)
        or not isinstance(view.get("source_id"), str)
        or not view["source_id"]
        for view in views.values()
    ):
        raise ValueError("candidate has malformed synchronized view sources")
    return [view["source_id"] for view in views.values()]


def require_synchronization(candidate: SceneCandidate, sync: Synchronization) -> None:
    """Bind a candidate to the exact retained synchronization revision."""
    claimed = candidate.evidence.get("synchronization", {})
    if not isinstance(claimed, dict) or claimed.get("artifact_id") != sync.id:
        raise ValueError("candidate and synchronization revisions disagree")
    retained = {row.source_id: row for row in sync.offsets if row.retained}
    views = candidate.evidence.get("views", {})
    if not isinstance(views, dict) or not views:
        raise ValueError("candidate lacks synchronized views")
    sources = _view_source_ids(views)
    if (
        len(set(sources)) != len(sources)
        or set(sources) != set(retained)
        or claimed.get("retained_source_ids") != sorted(retained)
    ):
        raise ValueError("candidate omits retained synchronized source")
    claimed_offsets = claimed.get("offsets")
    if not isinstance(claimed_offsets, dict):
        raise ValueError("candidate lacks synchronized offsets")
    for view in views.values():
        if not isinstance(view, dict) or view.get("source_id") not in retained:
            raise ValueError("candidate includes unsynchronized or excluded source")
        source = view["source_id"]
        offset = float(claimed_offsets.get(source, float("nan")))
        if (
            not np.isfinite(offset)
            or abs(offset - retained[source].effective_seconds) > 1e-9
        ):
            raise ValueError("candidate and synchronization offsets disagree")


def assess_scene(
    candidate: SceneCandidate,
    thresholds: QualityThresholds = QualityThresholds(),
) -> SceneAssessment:
    """Recompute per-view residuals and shared coverage before publication.

    The candidate's bundle diagnostics cover conditioning and ray geometry;
    observations are checked again against its published poses. Exclusion is
    permitted only when a coherent independent view set remains.
    """
    if candidate.status not in {"candidate", "weak"} or len(candidate.cameras) < 2:
        raise ValueError("scene camera candidate is unavailable or weak")
    evidence = candidate.evidence
    bundle = evidence.get("bundle")
    views = evidence.get("views")
    sync = evidence.get("synchronization")
    if (
        not isinstance(bundle, dict)
        or not isinstance(views, dict)
        or not isinstance(sync, dict)
    ):
        raise ValueError(
            "candidate lacks mandatory bundle, view or synchronization evidence"
        )
    if not sync.get("artifact_id") or set(candidate.cameras) - set(views):
        raise ValueError("candidate lacks source-bound synchronized view evidence")
    candidate_sources = _view_source_ids(views)
    declared_sources = sync.get("retained_source_ids")
    if (
        not isinstance(declared_sources, list)
        or len(set(candidate_sources)) != len(candidate_sources)
        or sorted(candidate_sources) != declared_sources
    ):
        raise ValueError("candidate omits retained synchronized source")
    alignment = evidence.get("frame_alignment", {})
    spread = (
        float(alignment.get("max_global_time_spread_seconds", float("nan")))
        if isinstance(alignment, dict)
        else float("nan")
    )
    if (
        not isinstance(alignment, dict)
        or alignment.get("state") != "aligned"
        or not np.isfinite(spread)
        or spread > 0.05
    ):
        raise ValueError("candidate lacks verified aligned frame timing")
    offsets = sync.get("offsets", {})
    if not isinstance(offsets, dict):
        raise ValueError("candidate lacks synchronized offsets")
    global_times = []
    for view in views.values():
        if not isinstance(view, dict):
            raise ValueError("candidate has malformed view timing")
        seconds = view.get("frame_seconds")
        native_ids = view.get("frame_native_ids")
        source = view.get("source_id")
        if (
            not isinstance(seconds, (list, tuple))
            or not seconds
            or not isinstance(native_ids, (list, tuple))
            or len(seconds) != len(native_ids)
            or source not in offsets
        ):
            raise ValueError("candidate lacks native frame timing")
        for value, native_id in zip(seconds, native_ids, strict=True):
            try:
                pts_text, rational = native_id.split(":", 1)
                num_text, den_text = rational.split("/", 1)
                if (
                    abs(float(value) - int(pts_text) * int(num_text) / int(den_text))
                    > 1e-9
                ):
                    raise ValueError("native PTS disagrees with frame time")
            except (AttributeError, ValueError, ZeroDivisionError) as exc:
                raise ValueError("candidate has invalid native frame timing") from exc
        global_times.append(
            np.asarray(seconds, dtype=np.float64) + float(offsets[source])
        )
    if not all(np.isfinite(row).all() for row in global_times):
        raise ValueError("candidate has nonfinite synchronized frame timing")
    if len({len(row) for row in global_times}) != 1 or (
        float(np.max(np.ptp(global_times, axis=0))) > 0.05
    ):
        raise ValueError("candidate frames are not aligned on global timeline")
    unsolved = evidence.get("excluded_cameras", {})
    if not isinstance(unsolved, dict) or set(views) - set(candidate.cameras) != set(
        unsolved
    ):
        raise ValueError("candidate has unaccounted synchronized views")
    if set(evidence.get("missing_intrinsics", {})) - set(unsolved):
        raise ValueError("candidate has unresolved intrinsic compatibility")
    residual = bundle.get("residual_px", {})
    angles = bundle.get("triangulation_angle_deg", {})
    if (
        not isinstance(residual, dict)
        or not isinstance(angles, dict)
        or any(
            not np.isfinite(float(value))
            for value in (
                residual.get("p90", float("nan")),
                angles.get("p10", float("nan")),
                bundle.get("point_condition_number", float("nan")),
            )
        )
        or not isinstance(bundle.get("behind_camera_observations"), int)
    ):
        raise ValueError("candidate lacks finite bundle geometry diagnostics")
    models: dict[str, CameraModel] = {}
    excluded: dict[str, str] = dict(unsolved)
    diagnostics: dict[str, Any] = {
        "bundle": bundle,
        "cameras": {},
        "shared_points": 0,
        "thresholds": vars(thresholds),
        "projection_samples": [],
    }
    sources: set[str] = set()
    good_observations: dict[int, set[str]] = {}
    for camera_id, record in sorted(candidate.cameras.items()):
        view = views[camera_id]
        source_id = record.get("source_id")
        if (
            not isinstance(view, dict)
            or not source_id
            or view.get("source_id") != source_id
            or source_id in sources
            or (
                view.get("source_sha256")
                and source_id != f"source:{view['source_sha256']}"
            )
            or not view.get("frame_native_ids")
            or record.get("intrinsic_source") not in {None, "supplied"}
        ):
            raise ValueError(
                f"camera {camera_id} has incompatible source or intrinsic evidence"
            )
        sources.add(source_id)
        model = CameraModel(
            Intrinsics.model_validate(record["intrinsics"]),
            np.asarray(record["world_to_camera"], dtype=np.float64),
        )
        models[camera_id] = model
        coverage = record.get("spatial_coverage", {})
        if not isinstance(coverage, dict):
            raise ValueError(f"camera {camera_id} lacks spatial coverage")
        hull = float(coverage.get("image_hull_fraction", float("nan")))
        cells = float(coverage.get("occupied_grid_cells_4x4", float("nan")))
        if not np.isfinite(hull) or not np.isfinite(cells):
            raise ValueError(f"camera {camera_id} has invalid coverage diagnostics")
        reasons = []
        if hull < thresholds.min_hull_fraction or cells < thresholds.min_grid_cells:
            reasons.append("poor image coverage")
        points = []
        pixels = []
        point_indices = []
        for index, point in enumerate(candidate.static_points):
            observation = point.get("observations", {}).get(camera_id)
            if observation is not None:
                points.append(point["xyz"])
                pixels.append(observation)
                point_indices.append(index)
        if len(points) < thresholds.min_camera_points:
            reasons.append("insufficient point correspondences")
        p90: float | None = None
        positive = 0.0
        if points:
            xyz = np.asarray(points, dtype=np.float64)
            observed = np.asarray(pixels, dtype=np.float64)
            if (
                xyz.shape != (len(points), 3)
                or observed.shape != (len(points), 2)
                or not np.isfinite(xyz).all()
                or not np.isfinite(observed).all()
            ):
                raise ValueError("malformed scene correspondences")
            pose = model.world_to_camera
            depth = xyz @ pose[:3, :3][2] + pose[2, 3]
            positive = float(np.mean(depth > 0))
            if positive >= thresholds.min_cheirality:
                valid = depth > 0
                errors = np.linalg.norm(
                    model.project(xyz[valid]) - observed[valid], axis=1
                )
                p90 = float(np.percentile(errors, 90))
                for index, error in zip(
                    np.asarray(point_indices)[valid], errors, strict=True
                ):
                    if error <= thresholds.max_camera_p90_px:
                        good_observations.setdefault(int(index), set()).add(camera_id)
        if positive < thresholds.min_cheirality:
            reasons.append("insufficient cheirality")
        if p90 is None or p90 > thresholds.max_camera_p90_px:
            reasons.append("large camera reprojection residual")
        diagnostics["cameras"][camera_id] = {
            "correspondences": len(points),
            "p90_reprojection_px": p90,
            "positive_depth_fraction": positive,
            "coverage": coverage,
        }
        if reasons:
            excluded[camera_id] = "; ".join(reasons)
    retained = tuple(sorted(set(models) - set(excluded)))
    if len(retained) < 2:
        raise ValueError("fewer than two usable independent cameras remain")
    shared_indices = tuple(
        index
        for index, cameras in good_observations.items()
        if len(cameras & set(retained)) >= 2
    )
    shared = [candidate.static_points[index] for index in shared_indices]
    diagnostics["shared_points"] = len(shared)
    if len(shared) < thresholds.min_shared_points:
        raise ValueError("retained cameras lack coherent shared coverage")
    for index in shared_indices:
        if len(diagnostics["projection_samples"]) >= 24:
            break
        point = candidate.static_points[index]
        visible = [camera for camera in retained if camera in good_observations[index]]
        if len(visible) < 2:
            continue
        sample = {"point_index": index, "xyz_source": point["xyz"], "views": {}}
        for camera in visible:
            projected = models[camera].project(np.asarray([point["xyz"]]))[0]
            observed = point["observations"][camera]
            sample["views"][camera] = {
                "observed_px": observed,
                "projected_px": projected.tolist(),
                "error_px": float(np.linalg.norm(projected - observed)),
            }
        diagnostics["projection_samples"].append(sample)
    xyz = np.asarray([point["xyz"] for point in shared], dtype=np.float64)
    singular = np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    ray_angles = []
    for index, point in zip(shared_indices, shared, strict=True):
        location = np.asarray(point["xyz"], dtype=np.float64)
        visible = [camera for camera in retained if camera in good_observations[index]]
        for a, b in combinations(visible, 2):
            pose_a = models[a].world_to_camera
            pose_b = models[b].world_to_camera
            center_a = -pose_a[:3, :3].T @ pose_a[:3, 3]
            center_b = -pose_b[:3, :3].T @ pose_b[:3, 3]
            ray_a = location - center_a
            ray_b = location - center_b
            cosine = float(
                np.dot(ray_a, ray_b) / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b))
            )
            ray_angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    p10_angle = float(np.percentile(ray_angles, 10))
    diagnostics["retained_geometry"] = {
        "point_condition_number": condition,
        "p10_triangulation_angle_deg": p10_angle,
    }
    if condition > thresholds.max_condition or p10_angle < thresholds.min_angle_deg:
        raise ValueError("retained cameras have weak ray geometry or conditioning")
    if not excluded and (
        float(residual["p90"]) > thresholds.max_bundle_p90_px
        or float(angles["p10"]) < thresholds.min_angle_deg
        or float(bundle["point_condition_number"]) > thresholds.max_condition
        or bundle["behind_camera_observations"] != 0
    ):
        raise ValueError("candidate fails bundle geometry diagnostics")
    if len(retained) > 2:
        adjacency: dict[str, set[str]] = {camera: set() for camera in retained}
        for a, b in combinations(retained, 2):
            count = sum(
                a in good_observations[index] and b in good_observations[index]
                for index in shared_indices
            )
            if count >= thresholds.min_shared_points:
                adjacency[a].add(b)
                adjacency[b].add(a)
        visited = {retained[0]}
        frontier = [retained[0]]
        while frontier:
            for neighbor in adjacency[frontier.pop()] - visited:
                visited.add(neighbor)
                frontier.append(neighbor)
        if visited != set(retained):
            raise ValueError("retained camera overlap graph is disconnected")
    flags = tuple(
        [f"excluded {camera}: {reason}" for camera, reason in sorted(excluded.items())]
    )
    return SceneAssessment(retained, excluded, flags, shared_indices, diagnostics)
