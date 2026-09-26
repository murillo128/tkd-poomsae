"""Deterministic N-view consensus and weighted distorted-pixel refinement."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from calibration.cameras import CameraModel
from contracts.models import (
    Calibration,
    CameraCalibration,
    EvidenceState,
    Landmark,
    Landmark2D,
    Landmark3D,
    Quality,
    StrictModel,
)
from sync.alignment import CameraQuery, TimeQuery

REVISION = "raw-nview-triangulation-v1"


class TriangulationConfig(StrictModel):
    pixel_sigma: float = Field(default=2.0, gt=0)
    max_reprojection_px: float = Field(default=8.0, gt=0)
    min_ray_angle_degrees: float = Field(default=1.0, gt=0, lt=90)
    max_condition: float = Field(default=1e6, gt=1)
    min_source_score: float = Field(default=0.05, ge=0, le=1)
    # Assumed pixel speed bound, used to propagate timing/bracket uncertainty.
    image_speed_px_s: float = Field(default=1000.0, gt=0)
    unknown_sync_sigma_seconds: float = Field(default=0.02, gt=0)
    unknown_calibration_sigma_px: float = Field(default=4.0, gt=0)


@dataclass
class View:
    query: CameraQuery
    point: Landmark2D
    camera: CameraModel
    sigma: float
    center: NDArray[np.float64]
    ray: NDArray[np.float64]
    diagnostic: dict[str, Any]


def _weak(quality: Quality, config: TriangulationConfig) -> bool:
    return quality.state == "unknown" or (
        quality.score is not None and quality.score < config.min_source_score
    )


def _view(
    query: CameraQuery,
    point: Landmark2D,
    calibration: CameraCalibration | None,
    config: TriangulationConfig,
) -> tuple[View | None, dict[str, Any]]:
    diagnostic: dict[str, Any] = {
        "camera_id": query.camera_id,
        "source_id": query.source_id,
        "landmark": point.name,
        "state": query.state,
        "endpoints": [
            {
                "observation_id": obs.id,
                "frame": obs.frame.model_dump(mode="json"),
                "weight": weight,
                "provenance": obs.provenance.model_dump(mode="json"),
            }
            for obs, weight in zip(query.endpoints, query.weights, strict=True)
        ],
        "xy_px": point.xy_px,
        "raw_score": (
            point.raw_score.model_dump(mode="json") if point.raw_score else None
        ),
        "source_quality": point.quality.model_dump(mode="json"),
        "synchronization_quality": query.synchronization_quality.model_dump(
            mode="json"
        ),
        "reasons": [],
        "used": False,
        "residual_px": None,
    }
    reasons = diagnostic["reasons"]
    if query.state == "unknown" or query.reasons:
        reasons.extend(query.reasons or ["unknown_time_join"])
    if point.xy_px is None or point.quality.state not in ("observed", "interpolated"):
        reasons.append("unsupported_source_landmark")
    if _weak(point.quality, config) or (
        point.raw_visibility is not None and point.raw_visibility <= 0
    ):
        reasons.append("insufficient_source_quality")
    if _weak(query.synchronization_quality, config):
        reasons.append("insufficient_synchronization_quality")
    if calibration is None or calibration.source_id != query.source_id:
        reasons.append("missing_or_mismatched_calibration")
    elif _weak(calibration.quality, config):
        reasons.append("insufficient_calibration_quality")
    if reasons:
        return None, diagnostic
    assert calibration is not None and point.xy_px is not None
    diagnostic["calibration_quality"] = calibration.quality.model_dump(mode="json")
    # Quality scores only modulate a declared noise model, never a probability.
    score = point.quality.score if point.quality.score is not None else 0.5
    sigma2 = config.pixel_sigma**2 / max(score, 0.05)
    calibration_sigma = calibration.rms_reprojection_px
    if calibration_sigma is None:
        calibration_sigma = config.unknown_calibration_sigma_px
        reasons.append("assumed_calibration_pixel_error")
    calibration_score = calibration.quality.score
    sigma2 += calibration_sigma**2 / max(
        calibration_score if calibration_score is not None else 0.5, 0.05
    )
    timing = query.synchronization_quality.uncertainty
    if timing is None:
        timing = config.unknown_sync_sigma_seconds
        reasons.append("assumed_synchronization_seconds_error")
    sync_score = query.synchronization_quality.score
    sigma2 += (config.image_speed_px_s * timing) ** 2 / max(
        sync_score if sync_score is not None else 0.5, 0.05
    )
    if query.state == "bracket":
        span = abs(
            query.endpoints[-1].frame.source_seconds
            - query.endpoints[0].frame.source_seconds
        )
        sigma2 += (config.image_speed_px_s * span / 2) ** 2
        reasons.append("interpolation_motion_bound")
    camera = CameraModel(calibration.intrinsics, np.array(calibration.world_to_camera))
    centers, rays = camera.rays([point.xy_px])
    sigma = float(np.sqrt(sigma2))
    diagnostic["assumed_sigma_px"] = sigma
    return View(
        query, point, camera, sigma, centers[0], rays[0], diagnostic
    ), diagnostic


def _geometry(
    views: list[View], config: TriangulationConfig
) -> tuple[NDArray[np.float64] | None, float, float]:
    angles = [
        float(np.degrees(np.arccos(np.clip(abs(a.ray @ b.ray), 0, 1))))
        for a, b in combinations(views, 2)
        if np.linalg.norm(a.center - b.center) > 1e-9
    ]
    angle = max(angles, default=0.0)
    matrices = [(np.eye(3) - np.outer(v.ray, v.ray)) / v.sigma**2 for v in views]
    normal = sum(matrices, np.zeros((3, 3)))
    condition = float(np.linalg.cond(normal))
    if angle < config.min_ray_angle_degrees or condition > config.max_condition:
        return None, angle, condition
    rhs = sum((m @ v.center for m, v in zip(matrices, views)), np.zeros(3))
    return np.asarray(np.linalg.solve(normal, rhs), dtype=np.float64), angle, condition


def _residual(view: View, xyz: NDArray[np.float64]) -> float | None:
    try:
        projected = view.camera.project([xyz])[0]
    except ValueError:
        return None
    return float(np.linalg.norm(projected - view.point.xy_px))


def triangulate_point(
    query: TimeQuery,
    name: Landmark,
    calibration: Calibration,
    config: TriangulationConfig | None = None,
) -> tuple[Landmark3D, dict[str, Any]]:
    config = config or TriangulationConfig()
    cameras = {c.camera_id: c for c in calibration.cameras}
    if len(cameras) != len(calibration.cameras):
        raise ValueError("calibration contains duplicate cameras")
    diagnostics: dict[str, Any] = {
        "name": name,
        "views": [],
        "reasons": [],
        "ray_angle_degrees": None,
        "condition": None,
        "covariance_world": None,
        "uncertainty_kind": "conditional_local_linearized_noise_bound",
    }
    views: list[View] = []
    seen: set[str] = set()
    for camera_query in query.cameras:
        point = next((p for p in camera_query.landmarks if p.name == name), None)
        if point is None:
            point = Landmark2D(name=name, xy_px=None, quality=Quality(state="unknown"))
        view, diagnostic = _view(
            camera_query, point, cameras.get(camera_query.camera_id or ""), config
        )
        diagnostics["views"].append(diagnostic)
        if view is not None:
            camera_id = camera_query.camera_id or ""
            if camera_id in seen:
                raise ValueError("multiple observations from one camera at one time")
            seen.add(camera_id)
            global_score = calibration.quality.score
            view.sigma /= np.sqrt(
                max(global_score if global_score is not None else 0.5, 0.05)
            )
            diagnostic["assumed_sigma_px"] = view.sigma
            views.append(view)

    def unknown(reason: str) -> tuple[Landmark3D, dict[str, Any]]:
        diagnostics["reasons"].append(reason)
        for view in views:
            view.diagnostic["used"] = False
            view.diagnostic["reasons"].append(reason)
        return Landmark3D(
            name=name, xyz_world=None, quality=Quality(state="unknown")
        ), diagnostics

    if calibration.camera_status != "resolved" or _weak(calibration.quality, config):
        return unknown("unresolved_or_unsupported_calibration")
    if len(views) < 2:
        return unknown("insufficient_independent_views")
    # Every geometrically useful pair proposes a hypothesis; consensus is over N views.
    candidates: list[tuple[list[View], NDArray[np.float64], float]] = []
    for pair in combinations(views, 2):
        xyz, _, _ = _geometry(list(pair), config)
        if xyz is None:
            continue
        residuals = [_residual(v, xyz) for v in views]
        retained = [
            v
            for v, r in zip(views, residuals)
            if r is not None and r <= config.max_reprojection_px
        ]
        if len(retained) >= 2:
            candidates.append(
                (
                    retained,
                    xyz,
                    sum(
                        (r / v.sigma) ** 2
                        for v, r in zip(views, residuals)
                        if any(v is item for item in retained) and r is not None
                    ),
                )
            )
    if not candidates:
        return unknown("no_consistent_front_facing_geometry")
    candidates.sort(key=lambda c: (-len(c[0]), c[2]))
    retained, xyz, _ = candidates[0]
    # Equal-support incompatible solutions cannot establish which views are correct.
    for other, alternative, _ in candidates[1:]:
        if len(other) != len(retained):
            continue
        if {v.query.camera_id for v in other} == {v.query.camera_id for v in retained}:
            continue
        if any(
            (r := _residual(v, alternative)) is None or r > config.max_reprojection_px
            for v in retained
        ):
            return unknown("ambiguous_equal_support_consensus")
    # Refine all consensus observations, rechecking gates after robust fitting.
    fit = None
    subsets: set[tuple[str | None, ...]] = set()
    for _ in range(2 * len(views)):
        subset = tuple(v.query.camera_id for v in retained)
        if subset in subsets:
            return unknown("unstable_consensus_after_refinement")
        subsets.add(subset)
        seed, angle, condition = _geometry(retained, config)
        diagnostics.update(
            ray_angle_degrees=angle,
            condition=condition if np.isfinite(condition) else None,
        )
        if seed is None:
            return unknown("ill_conditioned_ray_geometry")

        def residuals_at(point: NDArray[np.float64]) -> NDArray[np.float64]:
            residuals = []
            for view in retained:
                try:
                    pixel = view.camera.project([point])[0]
                    residuals.extend((pixel - view.point.xy_px) / view.sigma)
                except ValueError:
                    residuals.extend([1e6, 1e6])
            return np.asarray(residuals, dtype=np.float64)

        fit = least_squares(
            residuals_at,
            seed,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=100,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        xyz = fit.x
        if not fit.success or not np.isfinite(xyz).all():
            return unknown("refinement_failed")
        filtered = [
            v
            for v in views
            if (r := _residual(v, xyz)) is not None and r <= config.max_reprojection_px
        ]
        if len(filtered) < 2:
            return unknown("insufficient_views_after_refinement")
        if tuple(v.query.camera_id for v in filtered) == subset:
            break
        retained = filtered
    else:
        return unknown("unstable_consensus_after_refinement")
    assert fit is not None
    # The Jacobian is weighted and robust-loss adjusted. Never shrink the assumed noise.
    normal = fit.jac.T @ fit.jac
    condition = float(np.linalg.cond(normal))
    diagnostics["condition"] = condition if np.isfinite(condition) else None
    if condition > config.max_condition:
        return unknown("ill_conditioned_reprojection_geometry")
    inflation = max(1.0, float(fit.fun @ fit.fun) / max(1, len(fit.fun) - 3))
    covariance = np.linalg.inv(normal) * inflation
    uncertainty = float(np.sqrt(np.linalg.eigvalsh(covariance)[-1]))
    diagnostics["covariance_world"] = covariance.tolist()
    source_ids: list[str] = []
    for view in views:
        r = _residual(view, xyz)
        view.diagnostic["residual_px"] = r
        used = any(view is v for v in retained)
        view.diagnostic["used"] = used
        if used:
            source_ids.extend(obs.id for obs in view.query.endpoints)
        else:
            view.diagnostic["reasons"].append(
                "behind_camera" if r is None else "reprojection_outlier"
            )
    state: EvidenceState = (
        "interpolated"
        if any(v.query.state == "bracket" for v in retained)
        else "observed"
    )
    return Landmark3D(
        name=name,
        xyz_world=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        quality=Quality(
            state=state,
            uncertainty=uncertainty,
            source_ids=list(dict.fromkeys(source_ids)),
        ),
    ), diagnostics
