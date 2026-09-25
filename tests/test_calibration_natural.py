"""Rendered static scene geometry and deliberate natural-scene ambiguities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from calibration.cameras import CameraModel
from calibration.natural import SceneView, estimate_scene, persist_scene_candidate
from contracts.models import Intrinsics


def scene(
    baselines: tuple[float, ...],
    *,
    planar: bool = False,
    intrinsics: bool = True,
) -> list[SceneView]:
    rng = np.random.default_rng(23)
    model = Intrinsics(fx=700, fy=710, cx=400, cy=300)
    count = 300
    z = np.full(count, 6.0) if planar else rng.uniform(4, 8, count)
    xyz = np.column_stack((rng.uniform(-2, 2, count), rng.uniform(-1.4, 1.4, count), z))
    views = []
    for camera_index, baseline in enumerate(baselines):
        pose = np.eye(4)
        pose[0, 3] = baseline
        pixels = CameraModel(model, pose).project(xyz)
        image: NDArray[np.uint8] = np.zeros((600, 800), np.uint8)
        for point_index, (u, v) in enumerate(pixels):
            x, y = int(round(u)), int(round(v))
            if 12 < x < 788 and 12 < y < 588:
                patch_rng = np.random.default_rng(point_index)
                patch = patch_rng.integers(0, 2, (17, 17), dtype=np.uint8) * 200 + 30
                current = image[y - 8 : y + 9, x - 8 : x + 9]
                image[y - 8 : y + 9, x - 8 : x + 9] = np.maximum(current, patch)
        views.append(
            SceneView(
                f"camera-{camera_index}",
                f"source-{camera_index}",
                (image, image, image),
                model if intrinsics else None,
            )
        )
    return views


def test_nonplanar_variable_camera_scene(tmp_path: Path) -> None:
    for baselines in ((0, 0.8), (0, 0.8, 1.4), (0, 0.8, 1.4, 2.0)):
        candidate = estimate_scene(scene(baselines))
        assert candidate.status == "candidate", candidate.reasons
        assert len(candidate.cameras) == len(baselines)
        assert len(candidate.static_points) >= 50
        assert candidate.evidence["bundle"]["residual_px"]["p90"] < 1
        assert candidate.evidence["bundle"]["triangulation_angle_deg"]["p10"] > 1
        assert all(
            camera["spatial_coverage"]["occupied_grid_cells_4x4"] >= 4
            for camera in candidate.cameras.values()
        )
        assert candidate.to_dict()["scale"] == "unresolved"
        assert candidate.to_dict()["ground"] == "unresolved"
        path = persist_scene_candidate(candidate, tmp_path)
        assert persist_scene_candidate(candidate, tmp_path) == path
        assert json.loads(path.read_text())["status"] == "candidate"
        first = np.asarray(candidate.cameras["camera-0"]["world_to_camera"])
        other = np.asarray(candidate.cameras["camera-1"]["world_to_camera"])
        assert abs(np.linalg.norm(other[:3, 3] - first[:3, 3]) - 1.0) < 1e-6


def test_planar_and_uncalibrated_geometry_rejected() -> None:
    planar = estimate_scene(scene((0, 0.8), planar=True))
    assert planar.status != "candidate"
    assert any("planar" in reason for reason in planar.reasons)
    unknown = estimate_scene(scene((0, 0.8), intrinsics=False))
    assert unknown.status == "unavailable"
    assert not unknown.cameras
    assert unknown.evidence["missing_intrinsics"]["camera-0"]["priors"]


def test_no_overlap_does_not_publish_cameras() -> None:
    views = scene((0, 0.8))
    blank = np.zeros_like(views[1].frames[0])
    views[1] = SceneView("camera-1", "source-1", (blank,) * 3, views[1].intrinsics)
    candidate = estimate_scene(views)
    assert candidate.status == "weak"
    assert not candidate.cameras


def test_near_zero_baseline_is_weak() -> None:
    candidate = estimate_scene(scene((0, 0.005)))
    assert candidate.status == "weak"
    assert not candidate.cameras


def test_unsynchronized_native_frames_are_unavailable() -> None:
    views = scene((0, 0.8))
    views = [
        SceneView(
            view.camera_id,
            view.source_id,
            view.frames,
            view.intrinsics,
            frame_seconds=(5.0 + index * 0.4,) * 3,
        )
        for index, view in enumerate(views)
    ]
    candidate = estimate_scene(views)
    assert candidate.status == "unavailable"
    assert candidate.reasons == ["native frames are not synchronized"]
