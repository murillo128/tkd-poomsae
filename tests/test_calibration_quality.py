"""Cross-component acceptance of scene cameras, ground, and measured scale."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from calibration import CameraModel, resolve_scene, scene_revision
from calibration.natural import SceneCandidate, estimate_scene
from calibration.quality import assess_scene, require_synchronization
from contracts.models import Calibration, Intrinsics
from pipeline.runner import CapabilityUnavailable, _scene_calibration
from storage import ArtifactHandle, ArtifactKey, hash_config
from tests.test_calibration_ground import evidence, measured, scene
from tests.test_calibration_natural import scene as rendered_scene
from tests.test_calibration_natural import synced


def multiview(count: int) -> SceneCandidate:
    candidate = scene()
    points = np.asarray([point["xyz"] for point in candidate.static_points])
    for index in range(2, count):
        camera_id = f"cam{index}"
        record = dict(candidate.cameras["cam0"])
        record["source_id"] = f"source{index}"
        pose = np.eye(4)
        pose[0, 3] = index * 0.8
        pose[2, 3] = 5
        record["world_to_camera"] = pose.tolist()
        candidate.cameras[camera_id] = record
        candidate.evidence["views"][camera_id] = {
            "source_id": record["source_id"],
            "frame_native_ids": ["1:1/30"],
            "frame_seconds": [1 / 30],
        }
        candidate.evidence["synchronization"]["offsets"][record["source_id"]] = 0.0
        candidate.evidence["synchronization"]["retained_source_ids"] = sorted(
            record["source_id"] for record in candidate.cameras.values()
        )
        pixels = CameraModel(
            candidate_camera_intrinsics(candidate, camera_id),
            pose,
        ).project(points)
        for point, pixel in zip(candidate.static_points, pixels, strict=True):
            point["observations"][camera_id] = pixel.tolist()
    return candidate


def candidate_camera_intrinsics(
    candidate: SceneCandidate,
    camera_id: str,
) -> Intrinsics:
    return Intrinsics.model_validate(candidate.cameras[camera_id]["intrinsics"])


@pytest.mark.parametrize("count", [2, 3, 4])
def test_valid_views_reach_metric_world_projection(count: int) -> None:
    candidate = multiview(count)
    calibration = resolve_scene(candidate, evidence(candidate), measured(candidate))
    assert calibration.publication_status == "complete"
    assert calibration.camera_status == "resolved"
    assert len(calibration.cameras) == count
    assert not calibration.excluded_cameras
    assert calibration.world_unit == "m"
    assert calibration.ground_frame is not None
    assert calibration.projection_debug[0]["projection_samples"]
    transform = np.asarray(calibration.ground_frame.source_to_world)
    source_xyz = np.asarray(candidate.static_points[40]["xyz"])
    world_xyz = source_xyz @ transform[:3, :3].T + transform[:3, 3]
    for record in calibration.cameras:
        projected = CameraModel(
            record.intrinsics,
            np.asarray(record.world_to_camera),
        ).project(world_xyz[None])[0]
        observed = candidate.static_points[40]["observations"][record.camera_id]
        assert np.allclose(projected, observed, atol=1e-6)


def test_bad_view_is_excluded_only_with_coherent_retained_pair() -> None:
    candidate = multiview(3)
    for point in candidate.static_points:
        point["observations"]["cam2"][0] += 30
    candidate.status = "weak"
    candidate.evidence["bundle"]["residual_px"]["p90"] = 20
    calibration = resolve_scene(candidate, evidence(candidate), measured(candidate))
    assert [camera.camera_id for camera in calibration.cameras] == ["cam0", "cam1"]
    assert "reprojection" in calibration.excluded_cameras["cam2"]
    assert calibration.publication_status == "complete"
    assert calibration.projection_debug[0]["shared_points"] >= 24
    pair = multiview(2)
    for point in pair.static_points:
        point["observations"]["cam1"][0] += 30
    with pytest.raises(ValueError, match="fewer than two usable"):
        resolve_scene(pair)


def test_partial_capabilities_and_ambiguous_floor() -> None:
    candidate = multiview(2)
    no_scale = resolve_scene(candidate, evidence(candidate))
    assert no_scale.publication_status == "partial"
    assert no_scale.camera_status == no_scale.ground_status == "resolved"
    assert no_scale.scale_status == "unresolved"
    assert "metric scale unresolved" in no_scale.quality_flags
    no_floor = resolve_scene(candidate, size=measured(candidate))
    assert no_floor.publication_status == "partial"
    assert no_floor.ground_status == "unresolved"
    assert no_floor.scale_status == "resolved"
    ambiguous = replace(
        evidence(candidate),
        above_indices=(0, 1),
        source_revision=scene_revision(candidate),
    )
    with pytest.raises(ValueError, match="vertical sign evidence is ambiguous"):
        resolve_scene(candidate, ambiguous, measured(candidate))


def test_disconnected_third_view_keeps_supported_pair() -> None:
    views = rendered_scene((0, 0.8, 1.4))
    blank = np.zeros_like(views[2].frames[0])
    views[2] = replace(views[2], frames=(blank,) * 3)
    candidate = estimate_scene(views, synced(views))
    assessment = assess_scene(candidate)
    assert assessment.retained == ("camera-0", "camera-1")
    assert "camera-2" in assessment.excluded
    assert assessment.diagnostics["shared_points"] >= 24


def test_candidate_is_bound_to_exact_sync_offsets() -> None:
    views = rendered_scene((0, 0.8))
    sync = synced(views)
    candidate = estimate_scene(views, sync)
    require_synchronization(candidate, sync)
    changed = synced(views, {views[1].source_id: 0.1})
    with pytest.raises(ValueError, match="offsets disagree"):
        require_synchronization(candidate, changed)


def test_omitted_retained_sync_camera_fails_closed() -> None:
    views = rendered_scene((0, 0.8, 1.4))
    sync = synced(views)
    candidate = estimate_scene(views[:2], sync)
    assert candidate.status == "candidate"
    with pytest.raises(ValueError, match="omits retained synchronized source"):
        require_synchronization(candidate, sync)
    with pytest.raises(ValueError, match="omits retained synchronized source"):
        resolve_scene(candidate)


def test_offline_stage_consumes_verified_candidate(
    tmp_path: Path,
) -> None:
    views = rendered_scene((0, 0.8))
    sync = synced(views)
    candidate = estimate_scene(views, sync)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate.to_dict()))
    key = ArtifactKey(
        layer="calibration",
        inputs={},
        schema_version="1.0.0",
        algorithm_revision="scene-quality-v1",
        config_digest=hash_config({}),
    )
    output = _scene_calibration(
        key,
        {"sync": ArtifactHandle(tmp_path, sync, {})},
        {"candidate": str(path)},
    )
    assert isinstance(output.artifact, Calibration)
    assert output.artifact.camera_status == "resolved"
    assert output.artifact.publication_status == "partial"
    assert output.artifact.scale_status == "unresolved"
    with pytest.raises(ValueError, match="offsets disagree"):
        _scene_calibration(
            key,
            {
                "sync": ArtifactHandle(
                    tmp_path, synced(views, {views[1].source_id: 0.1}), {}
                )
            },
            {"candidate": str(path)},
        )
    with pytest.raises(CapabilityUnavailable, match="candidate is missing"):
        _scene_calibration(
            key,
            {"sync": ArtifactHandle(tmp_path, sync, {})},
            {"candidate": str(tmp_path / "absent.json")},
        )
