"""Synthetic ground evidence, gauge conversion, and unavailable capabilities."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from calibration import (
    CameraModel,
    GroundEvidence,
    SizeEvidence,
    persist_scene_calibration,
    resolve_scene,
    scene_revision,
)
from calibration.natural import SceneCandidate
from contracts.models import Calibration, Intrinsics
from storage import ArtifactStore, StorageRoot


def scene() -> SceneCandidate:
    rng = np.random.default_rng(8)
    floor = np.column_stack((rng.uniform(-2, 2, 40), rng.uniform(-1, 1, 40),
                             rng.normal(0, 0.002, 40)))
    floor[3, 2] = 0.6  # Robust fit must discard this mislabeled point.
    above = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 2.0]])
    wall = np.column_stack((np.full(100, 3.0), rng.uniform(-2, 2, 100),
                            rng.uniform(0, 3, 100)))
    points = np.vstack((floor, above, wall))
    intrinsics = Intrinsics(fx=700, fy=700, cx=320, cy=240)
    cameras = {}
    for number in range(2):
        pose = np.eye(4)
        pose[0, 3] = number
        cameras[f"cam{number}"] = {
            "source_id": f"source{number}",
            "intrinsics": intrinsics.model_dump(),
            "world_to_camera": pose.tolist(),
        }
    static_points = [{"xyz": point.tolist()} for point in points]
    for index in (40, 41):
        static_points[index]["observations"] = {
            camera_id: CameraModel(
                intrinsics, np.asarray(record["world_to_camera"]),
            ).project(points[index:index + 1])[0].tolist()
            for camera_id, record in cameras.items()
        }
    return SceneCandidate("candidate", [], cameras, static_points, {
        "vertical_references": [{
            "id": "upright-1", "kind": "known_upright",
            "point_indices": [40, 41],
            "source_kind": "upright_target", "source_id": "fixture-upright",
            "producer": "synthetic-upright-detector-v1",
        }],
    })


def evidence(candidate: SceneCandidate) -> GroundEvidence:
    return GroundEvidence("floor-classification", scene_revision(candidate),
                          tuple(range(40)), (40, 41), (0, 1),
                          (40, 41), "upright-1",
                          producer="synthetic-floor-classifier-v1")


def measured(candidate: SceneCandidate, unit: str = "cm") -> SizeEvidence:
    return SizeEvidence("measured-edge", scene_revision(candidate), (0, 1),
                        200 if unit == "cm" else 2, unit,  # type: ignore[arg-type]
                        producer="tape-measure")


def test_floor_fit_ignores_larger_wall_and_preserves_handedness(tmp_path: Path) -> None:
    candidate = scene()
    ground = evidence(candidate)
    calibration = resolve_scene(candidate, ground, measured(candidate))
    assert calibration.ground_status == calibration.scale_status == "resolved"
    assert calibration.world_unit == "m"
    assert calibration.ground_frame is not None
    assert calibration.ground_frame.inlier_count >= 39
    assert calibration.ground_frame.inlier_count < calibration.ground_frame.sample_count
    assert 3 not in calibration.ground_frame.inlier_indices
    assert calibration.ground_frame.axis_uncertainty_rad < 0.01
    transform = np.asarray(calibration.ground_frame.source_to_world)
    points = np.asarray([p["xyz"] for p in candidate.static_points])
    mapped = points @ transform[:3, :3].T + transform[:3, 3]
    assert np.percentile(np.abs(mapped[:40, 2]), 90) < 0.02
    assert np.all(mapped[40:42, 2] > 0)
    assert np.linalg.det(transform[:3, :3]) > 0
    assert np.allclose(np.linalg.norm(mapped[1] - mapped[0]), 2)
    for camera in calibration.cameras:
        model = CameraModel(camera.intrinsics, np.asarray(camera.world_to_camera))
        original = CameraModel(camera.intrinsics, np.asarray(
            candidate.cameras[camera.camera_id]["world_to_camera"]))
        source_points = points[[2, 5, 40]]
        assert np.allclose(model.project(mapped[[2, 5, 40]]),
                           original.project(source_points), atol=1e-7)
    handle = persist_scene_calibration(ArtifactStore(StorageRoot(tmp_path)),
                                       candidate, calibration)
    assert handle.metadata == calibration


def test_unavailable_ground_and_scale_are_explicit() -> None:
    candidate = scene()
    unresolved = resolve_scene(candidate)
    assert unresolved.ground_status == unresolved.scale_status == "unresolved"
    assert unresolved.ground_frame is None and unresolved.ground_z is None
    assert unresolved.scale == unresolved.world_unit == "arbitrary"
    metric = resolve_scene(candidate, size=measured(candidate))
    assert metric.scale_status == "resolved" and metric.ground_status == "unresolved"
    assert resolve_scene(candidate, evidence(candidate)).scale_status == "unresolved"
    with pytest.raises(ValueError, match="floor samples"):
        GroundEvidence("wall-only", scene_revision(candidate), (42, 43), (40,),
                       (42, 43), (40, 41), "upright-1", producer="fixture")
    with pytest.raises(ValueError, match="sign evidence"):
        GroundEvidence("no-sign", scene_revision(candidate), tuple(range(40)),
                       (), (0, 1), (40, 41), "upright-1", producer="fixture")
    candidate.static_points[41]["xyz"] = [0.5, 0.1, -2.0]
    with pytest.raises(ValueError, match="vertical sign evidence is ambiguous"):
        resolve_scene(candidate, evidence(candidate))


def test_wall_only_plane_cannot_become_ground() -> None:
    original = scene()
    wall = SceneCandidate(
        "candidate", [], original.cameras,
        original.static_points[42:57] + original.static_points[:2]
        + original.static_points[40:42],
        {"vertical_references": [{
            "id": "upright-1", "kind": "known_upright",
            "point_indices": [17, 18],
            "source_kind": "upright_target", "source_id": "fixture-upright",
            "producer": "synthetic-upright-detector-v1",
        }]},
    )
    claimed_floor = GroundEvidence(
        "wall-claim", scene_revision(wall), tuple(range(15)), (15, 16),
        (0, 1), (17, 18), "upright-1",
        producer="synthetic-floor-classifier-v1",
    )
    with pytest.raises(ValueError, match="conflicts with independent vertical"):
        resolve_scene(wall, claimed_floor)
    assert resolve_scene(wall).ground_status == "unresolved"
    wall.evidence["vertical_references"] = []
    with pytest.raises(ValueError, match="independent upright cue"):
        resolve_scene(
            wall, replace(claimed_floor, source_revision=scene_revision(wall)),
        )


def test_scene_vertical_cue_needs_independent_multiview_support() -> None:
    candidate = scene()
    candidate.evidence["vertical_references"][0]["producer"] = (
        "synthetic-floor-classifier-v1"
    )
    with pytest.raises(ValueError, match="independent upright cue"):
        resolve_scene(candidate, evidence(candidate))
    candidate = scene()
    candidate.static_points[40]["observations"]["cam1"][0] += 10
    with pytest.raises(ValueError, match="inconsistent multiview geometry"):
        resolve_scene(candidate, evidence(candidate))


def test_revisions_units_and_manual_recovery() -> None:
    candidate = scene()
    base = resolve_scene(candidate, evidence(candidate), measured(candidate))
    metres = resolve_scene(candidate, evidence(candidate), measured(candidate, "m"))
    assert np.allclose(base.cameras[0].world_to_camera,
                       metres.cameras[0].world_to_camera)
    assert base.id != metres.id  # Raw evidence and its unit remain auditable.
    manual = replace(evidence(candidate), kind="manual", author="operator",
                     reason="floor marker checked against survey")
    changed = resolve_scene(candidate, manual, measured(candidate))
    assert changed.id != base.id
    assert changed.ground_frame is not None
    assert changed.ground_frame.evidence_kind == "manual"
    assert base.source_revision == changed.source_revision
    with pytest.raises(ValueError, match="positive"):
        replace(measured(candidate), length=0)
    with pytest.raises(ValueError, match="unit"):
        replace(measured(candidate), unit="px")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="manual ground"):
        replace(evidence(candidate), kind="manual")
    with pytest.raises(ValueError, match="incompatible scene revision"):
        resolve_scene(candidate, replace(evidence(candidate), source_revision="old"))
    with pytest.raises(ValueError, match="status and units"):
        Calibration.model_validate(base.model_dump() | {"scale_status": "unresolved"})
