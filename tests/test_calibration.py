"""Generated board correspondences test geometry and evidence gates offline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration import (
    BoardSpec,
    CalibrationFailure,
    CameraModel,
    Capture,
    Detection,
    IntrinsicProfile,
    back_project_ray,
    estimate_calibration,
    observe_capture,
    persist_calibration,
)
from calibration.manifest import calibrate_from_manifest, load_manifest
from contracts.models import Intrinsics
from storage import ArtifactStore, StorageRoot


def generated_views() -> tuple[BoardSpec, list[Detection], dict[str, np.ndarray]]:
    board = BoardSpec(8, 6, 0.04, 0.025)
    points = board.object_corners()
    intrinsics = Intrinsics(fx=720, fy=710, cx=320, cy=240,
                            distortion=[0.01, -0.003, 0, 0])
    observations: list[Detection] = []
    anchors: dict[str, np.ndarray] = {}
    placement = np.diag([1.0, -1.0, -1.0, 1.0])
    for camera_number in range(2):
        for frame in range(9):
            rotation, _ = cv2.Rodrigues(np.array([
                0.12 + 0.055 * frame + 0.04 * camera_number,
                0.16 + 0.04 * (frame % 3),
                0.03 * (frame % 4),
            ]))
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = [
                -0.13 + 0.035 * (frame % 4) + 0.03 * camera_number,
                0.10 + 0.035 * (frame // 3),
                1.0 + 0.06 * (frame % 3),
            ]
            model = CameraModel(intrinsics, transform)
            pixels = model.project(points)
            if frame == 0:
                anchors[f"cam{camera_number}"] = transform @ placement
            capture = Capture(
                id=f"cam{camera_number}-frame{frame}",
                source_id=f"source{camera_number}",
                camera_id=f"cam{camera_number}",
                sha256=hashlib.sha256(f"{camera_number}:{frame}".encode()).hexdigest(),
                image_size=(640, 480), crop_xywh=(0, 0, 640, 480),
                board_to_world=tuple(tuple(float(x) for x in row) for row in placement)
                if frame == 0 else None,
            )
            observations.append(Detection(
                capture, np.arange(len(points), dtype=np.int32), pixels,
            ))
    return board, observations, anchors


def test_generated_multi_pose_recovers_cameras_and_persists(tmp_path: Path) -> None:
    board, observations, anchors = generated_views()
    calibration = estimate_calibration(board, observations)
    assert calibration.scale == "metric"
    assert len(calibration.cameras) == 2
    for camera in calibration.cameras:
        assert camera.intrinsic_source == "estimated"
        assert camera.corner_count == 9 * len(board.object_corners())
        assert camera.rms_reprojection_px is not None
        assert camera.rms_reprojection_px < 0.5
        assert abs(camera.intrinsics.fx - 720) < 15
        assert abs(camera.intrinsics.fy - 710) < 15
        assert np.allclose(
            camera.world_to_camera, anchors[camera.camera_id], atol=0.025
        )
        model = CameraModel(camera.intrinsics, np.asarray(camera.world_to_camera))
        world = np.array([[0.08, 0.06, 0], [0.12, 0.04, 0.1]])
        pixels = model.project(world)
        origins, directions = model.rays(pixels)
        for point, origin, direction in zip(world, origins, directions, strict=True):
            residual = point - origin
            assert np.linalg.norm(np.cross(residual, direction)) < 1e-4
        origin, ray = back_project_ray(model, tuple(pixels[0]))
        assert np.allclose(origin, origins[0])
        assert np.allclose(ray, directions[0])
        assert np.allclose(model.undistort(pixels).shape, (2, 2))
        with pytest.raises(ValueError, match="behind camera"):
            model.project(np.array([origins[0] - ray]))
    store = ArtifactStore(StorageRoot(tmp_path))
    first = persist_calibration(store, calibration, observations)
    second = persist_calibration(store, calibration, observations)
    assert first.path == second.path
    assert first.metadata == calibration


def test_single_planar_view_and_incompatible_profile_fail() -> None:
    board, observations, _ = generated_views()
    with pytest.raises(CalibrationFailure, match="at least three board poses"):
        estimate_calibration(board, [observations[0], observations[9]])
    profile = IntrinsicProfile(
        "cam0", 640, 480, (0, 0, 640, 480), 0,
        Intrinsics(fx=720, fy=710, cx=320, cy=240, distortion=[0.01, -0.003, 0, 0]),
    )
    imported = estimate_calibration(board, [observations[0], observations[9]],
                                    {"cam0": profile, "cam1": IntrinsicProfile(
                                        "cam1", 640, 480, (0, 0, 640, 480), 0,
                                        profile.intrinsics,
                                    )})
    assert all(c.intrinsic_source == "imported" for c in imported.cameras)
    with pytest.raises(ValueError, match="incompatible intrinsic profile"):
        estimate_calibration(board, [observations[0], observations[9]], {
            "cam0": IntrinsicProfile("cam0", 640, 480, (1, 0, 639, 480), 0,
                                     profile.intrinsics),
            "cam1": IntrinsicProfile("cam1", 640, 480, (0, 0, 640, 480), 0,
                                     profile.intrinsics),
        })
    with pytest.raises(ValueError, match="nonmonotonic"):
        IntrinsicProfile(
            "cam0", 640, 480, (0, 0, 640, 480), 0,
            Intrinsics(fx=720, fy=710, cx=320, cy=240,
                       distortion=[-10, 0, 0, 0]),
        )


def test_detect_rendered_charuco_board(tmp_path: Path) -> None:
    board = BoardSpec(8, 6, 0.04, 0.025)
    image = board.board().generateImage((800, 600), marginSize=30)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path = tmp_path / "board.png"
    path.write_bytes(encoded.tobytes())
    capture = Capture(
        "board", "source", "cam", hashlib.sha256(path.read_bytes()).hexdigest(),
        (800, 600), (0, 0, 800, 600),
    )
    detection = observe_capture(path, capture, board)
    assert len(detection.corner_ids) >= 8
    with pytest.raises(CalibrationFailure, match="hash mismatch"):
        observe_capture(path, Capture(
            "bad", "source", "cam", "0" * 64, (800, 600), (0, 0, 800, 600),
        ), board)


def test_manifest_keeps_capture_geometry_and_placement(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "board": {"squares_x": 8, "squares_y": 6,
                  "square_length_m": 0.04, "marker_length_m": 0.025},
        "captures": [{
            "id": "one", "source_id": "source", "camera_id": "cam",
            "image_file": "one.png", "sha256": "a" * 64,
            "image_size": [640, 480], "crop_xywh": [10, 20, 600, 400],
            "rotation_cw": 90,
            "board_to_world": np.diag([1.0, -1.0, -1.0, 1.0]).tolist(),
        }],
        "profiles": {"cam": {
            "width_px": 640, "height_px": 480,
            "crop_xywh": [10, 20, 600, 400], "rotation_cw": 90,
            "intrinsics": {"fx": 700, "fy": 700, "cx": 200, "cy": 300},
        }},
    }
    path = tmp_path / "captures.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _, captures, profiles = load_manifest(path)
    image_file, capture = captures[0]
    assert image_file == tmp_path / "one.png"
    assert capture.decoded_size == (400, 600)
    placement = np.diag([1.0, -1.0, -1.0, 1.0])
    assert capture.board_to_world == tuple(map(tuple, placement.tolist()))
    profiles["cam"].require_compatible("cam", capture.image_size,
                                       capture.crop_xywh, capture.rotation_cw)


def test_weak_or_unplaced_views_fail_explicitly() -> None:
    board, observations, _ = generated_views()
    with pytest.raises(CalibrationFailure, match="no known board placement"):
        estimate_calibration(board, [
            Detection(
                Capture(
                    d.capture.id, d.capture.source_id, d.capture.camera_id,
                    d.capture.sha256, d.capture.image_size, d.capture.crop_xywh,
                ), d.corner_ids, d.pixels,
            ) for d in observations
        ])
    repeated = [
        Detection(d.capture, d.corner_ids, observations[0].pixels)
        if d.capture.camera_id == "cam0" else d
        for d in observations
    ]
    with pytest.raises(CalibrationFailure, match="coverage/diversity"):
        estimate_calibration(board, repeated)


def test_manifest_to_immutable_artifact_from_rendered_frames(tmp_path: Path) -> None:
    board = BoardSpec(8, 6, 0.04, 0.025)
    pattern = board.board().generateImage((800, 600), marginSize=0)
    outer = np.array([
        [0, 0, 0], [0.32, 0, 0], [0.32, 0.24, 0], [0, 0.24, 0],
    ])
    image_outer = np.array([
        [0, 0], [799, 0], [799, 599], [0, 599],
    ], dtype=np.float32)
    placement = np.diag([1.0, -1.0, -1.0, 1.0])
    manifest: dict[str, object] = {
        "version": 1,
        "board": {"squares_x": 8, "squares_y": 6,
                  "square_length_m": 0.04, "marker_length_m": 0.025},
        "captures": [], "profiles": {},
    }
    captures = manifest["captures"]
    assert isinstance(captures, list)
    for camera_number in range(2):
        for frame in range(9):
            rotation, _ = cv2.Rodrigues(np.array([
                0.12 + 0.055 * frame + 0.04 * camera_number,
                0.16 + 0.04 * (frame % 3), 0.03 * (frame % 4),
            ]))
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = [
                -0.13 + 0.035 * (frame % 4) + 0.03 * camera_number,
                0.10 + 0.035 * (frame // 3), 1.0 + 0.06 * (frame % 3),
            ]
            model = CameraModel(
                Intrinsics(fx=720, fy=710, cx=320, cy=240), transform,
            )
            projected = model.project(outer).astype(np.float32)
            warp = cv2.getPerspectiveTransform(image_outer, projected)
            image = cv2.warpPerspective(pattern, warp, (640, 480), borderValue=255)
            encoded_ok, encoded = cv2.imencode(".png", image)
            assert encoded_ok
            name = f"cam{camera_number}-{frame}.png"
            (tmp_path / name).write_bytes(encoded.tobytes())
            record: dict[str, object] = {
                "id": name, "source_id": f"source{camera_number}",
                "camera_id": f"cam{camera_number}", "image_file": name,
                "sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
                "image_size": [640, 480], "crop_xywh": [0, 0, 640, 480],
            }
            if frame == 0:
                record["board_to_world"] = placement.tolist()
            captures.append(record)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate, handle = calibrate_from_manifest(manifest_path, tmp_path / "data")
    assert handle.metadata == candidate
    assert len(candidate.cameras) == 2
    for camera in candidate.cameras:
        assert camera.intrinsic_source == "estimated"
        assert camera.corner_count is not None and camera.corner_count >= 200
        assert camera.rms_reprojection_px is not None
        assert camera.rms_reprojection_px < 1
        assert abs(camera.intrinsics.fx - 720) < 40
