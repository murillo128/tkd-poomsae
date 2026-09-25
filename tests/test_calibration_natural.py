"""Rendered static scene geometry and deliberate natural-scene ambiguities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from calibration.cameras import CameraModel
from calibration.natural import SceneView, estimate_scene, persist_scene_candidate
from calibration.natural_cli import (
    load_synchronization_artifact,
    nearest_global_refs,
)
from contracts.models import (
    Interval,
    Intrinsics,
    Provenance,
    Quality,
    Synchronization,
    SyncOffset,
)
from media import FrameRef


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
                frame_seconds=(5.0, 5.4, 5.8),
                frame_native_ids=("150:1/30", "162:1/30", "174:1/30"),
            )
        )
    return views


def synced(
    views: list[SceneView], offsets: dict[str, float] | None = None
) -> Synchronization:
    offsets = offsets or {}
    values = [offsets.get(view.source_id, 0.0) for view in views]
    return Synchronization(
        id="synthetic-sync",
        kind="synchronization",
        schema_version="1.0.0",
        provenance=Provenance(producer="sync.solver.v1", config_digest="a" * 64),
        reference_source_id=views[0].source_id,
        common_interval=Interval(start=max(values), end=min(10 + x for x in values)),
        offsets=[
            SyncOffset(
                source_id=view.source_id,
                automatic_seconds=offset,
                timing_reference=index == 0,
                retained=True,
                source_interval=Interval(start=0, end=10),
                global_interval=Interval(start=offset, end=10 + offset),
                quality=Quality(state="observed", score=0.9),
            )
            for index, (view, offset) in enumerate(zip(views, values, strict=True))
        ],
    )


def estimate(views: list[SceneView], offsets: dict[str, float] | None = None):  # type: ignore[no-untyped-def]
    return estimate_scene(views, synced(views, offsets))


def test_nonplanar_variable_camera_scene(tmp_path: Path) -> None:
    for baselines in ((0, 0.8), (0, 0.8, 1.4), (0, 0.8, 1.4, 2.0)):
        candidate = estimate(scene(baselines))
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
    planar = estimate(scene((0, 0.8), planar=True))
    assert planar.status != "candidate"
    assert any("planar" in reason for reason in planar.reasons)
    unknown = estimate(scene((0, 0.8), intrinsics=False))
    assert unknown.status == "unavailable"
    assert not unknown.cameras
    assert unknown.evidence["missing_intrinsics"]["camera-0"]["priors"]


def test_no_overlap_does_not_publish_cameras() -> None:
    views = scene((0, 0.8))
    blank = np.zeros_like(views[1].frames[0])
    views[1] = replace(views[1], frames=(blank,) * 3)
    candidate = estimate(views)
    assert candidate.status == "weak"
    assert not candidate.cameras


def test_near_zero_baseline_is_weak() -> None:
    candidate = estimate(scene((0, 0.005)))
    assert candidate.status == "weak"
    assert not candidate.cameras


def test_verified_nonzero_offset_recovers_aligned_scene() -> None:
    views = scene((0, 0.8))
    views[1] = replace(
        views[1],
        frame_seconds=(4.8, 5.2, 5.6),
        frame_native_ids=("144:1/30", "156:1/30", "168:1/30"),
    )
    candidate = estimate(views, {"source-1": 0.2})
    assert candidate.status == "candidate", candidate.reasons
    assert (
        candidate.evidence["frame_alignment"]["max_global_time_spread_seconds"] < 1e-9
    )
    assert candidate.evidence["synchronization"]["offsets"]["source-1"] == 0.2


def test_equal_pts_can_still_be_misaligned() -> None:
    views = scene((0, 0.8))
    candidate = estimate(views, {"source-1": 0.2})
    assert candidate.status == "unavailable"
    assert candidate.reasons == ["frames are not aligned in verified global time"]
    assert not candidate.cameras


def test_unverified_timing_is_unavailable() -> None:
    candidate = estimate_scene(scene((0, 0.8)))
    assert candidate.status == "unavailable"
    assert candidate.reasons == ["verified synchronization artifact is required"]


def test_cli_selects_native_pts_from_global_targets() -> None:
    frames = tuple(
        FrameRef(index, pts, 1, 30, True)
        for index, pts in enumerate((144, 150, 156, 162))
    )
    selected = nearest_global_refs(frames, [5.0, 5.2], offset=0.2)
    assert [ref.pts for ref in selected] == [144, 150]


def test_cli_rejects_modified_synchronization_artifact(tmp_path: Path) -> None:
    directory = tmp_path / "derived" / "synchronization" / ("f" * 64)
    directory.mkdir(parents=True)
    raw = synced(scene((0, 0.8))).model_dump_json().encode()
    (directory / "metadata.json").write_bytes(raw)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "key": directory.name,
                "metadata": {
                    "path": "metadata.json",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "arrays": {},
            }
        )
    )
    assert load_synchronization_artifact(directory).id == "synthetic-sync"
    (directory / "metadata.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_synchronization_artifact(directory)
