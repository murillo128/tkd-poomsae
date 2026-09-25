"""Independent numerical and adversarial checks for the fixture library."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.synthetic import RECIPES, Camera, Scene, make_scene


def distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.dist(a, b)


@pytest.mark.parametrize("name,count", [("clean_two", 2), ("clean_three", 3),
                                       ("clean_four", 4)])
def test_calibrated_camera_counts_and_ground(name: str, count: int) -> None:
    scene = make_scene(name)
    assert len(scene.cameras) == count
    assert scene.oracle()["ground_plane"] == (0, 0, 1, 0)
    for camera in scene.cameras:
        matrix = camera.world_to_camera()
        assert len(matrix) == 16
        # The camera center must map to the origin under the exported transform.
        for row in range(3):
            transformed = sum(matrix[row * 4 + col] * camera.center[col]
                              for col in range(3)) + matrix[row * 4 + 3]
            assert abs(transformed) < 1e-12
        assert abs(camera.project(camera.target)[0] - camera.width / 2) < 1e-12  # type: ignore[index]
        assert abs(camera.project(camera.target)[1] - camera.height / 2) < 1e-12  # type: ignore[index]


def test_projection_and_distortion_against_independent_equations() -> None:
    scene = make_scene("clean_three")
    point = (0.25, 0.3, 1.4)
    for camera in scene.cameras:
        transform = camera.world_to_camera()
        coordinates = [sum(transform[row * 4 + col] * point[col]
                           for col in range(3)) + transform[row * 4 + 3]
                       for row in range(3)]
        x, y = coordinates[0] / coordinates[2], coordinates[1] / coordinates[2]
        k1, k2, p1, p2 = camera.distortion
        radius = x * x + y * y
        expected = (
            camera.width / 2 + camera.focal * (
                x * (1 + k1 * radius + k2 * radius**2)
                + 2 * p1 * x * y + p2 * (radius + 2 * x**2)),
            camera.height / 2 + camera.focal * (
                y * (1 + k1 * radius + k2 * radius**2)
                + p1 * (radius + 2 * y**2) + 2 * p2 * x * y),
        )
        projected = camera.project(point)
        assert projected is not None
        assert distance(projected, expected) < 1e-10


def test_articulation_morphology_and_pivot_geometry() -> None:
    scene = make_scene("clean_three")
    start = scene.landmarks(1.65)
    end = scene.landmarks(2.3)
    assert distance(start["left_forefoot"], end["left_forefoot"]) < 1e-10
    assert distance(start["left_heel"], end["left_heel"]) > 0.05
    assert scene.landmarks(1.2)["right_heel"][2] > start["right_heel"][2] + 0.3
    assert scene.landmarks(1.8)["right_heel"][2] == pytest.approx(0.0)
    assert scene.oracle()["morphology"]["shoulder_width"] == pytest.approx(0.46)
    assert scene.landmarks(0)["left_index_3"] != scene.landmarks(0)["left_thumb_3"]
    assert scene.orientations(0)["root"] != scene.orientations(2.3)["root"]
    assert scene.orientations(2.5)["head"] != scene.orientations(2.5)["root"]
    assert scene.ground_state(1.2)["right_contact"] == "no_contact"
    assert scene.ground_state(1.8)["right_contact"] == "contact"
    assert scene.ground_state(2.0)["left_pivot_region"] == "forefoot"
    # At the crossing, the left arm is closer to the body's forward direction.
    at_crossing = scene.landmarks(2.65)
    assert at_crossing["left_wrist"][1] > at_crossing["right_wrist"][1]
    actions = scene.oracle()["events"]["actions"]
    assert actions[0]["interval"][0] < actions[1]["interval"][0]
    assert actions[0]["interval"][1] > actions[1]["interval"][0]
    assert len(actions[-1]["tracks"]) == 2


def test_source_time_is_native_and_oracle_only(tmp_path: Path) -> None:
    scene = make_scene("mixed_timing")
    manifest = scene.source_manifest()
    serialized = json.dumps(manifest)
    assert "offset" not in serialized
    assert "event" not in serialized
    assert "landmarks" not in serialized
    assert len({(s["width"], s["height"]) for s in manifest["sources"]}) == 4
    assert len({(c.rate_num, c.rate_den) for c in scene.cameras}) == 4
    for camera in scene.cameras:
        frame = camera.frames()[17]
        assert frame.pts / frame.time_base_den == pytest.approx(frame.source_seconds)
        assert frame.source_seconds + camera.offset == pytest.approx(
            scene.oracle()["frame_global_seconds"][camera.id][17])
    left = scene.cameras[0].frames()[20]
    right = scene.cameras[1].frames()[20]
    assert left.source_seconds + scene.cameras[0].offset != pytest.approx(
        right.source_seconds + scene.cameras[1].offset)
    out = tmp_path / "oracle.json"
    scene.write_oracle(out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    scene.write_oracle(out)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == digest
    assert json.loads(out.read_text())["recipe"]["version"] == 1
    mutable_oracle = scene.oracle()
    mutable_oracle["morphology"]["shoulder_width"] = -1
    assert scene.oracle()["morphology"]["shoulder_width"] > 0
    ppm = scene.render_ppm("cam0", 20)
    assert ppm.startswith(b"P6\n160 120\n255\n")
    assert ppm == make_scene("mixed_timing").render_ppm("cam0", 20)


def angle_between_rays(camera_a: Camera, camera_b: Camera) -> float:
    target = (0.0, 0.0, 1.0)
    a = tuple(target[i] - camera_a.center[i] for i in range(3))
    b = tuple(target[i] - camera_b.center[i] for i in range(3))
    cosine = sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.dist(a, (0, 0, 0)) * math.dist(b, (0, 0, 0)))
    return math.acos(max(-1.0, min(1.0, cosine)))


def test_negative_recipes_exercise_failure_modes() -> None:
    normal = make_scene("clean_three")
    bad = make_scene("bad_camera")
    point = bad.landmarks(1.0)["pelvis"]
    good_camera = bad.cameras[0]
    bad_camera = bad.cameras[2]
    assert distance(bad.observed(bad_camera, 1.0)["pelvis"],
                    bad_camera.project(point)) > 30  # type: ignore[arg-type]
    assert distance(bad.observed(good_camera, 1.0)["pelvis"],
                    good_camera.project(point)) < 1e-10  # type: ignore[arg-type]
    degenerate = make_scene("degenerate_baseline")
    assert angle_between_rays(*degenerate.cameras[:2]) < 0.01
    assert angle_between_rays(*normal.cameras[:2]) > 0.5
    assert make_scene("unknown_scale").oracle()["metres_per_world_unit"] is None
    assert make_scene("low_texture").static_features() == {}
    static = normal.static_features()
    assert len([p for p in static.values() if p[2] == 0]) == 6
    assert len([p for p in static.values() if p[2] > 0]) == 3
    occluded = make_scene("occluded")
    assert "right_wrist" not in occluded.observed(occluded.cameras[0], 1.2)
    assert "right_wrist" in occluded.observed(occluded.cameras[1], 1.2)
    noisy = make_scene("noisy_missing")
    expected = set(normal.observed(normal.cameras[0], 1.0))
    actual = set(noisy.observed(noisy.cameras[0], 1.0))
    assert expected - actual
    assert any(distance(normal.observed(normal.cameras[0], 1.0)[name],
                        noisy.observed(noisy.cameras[0], 1.0)[name]) > 0.1
               for name in expected & actual)
    outliers = make_scene("outliers")
    assert any(distance(normal.observed(normal.cameras[0], 1.0)[name],
                        outliers.observed(outliers.cameras[0], 1.0)[name]) > 30
               for name in set(normal.observed(normal.cameras[0], 1.0))
               & set(outliers.observed(outliers.cameras[0], 1.0)))


@pytest.mark.parametrize("name", list(RECIPES))
def test_recipe_is_repeatable(name: str) -> None:
    a, b = make_scene(name), make_scene(name)
    assert a.source_manifest() == b.source_manifest()
    assert a.oracle() == b.oracle()
    assert a.render_ppm("cam0", 20) == b.render_ppm("cam0", 20)


def test_recipe_version_gate() -> None:
    data: dict[str, Any] = {**RECIPES["clean_two"].__dict__, "version": 2}
    from tests.fixtures.synthetic import Recipe

    with pytest.raises(ValueError, match="unsupported"):
        Scene(Recipe(**data))
