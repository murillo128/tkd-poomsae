"""Versioned, seeded multiview fixtures; never imported by production code.

The source-facing manifest and rendered PPM bytes contain no oracle timing,
geometry, or semantic annotations. Tests access those through ``Scene.oracle``.
World XY is ground, +Z is up, and image +Y points down.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]

RECIPE_VERSION = 1
SEED = 20260925


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def unit(a: Vec3) -> Vec3:
    length = math.sqrt(dot(a, a))
    return (a[0] / length, a[1] / length, a[2] / length)


def rotate_z(p: Vec3, angle: float) -> Vec3:
    c, s = math.cos(angle), math.sin(angle)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])


def ramp(t: float, start: float, end: float) -> float:
    return max(0.0, min(1.0, (t - start) / (end - start)))


def pulse(t: float, start: float, peak: float, end: float) -> float:
    return min(ramp(t, start, peak), 1.0 - ramp(t, peak, end))


@dataclass(frozen=True)
class Camera:
    id: str
    center: Vec3
    target: Vec3
    width: int
    height: int
    focal: float
    rate_num: int
    rate_den: int
    offset: float  # oracle: global seconds = native source seconds + offset
    distortion: tuple[float, float, float, float]  # k1, k2, p1, p2

    def axes(self) -> tuple[Vec3, Vec3, Vec3]:
        forward = unit(sub(self.target, self.center))
        right = unit(cross(forward, (0.0, 0.0, 1.0)))
        down = cross(forward, right)
        return right, down, forward

    def world_to_camera(self) -> tuple[float, ...]:
        axes = self.axes()
        return tuple(v for row in axes for v in (*row, -dot(row, self.center))) + (
            0.0, 0.0, 0.0, 1.0)

    def project(self, point: Vec3) -> Vec2 | None:
        right, down, forward = self.axes()
        delta = sub(point, self.center)
        depth = dot(forward, delta)
        if depth <= 0:
            return None
        x, y = dot(right, delta) / depth, dot(down, delta) / depth
        k1, k2, p1, p2 = self.distortion
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return (self.focal * xd + self.width / 2,
                self.focal * yd + self.height / 2)

    def frames(self, duration: float = 4.5) -> tuple[Frame, ...]:
        rate = Fraction(self.rate_num, self.rate_den)
        count = math.ceil((duration - self.offset) * float(rate))
        return tuple(Frame(self.id, index, index * self.rate_den,
                           self.rate_num, index / float(rate))
                     for index in range(count + 1))


@dataclass(frozen=True)
class Frame:
    camera_id: str
    index: int
    pts: int
    time_base_den: int  # native source seconds = pts / time_base_den
    source_seconds: float


@dataclass(frozen=True)
class Recipe:
    name: str
    version: int = RECIPE_VERSION
    seed: int = SEED
    camera_count: int = 3
    mixed_sources: bool = False
    noise_px: float = 0.0
    missing_fraction: float = 0.0
    outlier_fraction: float = 0.0
    bad_camera: bool = False
    degenerate_baseline: bool = False
    metric_scale: float | None = 1.0
    low_texture: bool = False
    occluded: bool = False


RECIPES: dict[str, Recipe] = {
    "clean_two": Recipe("clean_two", camera_count=2),
    "clean_three": Recipe("clean_three"),
    "clean_four": Recipe("clean_four", camera_count=4),
    "mixed_timing": Recipe("mixed_timing", camera_count=4, mixed_sources=True),
    "noisy_missing": Recipe("noisy_missing", noise_px=1.5,
                            missing_fraction=0.12),
    "outliers": Recipe("outliers", outlier_fraction=0.08),
    "bad_camera": Recipe("bad_camera", bad_camera=True),
    "degenerate_baseline": Recipe("degenerate_baseline",
                                  degenerate_baseline=True),
    "unknown_scale": Recipe("unknown_scale", metric_scale=None),
    "low_texture": Recipe("low_texture", low_texture=True),
    "occluded": Recipe("occluded", occluded=True),
}


MORPHOLOGY = {
    "shoulder_width": 0.46,
    "hip_width": 0.32,
    "upper_arm_length": 0.31,
    "forearm_length": 0.27,
    "thigh_length": 0.44,
    "shin_length": 0.43,
    "foot_length": 0.26,
}

TARGET_COLORS: dict[str, tuple[int, int, int]] = {
    f"target_{column}_{row}": (40 + column * 90, 60 + row * 110,
                               210 - column * 50 - row * 20)
    for column in range(3) for row in range(2)
}

EVENTS: dict[str, Any] = {
    "execution_interval": (0.0, 4.0),
    "preroll": (-0.5, 0.0),
    "postroll": (4.0, 4.5),
    "actions": (
        {"id": "left_arm", "kind": "arm", "tracks": ("left_arm",),
         "interval": (0.25, 1.15)},
        {"id": "right_arm", "kind": "arm", "tracks": ("right_arm",),
         "interval": (0.6, 1.6)},
        {"id": "kick", "kind": "kick", "tracks": ("right_leg",),
         "interval": (0.8, 1.45)},
        {"id": "recovery_placement", "kind": "placement",
         "tracks": ("right_leg",), "interval": (1.45, 1.8)},
        {"id": "pivot", "kind": "pivot", "tracks": ("left_leg", "root"),
         "interval": (1.65, 2.3)},
        {"id": "formal_arms", "kind": "special",
         "tracks": ("left_arm", "right_arm"), "interval": (2.1, 3.3)},
    ),
    "crossing": {"time": 2.65, "front": "left_forearm",
                 "back": "right_forearm"},
    "contacts": {"right_foot_airborne": (0.8, 1.65),
                 "right_foot_placed": 1.8,
                 "left_forefoot_pivot": (1.65, 2.3)},
    "keyframes": {"kick_chamber": 0.95, "kick_extension": 1.2,
                  "kick_retraction": 1.45, "pivot_end": 2.3,
                  "formal_arm_center": 2.65},
}


class Scene:
    def __init__(self, recipe: Recipe):
        if recipe.version != RECIPE_VERSION or recipe.camera_count not in (2, 3, 4):
            raise ValueError("unsupported synthetic recipe")
        self.recipe = recipe
        centers: tuple[Vec3, ...] = (
            (0.0, -4.2, 2.2), (3.8, 0.2, 2.4), (-3.7, 0.8, 2.0),
            (0.6, 4.0, 2.6))
        if recipe.degenerate_baseline:
            centers = ((0.0, -4.2, 2.2), (0.015, -4.2, 2.2),
                       (0.03, -4.2, 2.2), (0.045, -4.2, 2.2))
        rates = ((24, 1), (30_000, 1001), (25, 1), (60, 1))
        sizes = ((160, 120), (192, 144), (128, 96), (176, 132))
        offsets = (-0.75, -0.537, -0.683, -0.612)
        self.cameras = tuple(
            Camera(f"cam{i}", centers[i], (0.0, 0.0, 1.0),
                   *(sizes[i] if recipe.mixed_sources else sizes[0]),
                   145.0 if recipe.mixed_sources else 150.0,
                   *(rates[i] if recipe.mixed_sources else (24, 1)),
                   offsets[i] if recipe.mixed_sources else -0.75,
                   (0.012, -0.004, 0.001, -0.0005))
            for i in range(recipe.camera_count))

    def root(self, t: float) -> tuple[Vec3, float]:
        def travel(at: float) -> Vec3:
            progress = ramp(at, 0.0, 3.7)
            return (0.28 * progress, 0.38 * progress, 0.9)

        yaw = 0.7 * ramp(t, 1.65, 2.3)
        if t < 1.65:
            return travel(t), yaw
        pivot_anchor = add(travel(1.65), (-0.16, 0.19, 0.0))
        pivot_root = sub(pivot_anchor, rotate_z((-0.16, 0.19, 0.0), yaw))
        if t <= 2.3:
            return pivot_root, yaw
        end_root = sub(pivot_anchor, rotate_z((-0.16, 0.19, 0.0), 0.7))
        return add(end_root, sub(travel(t), travel(2.3))), yaw

    def landmarks(self, t: float) -> dict[str, Vec3]:
        root, yaw = self.root(t)
        left_move = ramp(t, 0.25, 1.15)
        right_move = ramp(t, 0.6, 1.6)
        formal = pulse(t, 2.1, 2.65, 3.3)
        kick = pulse(t, 0.8, 1.2, 1.65)
        recovery_lift = 0.08 * pulse(t, 1.35, 1.65, 1.8)
        placement = ramp(t, 1.65, 1.8)
        local: dict[str, Vec3] = {
            "pelvis": (0.0, 0.0, 0.0),
            "left_shoulder": (-0.23, 0.0, 0.48),
            "right_shoulder": (0.23, 0.0, 0.48),
            "left_elbow": (-0.32 + 0.2 * formal, 0.18 + 0.1 * left_move,
                           0.30 + 0.12 * formal),
            "right_elbow": (0.32 - 0.2 * formal, 0.12 + 0.08 * right_move,
                            0.30 + 0.12 * formal),
            "left_wrist": (-0.35 + 0.6 * formal,
                           0.24 + 0.12 * left_move + 0.12 * formal, 0.2),
            "right_wrist": (0.35 - 0.6 * formal,
                            0.15 + 0.13 * right_move - 0.04 * formal, 0.2),
            "neck": (0.0, 0.0, 0.57),
            "head_center": (0.0, 0.01, 0.76),
            "head_front": (0.0, 0.13, 0.76),
            "left_hip": (-0.16, 0.0, -0.03),
            "right_hip": (0.16, 0.0, -0.03),
            "left_knee": (-0.16, 0.05, -0.43),
            "right_knee": (0.16, 0.06 + 0.32 * kick, -0.43 + 0.18 * kick),
            "left_ankle": (-0.16, 0.02, -0.82),
            "right_ankle": (0.16, 0.02 + 0.52 * kick + 0.24 * placement,
                            -0.82 + 0.42 * kick + recovery_lift),
            "left_heel": (-0.16, -0.05, -0.9),
            "left_forefoot": (-0.16, 0.19, -0.9),
            "right_heel": (0.16, -0.05 + 0.52 * kick + 0.24 * placement,
                           -0.9 + 0.42 * kick + recovery_lift),
            "right_forefoot": (0.16, 0.21 + 0.52 * kick + 0.24 * placement,
                               -0.9 + 0.42 * kick + recovery_lift),
        }
        for side in ("left", "right"):
            wrist = local[f"{side}_wrist"]
            for finger_index, finger in enumerate(
                ("thumb", "index", "middle", "ring", "pinky")
            ):
                for joint in range(1, 4):
                    local[f"{side}_{finger}_{joint}"] = (
                        wrist[0] + (finger_index - 2) * 0.018,
                        wrist[1] + joint * (0.025 if finger != "thumb" else 0.018),
                        wrist[2] + (finger_index - 2) * 0.004)
        return {name: add(root, rotate_z(point, yaw))
                for name, point in local.items()}

    def orientations(self, t: float) -> dict[str, tuple[float, float, float, float]]:
        _, yaw = self.root(t)
        head_yaw = yaw + 0.2 * pulse(t, 2.0, 2.5, 3.1)
        return {"root": (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
                "head": (math.cos(head_yaw / 2), 0.0, 0.0,
                         math.sin(head_yaw / 2))}

    def ground_state(self, t: float) -> dict[str, Any]:
        """Physical contact truth, independent of semantic action boundaries."""
        right_contact = "no_contact" if 0.8 < t < 1.8 else "contact"
        return {"left_contact": "contact", "right_contact": right_contact,
                "support": "left" if right_contact == "no_contact" else "both",
                "left_pivot_region": "forefoot" if 1.65 <= t <= 2.3 else None}

    def static_features(self) -> dict[str, Vec3]:
        if self.recipe.low_texture:
            return {}
        features: dict[str, Vec3] = {}
        for column, x in enumerate((-0.8, 0.0, 0.8)):
            for row, y in enumerate((-0.6, 0.6)):
                features[f"target_{column}_{row}"] = (x, y, 0.0)
        features.update({"wall_a": (-1.2, 1.4, 1.0),
                         "wall_b": (1.1, 1.3, 1.7),
                         "wall_c": (0.4, -1.4, 0.65)})
        return features

    def observed(self, camera: Camera, t: float) -> dict[str, Vec2]:
        points = {**self.static_features(), **self.landmarks(t)}
        result: dict[str, Vec2] = {}
        for name, point in points.items():
            if (self.recipe.occluded and camera.id == "cam0"
                    and 0.85 <= t <= 1.5 and name.startswith("right_")):
                continue
            key = f"{self.recipe.seed}/{camera.id}/{t:.9f}/{name}"
            digest = hashlib.sha256(key.encode()).digest()
            selector = int.from_bytes(digest[:8], "big") / 2**64
            if selector < self.recipe.missing_fraction:
                continue
            pixel = camera.project(point)
            if pixel is None:
                continue
            if self.recipe.noise_px:
                rng = random.Random(int.from_bytes(digest[8:16], "big"))
                pixel = (pixel[0] + rng.gauss(0, self.recipe.noise_px),
                         pixel[1] + rng.gauss(0, self.recipe.noise_px))
            if (self.recipe.bad_camera and camera.id == "cam2"
                    or selector < self.recipe.outlier_fraction):
                pixel = (pixel[0] + 35.0, pixel[1] - 24.0)
            if 0 <= pixel[0] < camera.width and 0 <= pixel[1] < camera.height:
                result[name] = pixel
        return result

    def source_manifest(self) -> dict[str, Any]:
        """Only native source identity and media recipe; no oracle fields."""
        return {"recipe_version": RECIPE_VERSION, "seed": self.recipe.seed,
                "calibration_target": None if (
                    self.recipe.low_texture or self.recipe.metric_scale is None
                ) else {
                    "pattern": "six-color-floor-grid-v1",
                    "column_spacing_m": 0.8,
                    "row_spacing_m": 1.2,
                    "colors_rgb": deepcopy(TARGET_COLORS),
                },
                "sources": [
                    {"id": camera.id, "width": camera.width,
                     "height": camera.height,
                     "rate_num": camera.rate_num,
                     "rate_den": camera.rate_den,
                     "frames": [asdict(frame) for frame in camera.frames()]}
                    for camera in self.cameras]}

    def oracle(self) -> dict[str, Any]:
        times = tuple(round(-0.5 + i * 0.05, 8) for i in range(101))
        return {
            "recipe": asdict(self.recipe),
            "ground_plane": (0.0, 0.0, 1.0, 0.0),
            "metres_per_world_unit": self.recipe.metric_scale,
            "morphology": deepcopy(MORPHOLOGY),
            "cameras": [{"id": c.id, "center": c.center,
                         "world_to_camera": c.world_to_camera(),
                         "intrinsics": (c.focal, c.focal, c.width / 2,
                                        c.height / 2),
                         "distortion": c.distortion,
                         "offset_seconds": c.offset} for c in self.cameras],
            "static_features": self.static_features(),
            "times": times,
            "landmarks": [self.landmarks(t) for t in times],
            "ideal_image_points": [
                {camera.id: {name: camera.project(point)
                             for name, point in self.landmarks(t).items()}
                 for camera in self.cameras}
                for t in times],
            "frame_global_seconds": {
                camera.id: [frame.source_seconds + camera.offset
                            for frame in camera.frames()]
                for camera in self.cameras},
            "orientations": [self.orientations(t) for t in times],
            "ground_state": [self.ground_state(t) for t in times],
            "events": deepcopy(EVENTS),
            "known_bad_camera": "cam2" if self.recipe.bad_camera else None,
        }

    def render_ppm(self, camera_id: str, frame_index: int) -> bytes:
        """Render a small deterministic RGB source frame without annotations."""
        camera = next(c for c in self.cameras if c.id == camera_id)
        frame = camera.frames()[frame_index]
        t = frame.source_seconds + camera.offset
        pixels = bytearray((12, 12, 12) * (camera.width * camera.height))
        static = self.static_features()
        for name, (x, y) in self.observed(camera, t).items():
            px, py = round(x), round(y)
            color = TARGET_COLORS.get(name, (220, 220, 220)) if name in static \
                else (245, 150, 45)
            for iy in range(max(0, py - 1), min(camera.height, py + 2)):
                for ix in range(max(0, px - 1), min(camera.width, px + 2)):
                    pos = 3 * (iy * camera.width + ix)
                    pixels[pos:pos + 3] = bytes(color)
        return f"P6\n{camera.width} {camera.height}\n255\n".encode() + pixels

    def write_oracle(self, path: Path) -> None:
        path.write_text(json.dumps(self.oracle(), sort_keys=True,
                                   separators=(",", ":")) + "\n")


def make_scene(name: str) -> Scene:
    return Scene(RECIPES[name])
