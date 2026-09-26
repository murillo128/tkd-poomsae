"""Deterministic, evidence-qualified geometry; no semantic correctness rules."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from contracts.models import (
    Landmark,
    Landmark3D,
    MotionSample,
    Quality,
    Quaternion,
    StrictModel,
)

Vector = tuple[float, float, float]
Side = Literal["left", "right"]
SIDES: tuple[Side, Side] = ("left", "right")
RelationAxis = Literal["right", "front", "up", "crossing"]
RelationValue = Literal[
    "right_of",
    "left_of",
    "in_front_of",
    "behind",
    "above",
    "below",
    "crossed",
    "not_crossed",
    "unknown",
]
REVISION = "detailed-geometry-v1"


class GeometryConfig(StrictModel):
    min_length: float = Field(default=1e-8, gt=0)
    min_sine: float = Field(default=0.1, gt=0, le=1)
    max_angle_uncertainty: float = Field(default=0.25, gt=0, le=1)
    uncertainty_multiplier: float = Field(default=3, ge=1)
    open_max_flexion: float = Field(default=0.35, ge=0, lt=1)
    fist_min_flexion: float = Field(default=1.0, ge=1, lt=3.14)


class Quantity(StrictModel):
    value: float | None = None
    quality: Quality


class Direction(StrictModel):
    axis_world: Vector | None = None
    quality: Quality


class Frame(StrictModel):
    name: str
    origin_world: Vector | None = None
    orientation: Quaternion | None = None
    parent: Literal["world"] = "world"
    quality: Quality


class Finger(StrictModel):
    bends_rad: list[Quantity]
    configuration: Literal["extended", "flexed", "indeterminate"]
    quality: Quality


class Hand(StrictModel):
    frame: Frame
    longitudinal: Direction
    fingers: dict[str, Finger]
    wrist_alignment_rad: Quantity
    descriptor: Literal["open", "fist", "indeterminate"]
    quality: Quality


class Foot(StrictModel):
    longitudinal: Direction
    frame: Frame


class Head(StrictModel):
    frame: Frame
    orientation_in_torso: Quaternion | None
    orientation_quality: Quality
    neck_direction_in_torso: Vector | None
    neck_quality: Quality


class Relation(StrictModel):
    subject: str
    object: str
    reference_frame: str
    axis: RelationAxis
    value: RelationValue
    quality: Quality
    front_entity: str | None = None
    front_quality: Quality = Field(default_factory=lambda: Quality(state="unknown"))


class DetailedSample(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    global_seconds: float
    reconstruction_id: str
    representation: Literal["raw", "fitted", "regularized"]
    algorithm_revision: str = REVISION
    config: GeometryConfig
    body_frame: Frame
    hands: dict[str, Hand]
    feet: dict[str, Foot]
    head: Head
    relations: list[Relation]


def _known(value: float | None) -> float:
    assert value is not None
    return value


def _vec(v: Any) -> Vector:
    return float(v[0]), float(v[1]), float(v[2])


def _sources(points: list[Landmark3D]) -> list[str]:
    return list(dict.fromkeys(s for p in points for s in p.quality.source_ids))


def _quality(points: list[Landmark3D], uncertainty: float | None = None) -> Quality:
    return Quality(
        state="unknown" if uncertainty is None else "inferred",
        uncertainty=uncertainty,
        source_ids=_sources(points),
    )


def _usable(points: list[Landmark3D], count: int) -> bool:
    return len(points) == count and all(
        p.xyz_world is not None
        and p.quality.state != "unknown"
        and p.quality.uncertainty is not None
        and p.quality.source_ids
        for p in points
    )


def _sigma(points: list[Landmark3D]) -> float:
    # Sum rather than sqrt(N): no independence assumption or precision gain.
    return sum(float(p.quality.uncertainty or 0) for p in points)


def _matrix(frame: Frame) -> np.ndarray[Any, Any]:
    assert frame.orientation is not None
    w, x, y, z = frame.orientation.wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()  # type: ignore[no-any-return]


def _quaternion(matrix: np.ndarray[Any, Any]) -> Quaternion:
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return Quaternion(wxyz=(float(w), float(x), float(y), float(z)))


def direction(points: list[Landmark3D], config: GeometryConfig) -> Direction:
    """Endpoint direction only; axial twist deliberately remains unconstrained."""
    result = Direction(quality=_quality(points))
    if not _usable(points, 2):
        return result
    delta = np.array(points[1].xyz_world) - points[0].xyz_world
    length = float(np.linalg.norm(delta))
    if length <= config.min_length:
        return result
    uncertainty = _sigma(points) / length
    if uncertainty > config.max_angle_uncertainty:
        return result
    return Direction(
        axis_world=_vec(delta / length), quality=_quality(points, uncertainty)
    )


def frame_from_axes(
    name: str,
    origin: Vector,
    x: np.ndarray[Any, Any],
    y_hint: np.ndarray[Any, Any],
    points: list[Landmark3D],
    config: GeometryConfig,
) -> Frame:
    """Right-handed frame from two independently constrained labeled axes."""
    result = Frame(name=name, origin_world=origin, quality=_quality(points))
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y_hint))
    if (
        min(nx, ny) <= config.min_length
        or not _usable(points, len(points))
        or not points
    ):
        return result
    x, y_hint = x / nx, y_hint / ny
    z = np.cross(x, y_hint)
    sine = float(np.linalg.norm(z))
    if sine < config.min_sine:
        return result
    uncertainty = 2 * _sigma(points) / (min(nx, ny) * sine)
    if uncertainty > config.max_angle_uncertainty:
        return result
    z /= sine
    y = np.cross(z, x)
    return Frame(
        name=name,
        origin_world=origin,
        orientation=_quaternion(np.column_stack([x, y, z])),
        quality=_quality(points, uncertainty),
    )


def _pick(sample: MotionSample, *names: str) -> list[Landmark3D]:
    lookup = {p.name: p for p in sample.landmarks}
    return [lookup[n] for n in names if n in lookup]


def body_frame(sample: MotionSample, config: GeometryConfig) -> Frame:
    """x anatomical right, y anatomical front, z torso up; origin hip midpoint."""
    p = _pick(sample, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if not _usable(p, 4):
        return Frame(name="body", quality=_quality(p))
    lh, rh, ls, rs = np.array([v.xyz_world for v in p])
    origin = (lh + rh) / 2
    up = (ls + rs) / 2 - origin
    right = rh - lh
    # right cross front = up; front = up cross right.
    length = float(np.linalg.norm(right))
    if length <= config.min_length:
        return Frame(name="body", origin_world=_vec(origin), quality=_quality(p))
    up_length = float(np.linalg.norm(up))
    if up_length <= config.min_length:
        return Frame(name="body", origin_world=_vec(origin), quality=_quality(p))
    sine = float(np.linalg.norm(np.cross(up / up_length, right / length)))
    if sine < config.min_sine:
        return Frame(name="body", origin_world=_vec(origin), quality=_quality(p))
    return frame_from_axes(
        "body",
        _vec(origin),
        right,
        np.cross(up, right / length),
        p,
        config,
    )


def bend(points: list[Landmark3D], config: GeometryConfig) -> Quantity:
    result = Quantity(quality=_quality(points))
    if not _usable(points, 3):
        return result
    a, b, c = np.array([p.xyz_world for p in points])
    u, v = b - a, c - b
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if min(nu, nv) <= config.min_length:
        return result
    uncertainty = 2 * _sigma(points) / min(nu, nv)
    if uncertainty > config.max_angle_uncertainty:
        return result
    angle = float(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1)))
    return Quantity(value=angle, quality=_quality(points, uncertainty))


def hand_geometry(sample: MotionSample, side: Side, config: GeometryConfig) -> Hand:
    p = _pick(sample, f"{side}_wrist", f"{side}_index_mcp", f"{side}_pinky_mcp")
    frame = Frame(name=f"{side}_hand", quality=_quality(p))
    longitudinal = Direction(quality=_quality(p))
    alignment = Quantity(quality=_quality(p))
    if _usable(p, 3):
        wrist, index, pinky = np.array([v.xyz_world for v in p])
        center = (index + pinky) / 2
        # x across palm pinky->index, y wrist->MCP midpoint; z their cross.
        frame = frame_from_axes(
            f"{side}_hand",
            _vec(wrist),
            index - pinky,
            center - wrist,
            p,
            config,
        )
        midpoint = Landmark3D(
            name=cast(Landmark, f"{side}_middle_mcp"),
            xyz_world=_vec(center),
            quality=_quality(p[1:], _sigma(p[1:])),
        )
        longitudinal = direction([p[0], midpoint], config)
        elbow = _pick(sample, f"{side}_elbow")
        alignment = bend(elbow + [p[0], midpoint], config)
    fingers = {}
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        joints = (
            ("cmc", "mcp", "ip", "tip")
            if finger == "thumb"
            else (
                "mcp",
                "pip",
                "dip",
                "tip",
            )
        )
        names = [f"{side}_{finger}_{j}" for j in joints]
        bends = [bend(_pick(sample, *names[i : i + 3]), config) for i in (0, 1)]
        supported = all(b.value is not None for b in bends)
        state: Literal["extended", "flexed", "indeterminate"] = "indeterminate"
        if supported:
            if all(
                _known(b.value)
                + config.uncertainty_multiplier * _known(b.quality.uncertainty)
                <= config.open_max_flexion
                for b in bends
            ):
                state = "extended"
            elif all(
                _known(b.value)
                - config.uncertainty_multiplier * _known(b.quality.uncertainty)
                >= config.fist_min_flexion
                for b in bends
            ):
                state = "flexed"
        fp = _pick(sample, *names)
        fingers[finger] = Finger(
            bends_rad=bends,
            configuration=state,
            quality=_quality(fp, max(_known(b.quality.uncertainty) for b in bends))
            if state != "indeterminate"
            else _quality(fp),
        )
    descriptor: Literal["open", "fist", "indeterminate"] = "indeterminate"
    # All five digits are required. This is a bend descriptor, not martial taxonomy.
    if all(f.configuration == "extended" for f in fingers.values()):
        descriptor = "open"
    elif all(f.configuration == "flexed" for f in fingers.values()):
        descriptor = "fist"
    all_points = _pick(
        sample,
        *[
            f"{side}_{f}_{j}"
            for f in fingers
            for j in (
                ("cmc", "mcp", "ip", "tip")
                if f == "thumb"
                else ("mcp", "pip", "dip", "tip")
            )
        ],
    )
    quality = _quality(all_points)
    if descriptor != "indeterminate":
        quality = _quality(
            all_points, max(_known(f.quality.uncertainty) for f in fingers.values())
        )
    return Hand(
        frame=frame,
        longitudinal=longitudinal,
        fingers=fingers,
        wrist_alignment_rad=alignment,
        descriptor=descriptor,
        quality=quality,
    )


def foot_geometry(sample: MotionSample, side: Side, config: GeometryConfig) -> Foot:
    p = _pick(sample, f"{side}_heel", f"{side}_forefoot")
    axis = direction(p, config)
    full = _pick(sample, f"{side}_heel", f"{side}_forefoot", f"{side}_foot_outer")
    frame = Frame(name=f"{side}_foot", quality=_quality(full))
    if _usable(full, 3):
        heel, toe, outer = np.array([p.xyz_world for p in full])
        # x heel->forefoot, y toward anatomical medial (left) / outer (right).
        lateral = (outer - toe) * (-1 if side == "left" else 1)
        frame = frame_from_axes(
            frame.name, _vec(heel), toe - heel, lateral, full, config
        )
    return Foot(longitudinal=axis, frame=frame)


def head_geometry(sample: MotionSample, torso: Frame, config: GeometryConfig) -> Head:
    p = _pick(sample, "left_ear", "right_ear", "nose")
    frame = Frame(name="head", quality=_quality(p))
    if _usable(p, 3):
        left, right, nose = np.array([v.xyz_world for v in p])
        center = (left + right) / 2
        frame = frame_from_axes(
            "head", _vec(center), right - left, nose - center, p, config
        )
    relative = None
    q = Quality(
        state="unknown",
        source_ids=list(
            dict.fromkeys(
                frame.quality.source_ids + torso.quality.source_ids,
            )
        ),
    )
    if frame.orientation is not None and torso.orientation is not None:
        uncertainty = _known(frame.quality.uncertainty) + _known(
            torso.quality.uncertainty
        )
        if uncertainty <= config.max_angle_uncertainty:
            relative = _quaternion(_matrix(torso).T @ _matrix(frame))
            q = q.model_copy(update={"state": "inferred", "uncertainty": uncertainty})
    neck = direction(_pick(sample, "neck", "head"), config)
    neck_axis = None
    neck_q = Quality(
        state="unknown",
        source_ids=list(
            dict.fromkeys(
                neck.quality.source_ids + torso.quality.source_ids,
            )
        ),
    )
    if neck.axis_world is not None and torso.orientation is not None:
        uncertainty = _known(neck.quality.uncertainty) + _known(
            torso.quality.uncertainty
        )
        if uncertainty <= config.max_angle_uncertainty:
            neck_axis = _vec(_matrix(torso).T @ neck.axis_world)
            neck_q = neck_q.model_copy(
                update={"state": "inferred", "uncertainty": uncertainty}
            )
    return Head(
        frame=frame,
        orientation_in_torso=relative,
        orientation_quality=q,
        neck_direction_in_torso=neck_axis,
        neck_quality=neck_q,
    )


def point_relations(
    sample: MotionSample,
    subject: str,
    object: str,
    frame: Frame,
    config: GeometryConfig,
) -> list[Relation]:
    points = _pick(sample, subject, object)
    q = _quality(points)
    q.source_ids = list(dict.fromkeys(q.source_ids + frame.quality.source_ids))
    result = []
    for i, (axis, positive, negative) in enumerate(
        [
            ("right", "right_of", "left_of"),
            ("front", "in_front_of", "behind"),
            ("up", "above", "below"),
        ]
    ):
        value = "unknown"
        quality = q.model_copy(deep=True)
        if frame.orientation is not None and _usable(points, 2):
            delta = np.array(points[0].xyz_world) - points[1].xyz_world
            relative = _matrix(frame).T @ delta
            uncertainty = _sigma(points) + float(np.linalg.norm(delta)) * _known(
                frame.quality.uncertainty
            )
            if (
                abs(relative[i])
                > config.uncertainty_multiplier * uncertainty + config.min_length
            ):
                value = positive if relative[i] > 0 else negative
                quality.state, quality.uncertainty = "inferred", uncertainty
        result.append(
            Relation(
                subject=subject,
                object=object,
                reference_frame=frame.name,
                axis=cast(RelationAxis, axis),
                value=cast(RelationValue, value),
                quality=quality,
            )
        )
    return result


def forearm_crossing(
    sample: MotionSample, frame: Frame, config: GeometryConfig
) -> Relation:
    """Intersect forearms in body right/up plane, then compare body-front depth."""
    p = _pick(sample, "left_elbow", "left_wrist", "right_elbow", "right_wrist")
    q = _quality(p)
    q.source_ids = list(dict.fromkeys(q.source_ids + frame.quality.source_ids))
    result = Relation(
        subject="left_forearm",
        object="right_forearm",
        reference_frame=frame.name,
        axis="crossing",
        value="unknown",
        quality=q,
        front_quality=q.model_copy(deep=True),
    )
    if frame.orientation is None or not _usable(p, 4):
        return result
    xyz = np.array([point.xyz_world for point in p]) @ _matrix(frame)
    a, b, c, d = xyz[:, [0, 2]]
    u, v = b - a, d - c
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if min(nu, nv) <= config.min_length:
        return result
    matrix = np.column_stack([u, -v])
    sine = abs(float(np.linalg.det(matrix))) / (nu * nv)
    if sine < config.min_sine:
        return result
    angular = _known(frame.quality.uncertainty)
    positional = (
        _sigma(p)
        + max(
            float(np.linalg.norm(xyz[1] - xyz[0])),
            float(np.linalg.norm(xyz[3] - xyz[2])),
            float(np.linalg.norm(xyz[2] - xyz[0])),
        )
        * angular
    )
    parameter_error = config.uncertainty_multiplier * positional / (min(nu, nv) * sine)
    t, s = np.linalg.solve(matrix, c - a)
    if any(v < -parameter_error or v > 1 + parameter_error for v in (t, s)):
        result.value = "not_crossed"
        result.quality = q.model_copy(
            update={"state": "inferred", "uncertainty": positional}
        )
        return result
    if not all(parameter_error < v < 1 - parameter_error for v in (t, s)):
        return result
    result.value = "crossed"
    result.quality = q.model_copy(
        update={"state": "inferred", "uncertainty": positional}
    )
    depth = (xyz[0, 1] + t * (xyz[1, 1] - xyz[0, 1])) - (
        xyz[2, 1] + s * (xyz[3, 1] - xyz[2, 1])
    )
    depth_error = positional + parameter_error * (
        abs(xyz[1, 1] - xyz[0, 1]) + abs(xyz[3, 1] - xyz[2, 1])
    )
    if abs(depth) > config.uncertainty_multiplier * depth_error + config.min_length:
        result.front_entity = "left_forearm" if depth > 0 else "right_forearm"
        result.front_quality = q.model_copy(
            update={"state": "inferred", "uncertainty": depth_error}
        )
    return result


def derive_sample(
    sample: MotionSample,
    *,
    reconstruction_id: str,
    representation: Literal["raw", "fitted", "regularized"],
    config: GeometryConfig | None = None,
) -> DetailedSample:
    """Recompute on supplied final geometry, without mutating landmarks or frames."""
    config = config or GeometryConfig()
    torso = body_frame(sample, config)
    return DetailedSample(
        global_seconds=sample.global_seconds,
        reconstruction_id=reconstruction_id,
        representation=representation,
        config=config,
        body_frame=torso,
        hands={side: hand_geometry(sample, side, config) for side in SIDES},
        feet={side: foot_geometry(sample, side, config) for side in SIDES},
        head=head_geometry(sample, torso, config),
        relations=point_relations(sample, "left_wrist", "right_wrist", torso, config)
        + [forearm_crossing(sample, torso, config)],
    )
