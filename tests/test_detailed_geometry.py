"""Synthetic geometric contracts, with no model/data or accuracy claims."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from contracts.models import Landmark, Landmark3D, Quality, Reconstruction
from reconstruction.articulated import publish_articulated_fit
from reconstruction.detailed import (
    GeometryConfig,
    body_frame,
    derive_sample,
    foot_geometry,
    forearm_crossing,
    hand_geometry,
    head_geometry,
    load_detailed_geometry,
    point_relations,
    publish_detailed_geometry,
)
from storage import ArtifactStore, StorageRoot
from tests.test_articulated import participant, point, raw_handle

CONFIG = GeometryConfig()


def put(sample: Any, name: str, xyz: Any, sigma: float = 1e-5) -> None:
    sample.landmarks = [p for p in sample.landmarks if p.name != name]
    sample.landmarks.append(
        Landmark3D(
            name=cast(Landmark, name),
            xyz_world=tuple(xyz),
            quality=Quality(
                state="observed", uncertainty=sigma, source_ids=[f"view:{name}"]
            ),
        )
    )


def hands(flexed: bool = False, side: Literal["left", "right"] = "left") -> Any:
    sample = participant().samples[0]
    for name in [f"{side}_wrist", f"{side}_elbow"]:
        put(sample, name, [0, 0, 0] if name.endswith("wrist") else [0, -0.2, 0])
    for i, finger in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        joints = (
            ("cmc", "mcp", "ip", "tip")
            if finger == "thumb"
            else ("mcp", "pip", "dip", "tip")
        )
        x = (2.5 - i) * 0.02
        coords = [[x, 0.08, 0], [x, 0.11, 0], [x, 0.14, 0], [x, 0.17, 0]]
        if flexed:
            coords = [[x, 0.08, 0], [x, 0.11, 0], [x, 0.11, -0.03], [x, 0.08, -0.03]]
        for j, xyz in zip(joints, coords, strict=True):
            put(sample, f"{side}_{finger}_{j}", xyz)
    return sample


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("flexed", [False, True])
def test_open_and_flexed_digit_evidence(
    side: Literal["left", "right"], flexed: bool
) -> None:
    sample = hands(flexed, side)
    hand = hand_geometry(sample, side, CONFIG)
    assert hand.descriptor == ("fist" if flexed else "open")
    for finger in hand.fingers.values():
        assert finger.configuration == ("flexed" if flexed else "extended")
        assert [b.value for b in finger.bends_rad] == pytest.approx(
            [np.pi / 2] * 2 if flexed else [0, 0]
        )
        assert finger.quality.source_ids
    assert hand.frame.orientation is not None
    assert hand.longitudinal.axis_world is not None
    assert hand.wrist_alignment_rad.value is not None
    put(sample, f"{side}_elbow", [0.2, 0, 0])
    assert hand_geometry(
        sample, side, CONFIG
    ).wrist_alignment_rad.value == pytest.approx(np.pi / 2)
    sample.landmarks = [p for p in sample.landmarks if p.name != f"{side}_thumb_tip"]
    assert hand_geometry(sample, side, CONFIG).descriptor == "indeterminate"


@pytest.mark.parametrize("angle", [0, 0.4, 1.6])
def test_foot_direction_and_partial_orientation(angle: float) -> None:
    sample = participant().samples[0]
    rotation = Rotation.from_euler("z", angle)
    sides: tuple[Literal["left", "right"], ...] = ("left", "right")
    for side in sides:
        for name, xyz in [
            ("heel", [0, 0, 0]),
            ("forefoot", [0, 0.2, 0]),
            ("foot_outer", [-0.05 if side == "left" else 0.05, 0.18, 0]),
        ]:
            put(sample, f"{side}_{name}", rotation.apply(xyz))
        foot = foot_geometry(sample, side, CONFIG)
        assert foot.longitudinal.axis_world is not None
        np.testing.assert_allclose(
            foot.longitudinal.axis_world, rotation.apply([0, 1, 0]), atol=1e-12
        )
        assert foot.frame.orientation is not None
        sample.landmarks = [
            p for p in sample.landmarks if p.name != f"{side}_foot_outer"
        ]
        partial = foot_geometry(sample, side, CONFIG)
        assert (
            partial.frame.orientation is None
            and partial.longitudinal.axis_world is not None
        )


def test_head_turn_relative_to_torso_and_neck() -> None:
    sample = participant().samples[0]
    torso = body_frame(sample, CONFIG)
    for angle in [0, 0.6, -1.2]:
        rotation = Rotation.from_euler("z", angle)
        for name, xyz in [
            ("left_ear", [-0.08, 0, 0]),
            ("right_ear", [0.08, 0, 0]),
            ("nose", [0, 0.1, 0]),
        ]:
            put(sample, name, rotation.apply(xyz) + [0, 0, 1.8])
        put(sample, "head", [0, 0, 1.8])
        head = head_geometry(sample, torso, CONFIG)
        assert head.orientation_in_torso is not None
        w, x, y, z = head.orientation_in_torso.wxyz
        np.testing.assert_allclose(
            Rotation.from_quat([x, y, z, w]).as_matrix(),
            rotation.as_matrix(),
            atol=1e-10,
        )
        assert head.neck_direction_in_torso is not None
        np.testing.assert_allclose(head.neck_direction_in_torso, [0, 0, 1], atol=1e-10)
    sample.landmarks = [p for p in sample.landmarks if p.name != "nose"]
    assert head_geometry(sample, torso, CONFIG).orientation_in_torso is None


def crossing(depth: float = 0.12) -> Any:
    sample = participant().samples[0]
    for name, xyz in [
        ("left_elbow", [-0.3, depth, 1.6]),
        ("left_wrist", [0.3, depth, 1.2]),
        ("right_elbow", [0.3, -depth, 1.6]),
        ("right_wrist", [-0.3, -depth, 1.2]),
    ]:
        put(sample, name, xyz)
    return sample


@pytest.mark.parametrize(
    "depth,front",
    [(0.12, "left_forearm"), (-0.12, "right_forearm"), (0, None), (0.001, None)],
)
def test_crossing_uses_depth_with_uncertainty(depth: float, front: str | None) -> None:
    sample = crossing(depth)
    result = forearm_crossing(sample, body_frame(sample, CONFIG), CONFIG)
    assert result.value == "crossed"
    assert result.front_entity == front
    assert result.front_quality.state == ("unknown" if front is None else "inferred")


def test_world_transform_preserves_relations_and_anatomical_sides() -> None:
    sample = crossing()
    original = sample.model_dump_json()
    before = derive_sample(sample, reconstruction_id="raw", representation="raw")
    rotation = Rotation.from_euler("xyz", [0.8, -0.3, 1.2])
    shifted = sample.model_copy(deep=True)
    for p in shifted.landmarks:
        p.xyz_world = tuple(rotation.apply(p.xyz_world) + [4, -7, 2])
    after = derive_sample(shifted, reconstruction_id="raw", representation="raw")
    assert [r.value for r in before.relations] == [r.value for r in after.relations]
    assert after.relations[-1].front_entity == before.relations[-1].front_entity
    assert after.relations[0].value == "right_of"
    assert (
        before.reconstruction_id == "raw"
        and before.global_seconds == sample.global_seconds
    )
    assert sample.model_dump_json() == original
    assert after.config == CONFIG
    # Unit changes preserve geometry decisions when the uncertainty scales too.
    for p in shifted.landmarks:
        p.xyz_world = tuple(np.array(p.xyz_world) * 10)
        p.quality.uncertainty = float(p.quality.uncertainty) * 10
    scaled = derive_sample(shifted, reconstruction_id="raw", representation="raw")
    assert [r.value for r in scaled.relations] == [r.value for r in after.relations]


@pytest.mark.parametrize(
    "weakness", ["missing", "uncertainty", "no_sources", "no_sigma", "collinear"]
)
def test_weak_hand_evidence_stays_unknown(weakness: str) -> None:
    sample = hands()
    p = point(sample, "left_index_mcp")
    if weakness == "missing":
        p.xyz_world, p.quality = None, Quality(state="unknown", source_ids=["occluded"])
    elif weakness == "uncertainty":
        p.quality.uncertainty = 1
    elif weakness == "no_sources":
        p.quality.source_ids = []
    elif weakness == "no_sigma":
        p.quality.uncertainty = None
    else:
        p.xyz_world = (0, 0.1, 0)
        point(sample, "left_pinky_mcp").xyz_world = (0, 0.2, 0)
    result = hand_geometry(sample, "left", CONFIG)
    assert result.frame.orientation is None and result.frame.quality.state == "unknown"
    if weakness != "collinear":
        assert result.descriptor == "indeterminate"


def test_unknown_and_non_crossing_are_distinct() -> None:
    sample = crossing()
    put(sample, "right_elbow", [1, 0, 1.6])
    put(sample, "right_wrist", [1.2, 0, 1.2])
    assert (
        forearm_crossing(sample, body_frame(sample, CONFIG), CONFIG).value
        == "not_crossed"
    )
    put(sample, "right_wrist", [1.6, 0, 1.2])  # parallel projected forearms
    assert (
        forearm_crossing(sample, body_frame(sample, CONFIG), CONFIG).value == "unknown"
    )
    point(sample, "left_hip").xyz_world = point(sample, "right_hip").xyz_world
    assert body_frame(sample, CONFIG).orientation is None
    assert all(
        r.value == "unknown"
        for r in point_relations(
            sample, "left_wrist", "right_wrist", body_frame(sample, CONFIG), CONFIG
        )
    )


def test_orders_and_missing_endpoint() -> None:
    sample = crossing()
    put(sample, "left_wrist", [0.4, 0.3, 1.8])
    put(sample, "right_wrist", [-0.4, -0.3, 1.2])
    frame = body_frame(sample, CONFIG)
    assert [
        r.value
        for r in point_relations(sample, "left_wrist", "right_wrist", frame, CONFIG)
    ] == ["right_of", "in_front_of", "above"]
    assert [
        r.value
        for r in point_relations(sample, "right_wrist", "left_wrist", frame, CONFIG)
    ] == ["left_of", "behind", "below"]
    assert all(
        r.value == "unknown"
        for r in point_relations(sample, "left_wrist", "head", frame, CONFIG)
    )


def test_artifact_raw_fitted_lineage_cache_and_recompute(tmp_path: Path) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    raw_source = participant()
    raw_source.provenance.producer = "reconstruction.triangulation"
    raw = raw_handle(store, raw_source)
    fitted = publish_articulated_fit(store, raw).pose
    raw_before = raw.metadata.model_dump_json()
    sources: list[tuple[Any, Literal["raw", "fitted"]]] = [
        (raw, "raw"),
        (fitted, "fitted"),
    ]
    for source, representation in sources:
        output = publish_detailed_geometry(store, source, representation=representation)
        assert isinstance(output.metadata, Reconstruction)
        assert output.metadata.samples == source.metadata.samples
        derived = load_detailed_geometry(output)
        assert derived[0].representation == representation
        assert derived[0].reconstruction_id == source.metadata.id
        assert derived[0] == derive_sample(
            source.metadata.samples[0],
            reconstruction_id=source.metadata.id,
            representation=representation,
        )
        for a in source.metadata.arrays:
            np.testing.assert_array_equal(
                output.read_array(a.id), source.read_array(a.id)
            )
        assert (
            publish_detailed_geometry(store, source, representation=representation).path
            == output.path
        )
        assert (
            publish_detailed_geometry(
                store,
                source,
                representation=representation,
                config=GeometryConfig(min_sine=0.2),
            ).path
            != output.path
        )
    assert raw.metadata.model_dump_json() == raw_before
    with pytest.raises(ValueError, match="representation"):
        publish_detailed_geometry(store, fitted, representation="raw")
    with pytest.raises(ValueError, match="source reconstruction"):
        publish_detailed_geometry(store, output, representation="fitted")


def test_duplicate_landmark_names_rejected_at_reconstruction_boundary() -> None:
    source = participant()
    source.samples[0].landmarks.append(source.samples[0].landmarks[0])
    with pytest.raises(ValueError, match="unique"):
        Reconstruction.model_validate(source.model_dump())


def test_config_rejects_nonfinite_and_invalid_settings() -> None:
    for settings in [
        {"min_sine": 0},
        {"uncertainty_multiplier": 0},
        {"max_angle_uncertainty": float("nan")},
        {"contact": True},
    ]:
        with pytest.raises(ValueError):
            GeometryConfig.model_validate(settings)


def test_crossing_depth_at_intersection_and_occlusion() -> None:
    sample = crossing()
    for p in sample.landmarks:
        p.quality.uncertainty = 1e-5
    # Intersection is at t=1/4 on left and s=1/2 on right. Left midpoint
    # is behind, while its actual crossing point is in front.
    put(sample, "left_elbow", [-0.2, 0.3, 1.6])
    put(sample, "left_wrist", [0.6, -0.7, 1.2])
    put(sample, "right_elbow", [0.2, 0, 1.7])
    put(sample, "right_wrist", [-0.2, 0, 1.3])
    result = forearm_crossing(sample, body_frame(sample, CONFIG), CONFIG)
    assert result.value == "crossed" and result.front_entity == "left_forearm"
    p = point(sample, "left_elbow")
    p.xyz_world, p.quality = None, Quality(state="unknown", source_ids=["occluded"])
    result = forearm_crossing(sample, body_frame(sample, CONFIG), CONFIG)
    assert result.value == "unknown" and result.front_entity is None
    assert "occluded" in result.quality.source_ids


def test_near_degenerate_body_and_crossing_endpoint_are_unknown() -> None:
    sample = crossing()
    put(sample, "left_shoulder", [-0.2, 0, 1.0])
    put(sample, "right_shoulder", [0.2, 0, 1.0001])
    assert body_frame(sample, CONFIG).orientation is None
    sample = crossing()
    put(sample, "left_wrist", [0.3, 0.12, 1.6])
    result = forearm_crossing(sample, body_frame(sample, CONFIG), CONFIG)
    assert result.value == "unknown" and result.front_entity is None


def test_unknown_orders_at_uncertainty_boundary_and_digit_threshold() -> None:
    sample = crossing()
    put(sample, "left_wrist", [0.001, 0.001, 1.4], sigma=0.001)
    put(sample, "right_wrist", [0, 0, 1.4], sigma=0.001)
    assert all(
        r.value == "unknown"
        for r in point_relations(
            sample,
            "left_wrist",
            "right_wrist",
            body_frame(sample, CONFIG),
            CONFIG,
        )
    )
    sample = hands()
    # Near the open threshold, uncertainty prevents a confident classification.
    theta = CONFIG.open_max_flexion
    put(
        sample,
        "left_index_dip",
        [0.03, 0.11 + 0.03 * np.cos(theta), 0.03 * np.sin(theta)],
    )
    hand = hand_geometry(sample, "left", CONFIG)
    assert hand.fingers["index"].configuration == "indeterminate"
    assert hand.descriptor == "indeterminate"
