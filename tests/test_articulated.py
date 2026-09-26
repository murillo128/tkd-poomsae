"""Numerical skeletal invariants; no real-world anthropometric claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from contracts.models import (
    DenseArray,
    Landmark,
    Landmark3D,
    Morphology,
    MotionSample,
    Provenance,
    Quality,
    Quaternion,
    Reconstruction,
    SegmentFrame,
)
from reconstruction.articulated import (
    FitConfig,
    load_fit_diagnostics,
    load_morphology_evidence,
    publish_articulated_fit,
)
from reconstruction.articulated.core import BONES, estimate_morphology, fit_sample
from reconstruction.triangulation import publish_triangulation
from reconstruction.triangulation.artifact import ARRAY_ID, load_diagnostics
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, StorageRoot, hash_config
from tests.test_triangulation import handles, join, scene

PROV = Provenance(producer="synthetic", config_digest="0" * 64)


def participant(
    upper: float = 0.3,
    forearm: float = 0.25,
    shoulder: float = 0.4,
    hip: float = 0.3,
    *,
    scale: float = 1,
    units: Literal["metric", "arbitrary"] = "metric",
    supplied_orientation: bool = False,
) -> Reconstruction:
    samples = []
    for i in range(9):
        theta = i * 0.15
        rotation = Rotation.from_euler("z", theta)
        root = np.array([i * 0.1, -i * 0.03, 1 + 0.01 * i])
        points: dict[Landmark, Any] = {
            "pelvis": [0, 0, 0],
            "neck": [0, 0, 0.65],
            "left_hip": [-hip / 2, 0, 0],
            "right_hip": [hip / 2, 0, 0],
            "left_shoulder": [-shoulder / 2, 0, 0.55],
            "right_shoulder": [shoulder / 2, 0, 0.55],
            "left_ear": [-0.08, 0, 0.8],
            "right_ear": [0.08, 0, 0.8],
            "nose": [0, -0.1, 0.8],
        }
        for side in ("left", "right"):
            sign = -1 if side == "left" else 1
            s = np.array(points[f"{side}_shoulder"])  # type: ignore[index]
            elbow = s + [sign * upper * np.cos(theta), 0, upper * np.sin(theta)]
            wrist = elbow + [
                sign * forearm * np.cos(2 * theta),
                forearm * np.sin(2 * theta),
                0,
            ]
            knee = np.array(points[f"{side}_hip"]) + [0, 0, -0.45]  # type: ignore[index]
            ankle = knee + [0, 0.4 * np.sin(theta), -0.4 * np.cos(theta)]
            detail = {
                f"{side}_elbow": elbow,
                f"{side}_wrist": wrist,
                f"{side}_index_tip": wrist + [0.02, -0.12, 0.03],
                f"{side}_index_mcp": wrist + [0.02, -0.05, 0],
                f"{side}_pinky_mcp": wrist + [-0.02, -0.05, 0],
                f"{side}_knee": knee,
                f"{side}_ankle": ankle,
                f"{side}_heel": ankle + [0, 0.05, -0.04],
                f"{side}_forefoot": ankle + [0, -0.15, -0.04],
                f"{side}_foot_outer": ankle + [sign * 0.05, -0.13, -0.04],
            }
            points.update(detail)  # type: ignore[arg-type]
        quaternion = rotation.as_quat()
        orientation = Quaternion(
            wxyz=(
                float(quaternion[3]),
                float(quaternion[0]),
                float(quaternion[1]),
                float(quaternion[2]),
            )
        )
        samples.append(
            MotionSample(
                global_seconds=i * 0.1,
                root_xyz_world=tuple(root * scale),
                root_orientation=orientation if supplied_orientation else None,
                landmarks=[
                    Landmark3D(
                        name=n,
                        xyz_world=tuple((rotation.apply(p) + root) * scale),
                        quality=Quality(
                            state="observed",
                            uncertainty=0.001 * scale,
                            source_ids=[f"view-a:{i}:{n}", f"view-b:{i}:{n}"],
                        ),
                    )
                    for n, p in points.items()
                ],
                segments=[
                    SegmentFrame(
                        segment="head",
                        parent="root",
                        orientation=orientation,
                        quality=Quality(state="observed", source_ids=["head"]),
                    )
                ]
                if supplied_orientation
                else [],
                quality=Quality(state="observed", source_ids=[f"raw:{i}"]),
            )
        )
    return Reconstruction(
        kind="reconstruction",
        id="raw-synthetic",
        schema_version="1.0.0",
        provenance=PROV,
        participant_id="participant",
        calibration_id="cal",
        scale=units,
        samples=samples,
    )


def morphology(raw: Reconstruction, config: FitConfig | None = None) -> Morphology:
    return estimate_morphology(
        raw, config or FitConfig(), identifier="shape", config_digest=PROV.config_digest
    )[0]


def measurements(shape: Morphology) -> dict[str, Any]:
    return {m.name: m for m in shape.measurements}


def point(sample: MotionSample, name: str) -> Landmark3D:
    return next(p for p in sample.landmarks if p.name == name)


def raw_handle(store: ArtifactStore, source: Reconstruction) -> ArtifactHandle:
    array = np.frombuffer(
        json.dumps(
            {
                "version": 1,
                "artifact_role": "immutable_raw_triangulated_motion",
                "world_unit": "m" if source.scale == "metric" else "arbitrary",
                "calibration_id": source.calibration_id,
                "samples": [{"source": "synthetic"}],
            }
        ).encode(),
        dtype=np.uint8,
    )
    source = source.model_copy(deep=True)
    source.arrays = [
        DenseArray(id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"])
    ]
    key = ArtifactKey(
        layer="reconstruction",
        inputs={"synthetic": hash_config(source.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision="synthetic-v1",
        config_digest=PROV.config_digest,
    )
    return store.get_or_create(key, lambda: (source, {ARRAY_ID: array}))


def test_different_proportions_survive_identical_pose_sequence() -> None:
    for upper, forearm, shoulder, hip in [
        (0.3, 0.25, 0.4, 0.3),
        (0.45, 0.2, 0.5, 0.38),
    ]:
        raw = participant(upper, forearm, shoulder, hip)
        shape = morphology(raw)
        values = measurements(shape)
        for name, expected in [
            ("left_upper_arm", upper),
            ("left_forearm", forearm),
            ("shoulder_width", shoulder),
            ("hip_width", hip),
        ]:
            assert values[name].value == pytest.approx(expected)
            assert values[name].quality.state == "inferred"
        original_shape = shape.model_dump_json()
        for sample in raw.samples:
            fit, diagnostics = fit_sample(sample, shape, FitConfig())
            for name, a, b in [(n, *p) for n, p in BONES.items()]:
                length = np.linalg.norm(
                    np.array(point(fit, b).xyz_world) - point(fit, a).xyz_world
                )
                assert length == pytest.approx(values[name].value, abs=1e-8)
            assert shape.model_dump_json() == original_shape
            assert diagnostics["status"] == "converged"


def test_outliers_do_not_change_shape_and_fit_is_bounded() -> None:
    raw = participant()
    noisy = point(raw.samples[4], "left_wrist")
    noisy.xyz_world = tuple(np.array(noisy.xyz_world) + [0.1, 0, 0])
    before = raw.model_dump_json()
    config = FitConfig(max_displacement_fraction=0.05)
    shape, evidence = estimate_morphology(
        raw, config, identifier="shape", config_digest=PROV.config_digest
    )
    assert measurements(shape)["left_forearm"].value == pytest.approx(0.25)
    assert evidence["left_forearm"]["rejected"][0]["reason"] == "length_outlier"
    fitted, d = fit_sample(raw.samples[4], shape, config)
    assert d["status"] == "converged"
    active_lengths = {
        n: length
        for n, length in ((c["name"], c["target_length"]) for c in d["constraints"])
    }
    for n, displacement in d["data_residuals"].items():
        smallest = min(
            length for bone, length in active_lengths.items() if n in BONES[bone]
        )
        assert displacement <= config.max_displacement_fraction * smallest + 1e-8
        if displacement > smallest * 1e-10:
            p = point(fitted, n)
            assert p.quality.state == "inferred" and p.quality.score is None
            assert p.quality.uncertainty is not None
            assert p.quality.uncertainty >= float(
                point(raw.samples[4], n).quality.uncertainty or 0
            )
            assert p.quality.uncertainty >= displacement
            assert raw.id in p.quality.source_ids
            assert shape.id in p.quality.source_ids
    assert raw.model_dump_json() == before
    # Constraint residual improves but bounds prevent large unsupported corrections.
    constraint = next(c for c in d["constraints"] if c["name"] == "left_forearm")
    before_error = abs(
        np.linalg.norm(
            np.array(noisy.xyz_world) - point(raw.samples[4], "left_elbow").xyz_world
        )
        - 0.25
    )
    assert 0 < abs(constraint["residual"]) < before_error


@pytest.mark.parametrize(
    "weakness",
    [
        "missing",
        "inferred",
        "interpolated",
        "uncertainty",
        "no_uncertainty",
        "no_sources",
        "short",
        "span",
    ],
)
def test_weak_geometry_remains_unavailable(weakness: str) -> None:
    raw = participant()
    if weakness == "short":
        raw.samples = raw.samples[:4]
    elif weakness == "span":
        for i, s in enumerate(raw.samples):
            s.global_seconds = i * 0.001
    else:
        for s in raw.samples:
            p = point(s, "left_wrist")
            if weakness == "missing":
                p.xyz_world = None
                p.quality = Quality(state="unknown")
            elif weakness in {"inferred", "interpolated"}:
                p.quality.state = weakness  # type: ignore[assignment]
            elif weakness == "uncertainty":
                p.quality.uncertainty = 1
            elif weakness == "no_uncertainty":
                p.quality.uncertainty = None
            else:
                p.quality.source_ids = []
    shape = morphology(raw)
    m = measurements(shape)["left_forearm"]
    assert m.value is None and m.quality.state == "unknown"
    fit, _ = fit_sample(raw.samples[0], shape, FitConfig())
    assert point(fit, "left_wrist") == point(raw.samples[0], "left_wrist")


def test_unstable_sequence_rejected_and_sides_not_averaged() -> None:
    raw = participant()
    for i, s in enumerate(raw.samples):
        elbow, wrist = point(s, "left_elbow"), point(s, "left_wrist")
        wrist.xyz_world = tuple(np.array(elbow.xyz_world) + [0.1 + i * 0.05, 0, 0])
    shape, evidence = estimate_morphology(
        raw, FitConfig(), identifier="shape", config_digest=PROV.config_digest
    )
    assert measurements(shape)["left_forearm"].value is None
    assert evidence["left_forearm"]["reason"] == "unstable_geometry"
    assert measurements(shape)["right_forearm"].value == pytest.approx(0.25)


def test_reused_observations_do_not_supply_sequence_evidence() -> None:
    raw = participant()
    for s in raw.samples:
        for p in s.landmarks:
            p.quality.source_ids = [f"repeated:{p.name}"]
    shape, evidence = estimate_morphology(
        raw, FitConfig(), identifier="shape", config_digest=PROV.config_digest
    )
    assert all(m.value is None for m in shape.measurements)
    assert len(evidence["left_forearm"]["candidates"]) == 1
    assert len(evidence["left_forearm"]["rejected"]) == 8
    assert evidence["left_forearm"]["rejected"][0]["reason"] == "repeated_evidence"


def test_missing_root_and_detail_stay_unknown() -> None:
    raw = participant()
    shape = morphology(raw)
    s = raw.samples[0].model_copy(deep=True)
    s.root_xyz_world = None
    for name in ["pelvis", "left_index_tip", "left_heel", "nose"]:
        p = point(s, name)
        p.xyz_world = None
        p.quality = Quality(state="unknown", source_ids=["missing-source"])
    fit, d = fit_sample(s, shape, FitConfig())
    assert fit.root_xyz_world is None
    assert d["transforms"][0]["translation_quality"]["state"] == "unknown"
    for name in ["pelvis", "left_index_tip", "left_heel", "nose"]:
        assert point(fit, name) == point(s, name)
    frames = {f.segment: f for f in fit.segments}
    assert frames["left_foot"].orientation is None
    assert frames["head"].orientation is None
    # A named measured pelvis can supply missing translation without midpoint inference.
    s = raw.samples[0].model_copy(deep=True)
    s.root_xyz_world = None
    fit, d = fit_sample(s, shape, FitConfig())
    assert fit.root_xyz_world == point(s, "pelvis").xyz_world
    assert d["transforms"][0]["translation_quality"] == point(
        s, "pelvis"
    ).quality.model_dump(mode="json")


def test_orientation_evidence_and_world_motion() -> None:
    raw = participant()
    shape = morphology(raw)
    for i, s in enumerate(raw.samples):
        fit, d = fit_sample(s, shape, FitConfig())
        assert fit.root_xyz_world == s.root_xyz_world
        assert fit.root_orientation is not None
        w, x, y, z = fit.root_orientation.wxyz
        np.testing.assert_allclose(
            Rotation.from_quat([x, y, z, w]).as_matrix(),
            Rotation.from_euler("z", i * 0.15).as_matrix(),
            atol=1e-8,
        )
        assert d["transforms"][0]["parent"] == "world"
        for segment in fit.segments:
            if segment.segment in list(BONES)[:8]:
                assert (
                    segment.orientation is None and segment.quality.state == "unknown"
                )
            else:
                assert segment.orientation is not None
    sparse = raw.samples[0].model_copy(deep=True)
    sparse.landmarks = [p for p in sparse.landmarks if p.name != "neck"]
    fit, _ = fit_sample(sparse, shape, FitConfig())
    assert fit.root_orientation is None
    sparse = raw.samples[0].model_copy(deep=True)
    point(sparse, "neck").xyz_world = point(sparse, "left_hip").xyz_world
    assert fit_sample(sparse, shape, FitConfig())[0].root_orientation is None
    # Noncollinear points can still have too little angular evidence.
    weak = raw.samples[0].model_copy(deep=True)
    point(weak, "neck").xyz_world = (0, 0, 1.03)
    for name in ["left_hip", "right_hip", "neck"]:
        point(weak, name).quality.uncertainty = 0.015
    assert fit_sample(weak, shape, FitConfig())[0].root_orientation is None


def test_failed_optimizer_preserves_geometry() -> None:
    raw = participant()
    shape = morphology(raw)
    sample = raw.samples[0].model_copy(deep=True)
    p = point(sample, "left_wrist")
    p.xyz_world = tuple(np.array(p.xyz_world) + [0.02, 0, 0])
    fit, d = fit_sample(sample, shape, FitConfig(max_evaluations=1))
    assert d["status"] == "fit_rejected"
    assert fit.landmarks == sample.landmarks


@pytest.mark.parametrize("units,scale", [("metric", 1), ("arbitrary", 7)])
def test_persistence_detail_roots_evidence_cache_and_units(
    tmp_path: Path,
    units: Literal["metric", "arbitrary"],
    scale: float,
) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    source = participant(units=units, scale=scale, supplied_orientation=True)
    raw = raw_handle(store, source)
    manifest = (raw.path / "manifest.json").read_bytes()
    metadata = raw.metadata.model_dump_json()
    published = publish_articulated_fit(store, raw)
    fitted = published.pose.metadata
    assert isinstance(fitted, Reconstruction)
    assert isinstance(published.morphology.metadata, Morphology)
    assert measurements(published.morphology.metadata)[
        "left_upper_arm"
    ].value == pytest.approx(0.3 * scale)
    assert all(
        m.unit == ("m" if units == "metric" else "arbitrary")
        for m in published.morphology.metadata.measurements
    )
    for before, after in zip(source.samples, fitted.samples, strict=True):
        assert before.root_xyz_world == after.root_xyz_world
        assert before.root_orientation == after.root_orientation
        assert before.segments[0] == after.segments[0]
        detail = [
            p
            for p in before.landmarks
            if "tip" in p.name
            or "ear" in p.name
            or "heel" in p.name
            or "foot" in p.name
            or p.name == "nose"
        ]
        assert all(point(after, p.name) == p for p in detail)
    evidence = load_morphology_evidence(published.morphology)
    assert len(evidence["measurements"]["left_forearm"]["retained"]) == 9
    d = load_fit_diagnostics(published.pose)
    assert d["morphology_id"] == published.morphology.metadata.id
    assert d["raw_reconstruction_id"] == raw.metadata.id
    assert d["world_unit"] == ("m" if units == "metric" else "arbitrary")
    assert d["samples"][0]["transforms"][1]["translation"] is None
    assert load_diagnostics(published.pose) == load_diagnostics(raw)
    assert not published.pose.read_array(ARRAY_ID).flags.writeable
    assert publish_articulated_fit(store, raw).pose.path == published.pose.path
    changed = publish_articulated_fit(store, raw, FitConfig(length_weight=2))
    assert changed.pose.path != published.pose.path
    assert changed.morphology.path != published.morphology.path
    assert raw.metadata.model_dump_json() == metadata
    assert (raw.path / "manifest.json").read_bytes() == manifest
    # A fresh store loads and validates the persisted cache and its payloads.
    reopened = publish_articulated_fit(ArtifactStore(StorageRoot(tmp_path)), raw)
    assert reopened.morphology.metadata == published.morphology.metadata
    assert reopened.pose.metadata == published.pose.metadata
    assert load_morphology_evidence(reopened.morphology) == evidence
    assert load_fit_diagnostics(reopened.pose) == d


def test_actual_triangulation_publisher_input(tmp_path: Path) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    calibration, models = scene()
    cal, aligned = handles(
        store, calibration, join(models).query_many([0, 0.2, 0.4, 0.6, 0.8, 1])
    )
    raw = publish_triangulation(store, cal, aligned, "practitioner")
    published = publish_articulated_fit(store, raw)
    assert published.pose.metadata.participant_id == "practitioner"  # type: ignore[union-attr]
    assert load_diagnostics(published.pose) == load_diagnostics(raw)
    with pytest.raises(ValueError, match="raw reconstruction"):
        publish_articulated_fit(store, published.pose)


def test_config_rejects_invalid_parameters() -> None:
    for settings in [
        {"min_samples": 1},
        {"max_displacement_fraction": 0},
        {"length_weight": float("nan")},
        {"population_height": 1.7},
    ]:
        with pytest.raises(ValueError):
            FitConfig.model_validate(settings)
