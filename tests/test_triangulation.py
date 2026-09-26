"""Known geometry, rejection, all-view refinement and immutable raw motion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest

from calibration.cameras import CameraModel
from contracts.models import (
    Alignment,
    Calibration,
    CameraCalibration,
    DenseArray,
    Intrinsics,
    Landmark,
    Landmark2D,
    Observation,
    Quality,
    Reconstruction,
    Synchronization,
    SyncOffset,
)
from pipeline import Pipeline, Stage, StageOutput
from pipeline.runner import DEPENDENCIES, STAGE_LAYERS, default_stages
from reconstruction.triangulation import (
    REVISION,
    TriangulationConfig,
    load_diagnostics,
    produce_stage,
    publish_triangulation,
    triangulate_point,
)
from storage import ArtifactKey, ArtifactStore, StorageRoot, hash_config
from sync.alignment import ARRAY_ID, ObservationJoin, TimeQuery
from tests.fixtures.synthetic import Camera
from tests.test_alignment import PROV, observation
from tests.test_pipeline import artifact as fixture_artifact


def scene(n: int = 4) -> tuple[Calibration, list[CameraModel]]:
    centers = [(-3.0, -4.0, 2.0), (3.0, -4.0, 2.0), (4.0, 2.0, 3.0), (-4.0, 2.0, 3.0)]
    cameras = []
    for i, center in enumerate(centers[:n]):
        c = Camera(
            str(i),
            center,
            (0.0, 0.0, 1.0),
            1280,
            720,
            800,
            30,
            1,
            0.0,
            (0.02, -0.01, 0.001, -0.002),
        )
        cameras.append(
            CameraCalibration(
                camera_id=f"camera-{i}",
                source_id=str(i),
                intrinsics=Intrinsics(
                    fx=800, fy=800, cx=640, cy=360, distortion=list(c.distortion)
                ),
                world_to_camera=np.array(c.world_to_camera()).reshape(4, 4).tolist(),
                quality=Quality(state="observed", score=1),
                rms_reprojection_px=0.1,
            )
        )
    calibration = Calibration(
        kind="calibration",
        id="calibration-synthetic",
        schema_version="1.0.0",
        provenance=PROV,
        scale="arbitrary",
        world_unit="arbitrary",
        cameras=cameras,
        camera_status="resolved",
        quality=Quality(state="observed", score=1),
    )
    return calibration, [
        CameraModel(c.intrinsics, np.array(c.world_to_camera)) for c in cameras
    ]


def native(
    models: list[CameraModel], times: tuple[int, ...] = (0, 1000)
) -> list[Observation]:
    observations = []
    for i, model in enumerate(models):
        for pts in times:
            obs = observation(str(i), pts)
            points = []
            for k, name in enumerate(get_args(Landmark)):
                xyz = [
                    0.01 * k + 0.2 * pts / 1000,
                    0.1 * np.sin(pts / 1000),
                    1 + 0.003 * k,
                ]
                pixel = model.project([xyz])[0]
                points.append(
                    Landmark2D(
                        name=name,
                        xy_px=tuple(pixel),
                        quality=Quality(state="observed", score=1),
                    )
                )
            obs.landmarks = points
            obs.wholebody_landmarks = [p.model_copy(deep=True) for p in points]
            obs.refined_landmarks = [
                p.model_copy(deep=True) for p in points if "tip" in p.name
            ]
            obs.regional_geometry = []
            observations.append(obs)
    return observations


def join(
    models: list[CameraModel], times: tuple[int, ...] = (0, 1000)
) -> ObservationJoin:
    synchronization = Synchronization(
        kind="synchronization",
        id="sync-synthetic",
        schema_version="1.0.0",
        provenance=PROV,
        offsets=[
            SyncOffset(
                source_id=str(i),
                automatic_seconds=0,
                quality=Quality(state="observed", uncertainty=0, score=1),
            )
            for i in range(len(models))
        ],
    )
    return ObservationJoin(native(models, times), synchronization)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_known_geometry_uses_every_view(n: int) -> None:
    calibration, models = scene(n)
    q = join(models).query(1)
    for k, name in enumerate(get_args(Landmark)):
        point, diagnostic = triangulate_point(q, name, calibration)
        assert point.xyz_world is not None
        np.testing.assert_allclose(
            point.xyz_world, [0.01 * k + 0.2, 0.1 * np.sin(1), 1 + 0.003 * k], atol=1e-7
        )
        assert len([v for v in diagnostic["views"] if v["used"]]) == n
        assert point.quality.score is None
        assert point.quality.uncertainty is not None and point.quality.uncertainty > 0


def test_noisy_extra_views_influence_weighted_refinement() -> None:
    calibration, models = scene()
    q = join(models).query(0)
    q.cameras[2].landmarks[0].xy_px = tuple(
        np.array(q.cameras[2].landmarks[0].xy_px) + [2, -1]
    )
    q.cameras[3].landmarks[0].xy_px = tuple(
        np.array(q.cameras[3].landmarks[0].xy_px) + [-1, 2]
    )
    all_point, d = triangulate_point(q, "pelvis", calibration)
    pair, _ = triangulate_point(
        q.model_copy(update={"cameras": q.cameras[:2]}), "pelvis", calibration
    )
    assert all(v["used"] for v in d["views"])
    assert (
        np.linalg.norm(np.array(all_point.xyz_world) - np.array(pair.xyz_world)) > 1e-4
    )
    weak = q.model_copy(deep=True)
    weak.cameras[2].landmarks[0].quality.score = 0.1
    weighted, _ = triangulate_point(weak, "pelvis", calibration)
    assert (
        np.linalg.norm(np.array(weighted.xyz_world) - np.array(all_point.xyz_world))
        > 1e-4
    )


@pytest.mark.parametrize("n", [3, 4])
def test_outlier_rejected_with_provenance(n: int) -> None:
    calibration, models = scene(n)
    q = join(models).query(0)
    q.cameras[-1].landmarks[0].xy_px = (30, 30)
    point, d = triangulate_point(q, "pelvis", calibration)
    assert point.xyz_world is not None
    np.testing.assert_allclose(point.xyz_world, [0, 0, 1], atol=1e-7)
    rejected = d["views"][-1]
    assert not rejected["used"] and "reprojection_outlier" in rejected["reasons"]
    assert rejected["endpoints"][0]["observation_id"] == f"{n - 1}:0"
    assert f"{n - 1}:0" not in point.quality.source_ids
    assert rejected["residual_px"] > 8


def test_distinct_landmarks_use_distinct_subsets_and_insufficient_regions() -> None:
    calibration, models = scene()
    q = join(models).query(0)
    for i, c in enumerate(q.cameras):
        for p in c.landmarks:
            if (p.name == "left_index_tip" and i > 1) or (
                p.name == "right_heel" and i != 3
            ):
                p.xy_px = None
                p.quality = Quality(state="unknown")
    hand, hand_d = triangulate_point(q, "left_index_tip", calibration)
    foot, foot_d = triangulate_point(q, "right_heel", calibration)
    assert hand.xyz_world is not None
    assert sum(v["used"] for v in hand_d["views"]) == 2
    assert foot.xyz_world is None and foot.quality.state == "unknown"
    assert "insufficient_independent_views" in foot_d["reasons"]


def test_near_parallel_rays_and_duplicate_view_rejected() -> None:
    calibration, models = scene(2)
    calibration.cameras[1].world_to_camera = calibration.cameras[0].world_to_camera
    models[1] = models[0]
    q = join(models).query(0)
    point, _ = triangulate_point(q, "pelvis", calibration)
    assert point.xyz_world is None
    q.cameras[1] = q.cameras[0]
    with pytest.raises(ValueError, match="one camera"):
        triangulate_point(q, "pelvis", calibration)
    # Distinct centers but a baseline too small to constrain depth.
    calibration.cameras[1].world_to_camera = np.array(
        models[0].world_to_camera
    ).tolist()
    calibration.cameras[1].world_to_camera[0][3] += 1e-5
    near = CameraModel(
        calibration.cameras[1].intrinsics,
        np.array(calibration.cameras[1].world_to_camera),
    )
    q = join([models[0], near]).query(0)
    assert triangulate_point(q, "pelvis", calibration)[0].xyz_world is None


def test_behind_camera_rejected() -> None:
    calibration, _ = scene(2)
    # Parallel forward cameras; observations algebraically correspond to z=-5.
    models = []
    for i, c in enumerate(calibration.cameras):
        matrix = np.eye(4)
        matrix[0, 3] = -float(i)
        c.world_to_camera = matrix.tolist()
        c.intrinsics.distortion = []
        models.append(CameraModel(c.intrinsics, matrix))
    q = join(models).query(0)
    for i, query_camera in enumerate(q.cameras):
        query_camera.landmarks[0].xy_px = (640 + 800 * (0.2 - i) / -5, 360)
    point, d = triangulate_point(q, "pelvis", calibration)
    assert point.xyz_world is None
    assert "no_consistent_front_facing_geometry" in d["reasons"]


@pytest.mark.parametrize(
    "reason", ["sync", "source", "calibration", "mismatch", "unresolved"]
)
def test_unsupported_quality_rejected(reason: str) -> None:
    calibration, models = scene(2)
    q = join(models).query(0)
    if reason == "sync":
        q.cameras[1].synchronization_quality = Quality(state="unknown")
    elif reason == "source":
        q.cameras[1].landmarks[0].quality.score = 0.001
    elif reason == "calibration":
        calibration.cameras[1].quality = Quality(state="unknown")
    elif reason == "mismatch":
        calibration.cameras[1].source_id = "other"
    else:
        calibration.camera_status = "unresolved"
    assert triangulate_point(q, "pelvis", calibration)[0].xyz_world is None


def test_timing_calibration_and_interpolation_inflate_conditional_uncertainty() -> None:
    calibration, models = scene(3)
    q = join(models).query(0)
    base, _ = triangulate_point(q, "pelvis", calibration)
    for c in q.cameras:
        c.synchronization_quality.uncertainty = 0.05
        c.synchronization_quality.score = 0.1
    uncertain, _ = triangulate_point(q, "pelvis", calibration)
    assert uncertain.quality.uncertainty > base.quality.uncertainty * 10  # type: ignore[operator]
    q = join(models).query(0)
    for calibrated_camera in calibration.cameras:
        calibrated_camera.rms_reprojection_px = 10
    weak, _ = triangulate_point(q, "pelvis", calibration)
    assert weak.quality.uncertainty > base.quality.uncertainty  # type: ignore[operator]
    bracket = join(models).query(0.5)
    interpolated, d = triangulate_point(bracket, "pelvis", calibration)
    assert interpolated.quality.state == "interpolated"
    assert len(interpolated.quality.source_ids) == 6
    assert all(v["endpoints"][0]["weight"] == 0.5 for v in d["views"])
    assert all("interpolation_motion_bound" in v["reasons"] for v in d["views"])


def key(layer: str, name: str) -> ArtifactKey:
    return ArtifactKey(
        layer=layer,
        inputs={"fixture": hash_config(name)},
        schema_version="1.0.0",
        algorithm_revision="synthetic-v1",
        config_digest=PROV.config_digest,
    )


def handles(
    store: ArtifactStore, calibration: Calibration, queries: list[TimeQuery]
) -> tuple[Any, Any]:
    cal = store.get_or_create(key("calibration", "cal"), lambda: (calibration, {}))
    array = np.frombuffer(
        json.dumps(
            {
                "version": 1,
                "queries": [q.model_dump(mode="json") for q in queries],
                "coverage": [],
            }
        ).encode(),
        dtype=np.uint8,
    )
    alignment = Alignment(
        kind="alignment",
        id="aligned",
        schema_version="1.0.0",
        provenance=PROV,
        synchronization_id="sync-synthetic",
        observation_digests=["0" * 64],
        query_count=len(queries),
        arrays=[
            DenseArray(
                id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
            )
        ],
    )
    aligned = store.get_or_create(
        key("alignment", "aligned"), lambda: (alignment, {ARRAY_ID: array})
    )
    return cal, aligned


def test_motion_topology_provenance_and_immutable_cache(tmp_path: Path) -> None:
    calibration, models = scene()
    store = ArtifactStore(StorageRoot(tmp_path))
    cal, aligned = handles(store, calibration, join(models).query_many([0, 0.5, 1]))
    handle = publish_triangulation(store, cal, aligned, "practitioner")
    metadata = handle.metadata
    assert isinstance(metadata, Reconstruction)
    assert len(metadata.samples) == 3
    assert len(metadata.samples[0].landmarks) == len(get_args(Landmark))
    assert all(p.xyz_world is not None for s in metadata.samples for p in s.landmarks)
    assert metadata.samples[0].root_xyz_world != metadata.samples[-1].root_xyz_world
    d = load_diagnostics(handle)
    assert (
        d["samples"][1]["points"][0]["views"][0]["endpoints"][1]["observation_id"]
        == "0:1000"
    )
    assert "MMPose" in d["assumptions"][-1]
    assert not handle.read_array("raw_triangulation_diagnostics_json").flags.writeable
    again = publish_triangulation(store, cal, aligned, "practitioner")
    assert again.path == handle.path
    changed = publish_triangulation(
        store, cal, aligned, "practitioner", TriangulationConfig(pixel_sigma=3)
    )
    assert changed.path != handle.path
    assert load_diagnostics(handle) == d
    stage = next(s for s in default_stages() if s.name == "reconstruction")
    assert stage.revision == REVISION and stage.producer is produce_stage
    assert stage.capability_reason is None
    output = stage.producer(
        key("reconstruction", "stage"),
        {"calibration": cal, "attachment": aligned},
        {"participant_id": "practitioner"},
    )
    assert isinstance(output.artifact, Reconstruction)
    assert output.artifact.samples == metadata.samples


def test_actual_pipeline_reuses_raw_artifact_and_invalidates_upstream(
    tmp_path: Path,
) -> None:
    calibration, models = scene()
    store = ArtifactStore(StorageRoot(tmp_path / "store"))
    cal, aligned = handles(store, calibration, join(models).query_many([0, 1]))
    stages = []
    calls: list[str] = []
    for stage in default_stages():
        if stage.name == "reconstruction":
            stages.append(stage)
            continue

        def produce(
            k: ArtifactKey, inputs: Any, settings: Any, *, name: str = stage.name
        ) -> StageOutput:
            calls.append(name)
            if name in {"calibration", "attachment"}:
                original = cal if name == "calibration" else aligned
                metadata = original.metadata.model_copy(deep=True)
                metadata.provenance.config_digest = k.config_digest
                arrays = {
                    a.id: original.read_array(a.id)
                    for a in getattr(metadata, "arrays", [])
                }
                return StageOutput(metadata, arrays)
            return StageOutput(fixture_artifact(name, k))

        stages.append(
            Stage(
                stage.name, produce, STAGE_LAYERS[stage.name], DEPENDENCIES[stage.name]
            )
        )
    pipe = Pipeline(store, tuple(stages))
    source = tmp_path / "source.bin"
    source.write_bytes(b"synthetic input")
    pipe.register("synthetic", {"left": source, "right": source})
    config = {"reconstruction": {"participant_id": "practitioner"}}
    first = pipe.analyze("synthetic", "reconstruction", config=config)
    assert first["stages"]["reconstruction"]["status"] == "complete"
    assert first["stages"]["reconstruction"]["result_version"] == "1"
    second = pipe.analyze("synthetic", "reconstruction")
    assert second["stages"]["reconstruction"]["cached"]
    assert calls.count("attachment") == 1
    third = pipe.analyze("synthetic", "reconstruction", rerun="sync")
    assert third["stages"]["reconstruction"]["status"] == "complete"
    assert not third["stages"]["reconstruction"]["cached"]
    assert (
        third["stages"]["reconstruction"]["key"]
        != first["stages"]["reconstruction"]["key"]
    )
    assert calls.count("attachment") == 2
    assert calls.count("observations") == 1


def test_stage_rejects_calibration_from_other_sync(tmp_path: Path) -> None:
    calibration, models = scene(2)
    calibration.evidence_links = ["synchronization:obsolete"]
    store = ArtifactStore(StorageRoot(tmp_path))
    cal, aligned = handles(store, calibration, [join(models).query(0)])
    with pytest.raises(ValueError, match="synchronization revisions"):
        produce_stage(
            key("reconstruction", "mismatch"),
            {"calibration": cal, "attachment": aligned},
            {"participant_id": "practitioner"},
        )
