"""Offline integration of bounded persisted inspection and revision-safe edits."""

from __future__ import annotations

import io
import json
import tracemalloc
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from contracts.models import (
    DenseArray,
    Footprint,
    FrameTime,
    Ground,
    Landmark3D,
    MotionSample,
    Observation,
    Provenance,
    Quality,
    Reconstruction,
    Synchronization,
    SyncOffset,
)
from pipeline import Pipeline
from reconstruction.arms import publish_arm_actions
from reconstruction.lower_body import publish_lower_body
from reconstruction.segmentation import publish_segmentation
from reconstruction.semantics import publish_semantics
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, StorageRoot, hash_file
from tests.test_arm_actions import persist_motion
from tests.test_media_reader import video
from tests.test_segmentation import persist_features
from tests.test_semantic_assembly import compound
from tkd_poomsae.api import create_app
from tkd_poomsae.inspection import MAX_BYTES, Inspection, InspectionError

BASE = "/api/projects/demo/inspection"
WRITE = {"x-tkd-local-request": "1"}
ATTRIBUTION = {"author": "operator", "source": "inspector", "reason": "inspect"}


def key_for(
    handle: ArtifactHandle, layer: str, inputs: dict[str, str], revision: str
) -> ArtifactKey:
    return ArtifactKey(
        layer=layer,
        inputs=inputs,
        schema_version="1.0.0",
        algorithm_revision=revision,
        config_digest=handle.metadata.provenance.config_digest,
    )


def persist(
    store: ArtifactStore, value: Any, arrays: Any = None
) -> tuple[ArtifactKey, ArtifactHandle]:
    key = ArtifactKey(
        layer=value.kind,
        inputs={"fixture": "0" * 64},
        schema_version="1.0.0",
        algorithm_revision=value.id,
        config_digest=value.provenance.config_digest,
    )
    return key, store.get_or_create(key, lambda: (value, arrays or {}))


@pytest.fixture
def inspection(tmp_path: Path) -> tuple[Inspection, dict[str, ArtifactKey]]:
    store = ArtifactStore(StorageRoot(tmp_path / "store"))
    pipe = Pipeline(store)
    left = video(tmp_path / "left.mkv", [400, 440, 500, 580])
    right = video(tmp_path / "right.mkv", [900, 940, 980])
    pipe.register("demo", {"left": left, "right": right})
    q = Quality(state="observed", score=0.9, source_ids=[])
    sync = Synchronization(
        kind="synchronization",
        id="sync",
        schema_version="1.0.0",
        provenance=Provenance(producer="fixture", config_digest="0" * 64),
        offsets=[
            SyncOffset(
                source_id="source:" + hash_file(path),
                automatic_seconds=offset,
                quality=q,
            )
            for path, offset in ((left, 23.0), (right, 22.5))
        ],
    )
    sync_key, _ = persist(store, sync)
    obs = Observation(
        kind="observation",
        id="native-left",
        schema_version="1.0.0",
        provenance=sync.provenance,
        frame=FrameTime.model_validate(
            {
                "source_id": sync.offsets[0].source_id,
                "camera_id": "left",
                "pts": 400,
                "time_base_num": 1,
                "time_base_den": 1000,
                "source_seconds": 0.4,
                "offset_seconds": 0,
                "global_seconds": 0.4,
            }
        ),
        landmarks=[],
    )
    obs_key, _ = persist(store, obs)
    motion = Reconstruction(
        kind="reconstruction",
        id="motion",
        schema_version="1.0.0",
        provenance=sync.provenance,
        calibration_id="cal",
        participant_id="person",
        scale="arbitrary",
        samples=[
            MotionSample(
                global_seconds=23.4,
                root_xyz_world=None,
                root_orientation=None,
                quality=q,
                landmarks=[
                    Landmark3D(
                        name="left_wrist",
                        xyz_world=(1, 2, 3),
                        quality=q.model_copy(update={"source_ids": [obs.id]}),
                    ),
                    Landmark3D(
                        name="right_wrist",
                        xyz_world=None,
                        quality=Quality(state="unknown"),
                    ),
                ],
            )
        ],
        arrays=[
            DenseArray(
                id="trajectory",
                dtype="float64",
                shape=[1, 3],
                axes=["native_time", "xyz"],
                missing_mask_id="missing",
            ),
            DenseArray(
                id="missing", dtype="bool", shape=[1, 3], axes=["native_time", "xyz"]
            ),
        ],
    )
    motion_key, _ = persist(
        store,
        motion,
        {
            "trajectory": np.array([[1.0, 2.0, 3.0]]),
            "missing": np.zeros((1, 3), dtype=bool),
        },
    )
    ground = Ground(
        kind="ground",
        id="ground",
        schema_version="1.0.0",
        provenance=sync.provenance,
        reconstruction_id=motion.id,
        scale="arbitrary",
        samples=[],
        footprints=[
            Footprint.model_validate(
                {
                    "id": "left-placement",
                    "foot": "left",
                    "interval": {"start": 23.4, "end": 23.6},
                    "xy_ground": None,
                    "yaw_rad": None,
                    "quality": {"state": "unknown"},
                }
            )
        ],
    )
    ground_key, _ = persist(store, ground)
    index = Inspection(pipe)
    products = {"sync": sync_key, "reconstruction": motion_key, "ground": ground_key}
    index.register("demo", products, observations=[obs_key])
    return index, products


def client(index: Inspection) -> TestClient:
    return TestClient(
        create_app(
            index.pipe, allowed_roots={"local": index.pipe.store.root.path.parent}
        ),
        base_url="http://localhost",
    )


def test_windows_provenance_masks_arrays_and_validators(
    inspection: tuple[Inspection, dict[str, ArtifactKey]],
) -> None:
    index, _ = inspection
    with client(index) as http:
        url = BASE + "/reconstruction/window?start=23&end=24"
        response = http.get(url)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["unit"] == "arbitrary" and data["available"]
        assert data["rows"][0]["landmarks"][1]["xyz_world"] is None
        assert "NaN" not in response.text
        assert (
            http.get(
                url, headers={"if-none-match": response.headers["etag"]}
            ).status_code
            == 304
        )
        assert http.get(url + "&expected_revision=old").status_code == 409
        joint = http.get(
            BASE + "/entities", params={"id": "motion/samples/0/left_wrist"}
        ).json()
        assert joint["source_evidence"][0]["frame"]["pts"] == 400
        assert "internal consistency" in joint["diagnostic_meaning"]
        footprint = http.get(BASE + "/entities", params={"id": "left-placement"}).json()
        assert footprint["source_evidence"][0]["observation_id"] == "native-left"
        array = http.get(BASE + "/reconstruction/arrays/trajectory?start=23&end=24")
        assert array.status_code == 200, array.text
        with np.load(io.BytesIO(array.content), allow_pickle=False) as archive:
            assert archive["values"].tolist() == [[1, 2, 3]]
            assert not archive["missing_mask"].any()
        for extra in ("&limit=257", "&collection=private", "&cursor=-1"):
            assert http.get(url + extra).status_code == 422
        assert (
            http.get(BASE + "/reconstruction/window?start=0&end=31").status_code == 422
        )
        assert (
            http.get(url, headers={"origin": "https://evil.example"}).status_code == 403
        )
        assert (
            http.get(
                BASE + "/reconstruction/arrays/../../etc/passwd?start=0&end=1"
            ).status_code
            != 200
        )


def test_sync_conflict_native_time_mapping_and_stale_geometry(
    inspection: tuple[Inspection, dict[str, ArtifactKey]],
) -> None:
    index, products = inspection
    with client(index) as http:
        mapping = http.get(BASE + "/time/left?seconds=23.45")
        assert mapping.status_code == 200, mapping.text
        value = mapping.json()
        assert value["before"]["pts"] == 440 and value["after"]["pts"] == 500
        assert value["nearest_gap_seconds"] == pytest.approx(0.01)
        original = http.get(
            BASE + "/observations/window?collection=observations&start=23.3&end=23.5"
        ).json()
        assert original["rows"][0]["frame"]["global_seconds"] == 23.4
        edit = {
            "expected_revision": 0,
            "camera": "left",
            "offset_seconds": 24,
            **ATTRIBUTION,
        }
        assert http.post(BASE + "/sync-offset", json=edit).status_code == 403
        assert (
            http.post(BASE + "/sync-offset", json=edit, headers=WRITE).status_code
            == 200
        )
        assert (
            http.post(BASE + "/sync-offset", json=edit, headers=WRITE).status_code
            == 409
        )
        stale = http.get(BASE + "/reconstruction/window?start=23&end=24").json()
        assert not stale["available"] and not stale["rows"]
        assert (
            http.get(
                BASE + "/entities", params={"id": "motion/samples/0/left_wrist"}
            ).status_code
            == 409
        )
        native = http.get(
            BASE + "/observations/window?collection=observations&start=24.3&end=24.5"
        ).json()
        assert native["rows"][0]["frame"]["global_seconds"] == 24.4
        assert native["artifact_revisions"] == original["artifact_revisions"]
        mapped = http.get(BASE + "/time/left?seconds=24.4").json()
        assert mapped["nearest"]["pts"] == 400
        assert mapped["offset"]["automatic_seconds"] == 23
        # Even trusted re-registration cannot make the old timed product current.
        sync = index.pipe.store.get(products["sync"]).metadata.model_copy(deep=True)
        assert isinstance(sync, Synchronization)
        sync.id = "sync-edited"
        sync.offsets[0].manual_seconds = 24
        sync.offsets[0].manual_author = "operator"
        sync.offsets[0].manual_source = "inspector"
        sync.offsets[0].manual_reason = "inspect"
        key, _ = persist(index.pipe.store, sync)
        with pytest.raises(InspectionError):
            index.register("demo", products | {"sync": key})


def test_parser_edits_preserve_vision_and_automatic_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical, floor, geometry = compound()
    store = ArtifactStore(StorageRoot(tmp_path / "store"))
    features = persist_features(store, physical)
    coarse = publish_segmentation(store, features)
    motion = persist_motion(store, physical, geometry)
    arms = publish_arm_actions(store, features, coarse, motion)
    ground_key, ground = persist(store, floor)
    lower = publish_lower_body(store, features, ground, coarse)
    automatic = publish_semantics(store, features, coarse, arms, lower)
    from reconstruction.semantics import load_semantic_evidence

    evidence = load_semantic_evidence(automatic)
    semantics_key = key_for(
        automatic, "semantics", evidence["input_revisions"], "semantic-assembly-v1"
    )
    # Producers use their authoritative revision constants.
    from reconstruction.semantics.core import REVISION

    semantics_key = replace(semantics_key, algorithm_revision=REVISION)
    assert isinstance(motion.metadata, Reconstruction)
    final_motion = motion.metadata.model_copy(deep=True)
    final_motion.arrays = []
    final_motion.samples = [
        MotionSample(
            global_seconds=t,
            root_xyz_world=None,
            root_orientation=None,
            landmarks=[],
            quality=Quality(state="unknown"),
        )
        for t in evidence["motion_times"]
    ]
    motion_key, _ = persist(store, final_motion)
    pipe = Pipeline(store)
    for side in ("left", "right"):
        (tmp_path / side).write_bytes(side.encode())
    pipe.register("demo", {side: tmp_path / side for side in ("left", "right")})
    sync = Synchronization(
        kind="synchronization",
        id="sync",
        schema_version="1.0.0",
        provenance=Provenance(producer="fixture", config_digest="0" * 64),
        offsets=[
            SyncOffset(
                source_id="source:" + hash_file(tmp_path / side),
                automatic_seconds=0,
                quality=Quality(state="observed"),
            )
            for side in ("left", "right")
        ],
    )
    sync_key, _ = persist(store, sync)
    index = Inspection(pipe)
    index.register(
        "demo",
        {
            "sync": sync_key,
            "reconstruction": motion_key,
            "ground": ground_key,
            "semantics": semantics_key,
        },
    )
    original = hash_file(automatic.path / "metadata.json")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("inspection must never analyze")

    monkeypatch.setattr(pipe, "analyze", forbidden)
    with client(index) as http:
        actions = http.get(
            BASE + "/semantics/window?collection=actions&start=23&end=26"
        ).json()
        selected = http.get(
            BASE + "/entities", params={"id": actions["rows"][0]["id"]}
        ).json()
        assert selected["entity"]["interval"] == actions["rows"][0]["interval"]
        assert selected["source_evidence_reason"] is not None
        action = automatic.metadata.actions[0]  # type: ignore[union-attr]
        edit = {
            "expected_revision": 0,
            "automatic_revision": hash_file(automatic.path / "manifest.json"),
            "command": "apply",
            "operations": [
                {
                    "kind": "keyframe_add",
                    "keyframe": {
                        "id": "manual",
                        "action_id": action.id,
                        "global_seconds": action.interval.start,
                        "event": "inspection",
                        "motion_sample_indices": [],
                        "quality": {"state": "unknown"},
                    },
                }
            ],
            **ATTRIBUTION,
        }
        before = http.get(BASE + "/ground/window?start=23&end=26").json()
        changed = http.post(BASE + "/parser-edits", json=edit, headers=WRITE)
        assert changed.status_code == 200, changed.text
        assert changed.json()["origin"] == "manual"
        assert (
            http.post(BASE + "/parser-edits", json=edit, headers=WRITE).status_code
            == 409
        )
        after = http.get(
            BASE + "/semantics/window?collection=keyframes&start=23&end=26"
        ).json()
        assert any(r["id"] == "manual" for r in after["rows"])
        assert after["effective_edit_revision"] == 1
        assert (
            http.get(BASE + "/ground/window?start=23&end=26").json()[
                "artifact_revisions"
            ]
            == before["artifact_revisions"]
        )
        edit.update(
            expected_revision=1,
            operations=[
                {"kind": "keyframe_move", "target_id": "missing", "global_seconds": 24}
            ],
        )
        assert (
            http.post(BASE + "/parser-edits", json=edit, headers=WRITE).status_code
            == 422
        )
        edit.update(command="undo", operations=[])
        assert (
            http.post(BASE + "/parser-edits", json=edit, headers=WRITE).status_code
            == 200
        )
        assert hash_file(automatic.path / "metadata.json") == original


def test_long_recording_paging_memory_and_cancellation(
    inspection: tuple[Inspection, dict[str, ArtifactKey]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, products = inspection
    old = index.pipe.store.get(products["reconstruction"]).metadata
    assert isinstance(old, Reconstruction)
    value = old.model_copy(deep=True)
    value.id = "long"
    value.arrays = []
    value.samples = [
        old.samples[0].model_copy(update={"global_seconds": i / 1000})
        for i in range(50000)
    ]
    key, _ = persist(index.pipe.store, value)
    index.register("demo", {"sync": products["sync"], "reconstruction": key})

    # Queries must not reopen dense artifact metadata through the store.
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("query attempted a whole-artifact read")

    monkeypatch.setattr(index.pipe.store, "get", forbidden)
    tracemalloc.start()
    data = index.window("demo", "reconstruction", "samples", 0, 30, 256)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(data["rows"]) == 256 and data["next_cursor"] is not None
    assert len(json.dumps(data)) < MAX_BYTES
    assert peak < 4 * MAX_BYTES
    next_page = index.window(
        "demo", "reconstruction", "samples", 0, 30, 256, data["next_cursor"]
    )
    assert next_page["rows"][0]["global_seconds"] > data["rows"][-1]["global_seconds"]
    cancelled = Event()
    cancelled.set()
    with pytest.raises(InspectionError, match="cancelled"):
        index.window(
            "demo", "reconstruction", "samples", 0, 30, 256, cancelled=cancelled
        )


def test_finite_errors_source_authorization_and_changed_artifact(
    inspection: tuple[Inspection, dict[str, ArtifactKey]],
) -> None:
    index, products = inspection
    with client(index) as http:
        for value in ("nan", "inf", "-inf"):
            response = http.get(
                BASE + "/reconstruction/window", params={"start": value, "end": 24}
            )
            assert response.status_code == 422
            assert "NaN" not in response.text and "Infinity" not in response.text
        response = http.post(
            BASE + "/sync-offset",
            headers=WRITE | {"content-type": "application/json"},
            content='{"camera":"left","offset_seconds":NaN,"expected_revision":0,"author":"a","source":"s","reason":"r"}',
        )
        assert response.status_code == 422
        assert "NaN" not in response.text
        array = index.pipe.store.get(products["reconstruction"])
        array_path = array.path / array.files["trajectory"]
        array_path.write_bytes(array_path.read_bytes() + b"tampered")
        assert (
            http.get(BASE + "/reconstruction/window?start=23&end=24").status_code == 409
        )
    forbidden = TestClient(
        create_app(index.pipe, allowed_roots={}), base_url="http://localhost"
    )
    with forbidden as http:
        assert (
            http.get(BASE + "/reconstruction/window?start=23&end=24").status_code == 403
        )


def test_cli_registration_and_sync_compare_under_lock(
    inspection: tuple[Inspection, dict[str, ArtifactKey]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from pipeline.runner import RevisionConflict
    from tkd_poomsae.cli import main

    index, products = inspection
    bundle = tmp_path / "products.json"
    bundle.write_text(
        json.dumps(
            {
                "products": {
                    name: key.__dict__ | {"inputs": dict(key.inputs)}
                    for name, key in products.items()
                }
            }
        )
    )
    monkeypatch.setenv("TKD_DATA_ROOT", str(index.pipe.store.root.path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["tkd-poomsae", "inspection-register", "demo", "--artifacts", str(bundle)],
    )
    assert main() == 0

    def change() -> str:
        try:
            index.pipe.revise_sync_offset(
                "demo",
                "left",
                24,
                expected_revision=0,
                author=ATTRIBUTION["author"],
                source=ATTRIBUTION["source"],
                reason=ATTRIBUTION["reason"],
            )
            return "accepted"
        except RevisionConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: change(), range(2))) == [
            "accepted",
            "conflict",
        ]
