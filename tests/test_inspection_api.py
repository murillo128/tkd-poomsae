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
    store: ArtifactStore,
    value: Any,
    arrays: Any = None,
    *,
    inputs: dict[str, str] | None = None,
    sync_revision: str | None = None,
) -> tuple[ArtifactKey, ArtifactHandle]:
    key = ArtifactKey(
        layer=value.kind,
        inputs=inputs or {"fixture": "0" * 64},
        sync_revision=sync_revision,
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
    sync_key, sync_handle = persist(store, sync)
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
        inputs={
            name: hash_file(path) for name, path in (("left", left), ("right", right))
        },
        sync_revision=hash_file(sync_handle.path / "manifest.json"),
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
    sync_key, sync_handle = persist(store, sync)
    motion_key, _ = persist(
        store,
        final_motion,
        inputs={side: hash_file(tmp_path / side) for side in ("left", "right")},
        sync_revision=hash_file(sync_handle.path / "manifest.json"),
    )
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
    key, _ = persist(
        index.pipe.store,
        value,
        inputs=dict(products["reconstruction"].inputs),
        sync_revision=products["reconstruction"].sync_revision,
    )
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


def test_native_entity_revision_identifies_its_owning_window(
    inspection: tuple[Inspection, dict[str, ArtifactKey]],
) -> None:
    index, products = inspection
    headers, _ = index.headers("demo")
    first_key = ArtifactKey(
        **next(
            h["key"] for name, h in headers.items() if name.startswith("observations:")
        )
    )
    first = index.pipe.store.get(first_key).metadata
    assert isinstance(first, Observation)
    sync = index.pipe.store.get(products["sync"]).metadata
    assert isinstance(sync, Synchronization)
    windows = [(first_key, first)]
    for identifier, camera, pts, source in (
        ("second-window-observation", "left", 440, sync.offsets[0].source_id),
        ("other-camera-observation", "right", 900, sync.offsets[1].source_id),
    ):
        observation = first.model_copy(deep=True)
        observation.id = identifier
        observation.frame = first.frame.model_copy(
            update={
                "source_id": source,
                "camera_id": camera,
                "pts": pts,
                "source_seconds": pts / 1000,
                "global_seconds": pts / 1000,
            }
        )
        key, _ = persist(index.pipe.store, observation)
        windows.append((key, observation))
    index.register("demo", products, observations=[key for key, _ in windows])
    # Ownership survives a new service instance and does not depend on order.
    reopened = Inspection(index.pipe)
    with client(reopened) as http:
        revisions = set()
        for key, observation in reversed(windows):
            response = http.get(BASE + "/entities", params={"id": observation.id})
            assert response.status_code == 200
            selected = response.json()
            expected = hash_file(index.pipe.store.get(key).path / "manifest.json")
            assert selected["artifact_revision"] == expected
            assert selected["entity"]["frame"]["pts"] == observation.frame.pts
            revisions.add(selected["artifact_revision"])
        assert len(revisions) == len(windows)


def test_legacy_native_index_requires_owner_registration(
    inspection: tuple[Inspection, dict[str, ArtifactKey]],
) -> None:
    index, products = inspection
    headers, _ = index.headers("demo")
    key = ArtifactKey(
        **next(
            h["key"] for name, h in headers.items() if name.startswith("observations:")
        )
    )
    with index.connection("demo") as db:
        db.execute("DROP TABLE observation_owners")
    with client(index) as http:
        response = http.get(BASE + "/entities", params={"id": "native-left"})
        assert response.status_code == 409
        assert "re-register observations" in response.text
        index.register("demo", products, observations=[key])
        response = http.get(BASE + "/entities", params={"id": "native-left"})
        assert response.status_code == 200
        assert response.json()["artifact_revision"] == hash_file(
            index.pipe.store.get(key).path / "manifest.json"
        )


def test_observation_entity_uses_the_same_effective_frame_as_window(
    inspection: tuple[Inspection, dict[str, ArtifactKey]],
) -> None:
    index, _ = inspection
    with client(index) as http:
        for revision, start, expected in ((0, 23.3, 23.4), (1, 24.3, 24.4)):
            if revision:
                assert (
                    http.post(
                        BASE + "/sync-offset",
                        headers=WRITE,
                        json={
                            "expected_revision": 0,
                            "camera": "left",
                            "offset_seconds": 24,
                            **ATTRIBUTION,
                        },
                    ).status_code
                    == 200
                )
            window = http.get(
                BASE + "/observations/window",
                params={
                    "collection": "observations",
                    "start": start,
                    "end": start + 0.2,
                },
            ).json()
            selected = http.get(BASE + "/entities", params={"id": "native-left"}).json()
            assert selected["entity"]["frame"] == window["rows"][0]["frame"]
            assert selected["entity"]["frame"]["global_seconds"] == pytest.approx(
                expected
            )
            assert selected["entity"]["native_frame"]["global_seconds"] == 0.4
            assert (
                selected["source_evidence"][0]["frame"] == selected["entity"]["frame"]
            )
            assert selected["revision"] == window["revision"]


@pytest.mark.parametrize("change", ["sync", "source"])
def test_first_registration_rejects_preexisting_motion_with_stale_lineage(
    inspection: tuple[Inspection, dict[str, ArtifactKey]],
    change: str,
) -> None:
    index, products = inspection
    motion = index.pipe.store.get(products["reconstruction"]).metadata.model_copy(
        deep=True
    )
    motion.id = "unindexed-cached-motion"
    old_key, _ = persist(
        index.pipe.store,
        motion,
        inputs=dict(products["reconstruction"].inputs),
        sync_revision=products["reconstruction"].sync_revision,
        arrays={
            "trajectory": np.array([[1.0, 2.0, 3.0]]),
            "missing": np.zeros((1, 3), dtype=bool),
        },
    )
    sync = index.pipe.store.get(products["sync"]).metadata.model_copy(deep=True)
    assert isinstance(sync, Synchronization)
    with client(index) as http:
        if change == "sync":
            assert (
                http.post(
                    BASE + "/sync-offset",
                    headers=WRITE,
                    json={
                        "expected_revision": 0,
                        "camera": "left",
                        "offset_seconds": 24,
                        **ATTRIBUTION,
                    },
                ).status_code
                == 200
            )
            sync.offsets[0].manual_seconds = 24
            sync.offsets[0].manual_author = "operator"
            sync.offsets[0].manual_source = "inspector"
            sync.offsets[0].manual_reason = "inspect"
        else:
            project_file, _ = index.pipe._files("demo")
            path = Path(json.loads(project_file.read_text())["sources"]["left"])
            path.write_bytes(path.read_bytes() + b"changed-source")
            sync.offsets[0].source_id = "source:" + hash_file(path)
        sync.id = "current-sync"
        sync_key, _ = persist(index.pipe.store, sync)
        with pytest.raises(InspectionError, match="lineage"):
            index.register("demo", {"sync": sync_key, "reconstruction": old_key})
        # A failed registration retains the existing stale index, never old rows
        # certified under the new clock, even when this manifest was never indexed.
        result = http.get(BASE + "/reconstruction/window?start=23&end=25").json()
        assert result["available"] is False and result["rows"] == []


def test_partial_source_resolution_keeps_valid_evidence_and_missing_reason(
    inspection: tuple[Inspection, dict[str, ArtifactKey]],
) -> None:
    index, products = inspection
    headers, _ = index.headers("demo")
    observation_key = ArtifactKey(
        **next(
            h["key"] for name, h in headers.items() if name.startswith("observations:")
        )
    )
    motion = index.pipe.store.get(products["reconstruction"]).metadata.model_copy(
        deep=True
    )
    assert isinstance(motion, Reconstruction)
    motion.id = "partly-indexed-motion"
    motion.samples[0].landmarks[0].quality.source_ids = [
        "native-left",
        "native-right-not-indexed",
    ]
    motion.arrays = []
    key, _ = persist(
        index.pipe.store,
        motion,
        inputs=dict(products["reconstruction"].inputs),
        sync_revision=products["reconstruction"].sync_revision,
    )
    index.register(
        "demo",
        {"sync": products["sync"], "reconstruction": key},
        observations=[observation_key],
    )
    with client(index) as http:
        result = http.get(
            BASE + "/entities", params={"id": motion.id + "/samples/0/left_wrist"}
        ).json()
        assert [e["observation_id"] for e in result["source_evidence"]] == [
            "native-left"
        ]
        assert result["source_evidence_reason"] is not None
        assert result["source_evidence_unavailable_count"] == 1
        assert result["source_evidence_unavailable_ids"] == ["native-right-not-indexed"]
        assert result["evidence_truncated"] is False


def test_final_published_motion_verifies_transitive_native_sync_lineage(
    inspection: tuple[Inspection, dict[str, ArtifactKey]],
) -> None:
    """Real alignment/triangulation/fitting/temporal publishers remain inspectable."""
    from reconstruction.articulated.core import REVISION as FIT_REVISION
    from reconstruction.temporal import publish_temporal_motion
    from reconstruction.temporal.core import REVISION as TEMPORAL_REVISION
    from reconstruction.triangulation import REVISION as RAW_REVISION
    from reconstruction.triangulation import publish_triangulation
    from storage import hash_config
    from sync.alignment import REVISION as ALIGNMENT_REVISION
    from sync.alignment import JoinConfig, publish_alignment
    from tests.test_triangulation import native, scene

    index, products = inspection
    store = index.pipe.store
    sync_handle = store.get(products["sync"])
    sync = sync_handle.metadata
    assert isinstance(sync, Synchronization)
    calibration, models = scene(2)
    frames = native(models, times=(400, 440))
    keys, windows = [], []
    for i, camera in enumerate(("left", "right")):
        source_id = sync.offsets[i].source_id
        calibration.cameras[i].camera_id = camera
        calibration.cameras[i].source_id = source_id
        observations = frames[i * 2 : i * 2 + 2]
        for obs in observations:
            seconds = obs.frame.source_seconds + i * 0.5
            obs.frame = FrameTime(
                source_id=source_id,
                camera_id=camera,
                pts=obs.frame.pts + i * 500 if obs.frame.pts is not None else None,
                time_base_num=1,
                time_base_den=1000,
                source_seconds=seconds,
                global_seconds=seconds,
                offset_seconds=0,
            )
        payload = np.frombuffer(
            json.dumps(
                {
                    "version": 1,
                    "records": [
                        {
                            "observation": obs.model_dump(mode="json"),
                            "model_identity": {},
                            "inference_settings": {},
                            "hand_observations": {},
                            "hand_roi_to_source": {},
                            "source_orientation": {},
                            "tracker_state": {},
                        }
                        for obs in observations
                    ],
                }
            ).encode(),
            dtype=np.uint8,
        )
        metadata = observations[0].model_copy(deep=True)
        metadata.arrays = [
            DenseArray(
                id="window_records_json",
                dtype="uint8",
                shape=[len(payload)],
                axes=["json_byte"],
            )
        ]
        key, handle = persist(
            store,
            metadata,
            {"window_records_json": payload},
            inputs={"source": source_id.removeprefix("source:")},
        )
        keys.append(key)
        windows.append(handle)
    times = [23.4, 23.44]
    alignment = publish_alignment(store, windows, sync_handle, times)
    alignment_key = ArtifactKey(
        layer="alignment",
        inputs={
            f"window_{i}": digest
            for i, digest in enumerate(
                sorted(hash_file(w.path / "manifest.json") for w in windows)
            )
        },
        schema_version="1.0.0",
        algorithm_revision=ALIGNMENT_REVISION,
        config_digest=hash_config(
            {"join": JoinConfig().model_dump(mode="json"), "global_times": times}
        ),
        sync_revision=hash_file(sync_handle.path / "manifest.json"),
    )
    _, cal = persist(store, calibration)
    raw = publish_triangulation(store, cal, alignment, "practitioner")
    raw_key = key_for(
        raw,
        "reconstruction",
        {
            "calibration": hash_file(cal.path / "manifest.json"),
            "attachment": hash_file(alignment.path / "manifest.json"),
        },
        RAW_REVISION,
    )
    raw_key = replace(
        raw_key,
        calibration_revision=hash_file(cal.path / "manifest.json"),
        sync_revision=hash_file(alignment.path / "manifest.json"),
    )
    final = publish_temporal_motion(store, raw)
    fitted_key = key_for(
        final.fitted,
        "reconstruction",
        {
            "raw_reconstruction": hash_file(raw.path / "manifest.json"),
            "morphology": hash_file(final.morphology.path / "manifest.json"),
        },
        FIT_REVISION,
    )
    final_key = key_for(
        final.motion,
        "reconstruction",
        {
            "raw": hash_file(raw.path / "manifest.json"),
            "fit": hash_file(final.fitted.path / "manifest.json"),
            "morphology": hash_file(final.morphology.path / "manifest.json"),
        },
        TEMPORAL_REVISION,
    )
    lineage = [alignment_key, raw_key, fitted_key, *keys]
    with pytest.raises(InspectionError, match="lineage unavailable"):
        index.register("demo", {"sync": products["sync"], "reconstruction": final_key})
    index.register(
        "demo",
        {"sync": products["sync"], "reconstruction": final_key},
        observations=keys,
        lineage=lineage,
    )
    with client(index) as http:
        value = http.get(BASE + "/reconstruction/window?start=23.3&end=23.5").json()
        assert value["available"] and len(value["rows"]) == 2
        entity = http.get(
            BASE + "/entities",
            params={"id": final.motion.metadata.id + "/samples/0/left_wrist"},
        ).json()
        assert {e["frame"]["camera_id"] for e in entity["source_evidence"]} == {
            "left",
            "right",
        }
        assert entity["source_evidence_reason"] is None
        assert {e["frame"]["pts"] for e in entity["source_evidence"]} == {400, 900}
        # Upstream proof must be checked again on first registration in a fresh
        # index too: changing only the sync identity cannot certify old motion.
        sync = sync.model_copy(deep=True)
        sync.id = "other-automatic-sync"
        other_key, _ = persist(store, sync)
        with pytest.raises(InspectionError, match="synchronization"):
            index.register(
                "demo",
                {"sync": other_key, "reconstruction": final_key},
                lineage=lineage,
            )
