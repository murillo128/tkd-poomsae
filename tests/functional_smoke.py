"""Explicit local functional acceptance; never imported by default pytest.

Run with the application interpreter after pose.acceptance has produced receipts.
All network connects are denied. Dense artifacts and bundles stay in shared storage.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from calibration.cameras import CameraModel
from contracts.models import (
    DenseArray,
    FrameTime,
    Ground,
    Landmark2D,
    Observation,
    Provenance,
    Quality,
    Reconstruction,
    Semantics,
    Synchronization,
    SyncOffset,
)
from media import index_recording
from pipeline import Pipeline
from pose.observation_run import load_window, verify_receipt
from reconstruction.arms import publish_arm_actions
from reconstruction.features import publish_features
from reconstruction.footprints import publish_footprints
from reconstruction.ground import publish_contacts
from reconstruction.ground_view import publish_ground_view
from reconstruction.lower_body import publish_lower_body
from reconstruction.pivots import publish_pivots
from reconstruction.segmentation import publish_segmentation
from reconstruction.semantics import AssemblyConfig, publish_semantics
from reconstruction.temporal import publish_temporal_motion
from reconstruction.triangulation import TriangulationConfig, publish_triangulation
from storage import (
    ArtifactHandle,
    ArtifactKey,
    ArtifactStore,
    StorageRoot,
    hash_config,
    hash_file,
)
from sync import TimelineFailure
from sync.alignment import publish_alignment
from tests.fixtures.parser_motion import reconstructed
from tests.test_ground_contact import calibration
from tkd_poomsae.dataset_bootstrap import verify
from tkd_poomsae.inspection import Inspection
from tkd_poomsae.selections import resolve
from tkd_poomsae.vision.assets import registry, verified_paths


class InventoryStore(ArtifactStore):
    def __init__(self, root: StorageRoot | None = None) -> None:
        super().__init__(root)
        self.keys: dict[str, ArtifactKey] = {}
        self.created = 0

    def get_or_create(
        self, key: ArtifactKey, producer: Any, **kwargs: Any
    ) -> ArtifactHandle:
        produced = False

        def counted() -> Any:
            nonlocal produced
            result = producer()
            produced = True
            return result

        handle = super().get_or_create(key, counted, **kwargs)
        self.keys[key.digest] = key
        self.created += int(produced)
        return handle

    def key(self, handle: ArtifactHandle) -> ArtifactKey:
        return self.keys[handle.path.name]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def persist(store: InventoryStore, value: Any) -> ArtifactHandle:
    key = ArtifactKey(
        layer={"synchronization": "synchronization"}.get(value.kind, value.kind),
        inputs={"fixture": hash_config(value.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision="functional-synthetic-v1",
        config_digest=value.provenance.config_digest,
    )
    arrays = {}
    if isinstance(value, Observation):
        payload = {
            "version": 1,
            "records": [
                {
                    "observation": value.model_dump(mode="json"),
                    "model_identity": {"producer": "synthetic-projection; no model"},
                    "inference_settings": {},
                    "hand_observations": {},
                    "hand_roi_to_source": {},
                    "source_orientation": {},
                    "tracker_state": {},
                }
            ],
        }
        data = np.frombuffer(
            json.dumps(payload, sort_keys=True).encode(), dtype=np.uint8
        )
        value.arrays = [
            DenseArray(
                id="window_records_json",
                dtype="uint8",
                shape=[len(data)],
                axes=["json_byte"],
            )
        ]
        arrays = {"window_records_json": data}
        key = ArtifactKey(
            layer="observation",
            inputs=dict(key.inputs)
            | {"source": value.frame.source_id.removeprefix("source:")},
            schema_version="1.0.0",
            algorithm_revision="functional-synthetic-window-v1",
            config_digest=key.config_digest,
        )
    return store.get_or_create(key, lambda: (value, arrays))


def key_data(key: ArtifactKey) -> dict[str, Any]:
    return {
        "layer": key.layer,
        "inputs": dict(key.inputs),
        "schema_version": key.schema_version,
        "algorithm_revision": key.algorithm_revision,
        "config_digest": key.config_digest,
        "model_revision": key.model_revision,
        "calibration_revision": key.calibration_revision,
        "sync_revision": key.sync_revision,
    }


def bundle(
    store: InventoryStore,
    products: dict[str, ArtifactHandle],
    observations: list[ArtifactKey],
    lineage: list[ArtifactHandle],
) -> dict[str, Any]:
    return {
        "products": {name: key_data(store.key(h)) for name, h in products.items()},
        "observations": [key_data(k) for k in observations],
        "lineage": [key_data(store.key(h)) for h in lineage],
    }


def synthetic(store: InventoryStore, out: Path) -> dict[str, Any]:
    """Project dots with supplied synthetic metric cameras; no model inference."""
    pipe = Pipeline(store)
    project = "issue-50-synthetic"
    source, _ = reconstructed("compound", rate=20)
    cal = calibration()
    cal.cameras = cal.cameras[:2]
    cal.id = "functional-synthetic-calibration"
    cal.provenance = Provenance(
        producer="supplied-synthetic-cameras", config_digest="0" * 64
    )
    observations = []
    sources = {}
    for camera in cal.cameras:
        path = out / f"{camera.camera_id}.mp4"
        model = CameraModel(camera.intrinsics, np.array(camera.world_to_camera))
        pixels = []
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=20)
            stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
            stream.options = {"crf": "18", "preset": "ultrafast"}
            for sample in source.samples:
                points = [p for p in sample.landmarks if p.xyz_world is not None]
                xy = model.project([p.xyz_world for p in points])
                pixels.append((points, xy))
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
                for x, y in xy:
                    cv2.circle(image, (round(x), round(y)), 3, (255, 200, 40), -1)
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        digest = hash_file(path)
        camera.source_id = "source:" + digest
        sources[camera.camera_id] = path
        recording = index_recording(camera.camera_id, path)
        for i, (points, xy) in enumerate(pixels):
            value = Observation(
                kind="observation",
                id=f"synthetic:{camera.camera_id}:{i}",
                schema_version="1.0.0",
                provenance=cal.provenance,
                frame=FrameTime(
                    source_id=camera.source_id,
                    camera_id=camera.camera_id,
                    frame_index=i,
                    pts=recording.frames[i].pts,
                    time_base_num=recording.frames[i].time_base_num,
                    time_base_den=recording.frames[i].time_base_den,
                    source_seconds=recording.frames[i].seconds,
                    offset_seconds=0,
                    global_seconds=i / 20,
                ),
                landmarks=[
                    Landmark2D(
                        name=p.name,
                        xy_px=tuple(pixel),
                        quality=Quality(state="observed", score=1, uncertainty=0),
                    )
                    for p, pixel in zip(points, xy)
                ],
            )
            observations.append(persist(store, value))
    if not pipe._directory(project).exists():
        pipe.register(project, sources)
    else:
        registered = json.loads((pipe._directory(project) / "project.json").read_text())
        assert registered["sources"] == {k: str(v) for k, v in sources.items()}
    sync = persist(
        store,
        Synchronization(
            kind="synchronization",
            id="functional-synthetic-sync",
            schema_version="1.0.0",
            provenance=cal.provenance,
            offsets=[
                SyncOffset(
                    source_id=c.source_id,
                    automatic_seconds=0,
                    quality=Quality(state="observed", score=1, uncertainty=0),
                )
                for c in cal.cameras
            ],
        ),
    )
    cal.evidence_links = ["synchronization:" + sync.metadata.id]
    cal_handle = persist(store, cal)
    alignment = publish_alignment(
        store, observations, sync, [i / 20 for i in range(len(source.samples))]
    )
    # Ideal projected coordinates have known precision; this is fixture-only.
    triangulation = TriangulationConfig(pixel_sigma=0.1)
    raw = publish_triangulation(
        store, cal_handle, alignment, "synthetic-practitioner", triangulation
    )
    temporal = publish_temporal_motion(store, raw)
    motion = temporal.motion
    contacts = publish_contacts(store, motion, cal_handle)
    feet = publish_footprints(store, motion, contacts, cal_handle)
    pivots = publish_pivots(store, feet)
    ground = publish_ground_view(store, motion, cal_handle, pivots)
    features = publish_features(store, motion, ground)
    segmentation = publish_segmentation(store, features)
    arms = publish_arm_actions(store, features, segmentation, motion)
    legs = publish_lower_body(store, features, ground, segmentation)
    semantics = publish_semantics(store, features, segmentation, arms, legs)
    physical = [
        cal_handle,
        sync,
        alignment,
        raw,
        temporal.morphology,
        temporal.fitted,
        motion,
        contacts,
        feet,
        pivots,
        ground,
        *observations,
    ]
    before = {
        str(h.path / "manifest.json"): hash_file(h.path / "manifest.json")
        for h in physical
    }
    changed = publish_semantics(
        store, features, segmentation, arms, legs, AssemblyConfig(max_gap_seconds=0.2)
    )
    assert changed.path != semantics.path
    assert all(hash_file(Path(p)) == digest for p, digest in before.items())
    products = {
        "sync": sync,
        "calibration": cal_handle,
        "reconstruction": motion,
        "ground": ground,
        "semantics": semantics,
    }
    b = bundle(
        store,
        products,
        [store.key(h) for h in observations],
        [alignment, raw, temporal.morphology, temporal.fitted, *observations],
    )
    save(out / "synthetic-bundle.json", b)
    Inspection(pipe).register(
        project,
        {k: ArtifactKey(**v) for k, v in b["products"].items()},
        observations=[ArtifactKey(**v) for v in b["observations"]],
        lineage=[ArtifactKey(**v) for v in b["lineage"]],
    )
    assert isinstance(motion.metadata, Reconstruction)
    assert isinstance(raw.metadata, Reconstruction)
    assert isinstance(ground.metadata, Ground)
    assert isinstance(semantics.metadata, Semantics)
    assert any(
        p.xyz_world is not None for s in raw.metadata.samples for p in s.landmarks
    )
    assert ground.metadata.scale == "metric"
    assert any(sample.root_xyz_world is not None for sample in motion.metadata.samples)
    return {
        "project": project,
        "input": "projected compound motion with supplied synthetic metric cameras",
        "model_inference": False,
        "triangulation_config": triangulation.model_dump(mode="json"),
        "raw_samples": len(raw.metadata.samples),
        "steps": len(semantics.metadata.steps),
        "actions": len(semantics.metadata.actions),
        "ground_samples": len(ground.metadata.samples),
        "parser_config_changed_physical_hashes_unchanged": True,
        "products": {
            k: hash_file(h.path / "manifest.json") for k, h in products.items()
        },
        "product_configs": {k: key_data(store.key(h)) for k, h in products.items()},
    }


def real(store: InventoryStore, out: Path, receipts: list[Path]) -> dict[str, Any]:
    pipe = Pipeline(store)
    loaded = [verify_receipt(p, store) for p in receipts]
    keys: dict[str, list[ArtifactKey]] = {}
    full_projects = {
        entry["view"].split(":")[0]
        for receipt in loaded
        if receipt["identity"]["selection"] == "demo-full"
        for entry in receipt["windows"]
    }
    for receipt in loaded:
        for entry in receipt["windows"]:
            project = entry["view"].split(":")[0]
            # One tracking recipe per native stream: full and smoke overlap.
            if (
                project in full_projects
                and receipt["identity"]["selection"] != "demo-full"
            ):
                continue
            key = ArtifactKey(**entry["key"])
            if key not in keys.setdefault(project, []):
                keys[project].append(key)
    result = {}
    for project, native_keys in keys.items():
        print(project, flush=True)
        sync_reason = None
        try:
            sync = pipe.solve_sync(project)
            store.keys[sync.path.name] = pipe._expected_keys(
                json.loads((pipe._directory(project) / "project.json").read_text()),
                pipe.status(project),
            )["sync"]
        except TimelineFailure as exc:
            sync_reason = str(exc)
            sources = json.loads(
                (pipe._directory(project) / "project.json").read_text()
            )["sources"]
            hashes = {camera: hash_file(Path(path)) for camera, path in sources.items()}
            reference = "mendeley-frontal"
            digest = hash_config(
                {
                    "reason": sync_reason,
                    "reference": reference,
                    "reference_clock_policy": "explicit-timing-reference-v1",
                }
            )
            key = ArtifactKey(
                layer="synchronization",
                inputs=hashes,
                schema_version="1.0.0",
                algorithm_revision="native-inspection-reference-v1",
                config_digest=digest,
            )
            # Reference zero defines one source clock. It asserts no cross-view
            # alignment; other views stay excluded with unknown automatic offsets.
            native = Synchronization(
                kind="synchronization",
                id=key.digest,
                schema_version="1.0.0",
                provenance=Provenance(
                    producer="native-inspection-reference", config_digest=digest
                ),
                reference_source_id="source:" + hashes[reference],
                offsets=[
                    SyncOffset(
                        source_id="source:" + sha,
                        automatic_seconds=None,
                        timing_reference=camera == reference,
                        retained=camera == reference,
                        exclusion_reason=None if camera == reference else sync_reason,
                        quality=Quality(state="unknown"),
                    )
                    for camera, sha in hashes.items()
                ],
                diagnostics=["automatic synchronization unavailable: " + sync_reason],
            )
            sync = store.get_or_create(key, lambda: (native, {}))
        assert isinstance(sync.metadata, Synchronization)
        selection = "demo-full" if project.endswith("-1") else "all-forms-smoke"
        # For each bounded form, the production scene estimator consumes verified sync.
        from calibration.natural import (
            SceneCandidate,
            SceneView,
            estimate_scene,
            persist_scene_candidate,
        )
        from calibration.natural_cli import nearest_global_refs
        from media import MediaReader, index_recording

        windows = [w for w in resolve(selection) if w.execution_id == project]
        rows = {o.source_id: o for o in sync.metadata.offsets}
        retained_windows = [
            w for w in windows if rows["source:" + w.source_sha256].retained
        ]
        start = max(
            w.start_seconds + rows["source:" + w.source_sha256].effective_seconds
            for w in retained_windows
        )
        end = min(
            w.end_seconds + rows["source:" + w.source_sha256].effective_seconds
            for w in retained_windows
        )
        if sync_reason is not None:
            candidate = SceneCandidate(
                "unavailable", ["automatic synchronization unavailable: " + sync_reason]
            )
        elif len(retained_windows) < 2:
            candidate = SceneCandidate(
                "unavailable", ["automatic synchronization excluded a selected view"]
            )
        elif end <= start:
            candidate = SceneCandidate(
                "unavailable", ["selected windows have no common global interval"]
            )
        else:
            views = []
            targets = [start + (end - start) * (i + 0.5) / 5 for i in range(5)]
            for window in windows:
                rec = index_recording(window.camera_id, window.source_path)
                refs = nearest_global_refs(
                    rec.frames, targets, rows[rec.source_id].effective_seconds
                )
                reader = MediaReader(rec)
                views.append(
                    SceneView(
                        window.camera_id,
                        rec.source_id,
                        tuple(reader.frame(ref).rgb for ref in refs),
                        None,
                        rec.sha256,
                        tuple(ref.seconds for ref in refs),
                        tuple(
                            f"{ref.pts}:{ref.time_base_num}/{ref.time_base_den}"
                            for ref in refs
                        ),
                    )
                )
            candidate = estimate_scene(views, sync.metadata)
        candidate_path = persist_scene_candidate(candidate, out / project)
        state = pipe.analyze(
            project,
            through="parsing",
            config={"calibration": {"candidate": str(candidate_path)}},
        )
        b = bundle(store, {"sync": sync}, native_keys, [])
        save(out / f"{project}-bundle.json", b)
        Inspection(pipe).register(
            project, {"sync": store.key(sync)}, observations=native_keys
        )
        # Full native windows stay indexed; the downstream smoke uses at most
        # one verified window per camera, nearest the actual common start.
        by_camera: dict[str, list[ArtifactHandle]] = {}
        for key in native_keys:
            handle = store.get(key)
            assert isinstance(handle.metadata, Observation)
            if rows[handle.metadata.frame.source_id].retained:
                by_camera.setdefault(handle.metadata.frame.camera_id, []).append(handle)

        def distance(handle: ArtifactHandle) -> float:
            assert isinstance(handle.metadata, Observation)
            frame = handle.metadata.frame
            return abs(
                frame.source_seconds + rows[frame.source_id].effective_seconds - start
            )

        selected = [min(handles, key=distance) for handles in by_camera.values()]
        times = [
            obs.frame.source_seconds + rows[obs.frame.source_id].effective_seconds
            for handle in selected
            for obs in load_window(handle)
        ]
        times = sorted(set(times))
        # Calibration is unavailable: inspect a bounded grid of genuine native
        # global instants instead of materializing a dense hypothetical 3D run.
        selected_times = [
            times[i]
            for i in np.linspace(0, len(times) - 1, min(32, len(times)), dtype=int)
        ]
        alignment = publish_alignment(store, selected, sync, selected_times)
        result[project] = {
            "synchronization_status": "unavailable"
            if sync_reason
            else "partial"
            if len(retained_windows) < 2
            else "automatic-estimate",
            "synchronization_reason": sync_reason,
            "native_reference_only": len(retained_windows) < 2,
            "sync_manifest_sha256": hash_file(sync.path / "manifest.json"),
            "offsets": [o.model_dump(mode="json") for o in sync.metadata.offsets],
            "calibration": {
                "status": candidate.status,
                "reasons": candidate.reasons,
                "candidate_sha256": hash_file(candidate_path),
            },
            "alignment_query_count": len(selected_times),
            "alignment_manifest_sha256": hash_file(alignment.path / "manifest.json"),
            "stages": {
                k: {"status": v["status"], "diagnostics": v["diagnostics"]}
                for k, v in state["stages"].items()
            },
            "native_windows": len(native_keys),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_only and not args.receipt:
        parser.error("real acceptance requires explicit verified observation receipts")
    args.receipt = [p.resolve() for p in args.receipt]
    calls: Counter[str] = Counter()

    def deny(*_args: Any, **_kwargs: Any) -> Any:
        calls["network"] += 1
        raise AssertionError("functional acceptance must remain offline")

    socket.socket.connect = deny  # type: ignore[method-assign]
    socket.create_connection = deny
    store = InventoryStore()
    out = store.root.namespace("runs") / "functional-issue-50"
    out.mkdir(parents=True, exist_ok=True)
    before = {
        str(w.source_path): hash_file(w.source_path) for w in resolve("all-forms")
    }
    report: dict[str, Any] = {"version": 1, "synthetic": synthetic(store, out)}
    if not args.synthetic_only:
        report["real"] = real(store, out, args.receipt)
    assert all(hash_file(Path(p)) == digest for p, digest in before.items())
    report["models"] = {
        "verified_assets": len(verified_paths()),
        "registry_sha256": hash_config(registry()),
        "asset_sha256": {a["path"]: a["sha256"] for a in registry()["assets"]},
    }
    report["dataset_integrity"] = verify(store.root)
    report["original_source_hashes_unchanged"] = before
    physical = {
        str(store.get(k).path / "manifest.json"): hash_file(
            store.get(k).path / "manifest.json"
        )
        for k in store.keys.values()
    }
    first_created = store.created
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="tkd-functional-repeat-") as elsewhere:
            os.chdir(elsewhere)
            repeated = synthetic(store, out)
            assert repeated == report["synthetic"]
            if not args.synthetic_only:
                repeated_real = real(store, out, args.receipt)
                assert repeated_real == report["real"]
    finally:
        os.chdir(previous)
    assert store.created == first_created
    assert all(hash_file(Path(p)) == digest for p, digest in physical.items())
    report["offline_second_cwd"] = {
        "new_artifacts": 0,
        "network_connect_attempts": calls["network"],
        "verified_artifacts": len(physical),
        "identical_outcomes": True,
    }
    report["network_connect_attempts"] = calls["network"]
    assert not calls["network"]
    save(args.output, report)


if __name__ == "__main__":
    main()
