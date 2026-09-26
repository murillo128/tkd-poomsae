"""Canonical operator path, using supplied geometry/2D rather than model accuracy."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from contracts.models import (
    Calibration,
    DenseArray,
    Observation,
    Reconstruction,
    Semantics,
)
from pipeline.offline import configured_pipeline
from pipeline.runner import key_data
from pose.observation_run import ARRAY_ID, _record_bytes
from storage import ArtifactKey, hash_file
from tests.fixtures.controlled_acceptance import PROJECT, build
from tests.test_inspection_api import persist
from tkd_poomsae.cli import main
from tkd_poomsae.inspection import Inspection


def test_canonical_run_reuse_parser_and_capabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pipe, store, handles = build(tmp_path / "controlled")
    monkeypatch.setenv("TKD_DATA_ROOT", str(store.root.path))
    monkeypatch.setenv("TKD_VISION_ACTIVE", "1")
    cal = handles["calibration"].metadata.model_copy(deep=True)
    assert isinstance(cal, Calibration)
    cal.evidence_links = []  # Supplied cameras are independent of clock estimation.
    cal.id += "-independent"
    cal_key, _ = persist(store, cal)
    key_file = tmp_path / "calibration-key.json"
    key_file.write_text(json.dumps(key_data(cal_key)))
    project_data = json.loads(pipe._files(PROJECT)[0].read_text())
    cameras = sorted(project_data["sources"])
    windows = []
    for camera in cameras:
        observations = sorted(
            [
                h.metadata
                for h in (
                    store.get(k)
                    for k in list(store.keys.values())
                    if k.layer == "observation"
                )
                if isinstance(h.metadata, Observation)
                and h.metadata.frame.camera_id == camera
            ],
            key=lambda o: o.frame.source_seconds,
        )
        records = [
            {
                "observation": o.model_dump(mode="json"),
                "model_identity": {},
                "inference_settings": {},
                "hand_observations": {},
                "hand_roi_to_source": {},
                "source_orientation": {},
                "tracker_state": {},
            }
            for o in observations
        ]
        array = _record_bytes(records)
        metadata = observations[0].model_copy(deep=True)
        metadata.id = "supplied-window-" + camera
        metadata.arrays = [
            DenseArray(
                id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
            )
        ]
        key = ArtifactKey(
            layer="observation",
            inputs={"source": hash_file(Path(project_data["sources"][camera]))},
            schema_version="1.0.0",
            algorithm_revision="supplied-test-window",
            config_digest=metadata.provenance.config_digest,
        )
        store.get_or_create(key, lambda: (metadata, {ARRAY_ID: array}))
        windows.append(key)

    inference_calls = []

    def supplied_windows(
        _project: str, selected: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        inference_calls.append(1)
        assert {w.camera_id for w in selected} == set(cameras)
        return {"windows": [{"key": key_data(k)} for k in windows]}

    import pose.observation_run

    monkeypatch.setattr(pose.observation_run, "run_windows", supplied_windows)
    # Dot projections do not supply reliable motion/audio synchronization cues.
    # Supply declared synthetic cues while retaining the production solver.
    import sync
    from tests.test_sync_solver import _source

    def supplied_cues(recording: Any, **_kwargs: Any) -> Any:
        cues = _source(recording.source_id, 0).cues

        def bounded(stream: Any) -> Any:
            return replace(
                stream,
                samples=tuple(
                    sample for sample in stream.samples if sample.end_seconds <= 4
                ),
            )

        return replace(
            cues,
            source_sha256=recording.sha256,
            audio=bounded(cues.audio),
            motion=bounded(cues.motion),
        )

    original_extract = sync.extract_cues
    monkeypatch.setattr(sync, "extract_cues", supplied_cues)
    config: dict[str, Any] = {
        "sync": {
            "reference": cameras[0],
            "manual_offsets": {
                c: {
                    "offset_seconds": 0,
                    "author": "synthetic",
                    "source": "supplied clocks",
                    "reason": "controlled fixture",
                }
                for c in cameras
            },
        },
        "calibration": {"artifact": key_file.name},
        "reconstruction": {
            "participant_id": "controlled-practitioner",
            "triangulation": {"pixel_sigma": 0.001},
        },
    }
    config_file = tmp_path / "settings.json"
    config_file.write_text(json.dumps(config))
    cwd = tmp_path / "clean"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    def invoke(*args: str) -> tuple[int, dict[str, Any]]:
        monkeypatch.setattr(sys, "argv", ["tkd-poomsae", "run", PROJECT, *args])
        code = main()
        captured = capsys.readouterr()
        assert captured.out, captured.err
        return code, json.loads(captured.out)

    code, first = invoke("--config", str(config_file))
    assert code == 0, first
    assert first["inspection_registered"]
    assert all(s["status"] == "complete" for s in first["stages"].values())
    index = Inspection(configured_pipeline(PROJECT))
    with index.connection(PROJECT) as db:
        assert db.execute("SELECT count(*) FROM products").fetchone()[0] >= 5
    headers, clock = index.headers(PROJECT)
    assert all(index.current(header, clock) for header in headers.values())
    semantics = store.get(ArtifactKey(**headers["semantics"]["key"])).metadata
    assert isinstance(semantics, Semantics)
    assert semantics.provenance.producer == "reconstruction.semantics"
    # The real synchronization solver retains timing uncertainty, so entity
    # counts need not match the fixture's separately supplied exact clocks.
    from reconstruction.semantics import load_semantic_evidence

    motion = store.get(ArtifactKey(**headers["reconstruction"]["key"])).metadata
    assert isinstance(motion, Reconstruction)
    assert load_semantic_evidence(
        store.get(ArtifactKey(**headers["semantics"]["key"]))
    )["motion_times"] == [s.global_seconds for s in motion.samples]
    assert Path(first["inspection_bundle"]).is_file()
    before = {
        str(p): hash_file(p) for p in store.root.namespace("derived").rglob("*.json")
    }
    code, second = invoke()
    assert code == 0
    assert all(s["cached"] for s in second["stages"].values())
    assert inference_calls == [1]
    assert all(
        s["status"] == "complete" for s in pipe.status(PROJECT)["stages"].values()
    )
    assert before == {
        str(p): hash_file(p) for p in store.root.namespace("derived").rglob("*.json")
    }

    config["parsing"] = {"features": {"max_gap_seconds": 0.08}}
    config_file.write_text(json.dumps(config))
    code, parsed = invoke("--config", str(config_file), "--rerun", "parsing")
    assert code == 0, parsed
    assert inference_calls == [1]
    for stage in (
        "ingest",
        "sync",
        "calibration",
        "observations",
        "attachment",
        "reconstruction",
        "ground",
    ):
        assert parsed["stages"][stage]["key"] == first["stages"][stage]["key"]
        assert parsed["stages"][stage]["cached"]
    assert (
        parsed["stages"]["parsing"]["artifact_key"]
        != first["stages"]["parsing"]["artifact_key"]
    )

    # No geometry is synthesized when the operator has not supplied calibration.
    config.pop("calibration")
    config_file.write_text(json.dumps(config))
    code, partial = invoke("--config", str(config_file))
    assert code == 1
    assert partial["stages"]["calibration"]["status"] == "unavailable"
    assert partial["stages"]["observations"]["cached"]
    assert partial["inspection_registered"]
    assert partial["stages"]["ground"]["status"] == "unavailable"
    with index.connection(PROJECT) as db:
        assert (
            db.execute(
                "SELECT count(*) FROM products WHERE name='reconstruction'"
            ).fetchone()[0]
            == 0
        )

    # Retained manual offsets without reliable cues preserve unknown timing;
    # they cannot turn missing triangulation evidence into successful 3D.
    monkeypatch.setattr(sync, "extract_cues", original_extract)
    config["calibration"] = {"artifact": key_file.name}
    config_file.write_text(json.dumps(config))
    code, untimed = invoke("--config", str(config_file), "--rerun", "sync")
    assert code == 1
    assert untimed["stages"]["reconstruction"]["status"] == "unavailable"
    assert untimed["stages"]["observations"]["cached"]


def test_missing_vision_is_actionable_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pipeline import Pipeline
    from storage import ArtifactStore, StorageRoot
    from tests.test_media_reader import video

    root = tmp_path / "unprovisioned"
    monkeypatch.setenv("TKD_DATA_ROOT", str(root))
    monkeypatch.delenv("TKD_VISION_PYTHON", raising=False)
    left = video(tmp_path / "left.mkv", [0, 40, 80])
    right = video(tmp_path / "right.mkv", [0, 50, 100])
    Pipeline(ArtifactStore(StorageRoot(root))).register(
        "missing", {"left": left, "right": right}
    )
    monkeypatch.setattr(
        sys, "argv", ["tkd-poomsae", "run", "missing", "--through", "observations"]
    )
    assert main() == 1
    result = json.loads(capsys.readouterr().out)
    observations = result["stages"]["observations"]
    assert observations["status"] == "unavailable"
    assert "models bootstrap" in " ".join(observations["diagnostics"])
    assert "models runtime-bootstrap" in " ".join(observations["diagnostics"])
    assert not (root / "models").exists() or not list((root / "models").rglob("*.pth"))

    # An unavailable descendant from an earlier run does not fail a later
    # successful request for an upstream-only capability.
    monkeypatch.setattr(
        sys, "argv", ["tkd-poomsae", "run", "missing", "--through", "ingest"]
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["stages"]["ingest"]["cached"]


def test_scene_adapter_binds_runner_key_without_mutating_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline import Pipeline, StageOutput
    from storage import ArtifactStore, StorageRoot
    from tests.test_ground_contact import calibration
    from tests.test_pipeline import artifact

    store = ArtifactStore(StorageRoot(tmp_path / "store"))
    cal = calibration()
    cal.cameras = cal.cameras[:2]
    sources = {}
    for camera in cal.cameras:
        source = tmp_path / camera.camera_id
        source.write_text(camera.camera_id)
        camera.source_id = "source:" + hash_file(source)
        sources[camera.camera_id] = source
    Pipeline(store).register("scene", sources)
    pipe = configured_pipeline("scene", store)
    stage_keys = pipe._expected_keys({"sources": sources}, Pipeline._initial_state())
    sync_key = stage_keys["sync"]
    sync = store.get_or_create(sync_key, lambda: (artifact("sync", sync_key), {}))
    import pipeline.offline

    monkeypatch.setattr(
        pipeline.offline, "_scene_calibration", lambda *_args: StageOutput(cal)
    )
    original = cal.model_dump(mode="json")
    output = pipe.stages["calibration"].producer(
        stage_keys["calibration"], {"sync": sync}, {"candidate": "supplied-scene.json"}
    )
    assert isinstance(output, StageOutput)
    handle = store.get_or_create(
        stage_keys["calibration"], lambda: (output.artifact, output.arrays)
    )
    assert (
        handle.metadata.provenance.config_digest
        == stage_keys["calibration"].config_digest
    )
    assert cal.model_dump(mode="json") == original
