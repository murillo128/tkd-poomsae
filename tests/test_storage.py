"""Storage behavior with synthetic data only; no network or model assets."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import time
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import storage.store as store_module
from contracts.models import Observation
from storage import (
    ArtifactKey,
    ArtifactStore,
    Cancelled,
    CorruptArtifact,
    MissingResource,
    StorageRoot,
    WriteTimeout,
    hash_config,
    hash_file,
)

DIGEST = "a" * 64


def key(**changes: Any) -> ArtifactKey:
    values: dict[str, Any] = {
        "layer": "observation",
        "inputs": {"video": "b" * 64},
        "schema_version": "1.0.0",
        "algorithm_revision": "detector-1",
        "config_digest": DIGEST,
        "model_revision": "weights-1",
        "sync_revision": "offset-1",
    }
    values.update(changes)
    return ArtifactKey(**values)


def observation() -> Observation:
    return Observation.model_validate(
        {
            "kind": "observation",
            "id": "obs-1",
            "schema_version": "1.0.0",
            "provenance": {
                "producer": "synthetic",
                "model": "test",
                "model_version": "weights-1",
                "config_digest": DIGEST,
            },
            "frame": {
                "source_id": "src",
                "camera_id": "cam",
                "frame_index": 1,
                "source_seconds": 1.0,
                "offset_seconds": 0.1,
                "global_seconds": 1.1,
            },
            "landmarks": [],
            "arrays": [
                {
                    "id": "points",
                    "dtype": "float32",
                    "shape": [10000, 2],
                    "axes": ["time", "xy"],
                    "missing_mask_id": "unknown",
                },
                {
                    "id": "unknown",
                    "dtype": "bool",
                    "shape": [10000, 2],
                    "axes": ["time", "xy"],
                },
            ],
        }
    )


def produce() -> tuple[Observation, dict[str, np.ndarray[Any, Any]]]:
    return observation(), {
        "points": np.arange(20000, dtype=np.float32).reshape(10000, 2),
        "unknown": np.zeros((10000, 2), dtype=np.bool_),
    }


def worker(root: str, cwd: str, counter: str, result: mp.Queue[str]) -> None:
    os.chdir(cwd)
    os.environ["TKD_DATA_ROOT"] = root

    def counted() -> tuple[Observation, dict[str, np.ndarray[Any, Any]]]:
        with open(counter, "a", encoding="utf-8") as stream:
            stream.write("1\n")
        time.sleep(0.2)
        return produce()

    handle = ArtifactStore().get_or_create(key(), counted)
    result.put(str(handle.read_array("points", slice(2, 3))[0, 0]))


def die_while_writing(root: str) -> None:
    store = ArtifactStore(StorageRoot(Path(root)))

    original_write = store_module._write

    def abort_on_manifest(path: Path, data: bytes) -> None:
        if path.name == "manifest.json":
            os._exit(7)
        original_write(path, data)

    store_module._write = abort_on_manifest
    store.get_or_create(key(), produce)


def hold_lock(root: str, ready: Event) -> None:
    store = ArtifactStore(StorageRoot(Path(root)))

    def slow() -> tuple[Observation, dict[str, np.ndarray[Any, Any]]]:
        ready.set()
        time.sleep(1)
        return produce()

    store.get_or_create(key(), slow)


def test_two_processes_reuse_shared_root_from_distinct_cwds(tmp_path: Path) -> None:
    root, first, second = (
        tmp_path / name for name in ("shared", "checkout-a", "checkout-b")
    )
    first.mkdir()
    second.mkdir()
    counter = tmp_path / "calls"
    ctx = mp.get_context("fork")
    result: mp.Queue[str] = ctx.Queue()
    workers = [
        ctx.Process(target=worker, args=(str(root), str(cwd), str(counter), result))
        for cwd in (first, second)
    ]
    for process in workers:
        process.start()
    for process in workers:
        process.join(10)
        assert process.exitcode == 0
    assert [result.get(timeout=1) for _ in workers] == ["4.0", "4.0"]
    assert counter.read_text(encoding="utf-8") == "1\n"
    assert not list(first.iterdir()) and not list(second.iterdir())
    handle = ArtifactStore(StorageRoot(root)).get(key())
    assert handle.read_array("points", slice(100, 101)).shape == (1, 2)
    assert not handle.read_array("points").flags.writeable


def test_interruption_releases_lock_and_retry_publishes(tmp_path: Path) -> None:
    ctx = mp.get_context("fork")
    process = ctx.Process(target=die_while_writing, args=(str(tmp_path),))
    process.start()
    process.join(10)
    assert process.exitcode == 7
    store = ArtifactStore(StorageRoot(tmp_path))
    with pytest.raises(MissingResource):
        store.get(key())
    assert len(list((tmp_path / "derived/observation").glob(".*"))) == 1
    assert store.get_or_create(key(), produce).metadata.id == "obs-1"


def test_wait_is_bounded_and_cancellable_without_harming_writer(tmp_path: Path) -> None:
    ctx = mp.get_context("fork")
    ready = ctx.Event()
    process = ctx.Process(target=hold_lock, args=(str(tmp_path), ready))
    process.start()
    assert ready.wait(5)
    store = ArtifactStore(StorageRoot(tmp_path))
    with pytest.raises(WriteTimeout):
        store.get_or_create(key(), produce, timeout=0.02)
    with pytest.raises(Cancelled):
        store.get_or_create(key(), produce, cancelled=lambda: True)
    process.join(10)
    assert process.exitcode == 0
    assert store.get(key()).metadata.id == "obs-1"


def test_tampering_is_not_a_cache_hit(tmp_path: Path) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    handle = store.get_or_create(key(), produce)
    array_path = handle.path / handle.files["points"]
    with array_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(CorruptArtifact, match="hash mismatch"):
        store.get(key())
    with pytest.raises(CorruptArtifact):
        store.get_or_create(key(), produce)


def test_key_changes_only_for_relevant_inputs_and_revisions() -> None:
    original = key()
    assert original.digest == key(inputs={"video": "b" * 64}).digest
    assert original.digest != key(config_digest="c" * 64).digest
    assert original.digest != key(inputs={"video": "d" * 64}).digest
    assert original.digest != key(sync_revision="offset-2").digest
    assert original.digest != key(calibration_revision="camera-2").digest
    assert original.digest != key(model_revision="weights-2").digest


def test_parser_rerun_reuses_upstream_layers() -> None:
    observation_key = key()
    reconstruction_key = ArtifactKey(
        layer="reconstruction",
        inputs={"observation": observation_key.digest},
        schema_version="1.0.0",
        algorithm_revision="triangulation-1",
        config_digest=hash_config({"solver": "linear"}),
        calibration_revision="camera-1",
        sync_revision="offset-1",
    )

    def parser(threshold: float) -> ArtifactKey:
        return ArtifactKey(
            layer="semantics",
            inputs={"reconstruction": reconstruction_key.digest},
            schema_version="1.0.0",
            algorithm_revision="parser-1",
            config_digest=hash_config({"threshold": threshold}),
        )

    first = parser(0.3)
    second = parser(0.4)
    assert first.digest != second.digest
    assert observation_key.digest == key().digest
    assert (
        reconstruction_key.digest
        == ArtifactKey(
            layer="reconstruction",
            inputs={"observation": observation_key.digest},
            schema_version="1.0.0",
            algorithm_revision="triangulation-1",
            config_digest=hash_config({"solver": "linear"}),
            calibration_revision="camera-1",
            sync_revision="offset-1",
        ).digest
    )


def test_readers_validate_paths_and_never_create_or_fetch(tmp_path: Path) -> None:
    root = StorageRoot(tmp_path / "shared")
    with pytest.raises(MissingResource, match="provision"):
        root.existing("datasets", "camera/video.mp4")
    assert not root.path.exists()
    for bad in ("../escape", "/absolute", "folder/../escape", "a\\b"):
        with pytest.raises(ValueError):
            root.existing("models", bad)
    with pytest.raises(ValueError):
        root.existing("models", "weights.onnx", sha256="invalid")
    source = root.namespace("datasets") / "camera/video.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    assert root.existing("datasets", "camera/video.mp4") == source
    with pytest.raises(CorruptArtifact):
        root.existing("datasets", "camera/video.mp4", sha256="0" * 64)
    assert source.read_bytes() == b"original"
    assert hash_file(source) == hashlib.sha256(b"original").hexdigest()
    assert hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1})


def test_rejects_object_arrays_and_bad_masks(tmp_path: Path) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    metadata, arrays = produce()
    arrays["points"] = np.array([[object()] * 2] * 10000, dtype=object)
    with pytest.raises(ValueError, match="dtype/shape mismatch"):
        store.get_or_create(key(), lambda: (metadata, arrays))
    assert not store._path(key()).exists()
    metadata.arrays[0].missing_mask_id = "points"
    with pytest.raises(ValueError, match="missing mask"):
        store.get_or_create(key(), lambda: (metadata, produce()[1]))
