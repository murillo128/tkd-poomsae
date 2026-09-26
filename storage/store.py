"""Content-addressed, single-writer artifact storage.

The final directory appears only after its manifest and payloads have been
flushed. Lock files are deliberately retained: flock releases them on process
exit, including abrupt termination, without guessing whether a writer is stale.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from contracts.models import Artifact, ArtifactBase, DenseArray, validate_artifact

Namespace = Literal["datasets", "models", "derived", "runs"]
_NAMESPACES = frozenset({"datasets", "models", "derived", "runs"})
_LAYERS = frozenset(
    {
        "project",
        "source",
        "synchronization",
        "calibration",
        "observation",
        "alignment",
        "reconstruction",
        "morphology",
        "ground",
        "motion_features",
        "segmentation",
        "arm_actions",
        "semantics",
        "manual_edits",
    }
)


class StorageError(Exception):
    """Base class for storage failures."""


class MissingResource(StorageError):
    """An asset or completed artifact is absent; provision it outside the reader."""


class CorruptArtifact(StorageError):
    """A published artifact is incomplete, malformed, or has changed on disk."""


class WriteTimeout(StorageError):
    """Another process held the artifact writer lock past the deadline."""


class Cancelled(StorageError):
    """The caller cancelled while waiting for an artifact writer lock."""


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def hash_file(path: Path) -> str:
    """Hash existing source bytes for an ArtifactKey input (never modify them)."""
    if not path.is_file():
        raise MissingResource(f"Missing input file: {path}")
    return _digest(path)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def hash_config(value: object) -> str:
    """Hash only the effective configuration consumed by one producer."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _relative(path: str) -> Path:
    candidate = Path(path)
    if (
        not path
        or candidate.is_absolute()
        or any(p in {"", ".", ".."} for p in path.split("/"))
    ):
        raise ValueError(f"invalid relative artifact path: {path!r}")
    if "\\" in path or candidate.parts[0] == ".locks":
        raise ValueError(f"invalid relative artifact path: {path!r}")
    return candidate


def _safe_child(root: Path, path: str) -> Path:
    candidate = root / _relative(path)
    if candidate.is_symlink() or any(
        parent.is_symlink()
        for parent in candidate.parents
        if parent != root and root in parent.parents
    ):
        raise CorruptArtifact(f"symbolic link in artifact path: {path}")
    return candidate


@dataclass(frozen=True)
class StorageRoot:
    path: Path

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError("storage root must be absolute")
        object.__setattr__(self, "path", self.path.resolve())

    @classmethod
    def from_env(cls) -> StorageRoot:
        configured = os.environ.get("TKD_DATA_ROOT")
        return cls(
            Path(configured) if configured else Path.home() / ".local/share/tkd-poomsae"
        )

    def namespace(self, name: Namespace) -> Path:
        if name not in _NAMESPACES:
            raise ValueError(f"unknown namespace: {name}")
        return self.path / name

    def existing(
        self, name: Namespace, relative_path: str, sha256: str | None = None
    ) -> Path:
        """Resolve a local asset without fetching or creating it."""
        if sha256 is not None and not _sha256(sha256):
            raise ValueError("expected SHA-256 must be lowercase hex")
        path = _safe_child(self.namespace(name), relative_path)
        if not path.is_file():
            raise MissingResource(
                f"Missing {name}/{relative_path} under {self.path}; "
                "provision this asset first"
            )
        if sha256 is not None and _digest(path) != sha256:
            raise CorruptArtifact(f"SHA-256 mismatch for {name}/{relative_path}")
        return path


@dataclass(frozen=True)
class ArtifactKey:
    """Only revisions that affect this artifact should be included by its producer.

    Input values are content hashes (including upstream artifact identities).
    A dependent layer includes calibration/sync revisions when it uses them.
    Timings, worktree paths, and wall-clock timestamps have no place in this key.
    """

    layer: str
    inputs: Mapping[str, str]
    schema_version: str
    algorithm_revision: str
    config_digest: str
    model_revision: str | None = None
    calibration_revision: str | None = None
    sync_revision: str | None = None

    def __post_init__(self) -> None:
        if self.layer not in _LAYERS:
            raise ValueError("unknown motion layer")
        if not self.schema_version or not self.algorithm_revision:
            raise ValueError("schema and algorithm revisions are required")
        if not _sha256(self.config_digest) or any(
            not k or not _sha256(v) for k, v in self.inputs.items()
        ):
            raise ValueError(
                "configuration and input identities must be SHA-256 hashes"
            )
        for revision in (
            self.model_revision,
            self.calibration_revision,
            self.sync_revision,
        ):
            if revision == "":
                raise ValueError("revisions cannot be empty")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))

    @property
    def digest(self) -> str:
        data = {
            "layer": self.layer,
            "inputs": dict(self.inputs),
            "schema_version": self.schema_version,
            "algorithm_revision": self.algorithm_revision,
            "config_digest": self.config_digest,
            "model_revision": self.model_revision,
            "calibration_revision": self.calibration_revision,
            "sync_revision": self.sync_revision,
        }
        return hashlib.sha256(_json_bytes(data)).hexdigest()


@dataclass(frozen=True)
class ArtifactHandle:
    path: Path
    metadata: Artifact
    files: Mapping[str, str]

    def read_array(
        self, array_id: str, selection: Any = slice(None)
    ) -> np.ndarray[Any, Any]:
        """Return a read-only memory-mapped slice without loading the full recording."""
        if array_id not in self.files:
            raise KeyError(array_id)
        array = np.load(
            _safe_child(self.path, self.files[array_id]),
            mmap_mode="r",
            allow_pickle=False,
        )
        result: np.ndarray[Any, Any] = array[selection]
        result.setflags(write=False)
        return result


class ArtifactStore:
    def __init__(self, root: StorageRoot | None = None) -> None:
        self.root = root or StorageRoot.from_env()

    def _path(self, key: ArtifactKey) -> Path:
        return self.root.namespace("derived") / key.layer / key.digest

    def get(self, key: ArtifactKey) -> ArtifactHandle:
        directory = self._path(key)
        if not directory.exists():
            raise MissingResource(
                f"Missing derived/{key.layer}/{key.digest}; run its producer first"
            )
        if not directory.is_dir() or directory.is_symlink():
            raise CorruptArtifact(f"Invalid artifact directory: {directory}")
        try:
            manifest_path = _safe_child(directory, "manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or set(manifest) != {
                "version",
                "key",
                "metadata",
                "arrays",
            }:
                raise ValueError("invalid manifest fields")
            if manifest["version"] != 1 or manifest["key"] != key.digest:
                raise ValueError("manifest identity mismatch")
            metadata_entry = manifest["metadata"]
            if set(metadata_entry) != {"path", "sha256"} or not _sha256(
                metadata_entry["sha256"]
            ):
                raise ValueError("invalid metadata entry")
            metadata_path = _safe_child(directory, metadata_entry["path"])
            if (
                metadata_path.name != "metadata.json"
                or _digest(metadata_path) != metadata_entry["sha256"]
            ):
                raise ValueError("metadata hash mismatch")
            metadata = validate_artifact(
                json.loads(metadata_path.read_text(encoding="utf-8"))
            )
            if (
                metadata.kind != key.layer
                or metadata.schema_version != key.schema_version
                or metadata.provenance.config_digest != key.config_digest
                or (
                    key.model_revision is not None
                    and metadata.provenance.model_version != key.model_revision
                )
            ):
                raise ValueError("metadata/key mismatch")
            descriptors: list[DenseArray] = getattr(metadata, "arrays", [])
            arrays = manifest["arrays"]
            if not isinstance(arrays, dict) or set(arrays) != {
                item.id for item in descriptors
            }:
                raise ValueError("array inventory mismatch")
            if len(descriptors) != len(arrays):
                raise ValueError("duplicate array IDs")
            files: dict[str, str] = {}
            for item in descriptors:
                entry = arrays[item.id]
                if set(entry) != {"path", "sha256"} or not _sha256(entry["sha256"]):
                    raise ValueError("invalid array entry")
                expected_path = (
                    f"arrays/{hashlib.sha256(item.id.encode()).hexdigest()}.npy"
                )
                if entry["path"] != expected_path:
                    raise ValueError("unexpected array path")
                path = _safe_child(directory, entry["path"])
                if _digest(path) != entry["sha256"]:
                    raise ValueError("array hash mismatch")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if str(array.dtype) != item.dtype or list(array.shape) != item.shape:
                    raise ValueError("array dtype/shape mismatch")
                files[item.id] = entry["path"]
            _validate_masks(descriptors)
            return ArtifactHandle(directory, metadata, files)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise CorruptArtifact(
                f"Incomplete or corrupt artifact at {directory}: {exc}"
            ) from exc

    def get_or_create(
        self,
        key: ArtifactKey,
        producer: Callable[[], tuple[ArtifactBase, Mapping[str, np.ndarray[Any, Any]]]],
        *,
        timeout: float = 30.0,
        cancelled: Callable[[], bool] | None = None,
    ) -> ArtifactHandle:
        """Run producer once per key across processes; retry after a dead writer."""
        if timeout < 0:
            raise ValueError("timeout must be nonnegative")
        directory = self._path(key)
        lock_dir = self.root.namespace("derived") / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{key.digest}.lock"
        deadline = time.monotonic() + timeout
        with lock_path.open("a+b") as lock:
            while True:
                if cancelled is not None and cancelled():
                    raise Cancelled(f"Cancelled waiting for {key.digest}")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise WriteTimeout(
                            f"Timed out waiting for {key.digest}"
                        ) from None
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            try:
                if directory.exists():
                    return self.get(key)
                metadata, arrays = producer()
                self._publish(directory, key, metadata, arrays)
                return self.get(key)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _publish(
        self,
        directory: Path,
        key: ArtifactKey,
        metadata: ArtifactBase,
        arrays: Mapping[str, np.ndarray[Any, Any]],
    ) -> None:
        validated = validate_artifact(metadata.model_dump(mode="json"))
        if (
            validated.kind != key.layer
            or validated.schema_version != key.schema_version
            or validated.provenance.config_digest != key.config_digest
            or (
                key.model_revision is not None
                and validated.provenance.model_version != key.model_revision
            )
        ):
            raise ValueError("metadata does not match artifact key")
        descriptors: list[DenseArray] = getattr(validated, "arrays", [])
        if len({item.id for item in descriptors}) != len(descriptors):
            raise ValueError("duplicate array IDs")
        if set(arrays) != {item.id for item in descriptors}:
            raise ValueError("arrays do not match metadata descriptors")
        _validate_masks(descriptors)
        directory.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{key.digest}.", dir=directory.parent))
        try:
            (stage / "arrays").mkdir()
            metadata_path = stage / "metadata.json"
            _write(metadata_path, _json_bytes(validated.model_dump(mode="json")))
            entries: dict[str, dict[str, str]] = {}
            for descriptor in descriptors:
                array = arrays[descriptor.id]
                if (
                    str(array.dtype) != descriptor.dtype
                    or list(array.shape) != descriptor.shape
                ):
                    raise ValueError(f"dtype/shape mismatch: {descriptor.id}")
                if array.dtype.hasobject:
                    raise ValueError("object arrays are forbidden")
                relative = (
                    f"arrays/{hashlib.sha256(descriptor.id.encode()).hexdigest()}.npy"
                )
                path = stage / relative
                with path.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                entries[descriptor.id] = {"path": relative, "sha256": _digest(path)}
            manifest = {
                "version": 1,
                "key": key.digest,
                "metadata": {"path": "metadata.json", "sha256": _digest(metadata_path)},
                "arrays": entries,
            }
            _write(stage / "manifest.json", _json_bytes(manifest))
            _fsync_dir(stage / "arrays")
            _fsync_dir(stage)
            stage.rename(directory)
            _fsync_dir(directory.parent)
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def _validate_masks(descriptors: list[DenseArray]) -> None:
    by_id = {item.id: item for item in descriptors}
    for item in descriptors:
        if item.missing_mask_id is None:
            continue
        mask = by_id.get(item.missing_mask_id)
        if (
            mask is None
            or mask is item
            or mask.dtype != "bool"
            or mask.shape != item.shape
            or mask.axes != item.axes
        ):
            raise ValueError(f"invalid missing mask for {item.id}")


def _write(path: Path, data: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
