"""Pinned model assets shared across worktrees; readers never download."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from storage.store import StorageRoot


class ModelAssetError(RuntimeError):
    """A pinned model asset is absent, altered, or cannot be provisioned."""


def registry() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("registry.json").read_text())  # type: ignore[no-any-return]


def models_root() -> Path:
    return StorageRoot.from_env().namespace("models")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target(root: Path, relative: str) -> Path:
    parts = Path(relative).parts
    if (
        not parts
        or any(part in ("", ".", "..") for part in parts)
        or Path(relative).is_absolute()
    ):
        raise ModelAssetError(f"Invalid registry path: {relative}")
    path = root.joinpath(*parts)
    if root.is_symlink() or any(
        parent.is_symlink()
        for parent in [path, *path.parents]
        if parent != root and root in parent.parents
    ):
        raise ModelAssetError(f"Symlink in model path: {path}")
    return path


def verified_paths() -> dict[str, Path]:
    """Return all local paths only after verifying every pinned byte sequence."""
    root = models_root()
    paths: dict[str, Path] = {}
    for asset in registry()["assets"]:
        target = _target(root, asset["path"])
        if not target.is_file():
            raise ModelAssetError(
                f"Missing model asset {asset['path']}; "
                "run `tkd-poomsae models bootstrap`."
            )
        if _digest(target) != asset["sha256"]:
            raise ModelAssetError(
                f"Model asset changed: {target}; inspect it before re-bootstrap."
            )
        paths[asset["path"]] = target
    return paths


def bootstrap(*, opener: Callable[[str], Any] | None = None) -> list[str]:
    """Download missing assets with a process lock and atomic publication."""
    root = models_root()
    if root.is_symlink():
        raise ModelAssetError(f"Symlink in model root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    opener = opener or (lambda url: urllib.request.urlopen(url, timeout=60))
    lock_path = root / ".bootstrap.lock"
    if lock_path.is_symlink():
        raise ModelAssetError(f"Symlink in model lock: {lock_path}")
    installed: list[str] = []
    with lock_path.open("a+b") as lock:
        deadline = time.monotonic() + 300
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ModelAssetError(
                        "Timed out waiting for model bootstrap lock"
                    ) from None
                time.sleep(0.1)
        try:
            for asset in registry()["assets"]:
                target = _target(root, asset["path"])
                if target.exists():
                    if not target.is_file() or _digest(target) != asset["sha256"]:
                        raise ModelAssetError(f"Existing model asset changed: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                staged: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=target.parent, prefix=".download-", delete=False
                    ) as output:
                        staged = Path(output.name)
                        digest = hashlib.sha256()
                        with opener(asset["url"]) as response:
                            while chunk := response.read(1024 * 1024):
                                output.write(chunk)
                                digest.update(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    if digest.hexdigest() != asset["sha256"]:
                        raise ModelAssetError(
                            f"Downloaded model hash mismatch: {asset['path']}"
                        )
                    os.replace(staged, target)
                    installed.append(asset["path"])
                finally:
                    if staged is not None:
                        staged.unlink(missing_ok=True)
            descriptor = os.open(root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return installed
