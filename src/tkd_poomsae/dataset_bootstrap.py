"""Explicit, locked acquisition of the pinned Mendeley archive."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from storage import StorageRoot
from tkd_poomsae.dataset import load_registration

_DOWNLOAD = (
    "https://data.mendeley.com/public-files/datasets/bjy7vr4xkt/files/"
    "5a5bd049-cee1-4219-823b-b91305fdd63e/file_downloaded"
)
_RECEIPT = "acquisition.json"
_CHUNK = 1024 * 1024
_RETRIES = 3
_TIMEOUT = 30
_USER_AGENT = "tkd-poomsae/0.1 (public dataset acquisition)"


class AcquisitionError(RuntimeError):
    """Acquisition or local verification cannot establish the pinned bytes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _paths(root: StorageRoot) -> tuple[Path, Path, Path]:
    datasets = root.namespace("datasets")
    parent = datasets / "mendeley" / "bjy7vr4xkt"
    if any(path.is_symlink() for path in (datasets, parent.parent, parent)):
        raise AcquisitionError("dataset root contains a symbolic link")
    return parent, parent / "v1", parent / ".v1-acquiring"


def verify(root: StorageRoot | None = None) -> dict[str, Any]:
    """Verify the receipt and every published file without any network access."""
    storage = root or StorageRoot.from_env()
    registration = load_registration()
    _, destination, _ = _paths(storage)
    if destination.is_symlink():
        raise AcquisitionError("dataset publication cannot be a symbolic link")
    receipt_path = destination / _RECEIPT
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise AcquisitionError(
            "dataset has no completion receipt; explicit repair needed"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        archive = registration["inventory"]["archive"]
        if (
            receipt["schema_version"] != 1
            or receipt["dataset_id"] != registration["source"]["dataset_id"]
            or receipt["version"] != registration["source"]["version"]
            or receipt["archive"]["file_id"] != archive["file_id"]
            or receipt["archive"]["sha256"] != archive["sha256"]
            or receipt["archive"]["size_bytes"] != archive["size_bytes"]
        ):
            raise AcquisitionError("dataset receipt version or source mismatch")
        files = receipt["files"]
        members = registration["inventory"]["members"]
        if len(files) != len(members) or {item["path"] for item in files} != {
            item["path"] for item in members
        }:
            raise AcquisitionError("dataset receipt inventory mismatch")
        by_path = {item["path"]: item for item in files}
        total = 0
        for member in members:
            item = by_path[member["path"]]
            path = destination / member["path"]
            if (
                not path.is_file()
                or path.is_symlink()
                or any(
                    parent.is_symlink()
                    for parent in path.parents
                    if destination in parent.parents
                )
                or item["size_bytes"] != member["size_bytes"]
                or not isinstance(item["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
                or path.stat().st_size != item["size_bytes"]
                or _sha256(path) != item["sha256"]
            ):
                raise AcquisitionError(
                    f"dataset file differs from receipt: {member['path']}"
                )
            total += item["size_bytes"]
        if receipt["total_bytes"] != total:
            raise AcquisitionError("dataset receipt byte count mismatch")
        return {
            "state": "verified",
            "files": len(files),
            "bytes": total,
            "archive_sha256": archive["sha256"],
            "receipt_sha256": _sha256(receipt_path),
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, AcquisitionError):
            raise
        raise AcquisitionError("dataset receipt or files are invalid") from exc


def _head(url: str) -> tuple[int, str | None]:
    with urllib.request.urlopen(
        urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT}),
        timeout=_TIMEOUT,
    ) as response:
        length = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag")
        return length, etag if etag and not etag.startswith("W/") else None


def _download(url: str, stage: Path, expected_size: int) -> Path:
    archive = stage / "Data.zip.part"
    sidecar = stage / "transfer.json"
    for attempt in range(_RETRIES):
        try:
            size, etag = _head(url)
            if size != expected_size:
                raise AcquisitionError(
                    "publisher archive size differs from registration"
                )
            try:
                previous = json.loads(sidecar.read_text()) if sidecar.exists() else None
            except (OSError, ValueError):
                previous = None
            offset = archive.stat().st_size if archive.exists() else 0
            if offset > expected_size or not (
                previous == {"url": url, "size_bytes": size, "etag": etag} and etag
            ):
                offset = 0
                archive.unlink(missing_ok=True)
            _write_json(sidecar, {"url": url, "size_bytes": size, "etag": etag})
            if offset == size:
                return archive
            headers: dict[str, str] = {"User-Agent": _USER_AGENT}
            if offset and etag is not None:
                headers.update({"Range": f"bytes={offset}-", "If-Range": etag})
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
                if offset:
                    expected_range = f"bytes {offset}-{size - 1}/{size}"
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected_range
                    ):
                        raise AcquisitionError(
                            "publisher did not honor validated resume range"
                        )
                    if response.headers.get("ETag") != etag:
                        raise AcquisitionError(
                            "publisher changed during resumed transfer"
                        )
                elif response.status != 200:
                    raise AcquisitionError(
                        "publisher did not return the complete archive"
                    )
                with archive.open("ab" if offset else "wb") as stream:
                    transferred = offset
                    last_report = time.monotonic()
                    while block := response.read(_CHUNK):
                        transferred += len(block)
                        if transferred > size:
                            raise AcquisitionError(
                                "publisher transferred more than registered"
                            )
                        stream.write(block)
                        if time.monotonic() - last_report >= 5:
                            print(f"downloaded {transferred}/{size} bytes", flush=True)
                            last_report = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
            if archive.stat().st_size != size:
                raise OSError("download ended before registered archive size")
            return archive
        except (OSError, urllib.error.URLError) as exc:
            if attempt == _RETRIES - 1:
                raise AcquisitionError(
                    f"publisher transfer failed after {_RETRIES} attempts: {exc}"
                ) from exc
            time.sleep(min(2**attempt, 4))
    raise AssertionError("unreachable")


def _extract(
    archive: Path, stage: Path, registration: dict[str, Any]
) -> dict[str, Any]:
    target = stage / "extracted"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir()
    members = {item["path"]: item for item in registration["inventory"]["members"]}
    found: set[str] = set()
    output: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            name = entry.filename
            path = name.rstrip("/")
            if (
                not path
                or path.startswith("/")
                or "\\" in path
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or (entry.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise AcquisitionError(f"unsafe archive entry: {name}")
            if entry.is_dir():
                if not any(item.startswith(path + "/") for item in members):
                    raise AcquisitionError(f"unregistered archive directory: {name}")
                continue
            expected = members.get(path)
            if (
                expected is None
                or path in found
                or entry.file_size != expected["size_bytes"]
                or entry.compress_size != expected["compressed_size_bytes"]
                or f"{entry.CRC:08x}" != expected["crc32"]
                or entry.flag_bits & 1
                or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            ):
                raise AcquisitionError(
                    f"archive entry differs from registration: {name}"
                )
            found.add(path)
            destination = target / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with bundle.open(entry) as source, destination.open("xb") as sink:
                while block := source.read(_CHUNK):
                    count += len(block)
                    if count > expected["size_bytes"]:
                        raise AcquisitionError(
                            f"archive expansion limit exceeded: {name}"
                        )
                    sink.write(block)
                    digest.update(block)
                sink.flush()
                os.fsync(sink.fileno())
            if count != expected["size_bytes"]:
                raise AcquisitionError(f"archive entry truncated: {name}")
            output.append(
                {"path": path, "size_bytes": count, "sha256": digest.hexdigest()}
            )
    if found != set(members):
        raise AcquisitionError("archive lacks registered members")
    output.sort(key=lambda item: item["path"])
    return {
        "schema_version": 1,
        "dataset_id": registration["source"]["dataset_id"],
        "version": registration["source"]["version"],
        "archive": {
            "file_id": registration["inventory"]["archive"]["file_id"],
            "size_bytes": archive.stat().st_size,
            "sha256": _sha256(archive),
        },
        "files": output,
        "total_bytes": sum(item["size_bytes"] for item in output),
    }


def bootstrap(
    root: StorageRoot | None = None, *, repair: bool = False, url: str = _DOWNLOAD
) -> dict[str, Any]:
    """Acquire once; a verified hit makes no HTTP requests and transfers no bytes."""
    storage = root or StorageRoot.from_env()
    registration = load_registration()
    parent, destination, stage = _paths(storage)
    with _lock(parent / ".v1-bootstrap.lock"):
        if destination.exists():
            try:
                result = verify(storage)
            except AcquisitionError:
                if not repair:
                    raise AcquisitionError(
                        "local dataset is corrupt; rerun bootstrap with --repair"
                    ) from None
                backup = parent / f"v1-replaced-{uuid.uuid4().hex}"
                destination.rename(backup)
            else:
                return {**result, "downloaded_bytes": 0, "cache_hit": True}
        if stage.is_symlink() or (stage.exists() and not stage.is_dir()):
            raise AcquisitionError("acquisition staging path is unsafe")
        stage.mkdir(exist_ok=True)
        archive_meta = registration["inventory"]["archive"]
        archive_size = archive_meta["size_bytes"]
        expanded_size = sum(
            item["size_bytes"] for item in registration["inventory"]["members"]
        )
        partial = stage / "Data.zip.part"
        if partial.is_symlink():
            raise AcquisitionError("acquisition partial archive is a symbolic link")
        if shutil.disk_usage(parent).free < archive_size + expanded_size + 64 * _CHUNK:
            raise AcquisitionError("insufficient disk space for archive and extraction")
        archive = _download(url, stage, archive_size)
        if _sha256(archive) != archive_meta["sha256"]:
            archive.unlink()
            (stage / "transfer.json").unlink(missing_ok=True)
            raise AcquisitionError(
                "downloaded archive SHA-256 differs from pinned registration"
            )
        receipt = _extract(archive, stage, registration)
        extracted = stage / "extracted"
        _write_json(extracted / _RECEIPT, receipt)
        _sync_dir(extracted)
        extracted.rename(destination)
        _sync_dir(parent)
        archive.unlink(missing_ok=True)
        (stage / "transfer.json").unlink(missing_ok=True)
        stage.rmdir()
        result = verify(storage)
        return {**result, "downloaded_bytes": archive_size, "cache_hit": False}
