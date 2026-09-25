"""Read-only access to registered local recordings and bounded frame previews."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any, BinaryIO

import cv2

from media import MediaReader, Recording, index_recording
from pipeline import Pipeline

_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}
_RANGE = re.compile(r"bytes=(\d*)-(\d*)\Z")


class MediaAccessError(ValueError):
    """A requested resource is absent, forbidden or unsupported."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        super().__init__(detail)


@dataclass(frozen=True)
class RegisteredMedia:
    path: Path
    content_type: str
    recording: Recording
    device: int
    inode: int

    @property
    def browser_playback(self) -> bool:
        return (
            self.content_type == "video/mp4" and self.recording.codec == "h264"
        ) or (
            self.content_type == "video/webm"
            and self.recording.codec in {"vp8", "vp9", "av1"}
        )


def byte_range(value: str, size: int) -> tuple[int, int] | None:
    """Parse one RFC 9110 byte range; None means malformed or unsatisfiable."""
    match = _RANGE.fullmatch(value)
    if (
        match is None
        or not (match[1] or match[2])
        or size == 0
        or len(match[1]) > 20
        or len(match[2]) > 20
    ):
        return None
    if match[1]:
        start = int(match[1])
        end = min(int(match[2]), size - 1) if match[2] else size - 1
        return (start, end) if start < size and end >= start else None
    suffix = int(match[2])
    return (max(0, size - suffix), size - 1) if suffix > 0 else None


def frame_info(recording: Recording, ordinal: int) -> dict[str, Any]:
    ref = recording.frames[ordinal]
    return {
        "ordinal": ref.ordinal,
        "pts": ref.pts,
        "time_base_num": ref.time_base_num,
        "time_base_den": ref.time_base_den,
        "source_seconds": ref.seconds,
        "keyframe": ref.keyframe,
        "source_id": recording.source_id,
        "source_sha256": recording.sha256,
        "camera_id": recording.camera_id,
        "rotation_degrees": recording.rotation_degrees,
        "stored_width_px": recording.stored_width_px,
        "stored_height_px": recording.stored_height_px,
        "oriented_width_px": recording.source.width_px,
        "oriented_height_px": recording.source.height_px,
        "stored_to_oriented": recording.stored_to_oriented,
        "oriented_to_stored": recording.oriented_to_stored,
    }


class MediaAccess:
    """Bounded per-service indexes/previews, backed by shared project registration."""

    def __init__(self, pipe: Pipeline, roots: dict[str, Path]) -> None:
        self.pipe = pipe
        self.roots = tuple(
            {*roots.values(), pipe.store.root.namespace("datasets").resolve()}
        )
        self._lock = Lock()
        self._indexes: OrderedDict[
            tuple[str, str, str, int, int, int, int, int], Recording
        ] = OrderedDict()
        self._previews: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self.streams = BoundedSemaphore(4)
        self.decoders = BoundedSemaphore(2)
        self._preview_dir = pipe.store.root.namespace("derived") / "media-previews-v1"

    def _remember(self, key: tuple[str, int], content: bytes) -> None:
        with self._lock:
            if len(content) <= 64 * 1024 * 1024:
                self._previews[key] = content
                while (
                    len(self._previews) > 16
                    or sum(map(len, self._previews.values())) > 64 * 1024 * 1024
                ):
                    self._previews.popitem(last=False)

    def _disk_preview(
        self, key: tuple[str, int], content: bytes | None = None
    ) -> bytes | None:
        """Read or atomically publish a source-addressed preview under a shared cap."""
        directory = self._preview_dir
        if (
            directory.parent.is_symlink()
            or directory.is_symlink()
            or (directory / ".lock").is_symlink()
        ):
            raise MediaAccessError(409, "derived cache path is not safe")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key[0]}-{key[1]}.png"
        if path.is_symlink():
            raise MediaAccessError(409, "derived preview path is not safe")
        with (directory / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.is_file() and 0 < path.stat().st_size <= 64 * 1024 * 1024:
                cached = path.read_bytes()
                if cached.startswith(b"\x89PNG\r\n\x1a\n"):
                    os.utime(path, None)
                    return cached
            if content is None or len(content) > 64 * 1024 * 1024:
                return None
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stage:
                stage.write(content)
                stage.flush()
                os.fsync(stage.fileno())
                staged = Path(stage.name)
            os.replace(staged, path)
            entries = sorted(
                directory.glob("*.png"), key=lambda item: item.stat().st_mtime_ns
            )
            total = sum(item.stat().st_size for item in entries)
            while len(entries) > 16 or total > 64 * 1024 * 1024:
                oldest = entries.pop(0)
                total -= oldest.stat().st_size
                oldest.unlink()
            return content

    def open_stream(self, item: RegisteredMedia) -> BinaryIO:
        """Open the already resolved source without following a swapped leaf link."""
        try:
            fd = os.open(item.path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            raise MediaAccessError(409, "registered source cannot be opened") from exc
        stream = os.fdopen(fd, "rb")
        try:
            actual = Path(os.readlink(f"/proc/self/fd/{fd}")).resolve(strict=True)
            current = os.fstat(fd)
            if not stat.S_ISREG(current.st_mode) or not any(
                actual.is_relative_to(root) for root in self.roots
            ):
                raise MediaAccessError(403, "source outside permitted roots")
            if (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            ) != (
                item.device,
                item.inode,
                item.recording.size_bytes,
                item.recording.modified_ns,
            ):
                raise MediaAccessError(409, "source changed since indexing")
        except (OSError, ValueError) as exc:
            stream.close()
            if isinstance(exc, MediaAccessError):
                raise
            raise MediaAccessError(409, "registered source changed") from exc
        return stream

    def resolve(self, project: str, camera: str) -> RegisteredMedia:
        try:
            project_file, _ = self.pipe._files(project)
            if project_file.is_symlink():
                raise MediaAccessError(404, "unknown project")
            sources = json.loads(project_file.read_text(encoding="utf-8"))["sources"]
            if not isinstance(sources, dict) or camera not in sources:
                raise MediaAccessError(404, "unknown camera")
            registered = sources[camera]
            if not isinstance(registered, str) or not Path(registered).is_absolute():
                raise MediaAccessError(403, "invalid registered source")
            path = Path(registered).resolve(strict=True)
            if not path.is_file() or not any(
                path.is_relative_to(root) for root in self.roots
            ):
                raise MediaAccessError(403, "source outside permitted roots")
            content_type = _MEDIA_TYPES.get(path.suffix.lower())
            if content_type is None:
                raise MediaAccessError(
                    415, "source media type is not browser supported"
                )
            stat = path.stat()
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            if isinstance(exc, MediaAccessError):
                raise
            raise MediaAccessError(404, "registered source unavailable") from exc
        key = (
            project,
            camera,
            str(path),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        with self._lock:
            recording = self._indexes.get(key)
            if recording is None:
                recording = index_recording(camera, path)
                try:
                    after = path.stat()
                except OSError as exc:
                    raise MediaAccessError(
                        409, "source vanished while indexing"
                    ) from exc
                if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                ):
                    raise MediaAccessError(409, "source changed while indexing")
                self._indexes[key] = recording
                while len(self._indexes) > 8:
                    self._indexes.popitem(last=False)
            else:
                self._indexes.move_to_end(key)
        return RegisteredMedia(path, content_type, recording, stat.st_dev, stat.st_ino)

    def preview(self, recording: Recording, ordinal: int) -> bytes:
        key = (recording.sha256, ordinal)
        with self._lock:
            cached = self._previews.get(key)
            if cached is not None:
                self._previews.move_to_end(key)
                return cached
        cached = self._disk_preview(key)
        if cached is not None:
            self._remember(key, cached)
            return cached
        if recording.source.width_px * recording.source.height_px > 16_000_000:
            raise MediaAccessError(413, "decoded frame exceeds preview pixel limit")
        if not self.decoders.acquire(blocking=False):
            raise MediaAccessError(503, "frame decode capacity reached")
        try:
            rgb = (
                MediaReader(recording, max_decode_frames=1)
                .frame(recording.frames[ordinal])
                .rgb
            )
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            self.decoders.release()
        if not ok:
            raise MediaAccessError(500, "frame encoding failed")
        result = encoded.tobytes()
        self._disk_preview(key, result)
        self._remember(key, result)
        return result


def last_modified(modified_ns: int) -> str:
    return format_datetime(datetime.fromtimestamp(modified_ns / 1e9, UTC), usegmt=True)
