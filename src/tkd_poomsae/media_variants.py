"""Single-writer, content-addressed media transformations for test inputs."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
from av.video.stream import VideoStream

from media import MediaReader, index_recording
from storage import (
    CorruptArtifact,
    MissingResource,
    StorageRoot,
    hash_config,
    hash_file,
)
from tkd_poomsae.selections import Window


@dataclass(frozen=True)
class VariantRecipe:
    """Transform an original PTS window without claiming new temporal evidence."""

    version: int = 1
    codec: str = "mpeg4"
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    start_offset_seconds: float = 0.0
    pre_roll_seconds: float = 0.0
    post_roll_seconds: float = 0.0
    drop_source_ordinals: tuple[int, ...] = ()
    corrupt_tail_bytes: int = 0

    def validate(self) -> None:
        if self.version != 1 or self.codec not in {"mpeg4", "libx264"}:
            raise ValueError("unsupported media variant recipe")
        if self.fps is not None and (self.fps < 1 or self.fps > 30):
            raise ValueError("variant FPS must be between 1 and source maximum 30")
        if (self.width is None) != (self.height is None):
            raise ValueError("variant width and height must be supplied together")
        if self.width is not None and (
            self.width < 2
            or self.height is None
            or self.height < 2
            or self.width % 2
            or self.height % 2
        ):
            raise ValueError("variant dimensions must be positive even numbers")
        if any(
            value < 0
            for value in (
                self.start_offset_seconds,
                self.pre_roll_seconds,
                self.post_roll_seconds,
            )
        ):
            raise ValueError("offset and roll values must be nonnegative")
        if self.corrupt_tail_bytes < 0 or any(x < 0 for x in self.drop_source_ordinals):
            raise ValueError("drop ordinals and corrupt byte count must be nonnegative")


@dataclass(frozen=True)
class Variant:
    path: Path
    manifest: dict[str, Any]
    cache_hit: bool


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validated(
    directory: Path, digest: str, identity: dict[str, Any]
) -> dict[str, Any] | None:
    if not directory.exists():
        return None
    manifest_path = directory / "manifest.json"
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        if (
            manifest["key"] != digest
            or manifest["identity"] != identity
            or hash_file(directory / "clip.mkv") != manifest["output_sha256"]
        ):
            raise CorruptArtifact(f"changed media variant: {directory}")
    except (FileNotFoundError, MissingResource, KeyError, json.JSONDecodeError) as exc:
        raise CorruptArtifact(f"incomplete media variant: {directory}") from exc
    return manifest


def _encode(
    path: Path, window: Window, recipe: VariantRecipe
) -> tuple[list[dict[str, Any]], bool]:
    recording = index_recording(window.camera_id, window.source_path)
    first = max(
        recording.frames[0].seconds, window.start_seconds - recipe.pre_roll_seconds
    )
    last = min(
        recording.frames[-1].seconds, window.end_seconds + recipe.post_roll_seconds
    )
    refs = [ref for ref in recording.frames if first <= ref.seconds <= last]
    if recipe.fps is not None:
        native_gaps = [b.seconds - a.seconds for a, b in zip(refs, refs[1:])]
        if (
            native_gaps
            and recipe.fps > 1 / sorted(native_gaps)[len(native_gaps) // 2] + 0.01
        ):
            raise ValueError("upsampling cannot create higher-frequency evidence")
        interval = 1 / recipe.fps
        chosen = []
        next_time = first
        for ref in refs:
            if ref.seconds + 1e-9 >= next_time:
                chosen.append(ref)
                next_time += interval
        refs = chosen
    refs = [ref for ref in refs if ref.ordinal not in recipe.drop_source_ordinals]
    if not refs:
        raise ValueError("variant recipe selected no source frames")
    width = recipe.width or recording.source.width_px
    height = recipe.height or recording.source.height_px
    mapping: list[dict[str, Any]] = []
    reader = MediaReader(recording, max_decode_frames=len(recording.frames))
    wanted = {ref.ordinal for ref in refs}
    with av.open(str(path), "w") as output:
        stream = output.add_stream(recipe.codec, rate=recipe.fps or 30)
        assert isinstance(stream, VideoStream)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for decoded in reader.decode_window(
            refs[0].ordinal, refs[-1].ordinal - refs[0].ordinal + 1
        ):
            ref = decoded.ref
            if ref.ordinal not in wanted:
                continue
            frame = av.VideoFrame.from_ndarray(decoded.rgb, format="rgb24")
            if (frame.width, frame.height) != (width, height):
                frame = frame.reformat(width=width, height=height, format="yuv420p")
            frame.pts = round(
                (ref.seconds - first + recipe.start_offset_seconds) * 1000
            )
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
            mapping.append(
                {
                    "source_ordinal": ref.ordinal,
                    "source_pts": ref.pts,
                    "source_time_base": [ref.time_base_num, ref.time_base_den],
                    "source_seconds": ref.seconds,
                }
            )
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as result:
        decoded_times = []
        for output_frame in result.decode(video=0):
            if output_frame.pts is None or output_frame.time_base is None:
                raise CorruptArtifact("variant output has a frame without PTS")
            decoded_times.append(float(output_frame.pts * output_frame.time_base))
    if len(decoded_times) != len(mapping):
        raise CorruptArtifact("variant encoder changed the selected frame count")
    for item, seconds in zip(mapping, decoded_times):
        item["variant_seconds"] = seconds
    if recipe.corrupt_tail_bytes:
        size = path.stat().st_size
        if recipe.corrupt_tail_bytes >= size:
            raise ValueError("corrupt_tail_bytes would remove entire variant")
        with path.open("r+b") as file_stream:
            file_stream.truncate(size - recipe.corrupt_tail_bytes)
    return mapping, recording.audio_present


def materialize(
    window: Window, recipe: VariantRecipe, *, root: StorageRoot | None = None
) -> Variant:
    """Generate once in shared derived storage, or verify and reuse the result."""
    recipe.validate()
    root = root or StorageRoot.from_env()
    if hash_file(window.source_path) != window.source_sha256:
        raise CorruptArtifact(f"selection source changed: {window.source_path}")
    identity = {
        "source_sha256": window.source_sha256,
        "camera_id": window.camera_id,
        "window": [window.start_seconds, window.end_seconds],
        "recipe": {
            **asdict(recipe),
            "drop_source_ordinals": list(recipe.drop_source_ordinals),
        },
    }
    digest = hash_config(identity)
    parent = root.namespace("derived") / "media-variants"
    parent.mkdir(parents=True, exist_ok=True)
    lock_dir = root.namespace("derived") / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    directory = parent / digest
    with (lock_dir / f"{digest}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _validated(directory, digest, identity)
        if cached is not None:
            return Variant(directory / "clip.mkv", cached, True)
        stage = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}-", dir=parent))
        try:
            mappings, source_audio_present = _encode(stage / "clip.mkv", window, recipe)
            if hash_file(window.source_path) != window.source_sha256:
                raise CorruptArtifact(
                    "selection source changed during variant generation"
                )
            manifest = {
                "schema_version": 1,
                "key": digest,
                "identity": identity,
                "source_path": str(window.source_path),
                "output_sha256": hash_file(stage / "clip.mkv"),
                "time_mappings": mappings,
                "source_audio_present": source_audio_present,
                "output_audio_present": False,
                "geometry_independent_camera": False,
            }
            (stage / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n"
            )
            for name in ("clip.mkv", "manifest.json"):
                with (stage / name).open("rb") as stream:
                    os.fsync(stream.fileno())
            _fsync_dir(stage)
            os.replace(stage, directory)
            _fsync_dir(parent)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return Variant(directory / "clip.mkv", manifest, False)
