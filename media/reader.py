"""Bounded PyAV decoding with native presentation timestamps and pixel geometry."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import av
import numpy as np

from contracts.models import FrameTime, Provenance, Source
from storage.store import MissingResource, hash_config, hash_file


class IngestError(ValueError):
    """A recording cannot be used as a timed video source."""


class DecodeError(IngestError):
    """A stream failed during indexing or frame retrieval."""


class DecodeWindowExceeded(DecodeError):
    """The requested output has more frames than the configured window."""


@dataclass(frozen=True)
class FrameRef:
    """A presentation-order ordinal, never a claimed native container frame number."""

    ordinal: int
    pts: int
    time_base_num: int
    time_base_den: int
    keyframe: bool

    @property
    def seconds(self) -> float:
        return self.pts * self.time_base_num / self.time_base_den


@dataclass(frozen=True)
class Recording:
    camera_id: str
    path: Path
    sha256: str
    size_bytes: int
    modified_ns: int
    source: Source
    stream_index: int
    codec: str
    stored_width_px: int
    stored_height_px: int
    rotation_degrees: Literal[0, 90, 180, 270]
    stored_to_oriented: tuple[tuple[int, int, int], ...]
    oriented_to_stored: tuple[tuple[int, int, int], ...]
    audio_present: bool
    start_pts: int
    duration_seconds: float | None
    frames: tuple[FrameRef, ...]

    @property
    def source_id(self) -> str:
        return self.source.id

    def frame_time(self, ref: FrameRef, offset_seconds: float = 0.0) -> FrameTime:
        if ref.ordinal >= len(self.frames) or self.frames[ref.ordinal] != ref:
            raise ValueError("frame reference does not belong to this recording")
        return FrameTime(
            source_id=self.source_id,
            camera_id=self.camera_id,
            frame_index=None,  # Container video usually has no native frame number.
            pts=ref.pts,
            time_base_num=ref.time_base_num,
            time_base_den=ref.time_base_den,
            source_seconds=ref.seconds,
            offset_seconds=offset_seconds,
            global_seconds=ref.seconds + offset_seconds,
        )


@dataclass(frozen=True)
class DecodedFrame:
    ref: FrameRef
    rgb: np.ndarray


@dataclass(frozen=True)
class IngestManifest:
    """Versioned local source manifest; synchronization is a later layer."""

    recordings: tuple[Recording, ...]
    schema_version: Literal["1.0.0"] = "1.0.0"

    def __len__(self) -> int:
        return len(self.recordings)

    def __iter__(self) -> Iterator[Recording]:
        return iter(self.recordings)

    def __getitem__(self, index: int) -> Recording:
        return self.recordings[index]


def _geometry(
    width: int, height: int, degrees: int
) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    if degrees == 0:
        return ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    if degrees == 90:
        return ((0, 1, 0), (-1, 0, width - 1), (0, 0, 1)), (
            (0, -1, width - 1),
            (1, 0, 0),
            (0, 0, 1),
        )
    if degrees == 180:
        matrix = ((-1, 0, width - 1), (0, -1, height - 1), (0, 0, 1))
        return matrix, matrix
    if degrees == 270:
        return ((0, -1, height - 1), (1, 0, 0), (0, 0, 1)), (
            (0, 1, 0),
            (-1, 0, height - 1),
            (0, 0, 1),
        )
    raise IngestError(f"unsupported video rotation: {degrees} degrees")


def _rotation(
    frame: av.VideoFrame, metadata: dict[str, str]
) -> Literal[0, 90, 180, 270]:
    # FFmpeg exposes display-matrix rotation on decoded frames. Older files may
    # carry the legacy stream metadata tag instead.
    degrees = int(frame.rotation or metadata.get("rotate", "0")) % 360
    if degrees not in (0, 90, 180, 270):
        raise IngestError(f"unsupported video rotation: {degrees} degrees")
    return degrees  # type: ignore[return-value]


def _frame_ref(ordinal: int, frame: av.VideoFrame) -> FrameRef:
    if frame.pts is None or frame.time_base is None:
        raise IngestError(f"frame {ordinal} is missing native presentation timing")
    if frame.is_corrupt:
        raise DecodeError(f"frame {ordinal} is marked corrupt")
    base = Fraction(frame.time_base)
    if base <= 0:
        raise IngestError(f"frame {ordinal} has an invalid time base")
    return FrameRef(
        ordinal, frame.pts, base.numerator, base.denominator, bool(frame.key_frame)
    )


def _duration_tag_seconds(value: str) -> float:
    """Parse the Matroska-style stream DURATION tag, when one is present."""
    try:
        hours, minutes, seconds = value.split(":")
        result = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError as exc:
        raise IngestError(f"invalid stream DURATION metadata: {value!r}") from exc
    if result < 0 or not np.isfinite(result):
        raise IngestError(f"invalid stream DURATION metadata: {value!r}")
    return result


def _probe(camera_id: str, path: Path) -> Recording:
    if not camera_id:
        raise IngestError("camera ID must be nonempty")
    try:
        digest = hash_file(path)
        stat = path.stat()
    except MissingResource as exc:
        raise IngestError(str(exc)) from exc
    try:
        with av.open(str(path), mode="r") as container:
            videos = list(container.streams.video)
            if len(videos) != 1:
                raise IngestError(
                    f"{path}: expected one supported video stream, found {len(videos)}"
                )
            stream = videos[0]
            if stream.time_base is None or stream.time_base <= 0:
                raise IngestError(f"{path}: video stream has no valid time base")
            width, height = stream.codec_context.width, stream.codec_context.height
            if width <= 0 or height <= 0:
                raise IngestError(f"{path}: video dimensions are unavailable")
            frames: list[FrameRef] = []
            rotation: int | None = None
            final_frame_duration: float | None = None
            for frame in container.decode(stream):
                assert isinstance(frame, av.VideoFrame)
                ref = _frame_ref(len(frames), frame)
                if frames and ref.seconds < frames[-1].seconds:
                    raise IngestError(
                        f"{path}: decoded frames are not in presentation order"
                    )
                if frames and ref.seconds == frames[-1].seconds:
                    raise IngestError(
                        f"{path}: duplicate presentation PTS prevents exact retrieval"
                    )
                current_rotation = _rotation(frame, stream.metadata)
                if rotation is not None and rotation != current_rotation:
                    raise IngestError(f"{path}: rotation changes within the stream")
                rotation = current_rotation
                frames.append(ref)
                final_frame_duration = (
                    float(frame.duration * frame.time_base)
                    if frame.duration is not None and frame.time_base is not None
                    else None
                )
            if not frames:
                raise IngestError(f"{path}: no timed video frames could be decoded")
            assert rotation is not None
            forward, inverse = _geometry(width, height, rotation)
            oriented_width, oriented_height = (
                (height, width) if rotation in (90, 270) else (width, height)
            )
            base = Fraction(stream.time_base)
            declared_end: float | None = None
            if stream.duration is not None:
                stream_start = (
                    stream.start_time * base
                    if stream.start_time is not None
                    else frames[0].seconds
                )
                declared_end = float(stream_start + stream.duration * base)
            duration_tag = stream.metadata.get("DURATION")
            if duration_tag is not None:
                tagged_end = _duration_tag_seconds(duration_tag)
                # Some muxers report an absolute end; others report a span.
                if tagged_end < frames[0].seconds:
                    tagged_end += frames[0].seconds
                if declared_end is None or tagged_end > declared_end:
                    declared_end = tagged_end
            observed_end = (
                frames[-1].seconds + final_frame_duration
                if final_frame_duration is not None
                else None
            )
            if (
                declared_end is not None
                and observed_end is not None
                and final_frame_duration is not None
            ):
                tolerance = max(0.001, final_frame_duration * 0.5)
                if declared_end - observed_end > tolerance:
                    raise DecodeError(
                        f"{path}: truncated video: decoded frames end before "
                        "declared stream duration"
                    )
            duration = (
                declared_end - frames[0].seconds
                if declared_end is not None
                else (
                    observed_end - frames[0].seconds
                    if observed_end is not None
                    else None
                )
            )
            source = Source(
                id=f"source:{digest}",
                schema_version="1.0.0",
                kind="source",
                provenance=Provenance(
                    producer="media.reader",
                    config_digest=hash_config({"version": 1, "orientation": "display"}),
                ),
                camera_id=camera_id,
                width_px=oriented_width,
                height_px=oriented_height,
                time_base_num=base.numerator,
                time_base_den=base.denominator,
            )
            return Recording(
                camera_id=camera_id,
                path=path,
                sha256=digest,
                size_bytes=stat.st_size,
                modified_ns=stat.st_mtime_ns,
                source=source,
                stream_index=stream.index,
                codec=stream.codec_context.name,
                stored_width_px=width,
                stored_height_px=height,
                rotation_degrees=rotation,  # type: ignore[arg-type]
                stored_to_oriented=forward,
                oriented_to_stored=inverse,
                audio_present=bool(container.streams.audio),
                start_pts=frames[0].pts,
                duration_seconds=duration,
                frames=tuple(frames),
            )
    except (OSError, av.FFmpegError) as exc:
        raise DecodeError(f"{path}: video probe/decode failed: {exc}") from exc


def index_recording(
    camera_id: str, path: Path | str, *, preserve_path: bool = False
) -> Recording:
    """Index one source; preserve a caller-pinned descriptor path when requested."""
    candidate = Path(path).expanduser()
    return _probe(
        camera_id, candidate if preserve_path else candidate.resolve(strict=False)
    )


def ingest(sources: Sequence[tuple[str, Path | str]]) -> IngestManifest:
    """Probe local files without editing or copying them; require two distinct views."""
    if len(sources) < 2:
        raise IngestError("multiview ingest requires at least two recordings")
    recordings: list[Recording] = []
    ids: set[str] = set()
    hashes: set[str] = set()
    for camera_id, source_path in sources:
        if camera_id in ids:
            raise IngestError(f"duplicate camera ID: {camera_id}")
        path = Path(source_path).expanduser().resolve(strict=False)
        item = _probe(camera_id, path)
        if item.sha256 in hashes:
            raise IngestError(f"duplicate source recording: {path}")
        ids.add(camera_id)
        hashes.add(item.sha256)
        recordings.append(item)
    return IngestManifest(tuple(recordings))


class MediaReader:
    """Reuse a recording's compact index for timestamp lookup and bounded reads."""

    def __init__(self, recording: Recording, *, max_decode_frames: int = 256) -> None:
        if max_decode_frames < 1:
            raise ValueError("max_decode_frames must be positive")
        self.recording = recording
        self.max_decode_frames = max_decode_frames
        self._times = tuple(ref.seconds for ref in recording.frames)

    def bracket(self, seconds: float) -> tuple[FrameRef | None, FrameRef | None]:
        """Return presentation frames on either side of a native source time."""
        pos = bisect_left(self._times, seconds)
        if pos < len(self._times) and self._times[pos] == seconds:
            return self.recording.frames[pos], self.recording.frames[pos]
        before = self.recording.frames[pos - 1] if pos else None
        after = self.recording.frames[pos] if pos < len(self._times) else None
        return before, after

    def nearest(self, seconds: float) -> FrameRef:
        before, after = self.bracket(seconds)
        if before is None:
            assert after is not None
            return after
        if after is None or seconds - before.seconds <= after.seconds - seconds:
            return before
        return after

    def decode_window(
        self, start: int, count: int
    ) -> Generator[DecodedFrame, None, None]:
        """Decode a small presentation window without caching pixels."""
        refs = self.recording.frames
        if start < 0 or count < 1 or start + count > len(refs):
            raise ValueError("requested window is outside the recording")
        if count > self.max_decode_frames:
            raise DecodeWindowExceeded("requested output exceeds max_decode_frames")
        anchor = start
        while anchor > 0 and not refs[anchor].keyframe:
            anchor -= 1
        try:
            stat = self.recording.path.stat()
        except OSError as exc:
            raise DecodeError(f"{self.recording.path}: source is unavailable") from exc
        if (stat.st_size, stat.st_mtime_ns) != (
            self.recording.size_bytes,
            self.recording.modified_ns,
        ):
            raise DecodeError(f"{self.recording.path}: source changed since indexing")
        try:
            with av.open(str(self.recording.path), mode="r") as container:
                stream = container.streams[self.recording.stream_index]
                container.seek(refs[anchor].pts, backward=True, stream=stream)
                expected = anchor
                started = False
                for frame in container.decode(stream):
                    assert isinstance(frame, av.VideoFrame)
                    if not started:
                        if frame.pts != refs[anchor].pts:
                            if frame.pts is not None and frame.pts > refs[anchor].pts:
                                raise DecodeError(
                                    f"{self.recording.path}: seek missed keyframe"
                                )
                            continue
                        started = True
                    actual = _frame_ref(expected, frame)
                    if actual.pts != refs[expected].pts or (
                        actual.time_base_num,
                        actual.time_base_den,
                    ) != (
                        refs[expected].time_base_num,
                        refs[expected].time_base_den,
                    ):
                        raise DecodeError(
                            f"{self.recording.path}: indexed source timing changed"
                        )
                    if expected >= start:
                        rgb = frame.to_ndarray(format="rgb24")
                        rgb = np.rot90(
                            rgb,
                            k={0: 0, 90: 1, 180: 2, 270: 3}[
                                self.recording.rotation_degrees
                            ],
                        ).copy()
                        yield DecodedFrame(refs[expected], rgb)
                    expected += 1
                    if expected == start + count:
                        return
                raise DecodeError(
                    f"{self.recording.path}: truncated during requested decode"
                )
        except (OSError, av.FFmpegError) as exc:
            raise DecodeError(f"{self.recording.path}: decode failed: {exc}") from exc

    def frame(self, ref: FrameRef) -> DecodedFrame:
        if (
            ref.ordinal >= len(self.recording.frames)
            or self.recording.frames[ref.ordinal] != ref
        ):
            raise ValueError("frame reference does not belong to this recording")
        iterator = self.decode_window(ref.ordinal, 1)
        try:
            return next(iterator)
        finally:
            iterator.close()
