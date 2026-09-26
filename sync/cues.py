"""Bounded audio and visual cue extraction from an indexed native recording.

Times are source PTS seconds. This layer deliberately makes no offset or unique
event correspondence claim; consumers must evaluate the complete candidate set.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

import av
import numpy as np

from media import DecodeError, Recording
from storage import hash_config

_REVISION = "cues-v1"


@dataclass(frozen=True)
class CueConfig:
    video_hz: float = 10.0
    max_dimension: int = 96
    audio_window_ms: int = 20
    min_audio_rms: float = 0.01
    min_motion_energy: float = 0.025
    event_threshold: float = 0.45
    event_spacing_seconds: float = 0.08

    def __post_init__(self) -> None:
        if not 0 < self.video_hz <= 60 or not 16 <= self.max_dimension <= 512:
            raise ValueError("invalid bounded video analysis settings")
        if not 1 <= self.audio_window_ms <= 100:
            raise ValueError("audio_window_ms must be 1..100")
        if not 0 < self.min_audio_rms < 1 or not 0 < self.min_motion_energy < 1:
            raise ValueError("invalid minimum cue energy")
        if not 0 < self.event_threshold < 1 or self.event_spacing_seconds <= 0:
            raise ValueError("invalid event settings")


@dataclass(frozen=True)
class CueSample:
    start_seconds: float
    end_seconds: float
    amplitude: float
    normalized: float


@dataclass(frozen=True)
class CueEvent:
    source_seconds: float
    strength: float
    confidence: float


@dataclass(frozen=True)
class CueStream:
    kind: Literal["audio", "motion"]
    available: bool
    confidence: float
    samples: tuple[CueSample, ...]
    events: tuple[CueEvent, ...]
    missing_spans: tuple[tuple[float, float], ...]
    diagnostics: tuple[str, ...]
    sample_interval_seconds: float


@dataclass(frozen=True)
class CueSet:
    source_sha256: str
    config_digest: str
    algorithm_revision: str
    audio: CueStream
    motion: CueStream
    cache_hit: bool = False


def _gaps(samples: list[CueSample], expected: float) -> tuple[tuple[float, float], ...]:
    return tuple(
        (previous.end_seconds, current.start_seconds)
        for previous, current in zip(samples, samples[1:])
        if current.start_seconds - previous.end_seconds > max(2.5 * expected, 0.05)
    )


def _finish(
    kind: Literal["audio", "motion"],
    raw: list[tuple[float, float, float]],
    expected: float,
    minimum: float,
    diagnostics: list[str],
    quality: float = 1.0,
    threshold: float = 0.45,
    spacing: float = 0.08,
    extra_missing: tuple[tuple[float, float], ...] = (),
) -> CueStream:
    amplitudes = np.array([item[2] for item in raw], dtype=np.float64)
    if len(amplitudes):
        floor = float(np.percentile(amplitudes, 10))
        ceiling = float(np.percentile(amplitudes, 95))
        spread = ceiling - floor
        # Sparse transient streams can have a zero 95th percentile.
        scale = max(spread, float(np.max(amplitudes)) - floor)
        normalized = (
            np.clip((amplitudes - floor) / scale, 0, 1)
            if scale > 0
            else np.zeros_like(amplitudes)
        )
    else:
        normalized = amplitudes
    available = bool(len(amplitudes) and float(np.max(amplitudes)) >= minimum)
    if not available:
        diagnostics.append("silence" if kind == "audio" else "low_motion")
    elif scale < minimum * 0.5:
        diagnostics.append("low_contrast")
        available = False
    samples = [
        CueSample(start, end, amp, float(value))
        for (start, end, amp), value in zip(raw, normalized)
    ]
    missing = tuple(sorted(set(_gaps(samples, expected) + extra_missing)))
    if missing:
        diagnostics.append("missing_spans")
    if available:
        peaks: list[int] = []
        for i, sample in enumerate(samples):
            before = samples[i - 1].normalized if i else 0.0
            after = samples[i + 1].normalized if i + 1 < len(samples) else 0.0
            if sample.normalized >= threshold and (
                before < threshold
                or (sample.normalized >= before and sample.normalized > after)
            ):
                if (
                    peaks
                    and sample.end_seconds - samples[peaks[-1]].end_seconds < spacing
                ):
                    if sample.normalized > samples[peaks[-1]].normalized:
                        peaks[-1] = i
                else:
                    peaks.append(i)
        events = tuple(
            CueEvent(
                samples[i].end_seconds,
                samples[i].normalized,
                quality * samples[i].normalized,
            )
            for i in peaks
        )
    else:
        events = ()
    if len(events) > 1:
        diagnostics.append("multiple_candidates")
    # This is local cue quality, not a probability of a unique cross-view match.
    confidence = quality * (0.7 if len(events) > 1 else 1.0) if events else 0.0
    return CueStream(
        kind,
        available,
        confidence,
        tuple(samples),
        events,
        missing,
        tuple(diagnostics),
        expected,
    )


def _audio(recording: Recording, config: CueConfig) -> CueStream:
    expected = config.audio_window_ms / 1000
    if not recording.audio_present:
        return CueStream("audio", False, 0.0, (), (), (), ("absent_audio",), expected)
    raw: list[tuple[float, float, float]] = []
    diagnostics: list[str] = []
    clipped = 0
    total = 0
    try:
        with av.open(str(recording.path), mode="r") as container:
            stream = container.streams.audio[0]
            if len(container.streams.audio) > 1:
                diagnostics.append("multiple_audio_streams_first_used")
            for frame in container.decode(stream):
                assert isinstance(frame, av.AudioFrame)
                if (
                    frame.pts is None
                    or frame.time_base is None
                    or not frame.sample_rate
                ):
                    diagnostics.append("missing_audio_pts")
                    continue
                native = frame.to_ndarray()
                values = native.astype(np.float64)
                # Planar samples have a channel axis; packed samples interleave it.
                if frame.format.is_planar:
                    channels = values
                else:
                    channels = values.reshape(-1, frame.layout.nb_channels).T
                if np.issubdtype(native.dtype, np.unsignedinteger):
                    info = np.iinfo(cast(np.dtype[np.integer], native.dtype))
                    midpoint = (info.max + 1) / 2
                    channels = (channels - midpoint) / midpoint
                elif np.issubdtype(native.dtype, np.integer):
                    info = np.iinfo(cast(np.dtype[np.integer], native.dtype))
                    channels /= max(abs(info.min), info.max)
                start = float(frame.pts * frame.time_base)
                size = max(1, round(frame.sample_rate * expected))
                for offset in range(0, frame.samples, size):
                    chunk = channels[:, offset : offset + size]
                    if chunk.size == 0:
                        continue
                    clipped += int(np.count_nonzero(np.abs(chunk) >= 0.995))
                    total += chunk.size
                    value = float(np.sqrt(np.mean(np.square(chunk))))
                    raw.append(
                        (
                            start + offset / frame.sample_rate,
                            start + (offset + chunk.shape[1]) / frame.sample_rate,
                            value,
                        )
                    )
    except (OSError, av.FFmpegError) as exc:
        raise DecodeError(f"{recording.path}: audio decode failed: {exc}") from exc
    raw.sort(key=lambda item: item[0])
    if total and clipped / total > 0.01:
        diagnostics.append("clipping")
    coverage: list[tuple[float, float]] = []
    if raw:
        video_start = recording.frames[0].seconds
        video_end = recording.frames[-1].seconds
        if raw[0][0] - video_start > 2 * expected:
            coverage.append((video_start, raw[0][0]))
        if video_end - raw[-1][1] > 2 * expected:
            coverage.append((raw[-1][1], video_end))
    return _finish(
        "audio",
        raw,
        expected,
        config.min_audio_rms,
        diagnostics,
        0.5 if "clipping" in diagnostics else 1.0,
        config.event_threshold,
        config.event_spacing_seconds,
        tuple(coverage),
    )


def _gray(frame: av.VideoFrame, max_dimension: int) -> np.ndarray:
    scale = min(1.0, max_dimension / max(frame.width, frame.height))
    width = max(1, round(frame.width * scale))
    height = max(1, round(frame.height * scale))
    return frame.reformat(width=width, height=height, format="gray").to_ndarray()


def _shift_score(previous: np.ndarray, current: np.ndarray) -> tuple[float, bool]:
    a = previous.astype(np.float32)
    b = current.astype(np.float32)
    difference = np.abs(a - b)
    flattened = difference.ravel()
    top_count = max(1, len(flattened) // 10)
    energy = float(np.mean(np.partition(flattened, -top_count)[-top_count:]) / 255)
    baseline = (
        float(np.mean(difference[2:-2, 2:-2]))
        if min(a.shape) > 8
        else float(np.mean(difference))
    )
    if min(a.shape) <= 8 or baseline < 2:
        return energy, False
    best = baseline
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            if dx == dy == 0:
                continue
            shifted = float(
                np.mean(
                    np.abs(
                        a[2 + dy : a.shape[0] - 2 + dy, 2 + dx : a.shape[1] - 2 + dx]
                        - b[2:-2, 2:-2]
                    )
                )
            )
            best = min(best, shifted)
    return energy, best < 0.55 * baseline


def _motion(recording: Recording, config: CueConfig) -> CueStream:
    raw: list[tuple[float, float, float]] = []
    diagnostics: list[str] = []
    previous: np.ndarray | None = None
    previous_time: float | None = None
    last_selected: float | None = None
    shaken = 0
    total_pairs = 0
    missing: list[tuple[float, float]] = []
    frame_times = [ref.seconds for ref in recording.frames]
    native_intervals = np.diff(frame_times)
    expected = max(
        1 / config.video_hz,
        float(np.median(native_intervals)) if len(native_intervals) else 0.0,
    )
    decoded = 0
    try:
        with av.open(str(recording.path), mode="r") as container:
            stream = container.streams[recording.stream_index]
            for ordinal, frame in enumerate(container.decode(stream)):
                decoded += 1
                assert isinstance(frame, av.VideoFrame)
                if (
                    ordinal >= len(recording.frames)
                    or frame.pts != recording.frames[ordinal].pts
                ):
                    raise DecodeError(
                        f"{recording.path}: video PTS changed since indexing"
                    )
                seconds = recording.frames[ordinal].seconds
                if (
                    last_selected is not None
                    and seconds - last_selected < 1 / config.video_hz - 1e-9
                ):
                    continue
                current = _gray(frame, config.max_dimension)
                if previous is not None and previous_time is not None:
                    if seconds - previous_time > 2.5 * expected:
                        missing.append((previous_time, seconds))
                    else:
                        energy, shake = _shift_score(previous, current)
                        raw.append((previous_time, seconds, energy))
                        shaken += int(shake)
                        total_pairs += 1
                previous, previous_time, last_selected = current, seconds, seconds
            if decoded != len(recording.frames):
                raise DecodeError(f"{recording.path}: video truncated since indexing")
    except (OSError, av.FFmpegError) as exc:
        raise DecodeError(f"{recording.path}: visual decode failed: {exc}") from exc
    if total_pairs and shaken / total_pairs > 0.25:
        diagnostics.append("camera_shake_static_camera_violation")
    if raw and float(np.median([item[2] for item in raw])) > 0.08:
        diagnostics.append("camera_noise_or_continuous_motion")
    quality = 0.4 if "camera_shake_static_camera_violation" in diagnostics else 1.0
    return _finish(
        "motion",
        raw,
        expected,
        config.min_motion_energy,
        diagnostics,
        quality,
        config.event_threshold,
        config.event_spacing_seconds,
        tuple(missing),
    )


def _deserialize(data: dict[str, object]) -> CueSet:
    def stream(value: object) -> CueStream:
        assert isinstance(value, dict)
        return CueStream(
            kind=value["kind"],
            available=value["available"],
            confidence=value["confidence"],
            samples=tuple(CueSample(**item) for item in value["samples"]),
            events=tuple(CueEvent(**item) for item in value["events"]),
            missing_spans=tuple(tuple(item) for item in value["missing_spans"]),
            diagnostics=tuple(value["diagnostics"]),
            sample_interval_seconds=value["sample_interval_seconds"],
        )

    return CueSet(
        cast(str, data["source_sha256"]),
        cast(str, data["config_digest"]),
        cast(str, data["algorithm_revision"]),
        stream(data["audio"]),
        stream(data["motion"]),
        True,
    )


def extract_cues(
    recording: Recording,
    *,
    config: CueConfig = CueConfig(),
    cache_dir: Path | None = None,
) -> CueSet:
    """Extract native-time cues, optionally reusing a content-addressed disk cache."""
    try:
        stat = recording.path.stat()
    except OSError as exc:
        raise DecodeError(f"{recording.path}: source is unavailable") from exc
    if (stat.st_size, stat.st_mtime_ns) != (
        recording.size_bytes,
        recording.modified_ns,
    ):
        raise DecodeError(f"{recording.path}: source changed since indexing")
    digest = hash_config({"algorithm": _REVISION, "config": asdict(config)})
    cache_path = cache_dir / f"{recording.sha256}-{digest}.json" if cache_dir else None
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("source_sha256"),
            cached.get("config_digest"),
            cached.get("algorithm_revision"),
        ) == (recording.sha256, digest, _REVISION):
            return _deserialize(cached)
    result = CueSet(
        recording.sha256,
        digest,
        _REVISION,
        _audio(recording, config),
        _motion(recording, config),
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(result), allow_nan=False, separators=(",", ":"))
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=".cues-",
            delete=False,
        ) as temp:
            temporary = Path(temp.name)
            try:
                temp.write(payload)
                temp.flush()
                os.fsync(temp.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, cache_path)
    return replace(result, cache_hit=False)
