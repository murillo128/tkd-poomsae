"""Generated media proves cue timing, quality and offline reuse."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from media import DecodeError, ingest
from sync import CueConfig, extract_cues


def recording(
    path: Path,
    *,
    video_rate: int,
    audio_rate: int | None,
    pre_roll: float,
    events: tuple[float, ...],
    static: bool = False,
    silent: bool = False,
    camera_shift: bool = False,
    video_gap_after: int | None = None,
    clipped_audio: bool = False,
) -> Path:
    """Write a short clip with native millisecond video PTS and sample audio PTS."""
    duration = max(events, default=1.0) + 0.7
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=video_rate)
        video.width = 64
        video.height = 48
        video.pix_fmt = "yuv420p"
        video.time_base = Fraction(1, 1000)
        video.codec_context.time_base = Fraction(1, 1000)
        audio = None
        if audio_rate is not None:
            audio = container.add_stream("pcm_s16le", rate=audio_rate)
            audio.layout = "mono"
        for index in range(round(duration * video_rate) + 1):
            t = index / video_rate
            image = np.full((48, 64, 3), 16, dtype=np.uint8)
            if camera_shift:
                rng = np.random.default_rng(123)
                pattern = rng.integers(0, 256, size=(48, 64), dtype=np.uint8)
                image = np.repeat(
                    np.roll(pattern, index % 3, axis=1)[:, :, None], 3, axis=2
                )
            elif not static:
                # Two short, repeated actions. A still subject is visible between them.
                active = any(event <= t < event + 0.15 for event in events)
                image[12:36, 12 : 36 if active else 24] = 230
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            gap = 0.5 if video_gap_after is not None and index > video_gap_after else 0
            frame.pts = round((pre_roll + t + gap) * 1000)
            frame.time_base = Fraction(1, 1000)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if audio is not None and audio_rate is not None:
            # Keep each encoded frame under 20 ms so event onset has a clear bin.
            chunk_size = audio_rate // 100
            for offset in range(0, round(duration * audio_rate), chunk_size):
                count = min(chunk_size, round(duration * audio_rate) - offset)
                times = (offset + np.arange(count)) / audio_rate
                samples = np.zeros(count, dtype=np.int16)
                if not silent:
                    audio_active = np.zeros(count, dtype=bool)
                    for event in events:
                        audio_active |= (times >= event) & (times < event + 0.04)
                    if clipped_audio:
                        samples[audio_active] = 32767
                    else:
                        samples[audio_active] = (
                            12000 * np.sin(2 * np.pi * 700 * times[audio_active])
                        ).astype(np.int16)
                frame_a = av.AudioFrame.from_ndarray(
                    samples.reshape(1, -1), format="s16", layout="mono"
                )
                frame_a.sample_rate = audio_rate
                frame_a.pts = round(pre_roll * audio_rate) + offset
                frame_a.time_base = Fraction(1, audio_rate)
                for packet in audio.encode(frame_a):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
    return path


@pytest.mark.parametrize(
    "video_rate,audio_rate,pre_roll", [(10, 8000, 0.0), (25, 16000, 0.8)]
)
def test_native_timing_and_repeated_events(
    tmp_path: Path, video_rate: int, audio_rate: int, pre_roll: float
) -> None:
    events = (0.8, 1.7)
    first = recording(
        tmp_path / "first.mkv",
        video_rate=video_rate,
        audio_rate=audio_rate,
        pre_roll=pre_roll,
        events=events,
    )
    second = recording(
        tmp_path / "second.mkv",
        video_rate=10,
        audio_rate=None,
        pre_roll=0.2,
        events=events,
    )
    item = ingest([("one", first), ("two", second)])[0]
    cues = extract_cues(item, cache_dir=tmp_path / "cache")
    assert cues.cache_hit is False
    assert cues.audio.available and cues.motion.available
    assert len(cues.audio.events) >= 2
    assert len(cues.motion.events) >= 2
    assert "multiple_candidates" in cues.audio.diagnostics
    assert "multiple_candidates" in cues.motion.diagnostics
    for event in events:
        assert any(
            abs(candidate.source_seconds - (pre_roll + event)) < 0.08
            for candidate in cues.audio.events
        )
        assert any(
            abs(candidate.source_seconds - (pre_roll + event)) < 0.16
            for candidate in cues.motion.events
        )
    assert all(0 <= sample.normalized <= 1 for sample in cues.audio.samples)
    assert cues.source_sha256 == item.sha256
    assert extract_cues(item, cache_dir=tmp_path / "cache") == cues.__class__(
        **{**cues.__dict__, "cache_hit": True}
    )


def test_silent_static_and_absent_audio_are_insufficient(tmp_path: Path) -> None:
    silent = recording(
        tmp_path / "silent.mkv",
        video_rate=12,
        audio_rate=8000,
        pre_roll=0.4,
        events=(),
        static=True,
        silent=True,
    )
    absent = recording(
        tmp_path / "absent.mkv",
        video_rate=15,
        audio_rate=None,
        pre_roll=0.0,
        events=(),
        static=True,
    )
    first, second = ingest([("one", silent), ("two", absent)])
    quiet = extract_cues(first)
    no_audio = extract_cues(second)
    assert quiet.audio.confidence == quiet.motion.confidence == 0
    assert quiet.audio.events == quiet.motion.events == ()
    assert "silence" in quiet.audio.diagnostics
    assert "low_motion" in quiet.motion.diagnostics
    assert "absent_audio" in no_audio.audio.diagnostics
    assert no_audio.audio.confidence == no_audio.motion.confidence == 0


def test_config_changes_cache_identity(tmp_path: Path) -> None:
    source = recording(
        tmp_path / "source.mkv",
        video_rate=10,
        audio_rate=None,
        pre_roll=0.0,
        events=(0.8,),
    )
    other = recording(
        tmp_path / "other.mkv",
        video_rate=12,
        audio_rate=None,
        pre_roll=0.2,
        events=(0.8,),
    )
    item = ingest([("one", source), ("two", other)])[0]
    default = extract_cues(item, cache_dir=tmp_path / "cache")
    changed = extract_cues(
        item, config=CueConfig(video_hz=5), cache_dir=tmp_path / "cache"
    )
    assert default.config_digest != changed.config_digest
    assert changed.cache_hit is False


def test_cached_cues_do_not_decode_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = recording(
        tmp_path / "source.mkv",
        video_rate=10,
        audio_rate=8000,
        pre_roll=0.0,
        events=(0.8,),
    )
    other = recording(
        tmp_path / "other.mkv",
        video_rate=12,
        audio_rate=None,
        pre_roll=0.2,
        events=(0.8,),
    )
    item = ingest([("one", source), ("two", other)])[0]
    original = extract_cues(item, cache_dir=tmp_path / "cache")

    def reject_decode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cache hit must not decode media")

    monkeypatch.setattr(av, "open", reject_decode)
    cached = extract_cues(item, cache_dir=tmp_path / "cache")
    assert cached.cache_hit and cached.motion == original.motion
    assert cached.audio == original.audio


def test_changed_source_cannot_use_cached_cues(tmp_path: Path) -> None:
    source = recording(
        tmp_path / "source.mkv",
        video_rate=10,
        audio_rate=None,
        pre_roll=0.0,
        events=(0.8,),
    )
    other = recording(
        tmp_path / "other.mkv",
        video_rate=12,
        audio_rate=None,
        pre_roll=0.2,
        events=(0.8,),
    )
    item = ingest([("one", source), ("two", other)])[0]
    extract_cues(item, cache_dir=tmp_path / "cache")
    source.write_bytes(source.read_bytes() + b"modified")
    with pytest.raises(DecodeError, match="source changed"):
        extract_cues(item, cache_dir=tmp_path / "cache")


def test_camera_shake_and_missing_video_span(tmp_path: Path) -> None:
    source = recording(
        tmp_path / "source.mkv",
        video_rate=10,
        audio_rate=None,
        pre_roll=0.0,
        events=(),
        camera_shift=True,
        video_gap_after=7,
    )
    other = recording(
        tmp_path / "other.mkv",
        video_rate=12,
        audio_rate=None,
        pre_roll=0.2,
        events=(),
        static=True,
    )
    item = ingest([("one", source), ("two", other)])[0]
    cues = extract_cues(item)
    assert "camera_shake_static_camera_violation" in cues.motion.diagnostics
    assert cues.motion.confidence < 1
    assert any(end - start >= 0.5 for start, end in cues.motion.missing_spans)
    assert "missing_spans" in cues.motion.diagnostics


def test_audio_clipping_reduces_quality(tmp_path: Path) -> None:
    source = recording(
        tmp_path / "source.mkv",
        video_rate=10,
        audio_rate=8000,
        pre_roll=0.0,
        events=(0.8, 1.7),
        clipped_audio=True,
    )
    other = recording(
        tmp_path / "other.mkv",
        video_rate=12,
        audio_rate=None,
        pre_roll=0.2,
        events=(0.8,),
    )
    item = ingest([("one", source), ("two", other)])[0]
    audio = extract_cues(item).audio
    assert "clipping" in audio.diagnostics
    assert audio.available
    assert audio.confidence <= 0.5
