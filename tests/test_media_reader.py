"""Small generated recordings exercise native timing, seeking and orientation."""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import av
import imageio_ffmpeg  # type: ignore[import-untyped]
import numpy as np
import pytest

from media import DecodeError, DecodeWindowExceeded, IngestError, MediaReader, ingest


def video(path: Path, pts: list[int], *, b_frames: int = 0) -> Path:
    """Encode distinct frames at explicit millisecond presentation timestamps."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.codec_context.max_b_frames = b_frames
        stream.codec_context.gop_size = 5
        for i, timestamp in enumerate(pts):
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            image[:, :, 0] = i * 20
            image[:6, :6, 1] = 200
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = timestamp
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_cfr_vfr_and_b_frame_presentation_seek(tmp_path: Path) -> None:
    cfr = video(tmp_path / "cfr.mkv", [400 + i * 40 for i in range(12)])
    vfr = video(
        tmp_path / "vfr.mkv",
        [900, 940, 1020, 1060, 1160, 1200, 1240, 1320, 1360, 1400, 1500, 1540],
        b_frames=2,
    )
    manifest = ingest([("alpha", cfr), ("beta", vfr)])
    first, second = manifest
    assert manifest.schema_version == "1.0.0"
    assert first.sha256 != second.sha256
    assert first.start_pts == 400
    assert [r.pts for r in first.frames] == [400 + i * 40 for i in range(12)]
    assert [r.pts for r in second.frames] == [
        900,
        940,
        1020,
        1060,
        1160,
        1200,
        1240,
        1320,
        1360,
        1400,
        1500,
        1540,
    ]
    assert second.codec == "mpeg4"
    assert second.duration_seconds is not None
    assert second.audio_present is False
    assert second.source.width_px == 32
    assert second.source.height_px == 24
    reader = MediaReader(second, max_decode_frames=8)
    assert reader.bracket(1.10) == (second.frames[3], second.frames[4])
    assert reader.bracket(second.frames[3].seconds) == (
        second.frames[3], second.frames[3]
    )
    assert reader.nearest(1.10) == second.frames[3]
    for index in (0, 2, 5, 7, 11):
        decoded = reader.frame(second.frames[index])
        assert decoded.ref == second.frames[index]
        assert abs(int(decoded.rgb[12, 16, 0]) - index * 20) < 12
        timing = second.frame_time(decoded.ref, offset_seconds=-0.25)
        assert timing.pts == decoded.ref.pts
        assert timing.frame_index is None
        assert timing.global_seconds == pytest.approx(decoded.ref.seconds - 0.25)
    assert [d.ref.ordinal for d in reader.decode_window(5, 3)] == [5, 6, 7]
    with pytest.raises(DecodeWindowExceeded):
        list(MediaReader(second, max_decode_frames=1).decode_window(2, 2))


def test_display_matrix_rotation_and_inverse_transform(tmp_path: Path) -> None:
    plain = video(tmp_path / "plain.mkv", [0, 40, 80])
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(plain),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    reference, oriented = ingest([("plain", plain), ("oriented", rotated)])
    assert oriented.rotation_degrees == 90
    assert (oriented.stored_width_px, oriented.stored_height_px) == (32, 24)
    assert (oriented.source.width_px, oriented.source.height_px) == (24, 32)
    raw = MediaReader(reference).frame(reference.frames[0]).rgb
    output = MediaReader(oriented).frame(oriented.frames[0]).rgb
    np.testing.assert_array_equal(output, np.rot90(raw, k=1))
    x, y = 3, 4
    matrix = np.asarray(oriented.stored_to_oriented)
    inverse = np.asarray(oriented.oriented_to_stored)
    point = np.array([x, y, 1])
    np.testing.assert_array_equal(inverse @ matrix @ point, point)
    np.testing.assert_array_equal(matrix @ point, [y, 31 - x, 1])


def test_invalid_sources_and_partial_decode_are_explicit(tmp_path: Path) -> None:
    source = video(tmp_path / "source.mkv", [0, 40, 80, 120, 160])
    with pytest.raises(IngestError, match="at least two"):
        ingest([("only", source)])
    with pytest.raises(IngestError, match="duplicate camera ID"):
        ingest([("one", source), ("one", source)])
    with pytest.raises(IngestError, match="duplicate source"):
        ingest([("one", source), ("two", source)])
    with pytest.raises(IngestError, match="Missing input"):
        ingest([("one", source), ("two", tmp_path / "absent.mkv")])
    invalid = tmp_path / "invalid.mkv"
    invalid.write_bytes(b"not a media stream")
    with pytest.raises(DecodeError, match="probe/decode failed"):
        ingest([("one", source), ("two", invalid)])
    other = video(tmp_path / "other.mkv", [200, 240, 280, 320, 360])
    recording = ingest([("one", source), ("two", other)])[0]
    source.write_bytes(source.read_bytes()[:100])
    with pytest.raises(DecodeError, match="source changed"):
        MediaReader(recording).frame(recording.frames[-1])
