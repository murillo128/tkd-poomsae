"""Ephemeral generated clips and persisted observations for browser acceptance."""

from __future__ import annotations

import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import imageio_ffmpeg  # type: ignore[import-untyped]
import numpy as np

from contracts.models import (
    FrameTime,
    Landmark,
    Landmark2D,
    Observation,
    Provenance,
    Quality,
    RawScore,
    RegionOfInterest,
    Synchronization,
    SyncOffset,
)
from media import ingest
from pipeline import Pipeline
from storage import ArtifactKey, ArtifactStore, StorageRoot
from tests.test_inspection_api import persist
from tkd_poomsae.api import create_app
from tkd_poomsae.inspection import Inspection

scratch = tempfile.TemporaryDirectory(prefix="tkd-browser-")
root = Path(scratch.name)
pipe = Pipeline(ArtifactStore(StorageRoot(root / "store")))
index = Inspection(pipe)
landmark_names: list[Landmark] = ["left_wrist", "left_index_tip", "left_heel", "nose"]
provenance = Provenance(producer="generated-browser-fixture", config_digest="0" * 64)


def clip(path: Path, rate: int, start: int) -> Path:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = 160, 96
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.options = {"crf": "10", "preset": "ultrafast"}
        for i in range(64):
            image = np.zeros((96, 160, 3), dtype=np.uint8)
            image[:, :, 0] = (i * 3) % 256
            image[20:29, 30:39, 1] = 240
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = start + round(i * 1000 / rate)
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


sources = {
    "front": clip(root / "front.mp4", 25, 0),
    "rotated": clip(root / "unrotated.mkv", 30, 400),
    "slow": clip(root / "slow.mp4", 20, 0),
    "excluded": clip(root / "excluded.mp4", 24, 0),
}
subprocess.run(
    [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-copyts",
        "-display_rotation:v:0",
        "90",
        "-i",
        str(sources["rotated"]),
        "-c",
        "copy",
        str(root / "rotated.mp4"),
    ],
    check=True,
)
sources["rotated"] = root / "rotated.mp4"
recordings = ingest(list(sources.items()))
offsets = [0.0, -0.2, 0.1, 0.0]
for count in (2, 3, 4):
    project = f"cameras-{count}"
    pipe.register(project, dict(list(sources.items())[:count]))
    sync = Synchronization(
        kind="synchronization",
        id=f"sync-{count}",
        schema_version="1.0.0",
        provenance=provenance,
        offsets=[
            SyncOffset(
                source_id=recording.source_id,
                automatic_seconds=offsets[i],
                retained=i != 3,
                exclusion_reason="generated excluded view" if i == 3 else None,
                quality=Quality(state="observed", score=0.2 if i == 2 else 0.9),
            )
            for i, recording in enumerate(list(recordings)[:count])
        ],
    )
    sync_key, _ = persist(pipe.store, sync)
    observations: list[ArtifactKey] = []
    for recording in list(recordings)[:count]:
        for ref in recording.frames:
            # All points use display-oriented original pixels, as production does.
            matrix = np.asarray(recording.stored_to_oriented)
            point = (matrix @ np.array([34, 24, 1]))[:2]
            value = Observation(
                kind="observation",
                id=f"{project}-{recording.camera_id}-{ref.ordinal}",
                schema_version="1.0.0",
                provenance=provenance,
                frame=FrameTime(
                    source_id=recording.source_id,
                    camera_id=recording.camera_id,
                    frame_index=ref.ordinal,
                    pts=ref.pts,
                    time_base_num=ref.time_base_num,
                    time_base_den=ref.time_base_den,
                    source_seconds=ref.seconds,
                    offset_seconds=0,
                    global_seconds=ref.seconds,
                ),
                landmarks=[
                    Landmark2D(
                        name=name,
                        xy_px=(float(point[0]), float(point[1])),
                        raw_score=RawScore(
                            value=7, range_min=0, range_max=10, domain="fixture"
                        ),
                        quality=Quality(state="observed", score=0.9),
                    )
                    for name in landmark_names
                ]
                + [
                    Landmark2D(
                        name="right_wrist", xy_px=None, quality=Quality(state="unknown")
                    )
                ],
                regions=[RegionOfInterest(part="left_hand", xywh_px=(0, 0, 16, 16))],
            )
            key, _ = persist(pipe.store, value)
            observations.append(key)
    index.register(project, {"sync": sync_key}, observations=observations)
app = create_app(pipe, allowed_roots={"fixture": root})
