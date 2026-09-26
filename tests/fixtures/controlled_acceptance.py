"""Positive geometric integration fixture; no model or semantic oracle inputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from calibration.cameras import CameraModel
from contracts.models import (
    FrameTime,
    Landmark,
    Landmark2D,
    Observation,
    Provenance,
    Quality,
    Reconstruction,
    Semantics,
    Synchronization,
    SyncOffset,
)
from media import index_recording
from pipeline import Pipeline
from reconstruction.arms import publish_arm_actions
from reconstruction.features import FeatureConfig, publish_features
from reconstruction.footprints import publish_footprints
from reconstruction.ground import publish_contacts
from reconstruction.ground_view import publish_ground_view
from reconstruction.lower_body import publish_lower_body
from reconstruction.pivots import publish_pivots
from reconstruction.segmentation import publish_segmentation
from reconstruction.semantics import publish_semantics
from reconstruction.temporal import publish_temporal_motion
from reconstruction.triangulation import TriangulationConfig, publish_triangulation
from storage import ArtifactHandle, StorageRoot, hash_file
from sync.alignment import publish_alignment
from tests.fixtures.synthetic import RECIPES, Scene, Vec3, knee_between, ramp
from tests.functional_smoke import InventoryStore, bundle, persist, save
from tests.test_ground_contact import calibration
from tkd_poomsae.inspection import Inspection

PROJECT = "controlled-acceptance"
RATE = 25
REVISION = "controlled-motion-v1"


class ControlledScene(Scene):
    """Quiet torso, fixed-length arm extensions overlapping a physical foot lift."""

    def root(self, t: float) -> tuple[Vec3, float]:
        return (0.0, 0.0, 0.9), 0.0

    def landmarks(self, t: float) -> dict[str, Vec3]:
        points = super().landmarks(t)
        for side, sign, peak in (("left", -1, 1.3), ("right", 1, 1.6)):
            progress = max(0.0, 1 - abs(t - peak) / 0.5)
            distance = 0.3 + 0.4 * progress
            points[f"{side}_shoulder"] = (sign * 0.23, 0.0, 1.38)
            points[f"{side}_elbow"] = (
                sign * (0.23 + math.sqrt(0.4**2 - (distance / 2) ** 2)),
                distance / 2,
                1.38,
            )
            points[f"{side}_wrist"] = (sign * 0.23, distance, 1.38)
        lift = ramp(t, 0.8, 1.0)
        recovery = ramp(t, 1.8, 2.2)
        extension = max(0.0, 1 - abs(t - 1.4) / 0.4)
        ankle = (
            0.16,
            0.02 + 0.38 * lift + 0.42 * extension - 0.14 * recovery,
            0.08 + 0.79 * lift - 0.79 * recovery,
        )
        points["right_ankle"] = ankle
        points["right_knee"] = knee_between(points["right_hip"], ankle, 0.44, 0.43)
        for part, delta_y in (("heel", -0.07), ("forefoot", 0.19)):
            points[f"right_{part}"] = (ankle[0], ankle[1] + delta_y, ankle[2] - 0.08)
        points["right_foot_outer"] = (ankle[0] + 0.04, ankle[1] + 0.16, ankle[2] - 0.08)
        left = points["left_forefoot"]
        points["left_foot_outer"] = (left[0] - 0.04, left[1] - 0.03, left[2])
        # Use only the supported whole-body names below; omit the old hand recipe
        # because it is attached to a different arm trajectory.
        return points


def parse(
    store: InventoryStore,
    motion: ArtifactHandle,
    ground: ArtifactHandle,
    config: FeatureConfig | None = None,
) -> dict[str, ArtifactHandle]:
    features = publish_features(store, motion, ground, config)
    segmentation = publish_segmentation(store, features)
    arms = publish_arm_actions(store, features, segmentation, motion)
    legs = publish_lower_body(store, features, ground, segmentation)
    semantics = publish_semantics(store, features, segmentation, arms, legs)
    return dict(
        features=features,
        segmentation=segmentation,
        arms=arms,
        legs=legs,
        semantics=semantics,
    )


def build(
    root: Path, store: InventoryStore | None = None
) -> tuple[Pipeline, InventoryStore, dict[str, ArtifactHandle]]:
    root.mkdir(parents=True, exist_ok=True)
    store = store or InventoryStore(StorageRoot(root / "store"))
    pipe = Pipeline(store)
    scene = ControlledScene(RECIPES["clean_two"])
    cal = calibration()
    cal.cameras = cal.cameras[:2]
    cal.id = REVISION + "-calibration"
    cal.provenance = Provenance(
        producer="controlled-supplied-geometry",
        model=REVISION,
        config_digest="0" * 64,
    )
    times = [i / RATE for i in range(4 * RATE + 1)]
    names: list[Landmark] = [
        "pelvis",
        "neck",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_forefoot",
        "right_forefoot",
        "left_foot_outer",
        "right_foot_outer",
    ]
    observations, sources = [], {}
    for camera in cal.cameras:
        path = root / f"{camera.camera_id}.mp4"
        model = CameraModel(camera.intrinsics, np.array(camera.world_to_camera))
        pixels = [model.project([scene.landmarks(t)[n] for n in names]) for t in times]
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=RATE)
            stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
            stream.options = {"crf": "18", "preset": "ultrafast"}
            for xy in pixels:
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
                for x, y in xy:
                    cv2.circle(image, (round(x), round(y)), 3, (255, 200, 40), -1)
                for packet in stream.encode(
                    av.VideoFrame.from_ndarray(image, format="rgb24")
                ):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        camera.source_id = "source:" + hash_file(path)
        # These are exactly supplied cameras, not estimated video calibration.
        camera.rms_reprojection_px = 0
        sources[camera.camera_id] = path
        recording = index_recording(camera.camera_id, path)
        for i, (ref, xy) in enumerate(zip(recording.frames, pixels, strict=True)):
            observations.append(
                persist(
                    store,
                    Observation(
                        kind="observation",
                        id=f"{REVISION}:{camera.camera_id}:{i}",
                        schema_version="1.0.0",
                        provenance=cal.provenance,
                        frame=FrameTime(
                            source_id=camera.source_id,
                            camera_id=camera.camera_id,
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
                                name=n,
                                xy_px=tuple(pixel),
                                quality=Quality(
                                    state="observed", score=1, uncertainty=0
                                ),
                            )
                            for n, pixel in zip(names, xy, strict=True)
                        ],
                    ),
                )
            )
    if not pipe._directory(PROJECT).exists():
        pipe.register(PROJECT, sources)
    else:
        registered = json.loads(pipe._files(PROJECT)[0].read_text())
        assert registered["sources"] == {n: str(p) for n, p in sources.items()}
    sync = persist(
        store,
        Synchronization(
            kind="synchronization",
            id=REVISION + "-sync",
            schema_version="1.0.0",
            provenance=cal.provenance,
            offsets=[
                SyncOffset(
                    source_id=c.source_id,
                    automatic_seconds=0,
                    quality=Quality(state="observed", score=1, uncertainty=0),
                )
                for c in cal.cameras
            ],
        ),
    )
    cal.evidence_links = ["synchronization:" + sync.metadata.id]
    cal_handle = persist(store, cal)
    alignment = publish_alignment(store, observations, sync, times)
    raw = publish_triangulation(
        store,
        cal_handle,
        alignment,
        "controlled-practitioner",
        # Analytic 2D coordinates, not rounded dots decoded from the video.
        # Declare a conservative numerical precision floor for this input only.
        TriangulationConfig(pixel_sigma=0.001),
    )
    temporal = publish_temporal_motion(store, raw)
    motion = temporal.motion
    contacts = publish_contacts(store, motion, cal_handle)
    feet = publish_footprints(store, motion, contacts, cal_handle)
    pivots = publish_pivots(store, feet)
    ground = publish_ground_view(store, motion, cal_handle, pivots)
    handles = dict(
        sync=sync,
        calibration=cal_handle,
        alignment=alignment,
        raw=raw,
        morphology=temporal.morphology,
        fitted=temporal.fitted,
        reconstruction=motion,
        contacts=contacts,
        feet=feet,
        pivots=pivots,
        ground=ground,
    )
    handles.update(parse(store, motion, ground))
    products = {
        n: store.key(handles[n])
        for n in ("sync", "calibration", "reconstruction", "ground", "semantics")
    }
    lineage = [store.key(h) for h in [alignment, raw, temporal.fitted, *observations]]
    Inspection(pipe).register(
        PROJECT,
        products,
        observations=[store.key(h) for h in observations],
        lineage=lineage,
    )
    save(
        root / "bundle.json",
        bundle(
            store,
            {n: handles[n] for n in products},
            [store.key(h) for h in observations],
            [alignment, raw, temporal.fitted, *observations],
        ),
    )
    return pipe, store, handles


def report(handles: dict[str, ArtifactHandle]) -> dict[str, Any]:
    semantics = handles["semantics"].metadata
    motion = handles["reconstruction"].metadata
    assert isinstance(semantics, Semantics) and isinstance(motion, Reconstruction)
    return {
        "fixture_revision": REVISION,
        "project": PROJECT,
        "claim": "Controlled software integration; not real-video or parser accuracy",
        "supplied": [
            "metric cameras and ground",
            "zero-offset clocks",
            "projected 2D observations",
        ],
        "model_inference": False,
        "manual_labels": False,
        "motion_samples": len(motion.samples),
        "steps": len(semantics.steps),
        "action_categories": dict(Counter(a.category for a in semantics.actions)),
        "keyframes": len(semantics.keyframes),
        "execution": semantics.execution.model_dump() if semantics.execution else None,
        "products": {
            n: {
                "id": h.metadata.id,
                "producer": h.metadata.provenance.producer,
                "manifest_sha256": hash_file(h.path / "manifest.json"),
            }
            for n, h in handles.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    _, _, handles = build(args.root.resolve())
    result = report(handles)
    save(args.root / "report.json", result)
    print(json.dumps(result, indent=2))
    assert result["steps"] and result["action_categories"].get("arm", 0)
    assert result["action_categories"].get("kick", 0)


if __name__ == "__main__":
    main()
