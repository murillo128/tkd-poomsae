"""Constructed final motion inputs, never semantic labels or parser proposals."""

from __future__ import annotations

import json
import math
from typing import Literal

import numpy as np

from contracts.models import (
    Contact,
    DenseArray,
    Ground,
    GroundSample,
    Landmark3D,
    Provenance,
    Quality,
    Reconstruction,
)
from reconstruction.detailed import derive_sample
from reconstruction.temporal import TemporalConfig
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config
from tests.test_motion_features import sequence

Case = Literal["compound", "stationary_arms", "special", "crossing", "noise"]


def quality() -> Quality:
    return Quality(state="observed", uncertainty=1e-5, source_ids=["constructed-v1"])


def reconstructed(case: Case, rate: int = 50) -> tuple[Reconstruction, Ground]:
    source, floor = sequence(flat=True)
    template = source.samples[0]
    source.id = f"constructed:{case}:{rate}"
    floor.id = f"constructed-ground:{case}:{rate}"
    floor.reconstruction_id = source.id
    source.samples, floor.samples = [], []
    for i in range(7 * rate + 1):
        t = i / rate
        sample = template.model_copy(deep=True)
        sample.global_seconds = 23 + t
        points: dict[str, Landmark3D] = {p.name: p for p in sample.landmarks}
        for side, sign, peak in (("left", -1, 1.5), ("right", 1, 1.8)):
            pulse = max(0, 1 - abs(t - peak) / 0.5)
            distance = 0.3 + 0.4 * pulse
            if case in ("compound", "stationary_arms"):
                points[f"{side}_wrist"].xyz_world = (sign * 0.25, distance, 1.6)
                points[f"{side}_elbow"].xyz_world = (
                    sign * (0.25 + math.sqrt(0.4**2 - (distance / 2) ** 2)),
                    distance / 2,
                    1.6,
                )
            elif case == "special":
                delay = 0 if side == "left" else 0.08
                x = 0.5 - 0.45 * min(1, max(0, (t - 1 - delay) / 1))
                points[f"{side}_wrist"].xyz_world = (sign * x, 0.3, 1.6)
            elif case == "crossing":
                fraction = min(1, max(0, (t - 1) / 0.8))
                depth = 0.1 if side == "left" else 0
                points[f"{side}_elbow"].xyz_world = (sign * 0.3, depth, 1.6)
                points[f"{side}_wrist"].xyz_world = (
                    sign * (0.6 - 0.9 * fraction),
                    depth,
                    1.4,
                )
            elif case == "noise":
                points[f"{side}_wrist"].xyz_world = (
                    sign * 0.25 + 0.0002 * math.sin(i * 1.7),
                    0.3,
                    1.6,
                )
        airborne = case == "compound" and 0.8 <= t < 2.4
        if case == "compound":
            ratio = 0.5 + 0.48 * max(0, 1 - abs(t - 1.5) / 0.5)
            distance = 0.9 * ratio
            if t < 2.1:
                ankle = (-0.15, distance, 1.0)
                knee = (
                    -0.15 + math.sqrt(0.45**2 - (distance / 2) ** 2),
                    distance / 2,
                    1.0,
                )
            else:
                progress = min(1, (t - 2.1) / 0.3)
                ankle = (-0.15, 0.45 - 0.2 * progress, 1 - 0.9 * progress)
                knee = (0.25, 0.225, 0.75 - 0.2 * progress)
            points["left_ankle"].xyz_world = ankle
            points["left_knee"].xyz_world = knee
        for point in sample.landmarks:
            point.quality.source_ids = [f"constructed:{i}:{point.name}"]
        sample.quality = quality()
        source.samples.append(sample)
        floor.samples.append(
            GroundSample(
                global_seconds=sample.global_seconds,
                left=Contact(
                    state="no_contact" if airborne else "contact", quality=quality()
                ),
                right=Contact(state="contact", quality=quality()),
                support="right" if airborne else "both",
            )
        )
    return source, floor


def persist_inputs(
    store: ArtifactStore, source: Reconstruction, floor: Ground
) -> tuple[ArtifactHandle, ArtifactHandle]:
    """Serialize constructed final motion at the parser's persisted input boundary.

    Detailed geometry is derived from landmarks. No fit, vision, dataset, action,
    feature, event or segmentation oracle is substituted into the parser.
    """
    geometry = [
        derive_sample(s, reconstruction_id=source.id, representation="regularized")
        for s in source.samples
    ]
    payload = {
        "version": 1,
        "artifact_role": "temporally_coherent_motion",
        "reconstruction_id": source.id,
        "settings": TemporalConfig().model_dump(mode="json"),
        "detailed": [g.model_dump(mode="json") for g in geometry],
        "kinematics": [],
        "supplied_root": [False] * len(geometry),
        "root_translation_quality": [quality().model_dump(mode="json")] * len(geometry),
        "root_orientation_quality": [
            g.body_frame.quality.model_dump(mode="json") for g in geometry
        ],
    }
    array = np.frombuffer(json.dumps(payload, sort_keys=True).encode(), dtype=np.uint8)
    source = source.model_copy(deep=True)
    source.provenance = Provenance(
        producer="reconstruction.temporal",
        model="constructed-parser-input-v1",
        config_digest="0" * 64,
    )
    source.arrays = [
        DenseArray(
            id="temporal_motion_json",
            dtype="uint8",
            shape=[len(array)],
            axes=["json_byte"],
        )
    ]
    key = ArtifactKey(
        layer="reconstruction",
        inputs={"fixture": hash_config(source.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision="constructed-parser-input-v1",
        config_digest="0" * 64,
    )
    motion = store.get_or_create(key, lambda: (source, {"temporal_motion_json": array}))
    ground_key = ArtifactKey(
        layer="ground",
        inputs={"fixture": hash_config(floor.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision="constructed-parser-input-v1",
        config_digest="0" * 64,
    )
    ground = store.get_or_create(ground_key, lambda: (floor, {}))
    return motion, ground


def pivot_motion() -> tuple[Reconstruction, Ground]:
    """Foot rotation becomes ground evidence through the actual physical detector."""
    from reconstruction.footprints import derive_footprints
    from reconstruction.pivots import derive_pivots
    from tests.test_ground_contact import calibration

    source, floor = reconstructed("stationary_arms")
    from tests.test_pivots import sequence as foot_sequence

    feet, _ = foot_sequence()
    for index, sample in enumerate(source.samples):
        foot_sample = feet.samples[
            min(50, max(0, round((sample.global_seconds - 23.7) * 50)))
        ]
        for point in sample.landmarks:
            original = next(
                (p for p in foot_sample.landmarks if p.name == point.name), None
            )
            if original is not None:
                point.xyz_world = original.xyz_world
        floor.samples[index].left.region = "heel"
    footprints = derive_footprints(source, floor, calibration())
    pivots = derive_pivots(footprints)
    floor.footprints = [e.footprint for e in footprints.events]
    floor.pivots = [e.pivot for e in pivots.events]
    assert floor.pivots, "constructed supported rotation must reach the ground detector"
    return source, floor
