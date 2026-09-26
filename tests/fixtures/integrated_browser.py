"""Known synthetic geometry and semantics; never a model inference result."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contracts.models import (
    Action,
    DenseArray,
    Interval,
    Keyframe,
    Landmark3D,
    Provenance,
    Quality,
    Semantics,
    SequenceStep,
    Track,
)
from reconstruction.footprints import publish_footprints
from reconstruction.ground import publish_contacts
from reconstruction.ground_view import publish_ground_view
from reconstruction.pivots import publish_pivots
from reconstruction.semantics import AssemblyConfig
from reconstruction.semantics.core import REVISION
from storage import ArtifactKey, hash_config, hash_file
from tests.test_ground_contact import calibration
from tests.test_inspection_api import persist
from tests.test_pivots import sequence
from tkd_poomsae.inspection import Inspection


def register_integrated(
    index: Inspection,
    project: str,
    sync_key: ArtifactKey,
    observations: list[ArtifactKey],
    sources: dict[str, Path],
) -> None:
    """Bind generated media to positive native motion, ground and event evidence."""
    store = index.pipe.store
    sync = store.get(sync_key)
    source, _ = sequence()
    source.id = f"synthetic-motion-{project}"
    cal = calibration()
    cal.cameras = cal.cameras[:2]
    cal.provenance.producer = "generated-browser-fixture"
    for camera, (name, path) in zip(cal.cameras, list(sources.items())[:2]):
        camera.camera_id, camera.source_id = name, "source:" + hash_file(path)
    cal_key, cal_handle = persist(store, cal)
    for i, sample in enumerate(source.samples):
        # Native media ordinals are deliberately independent of the 50 Hz motion.
        ordinal = round(sample.global_seconds * 25)
        ids = [f"{project}-front-{ordinal}", f"synthetic-native-{project}-{i}"]
        quality = Quality(
            state="observed", score=0.9, uncertainty=0.00001, source_ids=ids
        )
        sample.quality = quality
        for point in sample.landmarks:
            point.quality = quality
        sample.landmarks.append(
            Landmark3D(
                name="left_wrist",
                xyz_world=(sample.global_seconds, 0, 1.2),
                quality=quality,
            )
        )
    motion_key, motion = persist(
        store,
        source,
        inputs={name: hash_file(path) for name, path in sources.items()},
        sync_revision=hash_file(sync.path / "manifest.json"),
    )
    contacts = publish_contacts(store, motion, cal_handle)
    placements = publish_footprints(store, motion, contacts, cal_handle)
    pivots = publish_pivots(store, placements)
    ground = publish_ground_view(store, motion, cal_handle, pivots)
    config = AssemblyConfig().model_dump(mode="json")
    inputs = {"fixture": hash_file(motion.path / "manifest.json")}
    digest = hash_config(config)
    semantic_key = ArtifactKey(
        layer="semantics",
        inputs=inputs,
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )
    quality = Quality(state="inferred", score=0.8, source_ids=[source.id])
    intervals = [Interval(start=0.2, end=0.8), Interval(start=0.4, end=0.9)]
    tracks: list[Track] = ["left_arm", "right_arm"]
    actions = [
        Action(
            id=f"synthetic-arm-{i}",
            interval=interval,
            step_id="synthetic-step",
            tracks=[track],
            category="arm",
            quality=quality,
        )
        for i, (interval, track) in enumerate(zip(intervals, tracks))
    ]
    evidence = np.frombuffer(
        json.dumps(
            {
                "version": 1,
                "input_revisions": inputs,
                "config": config,
                "motion_times": [s.global_seconds for s in source.samples],
                "events": [],
            }
        ).encode(),
        dtype=np.uint8,
    )
    semantic = Semantics(
        kind="semantics",
        id=f"semantics:{semantic_key.digest}",
        schema_version="1.0.0",
        provenance=Provenance(
            producer="reconstruction.semantics", model=REVISION, config_digest=digest
        ),
        reconstruction_id=source.id,
        ground_id=ground.metadata.id,
        execution=Interval(start=0, end=1),
        motion_sample_indices=list(range(51)),
        steps=[
            SequenceStep(
                id="synthetic-step",
                interval=Interval(start=0, end=1),
                action_ids=[a.id for a in actions],
                motion_sample_indices=list(range(51)),
            )
        ],
        actions=actions,
        phases=[],
        stances=[],
        relations=[],
        quality=quality,
        keyframes=[
            Keyframe(
                id="synthetic-event",
                action_id=actions[0].id,
                global_seconds=0.4,
                event="inspection",
                track="left_arm",
                quality=quality,
                motion_sample_indices=[20],
            )
        ],
        arrays=[
            DenseArray(
                id="semantic_evidence_json",
                dtype="uint8",
                shape=[len(evidence)],
                axes=["json_byte"],
            )
        ],
    )
    store.get_or_create(
        semantic_key, lambda: (semantic, {"semantic_evidence_json": evidence})
    )
    index.register(
        project,
        {
            "sync": sync_key,
            "calibration": cal_key,
            "reconstruction": motion_key,
            "ground": ArtifactKey(
                layer="ground",
                inputs={
                    "motion": hash_file(motion.path / "manifest.json"),
                    "calibration": hash_file(cal_handle.path / "manifest.json"),
                    "pivots": hash_file(pivots.path / "manifest.json"),
                },
                schema_version="1.0.0",
                algorithm_revision="ground-view-v2",
                config_digest=ground.metadata.provenance.config_digest,
            ),
            "semantics": semantic_key,
        },
        observations=observations,
    )
