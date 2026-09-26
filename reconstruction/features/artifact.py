"""Independent content-addressed feature publication from persisted products."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from contracts.models import (
    DenseArray,
    Ground,
    MotionFeatures,
    Provenance,
    Quality,
    Reconstruction,
)
from reconstruction.detailed import DetailedSample
from reconstruction.footprints import load_footprint_evidence
from reconstruction.pivots import load_pivot_evidence
from reconstruction.temporal import load_temporal_motion
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import REVISION, FeatureConfig, FeatureSeries, derive_features

ARRAY_ID = "motion_features_json"


def publish_features(
    store: ArtifactStore,
    motion: ArtifactHandle,
    ground: ArtifactHandle,
    config: FeatureConfig | None = None,
) -> ArtifactHandle:
    """Read existing final motion/ground only; no upstream producer is called."""
    source, floor = motion.metadata, ground.metadata
    if not isinstance(source, Reconstruction) or not isinstance(floor, Ground):
        raise ValueError("persisted reconstruction and ground required")
    payload = load_temporal_motion(motion)
    config = config or FeatureConfig()
    revisions = {
        "motion": hash_file(motion.path / "manifest.json"),
        "ground": hash_file(ground.path / "manifest.json"),
    }
    digest = hash_config(config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="motion_features",
        inputs=revisions,
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )

    def produce() -> tuple[MotionFeatures, dict[str, np.ndarray[Any, Any]]]:
        placements = None
        if floor.provenance.producer == "reconstruction.pivots":
            placements = load_pivot_evidence(ground).placements
        elif floor.provenance.producer == "reconstruction.footprints":
            placements = load_footprint_evidence(ground)
        result = derive_features(
            source,
            floor,
            config,
            geometry=[DetailedSample.model_validate(d) for d in payload["detailed"]],
            root_position_quality=[
                Quality.model_validate(q) for q in payload["root_translation_quality"]
            ],
            root_orientation_quality=[
                Quality.model_validate(q) for q in payload["root_orientation_quality"]
            ],
            root_orientation_from_body=[
                not supplied for supplied in payload["supplied_root"]
            ],
            placements=placements,
        )
        array = np.frombuffer(
            json.dumps(
                {
                    "input_revisions": revisions,
                    "features": result.model_dump(mode="json"),
                },
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
            dtype=np.uint8,
        )
        output = MotionFeatures(
            kind="motion_features",
            id=f"motion-features:{key.digest}",
            schema_version="1.0.0",
            provenance=Provenance(
                producer="reconstruction.features", model=REVISION, config_digest=digest
            ),
            reconstruction_id=source.id,
            ground_id=floor.id,
            arrays=[
                DenseArray(
                    id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
                )
            ],
        )
        return output, {ARRAY_ID: array}

    return store.get_or_create(key, produce)


def load_feature_evidence(handle: ArtifactHandle) -> FeatureSeries:
    metadata = handle.metadata
    if (
        not isinstance(metadata, MotionFeatures)
        or metadata.provenance.producer != "reconstruction.features"
    ):
        raise ValueError("physical motion feature artifact required")
    result = FeatureSeries.model_validate(
        json.loads(handle.read_array(ARRAY_ID).tobytes())["features"]
    )
    if (
        result.reconstruction_id,
        result.ground_id,
        result.algorithm_revision,
        hash_config(result.config.model_dump(mode="json")),
    ) != (
        metadata.reconstruction_id,
        metadata.ground_id,
        metadata.provenance.model,
        metadata.provenance.config_digest,
    ):
        raise ValueError("feature evidence disagrees with canonical artifact")
    return result
