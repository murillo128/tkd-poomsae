"""Immutable automatic segmentation from existing feature artifacts only."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from contracts.models import DenseArray, MotionFeatures, Provenance, Segmentation
from reconstruction.features import load_feature_evidence
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import REVISION, SegmentationConfig, SegmentationResult, segment_execution

ARRAY_ID = "segmentation_evidence_json"


def publish_segmentation(
    store: ArtifactStore,
    features: ArtifactHandle,
    config: SegmentationConfig | None = None,
) -> ArtifactHandle:
    """No vision, reconstruction, ground producer, labels or manual edits run."""
    source = features.metadata
    if not isinstance(source, MotionFeatures):
        raise ValueError("persisted physical motion features required")
    evidence = load_feature_evidence(features)
    config = config or SegmentationConfig()
    revision = hash_file(features.path / "manifest.json")
    digest = hash_config(config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="segmentation",
        inputs={"motion_features": revision},
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )

    def produce() -> tuple[Segmentation, dict[str, np.ndarray[Any, Any]]]:
        result = segment_execution(evidence, config)
        array = np.frombuffer(
            json.dumps(
                {
                    "input_revisions": {"motion_features": revision},
                    "segmentation": result.model_dump(mode="json"),
                },
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
            dtype=np.uint8,
        )
        output = Segmentation(
            kind="segmentation",
            id=f"segmentation:{key.digest}",
            schema_version="1.0.0",
            provenance=Provenance(
                producer="reconstruction.segmentation",
                model=REVISION,
                config_digest=digest,
            ),
            reconstruction_id=source.reconstruction_id,
            ground_id=source.ground_id,
            motion_features_id=source.id,
            execution=result.execution,
            steps=result.steps,
            quality=result.quality,
            arrays=[
                DenseArray(
                    id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
                )
            ],
        )
        return output, {ARRAY_ID: array}

    return store.get_or_create(key, produce)


def load_segmentation(handle: ArtifactHandle) -> SegmentationResult:
    metadata = handle.metadata
    if (
        not isinstance(metadata, Segmentation)
        or metadata.provenance.producer != "reconstruction.segmentation"
    ):
        raise ValueError("automatic segmentation artifact required")
    payload = json.loads(handle.read_array(ARRAY_ID).tobytes())
    result = SegmentationResult.model_validate(payload["segmentation"])
    if (
        result.reconstruction_id,
        result.ground_id,
        result.execution,
        result.steps,
        result.quality,
        result.algorithm_revision,
        hash_config(result.config.model_dump(mode="json")),
    ) != (
        metadata.reconstruction_id,
        metadata.ground_id,
        metadata.execution,
        metadata.steps,
        metadata.quality,
        metadata.provenance.model,
        metadata.provenance.config_digest,
    ):
        raise ValueError("segmentation evidence disagrees with canonical artifact")
    key = ArtifactKey(
        layer="segmentation",
        inputs=payload["input_revisions"],
        schema_version=metadata.schema_version,
        algorithm_revision=result.algorithm_revision,
        config_digest=metadata.provenance.config_digest,
    )
    if metadata.id != f"segmentation:{key.digest}" or handle.path.name != key.digest:
        raise ValueError("segmentation input revision disagrees with artifact identity")
    return result
