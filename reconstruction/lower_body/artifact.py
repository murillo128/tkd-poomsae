"""Immutable lower-body candidates from existing artifacts; no upstream runs."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from contracts.models import (
    DenseArray,
    Ground,
    LowerBodyParsing,
    Provenance,
    Segmentation,
)
from reconstruction.features import load_feature_evidence
from reconstruction.segmentation import load_segmentation
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import REVISION, LowerBodyConfig, LowerBodyResult, parse_lower_body

ARRAY_ID = "lower_body_evidence_json"


def publish_lower_body(
    store: ArtifactStore,
    features: ArtifactHandle,
    ground: ArtifactHandle,
    segmentation: ArtifactHandle,
    config: LowerBodyConfig | None = None,
) -> ArtifactHandle:
    floor, proposal = ground.metadata, segmentation.metadata
    if not isinstance(floor, Ground) or not isinstance(proposal, Segmentation):
        raise ValueError("persisted ground and segmentation required")
    physical = load_feature_evidence(features)
    coarse = load_segmentation(segmentation)
    if proposal.motion_features_id != features.metadata.id:
        raise ValueError("segmentation must reference exact feature artifact")
    config = config or LowerBodyConfig()
    revisions = {
        "motion_features": hash_file(features.path / "manifest.json"),
        "ground": hash_file(ground.path / "manifest.json"),
        "segmentation": hash_file(segmentation.path / "manifest.json"),
    }
    digest = hash_config(config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="lower_body_parsing",
        inputs=revisions,
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )

    def produce() -> tuple[LowerBodyParsing, dict[str, np.ndarray[Any, Any]]]:
        result = parse_lower_body(physical, floor, coarse, config)
        array = np.frombuffer(
            json.dumps(
                {
                    "input_revisions": revisions,
                    "lower_body": result.model_dump(mode="json"),
                },
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
            dtype=np.uint8,
        )
        metadata = LowerBodyParsing(
            kind="lower_body_parsing",
            id=f"lower-body:{key.digest}",
            schema_version="1.0.0",
            provenance=Provenance(
                producer="reconstruction.lower_body",
                model=REVISION,
                config_digest=digest,
            ),
            reconstruction_id=physical.reconstruction_id,
            ground_id=floor.id,
            motion_features_id=features.metadata.id,
            segmentation_id=proposal.id,
            arrays=[
                DenseArray(
                    id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
                )
            ],
        )
        return metadata, {ARRAY_ID: array}

    return store.get_or_create(key, produce)


def load_lower_body(handle: ArtifactHandle) -> LowerBodyResult:
    metadata = handle.metadata
    if not isinstance(metadata, LowerBodyParsing) or (
        metadata.provenance.producer != "reconstruction.lower_body"
    ):
        raise ValueError("automatic lower-body artifact required")
    result = LowerBodyResult.model_validate(
        json.loads(handle.read_array(ARRAY_ID).tobytes())["lower_body"]
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
        raise ValueError("lower-body evidence disagrees with canonical metadata")
    return result
