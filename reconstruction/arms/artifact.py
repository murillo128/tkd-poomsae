"""Immutable action candidates from persisted offline inputs."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from contracts.models import (
    ArmActions,
    DenseArray,
    Provenance,
    Reconstruction,
    Segmentation,
)
from reconstruction.detailed import DetailedSample, load_detailed_geometry
from reconstruction.features import load_feature_evidence
from reconstruction.segmentation import load_segmentation
from reconstruction.temporal import load_temporal_motion
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import REVISION, ArmConfig, ArmResult, parse_arm_actions

ARRAY_ID = "arm_action_evidence_json"


def publish_arm_actions(
    store: ArtifactStore,
    features: ArtifactHandle,
    segmentation: ArtifactHandle,
    motion: ArtifactHandle,
    config: ArmConfig | None = None,
) -> ArtifactHandle:
    """Load final motion/features/proposals; never run any upstream producer."""
    physical = load_feature_evidence(features)
    coarse = load_segmentation(segmentation)
    source, proposals = motion.metadata, segmentation.metadata
    if not isinstance(source, Reconstruction) or not isinstance(
        proposals, Segmentation
    ):
        raise ValueError("reconstructed motion and coarse proposals required")
    if (
        source.id != physical.reconstruction_id
        or proposals.motion_features_id != features.metadata.id
    ):
        raise ValueError("exact motion/features/segmentation input identity required")
    if source.provenance.producer == "reconstruction.temporal":
        geometry = [
            DetailedSample.model_validate(d)
            for d in load_temporal_motion(motion)["detailed"]
        ]
    elif source.provenance.producer == "reconstruction.detailed":
        geometry = load_detailed_geometry(motion)
        # The detailed publisher retains original source identity in its payload.
        geometry = [
            g.model_copy(update={"reconstruction_id": source.id}) for g in geometry
        ]
    else:
        raise ValueError("persisted detailed or temporal hand geometry required")
    config = config or ArmConfig()
    revisions = {
        name: hash_file(handle.path / "manifest.json")
        for name, handle in (
            ("features", features),
            ("segmentation", segmentation),
            ("motion", motion),
        )
    }
    digest = hash_config(config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="arm_actions",
        inputs=revisions,
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )

    def produce() -> tuple[ArmActions, dict[str, np.ndarray[Any, Any]]]:
        result = parse_arm_actions(physical, coarse, geometry, config)
        array = np.frombuffer(
            json.dumps(
                {
                    "input_revisions": revisions,
                    "arm_actions": result.model_dump(mode="json"),
                },
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
            dtype=np.uint8,
        )
        metadata = ArmActions(
            kind="arm_actions",
            id=f"arm-actions:{key.digest}",
            schema_version="1.0.0",
            provenance=Provenance(
                producer="reconstruction.arms", model=REVISION, config_digest=digest
            ),
            reconstruction_id=source.id,
            ground_id=physical.ground_id,
            motion_features_id=features.metadata.id,
            segmentation_id=proposals.id,
            arrays=[
                DenseArray(
                    id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
                )
            ],
        )
        return metadata, {ARRAY_ID: array}

    return store.get_or_create(key, produce)


def load_arm_actions(handle: ArtifactHandle) -> ArmResult:
    """Reload without reconstruction, feature extraction or automatic parsing."""
    metadata = handle.metadata
    if (
        not isinstance(metadata, ArmActions)
        or metadata.provenance.producer != "reconstruction.arms"
    ):
        raise ValueError("automatic arm action artifact required")
    payload = json.loads(handle.read_array(ARRAY_ID).tobytes())
    result = ArmResult.model_validate(payload["arm_actions"])
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
        raise ValueError("arm action evidence disagrees with canonical artifact")
    key = ArtifactKey(
        layer="arm_actions",
        inputs=payload["input_revisions"],
        schema_version=metadata.schema_version,
        algorithm_revision=result.algorithm_revision,
        config_digest=metadata.provenance.config_digest,
    )
    if metadata.id != f"arm-actions:{key.digest}" or handle.path.name != key.digest:
        raise ValueError("arm action input revision disagrees with artifact identity")
    return result
