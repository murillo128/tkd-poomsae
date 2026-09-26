"""Separately rerunnable semantic assembly from immutable persisted inputs."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from contracts.models import (
    Action,
    ArmActions,
    DenseArray,
    LowerBodyParsing,
    MotionFeatures,
    Phase,
    Segmentation,
    Semantics,
)
from reconstruction.arms import load_arm_actions
from reconstruction.features import load_feature_evidence
from reconstruction.lower_body import load_lower_body
from reconstruction.segmentation import load_segmentation
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import REVISION, AssemblyConfig, assemble_semantics

ARRAY_ID = "semantic_evidence_json"


def publish_semantics(
    store: ArtifactStore,
    features: ArtifactHandle,
    segmentation: ArtifactHandle,
    arms: ArtifactHandle,
    lower_body: ArtifactHandle,
    config: AssemblyConfig | None = None,
) -> ArtifactHandle:
    """Read existing parser products; never execute an upstream stage."""
    handles = {
        "features": features,
        "segmentation": segmentation,
        "arms": arms,
        "lower_body": lower_body,
    }
    physical, coarse, upper, lower = (
        features.metadata,
        segmentation.metadata,
        arms.metadata,
        lower_body.metadata,
    )
    if not (
        isinstance(physical, MotionFeatures)
        and isinstance(coarse, Segmentation)
        and isinstance(upper, ArmActions)
        and isinstance(lower, LowerBodyParsing)
    ):
        raise ValueError(
            "persisted features, segmentation and action proposals required"
        )
    lineage = (physical.reconstruction_id, physical.ground_id)
    for source in (coarse, upper, lower):
        if (source.reconstruction_id, source.ground_id) != lineage:
            raise ValueError("semantic input lineage disagrees")
        if source.motion_features_id != physical.id:
            raise ValueError("exact feature identity required")
    if upper.segmentation_id != coarse.id or lower.segmentation_id != coarse.id:
        raise ValueError("exact segmentation identity required")
    config = config or AssemblyConfig()
    revisions = {
        name: hash_file(h.path / "manifest.json") for name, h in handles.items()
    }
    digest = hash_config(config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="semantics",
        inputs=revisions,
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )

    def produce() -> tuple[Semantics, dict[str, np.ndarray[Any, Any]]]:
        feature_data = load_feature_evidence(features)
        coarse_data = load_segmentation(segmentation)
        arm_data = load_arm_actions(arms)
        lower_data = load_lower_body(lower_body)
        semantics = assemble_semantics(
            feature_data, coarse_data, arm_data, lower_data, config
        )
        semantics.id = f"semantics:{key.digest}"
        semantics.motion_features_id = physical.id
        semantics.segmentation_id = coarse.id
        semantics.arm_actions_id = upper.id
        semantics.lower_body_parsing_id = lower.id
        # Full native evidence retains dynamics, unknown geometry, pre/post-roll,
        # ground links, reasons and source candidates outside the execution.
        payload = {
            "version": 1,
            "input_revisions": revisions,
            "config": config.model_dump(mode="json"),
            "motion_times": [r.global_seconds for r in feature_data.trajectory[::6]],
            "coarse": coarse_data.model_dump(mode="json"),
            "arms": arm_data.model_dump(mode="json"),
            "lower_body": lower_data.model_dump(mode="json"),
            "events": [e.model_dump(mode="json") for e in feature_data.events],
        }
        array = np.frombuffer(
            json.dumps(
                payload, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode(),
            dtype=np.uint8,
        )
        semantics.arrays = [
            DenseArray(
                id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
            )
        ]
        return Semantics.model_validate(semantics.model_dump()), {ARRAY_ID: array}

    return store.get_or_create(key, produce)


def load_semantics(handle: ArtifactHandle) -> Semantics:
    metadata = handle.metadata
    if (
        not isinstance(metadata, Semantics)
        or metadata.provenance.producer != "reconstruction.semantics"
    ):
        raise ValueError("automatic semantic assembly artifact required")
    payload = load_semantic_evidence(handle)
    if payload.get("version") != 1:
        raise ValueError("unsupported semantic evidence version")
    config = AssemblyConfig.model_validate(payload["config"])
    digest = hash_config(config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="semantics",
        inputs=payload["input_revisions"],
        schema_version=metadata.schema_version,
        algorithm_revision=REVISION,
        config_digest=digest,
    )
    if (
        metadata.id != f"semantics:{key.digest}"
        or handle.path.name != key.digest
        or metadata.provenance.config_digest != digest
        or metadata.provenance.model != REVISION
    ):
        raise ValueError("semantic evidence disagrees with artifact identity")
    times = payload["motion_times"]
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("increasing native semantic clock required")
    for interval, indices in [
        (metadata.execution, metadata.motion_sample_indices),
        *[(s.interval, s.motion_sample_indices) for s in metadata.steps],
        *[(s.interval, s.motion_sample_indices) for s in metadata.stances],
    ]:
        if interval is None:
            if indices:
                raise ValueError("indeterminate execution cannot carry interval links")
            continue
        if (
            not indices
            or indices[0] < 0
            or indices[-1] >= len(times)
            or indices != list(range(indices[0], indices[-1] + 1))
            or (interval.start, interval.end) != (times[indices[0]], times[indices[-1]])
        ):
            raise ValueError("semantic interval references disagree with native clock")
    items: list[Action | Phase] = [*metadata.actions, *metadata.phases]
    for item in items:
        for link in item.motion_links:
            indices = link.motion_sample_indices
            if (
                indices[-1] >= len(times)
                or link.interval.start != times[indices[0]]
                or link.interval.end != times[indices[-1]]
            ):
                raise ValueError("semantic dense references disagree with native clock")
    for frame in metadata.keyframes:
        if any(
            i < 0 or i >= len(times) for i in frame.motion_sample_indices
        ) or frame.global_seconds not in [
            times[i] for i in frame.motion_sample_indices
        ]:
            raise ValueError("semantic event references disagree with native clock")
    return metadata


def load_semantic_evidence(handle: ArtifactHandle) -> dict[str, Any]:
    if not isinstance(handle.metadata, Semantics):
        raise ValueError("semantic artifact required")
    return dict(json.loads(handle.read_array(ARRAY_ID).tobytes()))
