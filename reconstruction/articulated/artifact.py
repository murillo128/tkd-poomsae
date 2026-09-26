"""Separate immutable morphology/evidence and fitted-pose publications."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.models import DenseArray, Morphology, Provenance, Reconstruction
from reconstruction.triangulation.artifact import (
    load_diagnostics as load_raw_diagnostics,
)
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import REVISION, FitConfig, estimate_morphology, fit_sample

MORPHOLOGY_ARRAY = "articulated_morphology_evidence_json"
POSE_ARRAY = "articulated_fit_diagnostics_json"
ASSUMPTIONS = [
    "Explicit labeled skeletal model; no population proportions, symmetry or mesh.",
    "Morphology uses direct observed endpoints only, median/MAD outlier rejection, "
    "minimum sample count/time span and relative uncertainty/dispersion gates.",
    "Repeated samples may share systematic errors; uncertainty is not reduced by N.",
    "Pose fits weighted data and constant lengths within per-joint "
    "displacement bounds; "
    "no contact, joint-angle, temporal or gap-filling prior.",
    "Adjusted points and sequence shape estimates are inferred, never measurements. "
    "Uncertainty floors are conservative diagnostics, not calibrated accuracy.",
    "Full orientations require sufficiently precise noncollinear labeled triplets; "
    "two endpoints constrain direction but leave axial twist unknown.",
    "Root and derived segment orientations are active local-to-world quaternions; "
    "Orientation uncertainty is radians; "
    "translation/length uncertainty is world units.",
    "Refined geometry, source evidence, supplied root/segment orientations and "
    "all source payloads survive; raw triangulation remains immutable.",
]


@dataclass(frozen=True)
class ArticulatedFit:
    morphology: ArtifactHandle
    pose: ArtifactHandle


def _array(payload: dict[str, Any]) -> np.ndarray[Any, Any]:
    return np.frombuffer(
        json.dumps(
            payload, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode(),
        dtype=np.uint8,
    )


def _descriptor(name: str, array: np.ndarray[Any, Any]) -> DenseArray:
    return DenseArray(id=name, dtype="uint8", shape=[len(array)], axes=["json_byte"])


def publish_articulated_fit(
    store: ArtifactStore,
    raw: ArtifactHandle,
    config: FitConfig | None = None,
) -> ArticulatedFit:
    """Fit only an identified raw triangulation artifact, without modifying it."""
    config = config or FitConfig()
    source = raw.metadata
    if (
        not isinstance(source, Reconstruction)
        or not source.samples
        or any(a.id == POSE_ARRAY for a in source.arrays)
    ):
        raise ValueError("nonempty raw reconstruction required")
    raw_evidence = load_raw_diagnostics(raw)
    if (
        raw_evidence.get("world_unit")
        != ("m" if source.scale == "metric" else "arbitrary")
        or raw_evidence.get("calibration_id") != source.calibration_id
    ):
        raise ValueError(
            "raw diagnostic units/calibration disagree with reconstruction"
        )
    revision = hash_file(raw.path / "manifest.json")
    config_digest = hash_config(config.model_dump(mode="json"))
    morphology_key = ArtifactKey(
        layer="morphology",
        inputs={"raw_reconstruction": revision},
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=config_digest,
    )

    def produce_morphology() -> tuple[Morphology, dict[str, np.ndarray[Any, Any]]]:
        morphology, evidence = estimate_morphology(
            source,
            config,
            identifier=f"articulated-morphology:{morphology_key.digest}",
            config_digest=config_digest,
        )
        array = _array(
            {
                "version": 1,
                "artifact_role": "participant_morphology_evidence",
                "raw_reconstruction_id": source.id,
                "raw_revision": revision,
                "world_unit": raw_evidence["world_unit"],
                "settings": config.model_dump(mode="json"),
                "assumptions": ASSUMPTIONS,
                "measurements": evidence,
            }
        )
        morphology.arrays = [_descriptor(MORPHOLOGY_ARRAY, array)]
        return morphology, {MORPHOLOGY_ARRAY: array}

    morphology_handle = store.get_or_create(morphology_key, produce_morphology)
    morphology = morphology_handle.metadata
    if not isinstance(morphology, Morphology):
        raise ValueError("not a morphology artifact")
    pose_key = ArtifactKey(
        layer="reconstruction",
        inputs={
            "raw_reconstruction": revision,
            "morphology": hash_file(morphology_handle.path / "manifest.json"),
        },
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=config_digest,
    )

    def produce_pose() -> tuple[Reconstruction, dict[str, np.ndarray[Any, Any]]]:
        pairs = [fit_sample(s, morphology, config) for s in source.samples]
        array = _array(
            {
                "version": 1,
                "artifact_role": "participant_articulated_pose",
                "raw_reconstruction_id": source.id,
                "raw_revision": revision,
                "morphology_id": morphology.id,
                "world_unit": raw_evidence["world_unit"],
                "settings": config.model_dump(mode="json"),
                "assumptions": ASSUMPTIONS,
                "samples": [d for _, d in pairs],
            }
        )
        # Copy every source payload (including per-camera triangulation lineage).
        arrays = {a.id: raw.read_array(a.id) for a in source.arrays}
        arrays[POSE_ARRAY] = array
        fitted = source.model_copy(deep=True)
        fitted.id = f"articulated-pose:{pose_key.digest}"
        fitted.provenance = Provenance(
            producer="reconstruction.articulated",
            model=REVISION,
            config_digest=config_digest,
        )
        fitted.samples = [s for s, _ in pairs]
        fitted.arrays.append(_descriptor(POSE_ARRAY, array))
        return fitted, arrays

    return ArticulatedFit(
        morphology_handle, store.get_or_create(pose_key, produce_pose)
    )


def load_morphology_evidence(handle: ArtifactHandle) -> dict[str, Any]:
    if not isinstance(handle.metadata, Morphology):
        raise ValueError("not a morphology artifact")
    return _load(handle, MORPHOLOGY_ARRAY, "participant_morphology_evidence")


def load_fit_diagnostics(handle: ArtifactHandle) -> dict[str, Any]:
    if not isinstance(handle.metadata, Reconstruction):
        raise ValueError("not a reconstruction artifact")
    return _load(handle, POSE_ARRAY, "participant_articulated_pose")


def _load(handle: ArtifactHandle, name: str, role: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(handle.read_array(name).tobytes())
    if payload.get("version") != 1 or payload.get("artifact_role") != role:
        raise ValueError("unsupported articulated payload")
    return payload
