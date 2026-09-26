"""Immutable physical contact publication with exact input lineage."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from contracts.models import (
    Calibration,
    DenseArray,
    Ground,
    Morphology,
    Provenance,
    Reconstruction,
)
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import REVISION, ContactConfig, ContactSeries, derive_contacts

ARRAY_ID = "contact_evidence_json"


def publish_contacts(
    store: ArtifactStore,
    motion: ArtifactHandle,
    calibration: ArtifactHandle,
    morphology: ArtifactHandle | None = None,
    config: ContactConfig | None = None,
) -> ArtifactHandle:
    """Derive from final temporal motion, preserving physical/semantic separation."""
    source, calibrated = motion.metadata, calibration.metadata
    shape = morphology.metadata if morphology else None
    if not isinstance(source, Reconstruction) or (
        source.provenance.producer != "reconstruction.temporal"
    ):
        raise ValueError("final temporal reconstruction required")
    if not isinstance(calibrated, Calibration):
        raise ValueError("calibration artifact required")
    if shape is not None and not isinstance(shape, Morphology):
        raise ValueError("morphology artifact required")
    # Canonical Ground requires resolved ground. Pure derivation exposes explicit
    # unavailable diagnostics; publication fails rather than creating invalid graphs.
    if calibrated.ground_status != "resolved":
        raise ValueError("unresolved ground: contact publication unavailable")
    config = config or ContactConfig()
    result = derive_contacts(source, calibrated, shape, config)
    if not result.samples:
        raise ValueError("nonempty reconstructed motion required")
    revisions = {
        "motion": hash_file(motion.path / "manifest.json"),
        "calibration": hash_file(calibration.path / "manifest.json"),
    }
    if morphology:
        revisions["morphology"] = hash_file(morphology.path / "manifest.json")
    digest = hash_config(config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="ground",
        inputs=revisions,
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )

    def produce() -> tuple[Ground, dict[str, np.ndarray[Any, Any]]]:
        payload = {
            "input_revisions": revisions,
            "contacts": result.model_dump(mode="json"),
        }
        array = np.frombuffer(
            json.dumps(
                payload, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode(),
            dtype=np.uint8,
        )
        output = Ground(
            kind="ground",
            id=f"ground-contact:{key.digest}",
            schema_version="1.0.0",
            provenance=Provenance(
                producer="reconstruction.ground", model=REVISION, config_digest=digest
            ),
            reconstruction_id=source.id,
            scale=source.scale,
            samples=result.samples,
            arrays=[
                DenseArray(
                    id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
                )
            ],
        )
        return output, {ARRAY_ID: array}

    return store.get_or_create(key, produce)


def load_contact_evidence(handle: ArtifactHandle) -> ContactSeries:
    metadata = handle.metadata
    if not isinstance(metadata, Ground) or (
        metadata.provenance.producer != "reconstruction.ground"
    ):
        raise ValueError("physical contact artifact required")
    payload = json.loads(handle.read_array(ARRAY_ID).tobytes())
    result = ContactSeries.model_validate(payload["contacts"])
    if result.reconstruction_id != metadata.reconstruction_id or (
        result.samples != metadata.samples
        or hash_config(result.config.model_dump(mode="json"))
        != metadata.provenance.config_digest
        or result.algorithm_revision != metadata.provenance.model
    ):
        raise ValueError("contact evidence disagrees with canonical ground artifact")
    return result
