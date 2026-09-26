"""Immutable placement publication with dense source-trajectory evidence."""

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
from reconstruction.ground import load_contact_evidence
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import REVISION, FootprintConfig, FootprintSeries, derive_footprints

ARRAY_ID = "footprint_evidence_json"


def publish_footprints(
    store: ArtifactStore,
    motion: ArtifactHandle,
    contacts: ArtifactHandle,
    calibration: ArtifactHandle,
    morphology: ArtifactHandle | None = None,
    config: FootprintConfig | None = None,
) -> ArtifactHandle:
    source, ground, cal = motion.metadata, contacts.metadata, calibration.metadata
    shape = morphology.metadata if morphology else None
    if not isinstance(source, Reconstruction) or not isinstance(ground, Ground):
        raise ValueError("motion and contact artifacts required")
    if not isinstance(cal, Calibration):
        raise ValueError("calibration artifact required")
    if shape is not None and not isinstance(shape, Morphology):
        raise ValueError("morphology artifact required")
    evidence = load_contact_evidence(contacts)
    if evidence.calibration_id != cal.id:
        raise ValueError("contact calibration identity mismatch")
    expected = {
        "motion": hash_file(motion.path / "manifest.json"),
        "calibration": hash_file(calibration.path / "manifest.json"),
    }
    if morphology:
        expected["morphology"] = hash_file(morphology.path / "manifest.json")
    upstream = json.loads(contacts.read_array("contact_evidence_json").tobytes())
    # Compare consumed motion/calibration bytes, not merely reusable model IDs.
    if any(
        upstream["input_revisions"].get(k) != v
        for k, v in expected.items()
        if k != "morphology"
    ):
        raise ValueError("contact inputs disagree with supplied artifact revisions")
    revisions = {**expected, "contacts": hash_file(contacts.path / "manifest.json")}
    config = config or FootprintConfig()
    result = derive_footprints(source, ground, cal, shape, config)
    if not result.trajectory:
        raise ValueError("nonempty reconstructed motion required")
    digest = hash_config(config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="ground",
        inputs=revisions,
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )

    def produce() -> tuple[Ground, dict[str, np.ndarray[Any, Any]]]:
        array = np.frombuffer(
            json.dumps(
                {
                    "input_revisions": revisions,
                    "placements": result.model_dump(mode="json"),
                },
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
            dtype=np.uint8,
        )
        output = ground.model_copy(deep=True)
        output.id = f"ground-footprints:{key.digest}"
        output.provenance = Provenance(
            producer="reconstruction.footprints", model=REVISION, config_digest=digest
        )
        output.footprints = [event.footprint for event in result.events]
        # Relations remain paired and explicitly framed in the dense payload.
        output.arrays.append(
            DenseArray(
                id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
            )
        )
        arrays = {a.id: contacts.read_array(a.id) for a in ground.arrays}
        arrays[ARRAY_ID] = array
        return output, arrays

    return store.get_or_create(key, produce)


def load_footprint_evidence(handle: ArtifactHandle) -> FootprintSeries:
    ground = handle.metadata
    if (
        not isinstance(ground, Ground)
        or ground.provenance.producer != "reconstruction.footprints"
    ):
        raise ValueError("physical placement artifact required")
    payload = json.loads(handle.read_array(ARRAY_ID).tobytes())
    result = FootprintSeries.model_validate(payload["placements"])
    if (
        result.reconstruction_id != ground.reconstruction_id
        or [event.footprint for event in result.events] != ground.footprints
        or result.algorithm_revision != ground.provenance.model
        or hash_config(result.config.model_dump(mode="json"))
        != ground.provenance.config_digest
        or result.world_unit != ("m" if ground.scale == "metric" else "arbitrary")
        or len(result.trajectory) != 2 * len(ground.samples)
    ):
        raise ValueError("placement evidence disagrees with canonical ground artifact")
    for i, sample in enumerate(ground.samples):
        for offset, side in enumerate(("left", "right")):
            row = result.trajectory[2 * i + offset]
            if (
                row.global_seconds != sample.global_seconds
                or row.foot != side
                or (
                    row.contact != (sample.left if side == "left" else sample.right)
                    or row.support != sample.support
                    or row.motion_sample_index != i
                    or row.contact_sample_index != i
                )
            ):
                raise ValueError("placement trajectory disagrees with contact samples")
    return result
