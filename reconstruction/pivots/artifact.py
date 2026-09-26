"""Immutable pivots preserving upstream placement/contact payloads."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from contracts.models import DenseArray, Ground, Provenance
from reconstruction.footprints import load_footprint_evidence
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import REVISION, PivotConfig, PivotSeries, derive_pivots

ARRAY_ID = "pivot_evidence_json"


def publish_pivots(
    store: ArtifactStore,
    placements: ArtifactHandle,
    config: PivotConfig | None = None,
) -> ArtifactHandle:
    """Consume exact final placement geometry; no inference or data downloads."""
    source = placements.metadata
    if not isinstance(source, Ground):
        raise ValueError("physical placement artifact required")
    evidence = load_footprint_evidence(placements)
    config = config or PivotConfig()
    result = derive_pivots(evidence, config)
    revision = hash_file(placements.path / "manifest.json")
    digest = hash_config(config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="ground",
        inputs={"placements": revision},
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )

    def produce() -> tuple[Ground, dict[str, np.ndarray[Any, Any]]]:
        array = np.frombuffer(
            json.dumps(
                {
                    "input_revisions": {"placements": revision},
                    "pivots": result.model_dump(mode="json"),
                },
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
            dtype=np.uint8,
        )
        output = source.model_copy(deep=True)
        output.id = f"ground-pivots:{key.digest}"
        output.provenance = Provenance(
            producer="reconstruction.pivots", model=REVISION, config_digest=digest
        )
        output.pivots = [e.pivot for e in result.events]
        output.arrays.append(
            DenseArray(
                id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
            )
        )
        arrays = {a.id: placements.read_array(a.id) for a in source.arrays}
        arrays[ARRAY_ID] = array
        return output, arrays

    return store.get_or_create(key, produce)


def load_pivot_evidence(handle: ArtifactHandle) -> PivotSeries:
    ground = handle.metadata
    if not isinstance(ground, Ground) or (
        ground.provenance.producer != "reconstruction.pivots"
    ):
        raise ValueError("physical pivot artifact required")
    payload = json.loads(handle.read_array(ARRAY_ID).tobytes())
    result = PivotSeries.model_validate(payload["pivots"])
    upstream = json.loads(handle.read_array("footprint_evidence_json").tobytes())
    if (
        result.placements.model_dump(mode="json") != upstream["placements"]
        or result.placements.reconstruction_id != ground.reconstruction_id
        or [e.pivot for e in result.events] != ground.pivots
        or [e.footprint for e in result.placements.events] != ground.footprints
        or result.algorithm_revision != ground.provenance.model
        or hash_config(result.config.model_dump(mode="json"))
        != ground.provenance.config_digest
        or result.placements.world_unit
        != ("m" if ground.scale == "metric" else "arbitrary")
        or len(result.trajectory) != 2 * len(ground.samples)
    ):
        raise ValueError("pivot evidence disagrees with canonical ground artifact")
    for i, row in enumerate(result.placements.trajectory):
        sample = ground.samples[i // 2]
        if row.global_seconds != sample.global_seconds or (
            row.contact != (sample.left if row.foot == "left" else sample.right)
            or row.support != sample.support
            or row.motion_sample_index != i // 2
            or row.contact_sample_index != i // 2
        ):
            raise ValueError("pivot source disagrees with native contact evidence")
    return result
