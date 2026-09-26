"""Content-addressed ground-view publication, independent of semantic parsing."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from contracts.models import Calibration, DenseArray, Ground, Provenance, Reconstruction
from reconstruction.pivots import load_pivot_evidence
from reconstruction.pivots.core import PivotSeries
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import REVISION, GroundViewConfig, GroundViewSeries, derive_ground_view

ARRAY_ID = "ground_view_json"
ROOT_ARRAY_ID = "ground_view_root"
ROOT_MASK_ID = "ground_view_root_missing"


def _root_arrays(result: GroundViewSeries) -> dict[str, np.ndarray[Any, Any]]:
    # [global seconds, ground X, ground Y, vertical Z]. Missing values use zero
    # only in storage, with an explicit mask; they are null in the query contract.
    return {
        ROOT_ARRAY_ID: np.asarray(
            [
                [p.global_seconds, *(p.xy_ground or (0, 0)), p.z_ground or 0]
                for p in result.root_trajectory
            ],
            dtype=np.float64,
        ).reshape((-1, 4)),
        ROOT_MASK_ID: np.asarray(
            [[False, *([p.xy_ground is None] * 3)] for p in result.root_trajectory],
            dtype=np.bool_,
        ),
    }


def publish_ground_view(
    store: ArtifactStore,
    motion: ArtifactHandle,
    calibration: ArtifactHandle,
    pivots: ArtifactHandle,
    config: GroundViewConfig | None = None,
) -> ArtifactHandle:
    """Consume exact physical inputs; cache keys deliberately exclude semantics."""
    source, cal, ground = motion.metadata, calibration.metadata, pivots.metadata
    if not isinstance(source, Reconstruction) or not isinstance(cal, Calibration):
        raise ValueError("motion and calibration artifacts required")
    if not isinstance(ground, Ground):
        raise ValueError("physical pivot artifact required")
    physical = load_pivot_evidence(pivots)
    revisions = {
        "motion": hash_file(motion.path / "manifest.json"),
        "calibration": hash_file(calibration.path / "manifest.json"),
        "pivots": hash_file(pivots.path / "manifest.json"),
    }
    # IDs alone are insufficient: reject stale feet paired with revised root data.
    upstream = json.loads(pivots.read_array("footprint_evidence_json").tobytes())
    if any(
        upstream["input_revisions"].get(name) != revisions[name]
        for name in ("motion", "calibration")
    ):
        raise ValueError("ground-view inputs disagree with physical artifact revisions")
    config = GroundViewConfig.model_validate(
        (config or GroundViewConfig()).model_dump()
    )
    digest = hash_config(config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="ground",
        inputs=revisions,
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )

    def produce() -> tuple[Ground, dict[str, np.ndarray[Any, Any]]]:
        result = derive_ground_view(
            source, cal, physical, ground_id=ground.id, config=config
        )
        if result.ground_status != "available" or not result.root_trajectory:
            raise ValueError(
                "nonempty motion and usable ground required for publication"
            )
        payload = np.frombuffer(
            json.dumps(
                {
                    "input_revisions": revisions,
                    "ground_view": result.model_dump(mode="json"),
                },
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
            dtype=np.uint8,
        )
        arrays = {a.id: pivots.read_array(a.id) for a in ground.arrays}
        arrays.update(_root_arrays(result))
        arrays[ARRAY_ID] = payload
        output = ground.model_copy(deep=True)
        output.id = f"ground-view:{key.digest}"
        output.provenance = Provenance(
            producer="reconstruction.ground_view", model=REVISION, config_digest=digest
        )
        output.arrays.extend(
            [
                DenseArray(
                    id=ARRAY_ID, dtype="uint8", shape=[len(payload)], axes=["json_byte"]
                ),
                DenseArray(
                    id=ROOT_ARRAY_ID,
                    dtype="float64",
                    shape=[len(result.root_trajectory), 4],
                    axes=["native_time", "global_seconds_x_y_z"],
                    missing_mask_id=ROOT_MASK_ID,
                ),
                DenseArray(
                    id=ROOT_MASK_ID,
                    dtype="bool",
                    shape=[len(result.root_trajectory), 4],
                    axes=["native_time", "global_seconds_x_y_z"],
                ),
            ]
        )
        return output, arrays

    return store.get_or_create(key, produce)


def load_ground_view(handle: ArtifactHandle) -> GroundViewSeries:
    ground = handle.metadata
    if (
        not isinstance(ground, Ground)
        or ground.provenance.producer != "reconstruction.ground_view"
    ):
        raise ValueError("physical ground-view artifact required")
    payload = json.loads(handle.read_array(ARRAY_ID).tobytes())
    result = GroundViewSeries.model_validate(payload["ground_view"])
    physical = PivotSeries.model_validate(
        json.loads(handle.read_array("pivot_evidence_json").tobytes())["pivots"]
    )
    if (
        result.ground_status != "available"
        or result.physical != physical
        or result.reconstruction_id != ground.reconstruction_id
        or result.world_unit != ("m" if ground.scale == "metric" else "arbitrary")
        or [e.sample for e in result.contact_events] != ground.samples
        or [e.footprint for e in physical.placements.events] != ground.footprints
        or [e.pivot for e in physical.events] != ground.pivots
        or result.algorithm_revision != ground.provenance.model
        or hash_config(result.config.model_dump(mode="json"))
        != ground.provenance.config_digest
        or any(
            not np.array_equal(handle.read_array(name), array)
            for name, array in _root_arrays(result).items()
        )
    ):
        raise ValueError("ground-view evidence disagrees with canonical artifact")
    return result
