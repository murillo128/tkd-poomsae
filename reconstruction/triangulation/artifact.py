"""Immutable raw triangulation artifacts and stage-runner integration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from contracts.models import (
    Alignment,
    Calibration,
    DenseArray,
    MotionSample,
    Provenance,
    Quality,
    Reconstruction,
)
from pipeline.runner import StageOutput
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file
from sync.alignment import TimeQuery, load_alignment

from .core import REVISION, TriangulationConfig, triangulate_point

ARRAY_ID = "raw_triangulation_diagnostics_json"
ASSUMPTIONS = [
    "Raw triangulation only: no body fitting or temporal regularization.",
    "Local covariance assumes independent isotropic pixel noise and fixed cameras.",
    "Largest covariance eigenvalue gives a conditional world-unit standard deviation.",
    "Pixel floors, quality weighting and timing speed bounds are assumptions, "
    "not learned accuracy.",
    "Calibration RMS is a pixel proxy, not extrinsic covariance; "
    "shared systematic errors are unmodeled.",
    "Timing uncertainty is seconds; "
    "interpolation adds a half-bracket pixel motion bound.",
    "Raw detector scores and reprojection errors are not accuracy probabilities "
    "or MMPose validation.",
]


def reconstruct(
    key: ArtifactKey,
    calibration: Calibration,
    queries: Sequence[TimeQuery],
    participant_id: str,
    config: TriangulationConfig | None = None,
    *,
    alignment_id: str,
) -> StageOutput:
    config = config or TriangulationConfig()
    if not participant_id or not queries:
        raise ValueError("participant_id and nonempty aligned queries are required")
    names = list(
        dict.fromkeys(p.name for q in queries for c in q.cameras for p in c.landmarks)
    )
    if not names:
        raise ValueError("alignment contains no named landmarks")
    samples = []
    diagnostics = []
    observed = 0
    for query in queries:
        points, evidence = [], []
        for name in names:
            point, diagnostic = triangulate_point(query, name, calibration, config)
            points.append(point)
            evidence.append(diagnostic)
            observed += point.xyz_world is not None
        # No inferred root or orientation. A named pelvis can supply translation.
        root = next((p.xyz_world for p in points if p.name == "pelvis"), None)
        sources = list(dict.fromkeys(s for p in points for s in p.quality.source_ids))
        samples.append(
            MotionSample(
                global_seconds=query.global_seconds,
                root_xyz_world=root,
                root_orientation=None,
                landmarks=points,
                quality=Quality(
                    state="inferred" if sources else "unknown", source_ids=sources
                ),
            )
        )
        diagnostics.append({"global_seconds": query.global_seconds, "points": evidence})
    payload = {
        "version": 1,
        "artifact_role": "immutable_raw_triangulated_motion",
        "alignment_id": alignment_id,
        "calibration_id": calibration.id,
        "world_unit": calibration.world_unit,
        "assumptions": ASSUMPTIONS,
        "settings": config.model_dump(mode="json"),
        "calibration_quality": calibration.quality.model_dump(mode="json"),
        "calibration_flags": calibration.quality_flags,
        "samples": diagnostics,
    }
    array = np.frombuffer(
        json.dumps(
            payload, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode(),
        dtype=np.uint8,
    )
    artifact = Reconstruction(
        kind="reconstruction",
        id=f"raw-triangulation:{key.digest}",
        schema_version="1.0.0",
        provenance=Provenance(
            producer="reconstruction.triangulation", config_digest=key.config_digest
        ),
        calibration_id=calibration.id,
        participant_id=participant_id,
        scale=calibration.scale,
        samples=samples,
        arrays=[
            DenseArray(
                id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
            )
        ],
    )
    return StageOutput(
        artifact,
        {ARRAY_ID: array},
        diagnostics=(
            f"{observed} triangulated points; "
            f"{len(queries) * len(names) - observed} unsupported",
            "uncertainty is conditional; see persisted assumptions",
        ),
        version="1",
    )


def produce_stage(
    key: ArtifactKey,
    inputs: Mapping[str, ArtifactHandle],
    settings: Mapping[str, Any],
) -> StageOutput:
    if set(settings) - {"participant_id", "triangulation"}:
        raise ValueError("unsupported reconstruction settings")
    participant = settings.get("participant_id")
    if not isinstance(participant, str) or not participant:
        raise ValueError("reconstruction requires explicit participant_id")
    calibration = inputs["calibration"].metadata
    alignment = inputs["attachment"].metadata
    if not isinstance(calibration, Calibration) or not isinstance(alignment, Alignment):
        raise ValueError(
            "reconstruction requires calibration and aligned observation artifacts"
        )
    sync_links = [
        link.removeprefix("synchronization:")
        for link in calibration.evidence_links
        if link.startswith("synchronization:")
    ]
    if sync_links and sync_links != [alignment.synchronization_id]:
        raise ValueError("calibration and alignment synchronization revisions disagree")
    queries, _ = load_alignment(inputs["attachment"])
    config = TriangulationConfig.model_validate(settings.get("triangulation", {}))
    return reconstruct(
        key, calibration, queries, participant, config, alignment_id=alignment.id
    )


def publish_triangulation(
    store: ArtifactStore,
    calibration: ArtifactHandle,
    alignment: ArtifactHandle,
    participant_id: str,
    config: TriangulationConfig | None = None,
) -> ArtifactHandle:
    config = config or TriangulationConfig()
    calibration_revision = hash_file(calibration.path / "manifest.json")
    alignment_revision = hash_file(alignment.path / "manifest.json")
    settings = {
        "participant_id": participant_id,
        "triangulation": config.model_dump(mode="json"),
    }
    key = ArtifactKey(
        layer="reconstruction",
        inputs={"calibration": calibration_revision, "attachment": alignment_revision},
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=hash_config(settings),
        calibration_revision=calibration_revision,
        sync_revision=alignment_revision,
    )

    def produce() -> tuple[Any, dict[str, np.ndarray[Any, Any]]]:
        output = produce_stage(
            key, {"calibration": calibration, "attachment": alignment}, settings
        )
        return output.artifact, dict(output.arrays)

    return store.get_or_create(key, produce)


def load_diagnostics(handle: ArtifactHandle) -> dict[str, Any]:
    if not isinstance(handle.metadata, Reconstruction):
        raise ValueError("not a reconstruction artifact")
    payload: dict[str, Any] = json.loads(handle.read_array(ARRAY_ID).tobytes())
    if (
        payload.get("version") != 1
        or payload.get("artifact_role") != "immutable_raw_triangulated_motion"
    ):
        raise ValueError("unsupported raw triangulation diagnostics")
    return payload
