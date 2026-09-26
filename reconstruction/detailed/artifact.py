"""Immutable derived payloads linked to an exact raw or fitted reconstruction."""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np

from contracts.models import DenseArray, Provenance, Reconstruction
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import REVISION, DetailedSample, GeometryConfig, derive_sample

ARRAY_ID = "detailed_geometry_json"


def publish_detailed_geometry(
    store: ArtifactStore,
    source: ArtifactHandle,
    *,
    representation: Literal["raw", "fitted"],
    config: GeometryConfig | None = None,
) -> ArtifactHandle:
    """Preserve all geometry/evidence; derive only from the declared input revision."""
    metadata = source.metadata
    if not isinstance(metadata, Reconstruction) or not metadata.samples:
        raise ValueError("nonempty reconstruction required")
    if any(a.id == ARRAY_ID for a in metadata.arrays):
        raise ValueError("derive from source reconstruction, not a detailed artifact")
    # Existing publishers identify roles; no guessing from landmark evidence states.
    expected = (
        "reconstruction.triangulation"
        if representation == "raw"
        else "reconstruction.articulated"
    )
    if metadata.provenance.producer != expected:
        raise ValueError("representation disagrees with reconstruction producer")
    config = config or GeometryConfig()
    revision = hash_file(source.path / "manifest.json")
    digest = hash_config(
        {"representation": representation, "config": config.model_dump(mode="json")}
    )
    key = ArtifactKey(
        layer="reconstruction",
        inputs={"source_reconstruction": revision},
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )

    def produce() -> tuple[Reconstruction, dict[str, np.ndarray[Any, Any]]]:
        samples = [
            derive_sample(
                sample,
                reconstruction_id=metadata.id,
                representation=representation,
                config=config,
            ).model_dump(mode="json")
            for sample in metadata.samples
        ]
        payload = {
            "version": 1,
            "artifact_role": "derived_detailed_geometry",
            "source_revision": revision,
            "reconstruction_id": metadata.id,
            "representation": representation,
            "world_unit": "m" if metadata.scale == "metric" else "arbitrary",
            "samples": samples,
        }
        array = np.frombuffer(
            json.dumps(
                payload,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
            dtype=np.uint8,
        )
        output = metadata.model_copy(deep=True)
        output.id = f"detailed-geometry:{key.digest}"
        output.provenance = Provenance(
            producer="reconstruction.detailed",
            model=REVISION,
            config_digest=digest,
        )
        output.arrays.append(
            DenseArray(
                id=ARRAY_ID,
                dtype="uint8",
                shape=[len(array)],
                axes=["json_byte"],
            )
        )
        arrays = {a.id: source.read_array(a.id) for a in metadata.arrays}
        arrays[ARRAY_ID] = array
        return output, arrays

    return store.get_or_create(key, produce)


def load_detailed_geometry(handle: ArtifactHandle) -> list[DetailedSample]:
    if not isinstance(handle.metadata, Reconstruction):
        raise ValueError("not a reconstruction artifact")
    payload = json.loads(handle.read_array(ARRAY_ID).tobytes())
    if (
        payload.get("version") != 1
        or payload.get("artifact_role") != "derived_detailed_geometry"
    ):
        raise ValueError("unsupported detailed geometry payload")
    return [DetailedSample.model_validate(s) for s in payload["samples"]]
