"""Durable reconstructed motion assembled without rerunning vision."""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.models import (
    DenseArray,
    Landmark3D,
    MotionSample,
    Provenance,
    Quality,
    Reconstruction,
    SegmentFrame,
)
from reconstruction.articulated import (
    FitConfig,
    load_fit_diagnostics,
    publish_articulated_fit,
)
from reconstruction.detailed import DetailedSample, derive_sample
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import (
    REVISION,
    KinematicSample,
    TemporalConfig,
    combined,
    connected,
    refresh_frames,
    regularize,
    regularize_positions,
    rotation,
    slerp,
    supported,
    vector,
)

ARRAY_ID = "temporal_motion_json"


@dataclass(frozen=True)
class TemporalMotion:
    morphology: ArtifactHandle
    fitted: ArtifactHandle
    motion: ArtifactHandle


def publish_temporal_motion(
    store: ArtifactStore,
    raw: ArtifactHandle,
    config: TemporalConfig | None = None,
    fit_config: FitConfig | None = None,
) -> TemporalMotion:
    """Compose immutable raw, shape, articulated fit and temporal geometry."""
    config = config or TemporalConfig()
    source = raw.metadata
    if not isinstance(source, Reconstruction) or source.provenance.producer != (
        "reconstruction.triangulation"
    ):
        raise ValueError("raw triangulation artifact required")
    fit = publish_articulated_fit(store, raw, fit_config)
    fitted = fit.pose.metadata
    assert isinstance(fitted, Reconstruction)
    raw_revision = hash_file(raw.path / "manifest.json")
    fit_revision = hash_file(fit.pose.path / "manifest.json")
    shape_revision = hash_file(fit.morphology.path / "manifest.json")
    digest = hash_config(config.model_dump(mode="json"))
    key = ArtifactKey(
        layer="reconstruction",
        inputs={
            "raw": raw_revision,
            "fit": fit_revision,
            "morphology": shape_revision,
        },
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=digest,
    )

    def produce() -> tuple[Reconstruction, dict[str, np.ndarray[Any, Any]]]:
        transforms = [
            s["transforms"][0] for s in load_fit_diagnostics(fit.pose)["samples"]
        ]
        translations = [
            Quality.model_validate(t["translation_quality"]) for t in transforms
        ]
        rotations = [
            Quality.model_validate(t["orientation_quality"]) for t in transforms
        ]
        # The raw triangulator explicitly obtains root translation from named pelvis.
        # Fit diagnostics may only expose aggregate sample quality for this value.
        for i, q in enumerate(translations):
            if q.uncertainty is None:
                pelvis = next(
                    (p for p in fitted.samples[i].landmarks if p.name == "pelvis"), None
                )
                if pelvis and pelvis.xyz_world == fitted.samples[i].root_xyz_world:
                    translations[i] = pelvis.quality
        identifier = f"temporal-motion:{key.digest}"
        samples, detailed, kinematics = regularize(
            fitted,
            config,
            identifier,
            translations,
            source,
            rotations,
        )
        payload = {
            "version": 1,
            "artifact_role": "temporally_coherent_motion",
            "raw_revision": raw_revision,
            "fit_revision": fit_revision,
            "morphology_revision": shape_revision,
            "morphology_id": fit.morphology.metadata.id,
            "reconstruction_id": identifier,
            "algorithm_revision": REVISION,
            "settings": config.model_dump(mode="json"),
            "world_unit": "m" if fitted.scale == "metric" else "arbitrary",
            "supplied_root": [s.root_orientation is not None for s in source.samples],
            "supplied_segments": [
                [s.segment for s in sample.segments] for sample in source.samples
            ],
            "root_orientation_quality": [
                (
                    rotations[i]
                    if source.samples[i].root_orientation is not None
                    else detailed[i].body_frame.quality
                ).model_dump(mode="json")
                for i in range(len(samples))
            ],
            "root_translation_quality": [
                q.model_dump(mode="json")
                for q in regularize_positions(
                    [s.global_seconds for s in fitted.samples],
                    [s.root_xyz_world for s in fitted.samples],
                    translations,
                    config,
                )[1]
            ],
            "detailed": [s.model_dump(mode="json") for s in detailed],
            "kinematics": [s.model_dump(mode="json") for s in kinematics],
        }
        array = np.frombuffer(
            json.dumps(
                payload, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode(),
            dtype=np.uint8,
        )
        output = fitted.model_copy(deep=True)
        output.id = identifier
        output.samples = samples
        output.provenance = Provenance(
            producer="reconstruction.temporal", model=REVISION, config_digest=digest
        )
        output.arrays.append(
            DenseArray(
                id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
            )
        )
        arrays = {a.id: fit.pose.read_array(a.id) for a in fitted.arrays}
        arrays[ARRAY_ID] = array
        return output, arrays

    return TemporalMotion(fit.morphology, fit.pose, store.get_or_create(key, produce))


def load_temporal_motion(handle: ArtifactHandle) -> dict[str, Any]:
    if not isinstance(handle.metadata, Reconstruction) or (
        handle.metadata.provenance.producer != "reconstruction.temporal"
    ):
        raise ValueError("temporal reconstruction required")
    payload: dict[str, Any] = json.loads(handle.read_array(ARRAY_ID).tobytes())
    if (
        payload.get("version") != 1
        or payload.get("artifact_role") != ("temporally_coherent_motion")
        or payload.get("reconstruction_id") != handle.metadata.id
    ):
        raise ValueError("unsupported temporal payload")
    TemporalConfig.model_validate(payload["settings"])
    for sample in payload["detailed"]:
        DetailedSample.model_validate(sample)
    for sample in payload["kinematics"]:
        KinematicSample.model_validate(sample)
    return payload


@dataclass(frozen=True)
class MotionQuery:
    sample: MotionSample
    geometry: DetailedSample
    bracket_times: tuple[float, ...]


def query_motion(handle: ArtifactHandle, global_seconds: float) -> MotionQuery:
    """Bounded interpolation; query results never become source observations."""
    payload = load_temporal_motion(handle)
    config = TemporalConfig.model_validate(payload["settings"])
    source = handle.metadata
    assert isinstance(source, Reconstruction)
    times = [s.global_seconds for s in source.samples]
    if (
        not math.isfinite(global_seconds)
        or not times
        or not (times[0] <= global_seconds <= times[-1])
    ):
        raise ValueError("query outside reconstructed time domain")
    index = bisect.bisect_left(times, global_seconds)
    if times[index] == global_seconds:
        sample = source.samples[index].model_copy(deep=True)
        return MotionQuery(
            sample,
            derive_sample(
                sample,
                reconstruction_id=source.id,
                representation="regularized",
                config=config.geometry,
            ),
            (global_seconds,),
        )
    a, b = source.samples[index - 1 : index + 1]
    gap = b.global_seconds - a.global_seconds
    allowed = connected(a.global_seconds, b.global_seconds, config) and gap <= (
        config.short_gap_seconds
    )
    fraction = (global_seconds - a.global_seconds) / gap

    def position(av: Any, bv: Any, aq: Quality, bq: Quality) -> tuple[Any, Quality]:
        unknown = Quality(
            state="unknown", source_ids=sorted(set(aq.source_ids + bq.source_ids))
        )
        if (
            not allowed
            or av is None
            or bv is None
            or not all(supported(q, config) for q in (aq, bq))
        ):
            return None, unknown
        delta = np.array(bv) - av
        if (
            config.max_step_world is not None
            and np.linalg.norm(delta) > config.max_step_world
        ):
            return None, unknown
        sigma = (
            max(float(q.uncertainty or 0) for q in (aq, bq))
            + float(np.linalg.norm(delta)) / 2
        )
        if sigma > config.max_position_uncertainty:
            return None, unknown
        return vector(np.array(av) + fraction * delta), combined(
            [aq, bq], sigma, "interpolated"
        )

    landmarks = []
    am, bm = ({p.name: p for p in sample.landmarks} for sample in (a, b))
    for name in sorted(am.keys() | bm.keys()):
        ap = am.get(
            name,
            Landmark3D(name=name, xyz_world=None, quality=Quality(state="unknown")),
        )
        bp = bm.get(
            name,
            Landmark3D(name=name, xyz_world=None, quality=Quality(state="unknown")),
        )
        xyz, q = position(ap.xyz_world, bp.xyz_world, ap.quality, bp.quality)
        landmarks.append(Landmark3D(name=name, xyz_world=xyz, quality=q))
    root_qs = [Quality.model_validate(q) for q in payload["root_translation_quality"]]
    root, root_q = position(
        a.root_xyz_world, b.root_xyz_world, root_qs[index - 1], root_qs[index]
    )
    segments = []
    other = {s.segment: s for s in b.segments}
    for segment in a.segments:
        end = other.get(segment.segment)
        q = Quality(state="unknown")
        orientation = None
        if (
            allowed
            and end
            and end.parent == segment.parent
            and all(
                s.orientation is not None and supported(s.quality, config, angular=True)
                for s in (segment, end)
            )
        ):
            assert segment.orientation is not None and end.orientation is not None
            angle = float(
                (
                    rotation(end.orientation) * rotation(segment.orientation).inv()
                ).magnitude()
            )
            sigma = (
                max(float(s.quality.uncertainty or 0) for s in (segment, end))
                + angle / 2
            )
            if sigma <= config.geometry.max_angle_uncertainty:
                orientation = slerp(segment.orientation, end.orientation, fraction)
                q = combined([segment.quality, end.quality], sigma, "interpolated")
        segments.append(
            SegmentFrame(
                segment=segment.segment,
                parent=segment.parent,
                orientation=orientation,
                quality=q,
            )
        )
    supplied_root = (
        payload["supplied_root"][index - 1] and payload["supplied_root"][index]
    )
    root_orientation = None
    if supplied_root and allowed and a.root_orientation and b.root_orientation:
        aq, bq = [
            Quality.model_validate(payload["root_orientation_quality"][i])
            for i in (index - 1, index)
        ]
        angle = float(
            (
                rotation(b.root_orientation) * rotation(a.root_orientation).inv()
            ).magnitude()
        )
        if all(supported(q, config, angular=True) for q in (aq, bq)) and (
            max(float(q.uncertainty or 0) for q in (aq, bq)) + angle / 2
            <= config.geometry.max_angle_uncertainty
        ):
            root_orientation = slerp(a.root_orientation, b.root_orientation, fraction)
    sample = MotionSample(
        global_seconds=global_seconds,
        root_xyz_world=root,
        root_orientation=root_orientation,
        landmarks=landmarks,
        segments=segments,
        quality=root_q,
    )
    # Recompute final landmark frames and relations at the query time.
    geometry = refresh_frames(
        sample,
        source.id,
        config,
        set(payload["supplied_segments"][index - 1])
        | set(payload["supplied_segments"][index]),
        bool(payload["supplied_root"][index - 1] or payload["supplied_root"][index]),
    )
    return MotionQuery(sample, geometry, (a.global_seconds, b.global_seconds))
