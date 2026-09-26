"""Participant-specific skeletal fit; no population shape or contact prior."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field
from scipy.optimize import least_squares  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from contracts.models import (
    Landmark,
    Landmark3D,
    Measurement,
    Morphology,
    MotionSample,
    Provenance,
    Quality,
    Quaternion,
    Reconstruction,
    SegmentFrame,
    StrictModel,
)

REVISION = "participant-skeleton-v1"
# Each side is independent: no assumed symmetry or population proportions.
BONES: dict[str, tuple[Landmark, Landmark]] = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_shank": ("right_knee", "right_ankle"),
    "shoulder_width": ("left_shoulder", "right_shoulder"),
    "hip_width": ("left_hip", "right_hip"),
}


class FitConfig(StrictModel):
    min_samples: int = Field(default=5, ge=3)
    min_span_seconds: float = Field(default=0.1, gt=0)
    max_relative_uncertainty: float = Field(default=0.15, gt=0, le=0.5)
    max_relative_dispersion: float = Field(default=0.1, gt=0, le=0.5)
    outlier_mad_multiplier: float = Field(default=3.5, ge=1)
    length_weight: float = Field(default=4, gt=0, le=100)
    max_displacement_fraction: float = Field(default=0.1, gt=0, le=0.5)
    max_evaluations: int = Field(default=100, ge=1, le=1000)
    frame_min_sine: float = Field(default=0.1, gt=0, le=1)
    max_orientation_uncertainty_rad: float = Field(default=0.25, gt=0, le=1.5)


def _sources(points: list[Landmark3D]) -> list[str]:
    return list(dict.fromkeys(s for p in points for s in p.quality.source_ids))


def _supported(point: Landmark3D, config: FitConfig, length: float) -> bool:
    q = point.quality
    return (
        point.xyz_world is not None
        and q.state == "observed"
        and bool(q.source_ids)
        and q.uncertainty is not None
        and q.uncertainty <= config.max_relative_uncertainty * length
    )


def estimate_morphology(
    raw: Reconstruction, config: FitConfig, *, identifier: str, config_digest: str
) -> tuple[Morphology, dict[str, Any]]:
    """Median/MAD over direct, sufficiently precise, distinct-time evidence."""
    measurements, evidence = [], {}
    for name, (start, end) in BONES.items():
        candidates: list[dict[str, Any]] = []
        rejected = []
        seen_sources: set[tuple[str, ...]] = set()
        for sample in raw.samples:
            points = {p.name: p for p in sample.landmarks}
            a, b = points.get(start), points.get(end)
            if a is None or b is None or a.xyz_world is None or b.xyz_world is None:
                rejected.append({"time": sample.global_seconds, "reason": "missing"})
                continue
            length = float(np.linalg.norm(np.array(b.xyz_world) - a.xyz_world))
            if length <= 0 or not all(_supported(p, config, length) for p in [a, b]):
                rejected.append({"time": sample.global_seconds, "reason": "weak"})
                continue
            sources = _sources([a, b])
            source_key = tuple(sorted(sources))
            if source_key in seen_sources:
                rejected.append(
                    {"time": sample.global_seconds, "reason": "repeated_evidence"}
                )
                continue
            seen_sources.add(source_key)
            candidates.append(
                {
                    "time": sample.global_seconds,
                    "length": length,
                    "source_ids": sources,
                    "uncertainty": float(
                        np.hypot(a.quality.uncertainty or 0, b.quality.uncertainty or 0)
                    ),
                }
            )
        value = None
        quality = Quality(state="unknown")
        reason = "insufficient_observed_samples"
        retained = []
        if len(candidates) >= config.min_samples:
            lengths = np.array([c["length"] for c in candidates])
            median = float(np.median(lengths))
            mad = float(np.median(np.abs(lengths - median)))
            # A scale-relative numerical floor handles exact synthetic data.
            cutoff = config.outlier_mad_multiplier * max(1.4826 * mad, median * 1e-6)
            retained = [c for c in candidates if abs(c["length"] - median) <= cutoff]
            for c in candidates:
                if c not in retained:
                    rejected.append({**c, "reason": "length_outlier"})
            if len(retained) >= config.min_samples:
                span = retained[-1]["time"] - retained[0]["time"]
                values = np.array([c["length"] for c in retained])
                median = float(np.median(values))
                dispersion = float(1.4826 * np.median(np.abs(values - median)))
                if span < config.min_span_seconds:
                    reason = "insufficient_time_span"
                elif dispersion > median * config.max_relative_dispersion:
                    reason = "unstable_geometry"
                else:
                    value = median
                    # Do not divide by sqrt(N): correlated/systematic noise remains.
                    uncertainty = max(
                        dispersion,
                        float(np.median([c["uncertainty"] for c in retained])),
                    )
                    quality = Quality(
                        state="inferred",
                        uncertainty=uncertainty,
                        source_ids=[raw.id],
                    )
                    reason = "robust_sequence_estimate"
        measurements.append(
            Measurement(
                name=name,
                value=value,
                unit="m" if raw.scale == "metric" else "arbitrary",
                quality=quality,
            )
        )
        evidence[name] = {
            "reason": reason,
            "candidates": candidates,
            "retained": retained,
            "rejected": rejected,
        }
    return Morphology(
        kind="morphology",
        id=identifier,
        schema_version="1.0.0",
        provenance=Provenance(
            producer="reconstruction.articulated",
            model=REVISION,
            config_digest=config_digest,
        ),
        participant_id=raw.participant_id,
        measurements=measurements,
    ), evidence


def _orientation(
    origin: Landmark3D | None,
    lateral: Landmark3D | None,
    axial: Landmark3D | None,
    config: FitConfig,
) -> tuple[Quaternion | None, Quality]:
    points = [p for p in [origin, lateral, axial] if p is not None]
    if len(points) != 3 or any(p.xyz_world is None for p in points):
        return None, Quality(state="unknown")
    a, b, c = [np.array(p.xyz_world) for p in points]
    x, z = b - a, c - a
    nx, nz = float(np.linalg.norm(x)), float(np.linalg.norm(z))
    if min(nx, nz) <= 0 or not all(_supported(p, config, min(nx, nz)) for p in points):
        return None, Quality(state="unknown")
    x, z = x / nx, z / nz
    y = np.cross(z, x)
    sine = float(np.linalg.norm(y))
    if sine < config.frame_min_sine:
        return None, Quality(state="unknown")
    angular_uncertainty = max(float(p.quality.uncertainty or 0) for p in points) / (
        min(nx, nz) * sine
    )
    if angular_uncertainty > config.max_orientation_uncertainty_rad:
        return None, Quality(state="unknown")
    y /= sine
    z = np.cross(x, y)
    q = Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()
    return Quaternion(
        wxyz=(float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    ), Quality(
        state="inferred",
        source_ids=_sources(points),
        uncertainty=angular_uncertainty,
    )


def fit_sample(
    sample: MotionSample,
    morphology: Morphology,
    config: FitConfig,
) -> tuple[MotionSample, dict[str, Any]]:
    """Bounded weighted data/length least squares; never fill absent joints."""
    fitted = sample.model_copy(deep=True)
    points = {p.name: p for p in fitted.landmarks}
    measurements = {m.name: m for m in morphology.measurements}
    active = []
    for name, (a, b) in BONES.items():
        m = measurements[name]
        if (
            m.value is not None
            and a in points
            and b in points
            and all(_supported(points[n], config, m.value) for n in (a, b))
        ):
            active.append((name, a, b, m.value))
    names = sorted({n for _, a, b, _ in active for n in (a, b)})
    diagnostic: dict[str, Any] = {
        "time": sample.global_seconds,
        "constraints": [],
        "data_residuals": {},
        "status": "no_supported_constraints",
    }
    if names:
        initial = np.array([points[n].xyz_world for n in names], dtype=float)
        index = {n: i for i, n in enumerate(names)}
        local_scale = {
            n: min(length for _, a, b, length in active if n in (a, b)) for n in names
        }
        scales = np.array([local_scale[n] for n in names])
        sigmas = np.array(
            [
                max(float(points[n].quality.uncertainty or 0), local_scale[n] * 1e-6)
                for n in names
            ]
        )
        bounds = config.max_displacement_fraction * scales[:, None] / np.sqrt(3)

        def residual(vector: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            xyz = vector.reshape(-1, 3)
            data = ((xyz - initial) / sigmas[:, None]).ravel()
            lengths = [
                config.length_weight
                * (np.linalg.norm(xyz[index[b]] - xyz[index[a]]) - length)
                / max(float(measurements[n].quality.uncertainty or 0), length * 1e-6)
                for n, a, b, length in active
            ]
            return np.concatenate([data, lengths])

        result = least_squares(
            residual,
            initial.ravel(),
            bounds=((initial - bounds).ravel(), (initial + bounds).ravel()),
            max_nfev=config.max_evaluations,
        )
        diagnostic.update(
            status="converged" if result.success else "fit_rejected",
            evaluations=result.nfev,
            cost=float(result.cost),
            message=result.message,
        )
        xyz = result.x.reshape(-1, 3) if result.success else initial
        constraint_sources = list(
            dict.fromkeys(
                s for n, _, _, _ in active for s in measurements[n].quality.source_ids
            )
        )
        point_sources = _sources([points[n] for n in names])
        for n in names:
            p = points[n]
            displacement = float(np.linalg.norm(xyz[index[n]] - initial[index[n]]))
            diagnostic["data_residuals"][n] = displacement
            if result.success and displacement > local_scale[n] * 1e-10:
                p.xyz_world = tuple(float(v) for v in xyz[index[n]])  # type: ignore[assignment]
                p.quality = Quality(
                    state="inferred",
                    source_ids=list(
                        dict.fromkeys(
                            point_sources + constraint_sources + [morphology.id]
                        )
                    ),
                    uncertainty=max(
                        float(p.quality.uncertainty or 0),
                        displacement,
                        max(
                            float(measurements[k].quality.uncertainty or 0)
                            for k, a, b, _ in active
                            if n in (a, b)
                        ),
                    ),
                )
        diagnostic["constraints"] = [
            {
                "name": n,
                "target_length": length,
                "residual": float(
                    np.linalg.norm(xyz[index[b]] - xyz[index[a]]) - length
                ),
            }
            for n, a, b, length in active
        ]
    # Root stays in the world frame. A labeled triplet fixes orientation; pairs do not.
    original = {p.name: p for p in sample.landmarks}
    translation_quality = (
        sample.quality.model_copy(deep=True)
        if fitted.root_xyz_world is not None
        else Quality(state="unknown")
    )
    if fitted.root_xyz_world is None:
        pelvis = original.get("pelvis")
        if pelvis is not None:
            fitted.root_xyz_world = pelvis.xyz_world
            translation_quality = pelvis.quality.model_copy(deep=True)
    root_quality = sample.quality.model_copy(deep=True)
    if fitted.root_orientation is None:
        fitted.root_orientation, root_quality = _orientation(
            original.get("left_hip"),
            original.get("right_hip"),
            original.get("neck"),
            config,
        )
    transforms = [
        {
            "frame": "root",
            "parent": "world",
            "translation": fitted.root_xyz_world,
            "translation_quality": translation_quality.model_dump(mode="json"),
            "orientation": (
                fitted.root_orientation.model_dump(mode="json")
                if fitted.root_orientation
                else None
            ),
            "orientation_quality": root_quality.model_dump(mode="json"),
        }
    ]
    # Supplied explicit orientations are preserved. Derive only noncollinear triplets.
    existing = {s.segment for s in fitted.segments}
    triplets: dict[str, tuple[Landmark, Landmark, Landmark]] = {
        "torso": ("left_shoulder", "right_shoulder", "neck"),
        "head": ("left_ear", "right_ear", "nose"),
        "left_hand": ("left_wrist", "left_index_mcp", "left_pinky_mcp"),
        "right_hand": ("right_wrist", "right_index_mcp", "right_pinky_mcp"),
        "left_foot": ("left_heel", "left_forefoot", "left_foot_outer"),
        "right_foot": ("right_heel", "right_forefoot", "right_foot_outer"),
    }
    for name in list(BONES)[:8] + list(triplets):
        if name in existing:
            continue
        if name in triplets:
            a, b, c = triplets[name]
            rotation, quality = _orientation(
                original.get(a),
                original.get(b),
                original.get(c),
                config,
            )
        else:
            rotation, quality = None, Quality(state="unknown")
        fitted.segments.append(
            SegmentFrame(
                segment=name,
                parent="world",
                orientation=rotation,
                quality=quality,
            )
        )
    for segment in fitted.segments:
        anchor = (
            triplets[segment.segment][0]
            if segment.segment in triplets
            else BONES[segment.segment][0]
            if segment.segment in BONES
            else None
        )
        # Derived translations are world coordinates. Supplied frames with other
        # parents retain orientation but translation is unavailable, never mislabeled.
        point = points.get(anchor) if anchor else None
        transforms.append(
            {
                "frame": segment.segment,
                "parent": segment.parent,
                "translation": (
                    point.xyz_world if point and segment.parent == "world" else None
                ),
                "translation_quality": (
                    point.quality.model_dump(mode="json")
                    if point and segment.parent == "world"
                    else Quality(state="unknown").model_dump(mode="json")
                ),
                "orientation": (
                    segment.orientation.model_dump(mode="json")
                    if segment.orientation
                    else None
                ),
                "orientation_quality": segment.quality.model_dump(mode="json"),
            }
        )
    if names or fitted.root_orientation != sample.root_orientation:
        fitted.quality = Quality(
            state="inferred",
            source_ids=list(
                dict.fromkeys(
                    sample.quality.source_ids
                    + [s for p in fitted.landmarks for s in p.quality.source_ids]
                    + root_quality.source_ids
                )
            ),
        )
    diagnostic["transforms"] = transforms
    return fitted, diagnostic
