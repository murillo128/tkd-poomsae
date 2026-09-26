"""Offline temporal numerical/evidence contracts; no dataset accuracy claims."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from contracts.models import (
    Landmark3D,
    MotionSample,
    Quality,
    Quaternion,
    Reconstruction,
)
from reconstruction.detailed import derive_sample
from reconstruction.temporal import (
    TemporalConfig,
    load_temporal_motion,
    publish_temporal_motion,
    query_motion,
    regularize,
)
from reconstruction.temporal.core import (
    angular_derivatives,
    derivatives,
    quaternion,
    regularize_positions,
    rotation,
)
from storage import ArtifactKey, ArtifactStore, StorageRoot, hash_file
from tests.test_articulated import PROV, participant, point, raw_handle


def quality(sigma: float = 0.001, source: str = "camera") -> Quality:
    return Quality(state="observed", uncertainty=sigma, source_ids=[source])


def trajectory(times: Any, coordinates: Any) -> Reconstruction:
    return Reconstruction(
        kind="reconstruction",
        id="trajectory",
        schema_version="1.0.0",
        provenance=PROV,
        calibration_id="cal",
        participant_id="p",
        scale="metric",
        samples=[
            MotionSample(
                global_seconds=float(t),
                root_xyz_world=tuple(v),
                root_orientation=None,
                quality=quality(source=f"root:{i}"),
                landmarks=[
                    Landmark3D(
                        name="left_wrist",
                        xyz_world=tuple(v),
                        quality=quality(source=f"wrist:{i}"),
                    )
                ],
            )
            for i, (t, v) in enumerate(zip(times, coordinates, strict=True))
        ],
    )


def test_mixed_timestamps_quadratic_derivatives_and_endpoints() -> None:
    times = [0.0, 0.01, 0.027, 0.039, 0.06, 0.077]
    values: list[tuple[float, float, float] | None] = [
        (t * t, 3 * t, 1 + 2 * t * t) for t in times
    ]
    config = TemporalConfig(derivative_window_seconds=0.2, regularization=0)
    result = derivatives(times, values, [quality() for _ in times], config, "m")
    for t, d in zip(times, result, strict=True):
        assert d.velocity == pytest.approx([2 * t, 3, 4 * t], abs=1e-10)
        assert d.acceleration == pytest.approx([2, 0, 4], abs=1e-8)
        assert d.velocity_valid and d.acceleration_valid
        assert d.velocity_quality.source_ids
        assert d.velocity_quality.uncertainty is not None
        assert d.unit == "m" and d.time_unit == "s"
        assert d.support_times == times
    isolated = derivatives(
        [0.0, 0.5],
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        [quality(), quality()],
        config,
        "arbitrary",
    )
    assert all(not d.velocity_valid and not d.acceleration_valid for d in isolated)


@pytest.mark.parametrize("weakness", ["missing", "sigma", "sources", "unknown"])
def test_missing_and_weak_data_stay_unknown(weakness: str) -> None:
    times = [i * 0.01 for i in range(12)]
    values: list[Any] = [(t, 0.0, 0.0) for t in times]
    qs = [quality(source=str(i)) for i in range(len(times))]
    for i in range(3, 9):
        if weakness == "missing":
            values[i] = None
            qs[i] = Quality(state="unknown", source_ids=["occluded"])
        elif weakness == "sigma":
            qs[i].uncertainty = 1
        elif weakness == "sources":
            qs[i].source_ids = []
        else:
            qs[i].state = "unknown"
    config = TemporalConfig(short_gap_seconds=0.02)
    filtered, evidence = regularize_positions(times, values, qs, config)
    ds = derivatives(times, filtered, evidence, config, "m")
    assert filtered[3:9] == [None] * 6
    assert all(q.state == "unknown" for q in evidence[3:9])
    assert all(not d.velocity_valid and not d.acceleration_valid for d in ds[3:9])
    assert ds[2].support_times == times[:3]
    assert ds[9].support_times == times[9:]


def test_short_gap_has_uncertainty_sources_and_no_extrapolation() -> None:
    times = [0.0, 0.01, 0.02, 0.03, 0.04]
    values = [None, (0.0, 0.0, 0.0), None, (0.02, 0.0, 0.0), None]
    qs = [quality(source=str(i)) for i in range(5)]
    filtered, evidence = regularize_positions(
        times, values, qs, TemporalConfig(regularization=0)
    )
    assert filtered[0] is None and filtered[-1] is None
    assert filtered[2] == pytest.approx([0.01, 0, 0])
    assert evidence[2].state == "interpolated"
    assert evidence[2].source_ids == ["1", "3"]
    assert evidence[2].uncertainty == pytest.approx(0.011)
    # Large short-bracket motion is insufficiently constrained.
    values[3] = (1.0, 0.0, 0.0)
    assert regularize_positions(times, values, qs, TemporalConfig())[0][2] is None


@pytest.mark.parametrize("policy", ["break", "step", "interval"])
def test_discontinuities_split_filter_derivative_and_gap_windows(policy: str) -> None:
    times = [i * 0.01 for i in range(10)]
    if policy == "interval":
        times[5:] = [t + 1 for t in times[5:]]
    values: list[tuple[float, float, float] | None] = [
        (0.0 if i < 5 else 2.0, 0.0, 0.0) for i in range(10)
    ]
    config = TemporalConfig(
        **(
            {"break_times": [times[5]]}
            if policy == "break"
            else {"max_step_world": 0.5}
            if policy == "step"
            else {}
        )
    )
    filtered, qs = regularize_positions(times, values, [quality()] * 10, config)
    np.testing.assert_allclose(
        np.array(filtered, dtype=float), np.array(values, dtype=float), atol=1e-12
    )
    ds = derivatives(times, filtered, qs, config, "m")
    for i, d in enumerate(ds):
        if d.velocity_valid:
            assert d.velocity == pytest.approx([0, 0, 0], abs=1e-10)
            assert all((t < times[5]) == (i < 5) for t in d.support_times)


def test_quality_weights_reduce_low_quality_outlier_influence() -> None:
    times = [i * 0.01 for i in range(9)]
    values: list[tuple[float, float, float] | None] = [(t, 0.0, 0.0) for t in times]
    values[4] = (0.065, 0.0, 0.0)
    qs = [quality() for _ in times]
    qs[4] = quality(0.04)
    config = TemporalConfig(filter_window_seconds=0.08)
    filtered, evidence = regularize_positions(times, values, qs, config)
    assert filtered[4] is not None
    assert abs(filtered[4][0] - 0.04) < 0.001
    assert evidence[4].state == "inferred" and evidence[4].score is None
    assert evidence[4].uncertainty is not None and evidence[4].uncertainty >= 0.04


def transient_metrics() -> dict[str, float]:
    times = np.arange(0, 1.001, 0.005)
    # Rapid 70 ms extension/retraction and a separate direction reversal.
    pulse = np.exp(-(((times - 0.4) / 0.035) ** 2))
    turn = np.where(times <= 0.7, times, 1.4 - times)
    truth = np.column_stack([pulse, turn, np.ones(len(times))])
    rng = np.random.default_rng(28)
    noise = rng.normal(0, 0.004, truth.shape)
    noisy = truth + noise
    source = trajectory(times, noisy)
    config = TemporalConfig(filter_window_seconds=0.04)
    filtered, _, _ = regularize(source, config, "final")
    actual = np.array([s.landmarks[0].xyz_world for s in filtered])
    return {
        "noisy_rmse": float(np.sqrt(np.mean(noise**2))),
        "filtered_rmse": float(np.sqrt(np.mean((actual - truth) ** 2))),
        "peak_amplitude_error": float(abs(actual[:, 0].max() - 1)),
        "peak_timing_error_seconds": float(abs(times[np.argmax(actual[:, 0])] - 0.4)),
        "reversal_timing_error_seconds": float(
            abs(times[np.argmax(actual[:, 1])] - 0.7)
        ),
        "reversal_amplitude_error": float(abs(actual[:, 1].max() - 0.7)),
    }


def test_noisy_fast_extension_retraction_and_reversal_tolerances() -> None:
    metrics = transient_metrics()
    assert metrics["filtered_rmse"] < metrics["noisy_rmse"] * 0.8
    assert metrics["peak_amplitude_error"] < 0.04
    assert metrics["peak_timing_error_seconds"] <= 0.0051
    assert metrics["reversal_timing_error_seconds"] <= 0.0101
    assert metrics["reversal_amplitude_error"] < 0.012


def test_quaternion_sign_wrap_and_angular_acceleration() -> None:
    times = [i * 0.01 for i in range(15)]
    orientations: list[Quaternion | None] = [
        quaternion(Rotation.from_euler("z", np.pi - 0.1 + 2 * t + 3 * t * t))
        for t in times
    ]
    for i, q in enumerate(orientations):
        assert q is not None
        if i % 2:
            q.wxyz = tuple(-v for v in q.wxyz)  # type: ignore[assignment]
    config = TemporalConfig(derivative_window_seconds=0.1)
    ds = angular_derivatives(
        times, orientations, [quality()] * len(times), config, "root"
    )
    for t, d in zip(times, ds, strict=True):
        assert d.velocity == pytest.approx([0, 0, 2 + 6 * t], abs=1e-9)
        assert d.acceleration == pytest.approx([0, 0, 6], abs=1e-7)
        assert d.unit == "rad" and d.parent == "root"
    orientations[7] = None
    ds = angular_derivatives(
        times, orientations, [quality()] * len(times), config, "world"
    )
    assert not ds[7].velocity_valid
    assert all(t < times[7] for t in ds[6].support_times)


def test_publication_cache_morphology_geometry_query_reload(tmp_path: Path) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    source = participant()
    source.provenance.producer = "reconstruction.triangulation"
    # Dense native times support interpolation, while spanning morphology's minimum.
    for i, sample in enumerate(source.samples):
        sample.global_seconds = i * 0.02
    raw = raw_handle(store, source)
    before = hash_file(raw.path / "manifest.json")
    config = TemporalConfig(max_position_uncertainty=0.15)
    result = publish_temporal_motion(store, raw, config)
    final = result.motion.metadata
    assert isinstance(final, Reconstruction)
    payload = load_temporal_motion(result.motion)
    assert payload["morphology_id"] == result.morphology.metadata.id
    assert payload["raw_revision"] == before
    assert hash_file(raw.path / "manifest.json") == before
    assert len(final.samples) == len(source.samples)
    for a in raw.metadata.arrays:  # type: ignore[union-attr]
        np.testing.assert_array_equal(
            raw.read_array(a.id), result.motion.read_array(a.id)
        )
    for i, sample in enumerate(final.samples):
        assert {p.name for p in sample.landmarks} == {
            p.name for p in source.samples[i].landmarks
        }
        assert point(sample, "left_index_tip").xyz_world is not None
        assert point(sample, "left_heel").xyz_world is not None
        assert point(sample, "left_heel").xyz_world[2] != 0  # type: ignore[index]
        assert sample.root_xyz_world is not None
        assert payload["detailed"][i] == derive_sample(
            sample,
            reconstruction_id=final.id,
            representation="regularized",
            config=config.geometry,
        ).model_dump(mode="json")
    same = publish_temporal_motion(store, raw, config)
    assert same.motion.path == result.motion.path
    changed = publish_temporal_motion(
        store, raw, config.model_copy(update={"regularization": 0.01})
    )
    assert changed.motion.path != result.motion.path
    assert changed.morphology.path == result.morphology.path
    assert changed.fitted.path == result.fitted.path
    exact = query_motion(result.motion, 0.04)
    assert exact.sample == final.samples[2]
    interpolated = query_motion(result.motion, 0.05)
    assert interpolated.bracket_times == (0.04, 0.06)
    assert any(p.quality.state == "interpolated" for p in interpolated.sample.landmarks)
    reloaded = store.get(
        ArtifactKey(
            layer="reconstruction",
            inputs={
                "raw": payload["raw_revision"],
                "fit": payload["fit_revision"],
                "morphology": payload["morphology_revision"],
            },
            schema_version="1.0.0",
            algorithm_revision=payload["algorithm_revision"],
            config_digest=final.provenance.config_digest,
        )
    )
    assert query_motion(reloaded, 0.05) == interpolated
    assert load_temporal_motion(reloaded) == payload
    for t in [-0.01, 0.17, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="outside"):
            query_motion(result.motion, t)


def test_query_long_gap_and_supplied_rotation_slerp(tmp_path: Path) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    source = participant(supplied_orientation=True)
    source.provenance.producer = "reconstruction.triangulation"
    for i, sample in enumerate(source.samples):
        sample.global_seconds = i * 0.02
        sample.quality = quality(source=f"root:{i}")
        sample.root_orientation = quaternion(Rotation.from_euler("z", 3.1 + i * 0.05))
        sample.segments[0].quality = quality(source=f"head:{i}")
    result = publish_temporal_motion(
        store, raw_handle(store, source), TemporalConfig(max_position_uncertainty=0.2)
    )
    final = result.motion.metadata
    assert isinstance(final, Reconstruction)
    for a, b in zip(final.samples, final.samples[1:]):
        assert a.root_orientation and b.root_orientation
        assert np.dot(a.root_orientation.wxyz, b.root_orientation.wxyz) >= 0
    query = query_motion(result.motion, 0.01)
    assert query.sample.root_orientation is not None
    np.testing.assert_allclose(
        rotation(query.sample.root_orientation).as_matrix(),
        Rotation.from_euler("z", 3.125).as_matrix(),
        atol=1e-10,
    )
    assert (
        next(s for s in query.sample.segments if s.segment == "head").parent == "root"
    )
    source.samples[4].global_seconds += 0.5
    for i in range(5, len(source.samples)):
        source.samples[i].global_seconds += 0.5
    gapped = publish_temporal_motion(store, raw_handle(store, source))
    query = query_motion(gapped.motion, 0.3)
    assert all(p.xyz_world is None for p in query.sample.landmarks)
    assert query.sample.root_xyz_world is None and query.sample.root_orientation is None
    assert all(r.value == "unknown" for r in query.geometry.relations)


def test_invalid_temporal_configuration() -> None:
    for values in [
        {"regularization": -1},
        {"filter_window_seconds": 0},
        {"short_gap_seconds": 0.2},
        {"break_times": [1, 0]},
        {"max_step_world": float("nan")},
        {"floor_snap": True},
    ]:
        with pytest.raises(ValueError):
            TemporalConfig.model_validate(values)


def pivot_metrics() -> dict[str, float]:
    source = participant()
    base = source.samples[0]
    times = np.arange(0, 0.501, 0.005)
    angles = (np.pi / 4) * (1 + np.tanh((times - 0.25) / 0.025))
    samples = []
    for i, (t, angle) in enumerate(zip(times, angles, strict=True)):
        sample = base.model_copy(deep=True)
        sample.global_seconds = float(t)
        r = Rotation.from_euler("z", angle)
        for p in sample.landmarks:
            assert p.xyz_world is not None
            p.xyz_world = tuple(r.apply(p.xyz_world))
            p.quality = quality(0.0001, source=f"view:{i}:{p.name}")
        samples.append(sample)
    source.samples = samples
    config = TemporalConfig(filter_window_seconds=0.03, derivative_window_seconds=0.03)
    output, _, ds = regularize(source, config, "pivot")
    actual = np.array(
        [
            rotation(s.root_orientation).as_euler("xyz")[2]
            for s in output
            if s.root_orientation is not None
        ]
    )
    velocities = [s.angular["root"].velocity for s in ds]
    omega = np.array([v[2] if v is not None else np.nan for v in velocities])
    expected_peak = (np.pi / 4) / 0.025
    return {
        "max_angle_error_rad": float(np.max(np.abs(actual - angles))),
        "pivot_midpoint_error_seconds": float(
            abs(times[np.argmin(np.abs(actual - np.pi / 4))] - 0.25)
        ),
        "angular_peak_relative_error": float(abs(np.nanmax(omega) / expected_peak - 1)),
    }


def test_fast_pivot_geometry_and_angular_peak() -> None:
    metrics = pivot_metrics()
    assert metrics["max_angle_error_rad"] < 0.02
    assert metrics["pivot_midpoint_error_seconds"] <= 0.0051
    assert metrics["angular_peak_relative_error"] < 0.15


@pytest.mark.parametrize("scale,unit", [(1.0, "metric"), (10.0, "arbitrary")])
def test_morphology_lengths_and_vertical_root_survive(
    scale: float, unit: Any, tmp_path: Path
) -> None:
    source = participant(
        upper=0.41, forearm=0.32, shoulder=0.5, scale=scale, units=unit
    )
    source.provenance.producer = "reconstruction.triangulation"
    result = publish_temporal_motion(
        ArtifactStore(StorageRoot(tmp_path)),
        raw_handle(ArtifactStore(StorageRoot(tmp_path)), source),
        TemporalConfig(max_position_uncertainty=0.05 * scale),
    )
    shape = {m.name: m for m in result.morphology.metadata.measurements}  # type: ignore[union-attr]
    assert shape["left_upper_arm"].value == pytest.approx(0.41 * scale)
    assert shape["left_forearm"].value == pytest.approx(0.32 * scale)
    final = result.motion.metadata
    assert isinstance(final, Reconstruction)
    assert final.scale == unit
    np.testing.assert_allclose(
        np.array([s.root_xyz_world for s in final.samples], dtype=float),
        np.array([s.root_xyz_world for s in source.samples], dtype=float),
        atol=1e-10,
    )
    payload = load_temporal_motion(result.motion)
    expected_unit = "m" if unit == "metric" else "arbitrary"
    assert payload["world_unit"] == expected_unit
    assert payload["kinematics"][0]["linear"]["root"]["unit"] == expected_unit


def test_actual_n_view_raw_publisher_composes_without_vision(tmp_path: Path) -> None:
    from reconstruction.triangulation import load_diagnostics, publish_triangulation
    from tests.test_triangulation import handles, join, scene

    store = ArtifactStore(StorageRoot(tmp_path))
    calibration, models = scene()
    times = [i * 0.02 for i in range(9)]
    cal, aligned = handles(
        store,
        calibration,
        join(models, tuple(i * 20 for i in range(9))).query_many(times),
    )
    raw = publish_triangulation(store, cal, aligned, "participant")
    result = publish_temporal_motion(store, raw)
    assert load_diagnostics(result.motion) == load_diagnostics(raw)
    assert isinstance(result.motion.metadata, Reconstruction)
    assert result.motion.metadata.participant_id == "participant"
    assert all(
        point(s, "left_index_tip").xyz_world is not None
        for s in result.motion.metadata.samples
    )
    with pytest.raises(ValueError, match="raw triangulation"):
        publish_temporal_motion(store, result.motion)
