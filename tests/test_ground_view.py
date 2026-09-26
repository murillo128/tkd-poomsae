"""Ground-view acceptance on shared synthetic artifacts, without accuracy claims."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from contracts.models import (
    Ground,
    Interval,
    Landmark3D,
    Quality,
    Reconstruction,
    Semantics,
)
from reconstruction.footprints import derive_footprints, publish_footprints
from reconstruction.ground import publish_contacts
from reconstruction.ground_view import (
    GroundViewConfig,
    GroundViewSeries,
    derive_ground_view,
    load_ground_view,
    publish_ground_view,
)
from reconstruction.pivots import derive_pivots, publish_pivots
from storage import ArtifactStore, StorageRoot
from tests.test_footprints import ground
from tests.test_ground_contact import calibration, motion, persist, shape
from tests.test_pivots import sequence


def view(source: Reconstruction, contacts: Ground) -> GroundViewSeries:
    cal = calibration(source.scale == "metric")
    morphology = shape(7) if source.scale == "arbitrary" else None
    physical = derive_pivots(derive_footprints(source, contacts, cal, morphology))
    return derive_ground_view(source, cal, physical, ground_id="pivots")


def test_xy_z_handedness_dense_quality_and_no_double_calibration() -> None:
    source = motion()
    for i, sample in enumerate(source.samples):
        t = sample.global_seconds
        sample.root_xyz_world = (-2 + t, 3 - 2 * t, 1 + t / 2)
        sample.quality = Quality(
            state="observed", uncertainty=0.02, source_ids=[f"root-source:{i}"]
        )
    cal = calibration()
    # Source-to-world has already been consumed by reconstruction. Reapplying it
    # would translate the test root a second time.
    assert cal.ground_frame is not None
    cal.ground_frame.source_to_world[0][3] = 100
    physical = derive_pivots(derive_footprints(source, ground(source), cal))
    before = source.model_dump_json(), physical.model_dump_json()
    result = derive_ground_view(source, cal, physical, ground_id="pivots")
    assert len(result.root_trajectory) == len(source.samples)
    for i, point in enumerate(result.root_trajectory):
        xyz = source.samples[i].root_xyz_world
        assert xyz is not None
        assert point.xy_ground == xyz[:2]
        assert point.z_ground == xyz[2]
        assert point.quality == source.samples[i].quality
        assert point.motion_sample_index == i
    assert result.world_unit == "m"
    assert result.scene_bounds is not None
    assert result.scene_bounds.minimum_xy == (-2, 0)
    assert result.scene_bounds.maximum_xy == (0.2, 3)
    assert result.scene_bounds.minimum_z == 1
    assert result.scene_bounds.maximum_z == 1.5
    assert result.root_path_indices == [list(range(len(source.samples)))]
    assert before == (source.model_dump_json(), physical.model_dump_json())
    assert derive_ground_view(source, cal, physical, ground_id="pivots") == result
    assert GroundViewSeries.model_validate_json(result.model_dump_json()) == result


def test_dynamic_and_summary_agree_with_physical_ids_and_pivot_paths() -> None:
    source, contacts = sequence(sign=-1)
    result = view(source, contacts)
    assert result.physical is not None and len(result.physical.events) == 1
    for t in (0, 0.2, 0.4, 0.6, 1):
        snapshot = result.query(t)
        i = round(t * 50)
        assert snapshot.status == "native"
        assert snapshot.sampled_seconds == t
        assert snapshot.root == result.root_trajectory[i]
        assert snapshot.feet == result.physical.placements.trajectory[2 * i : 2 * i + 2]
        assert snapshot.contact_event == result.contact_events[i]
        assert snapshot.placement_ids == sorted(
            {r.event_id for r in snapshot.feet if r.event_id}
        )
    pivot = result.physical.events[0]
    assert result.query(0.4).pivot_ids == [pivot.pivot.id]
    assert result.query(0.8).pivot_ids == []
    assert pivot.pivot.rotation_rad == pytest.approx(-1)
    assert len(pivot.translations) == 3
    assert all(
        len(t.positions) == len(pivot.rotation_indices) for t in pivot.translations
    )
    assert all(e.geometry.polygon is None for e in result.physical.placements.events)
    snapshot = result.query(0.405)
    assert snapshot.status == "native_snapshot"
    assert snapshot.sampled_seconds == 0.4
    assert snapshot.bracket_seconds == (0.4, 0.42)
    snapshot.feet[0].reasons.append("caller-change")
    assert "caller-change" not in result.query(0.4).feet[0].reasons


def test_missing_root_pelvis_fallback_time_gaps_and_outside_execution() -> None:
    source = motion()
    source.samples[10].root_xyz_world = None
    source.samples[11].root_xyz_world = None
    source.samples[11].landmarks.append(
        Landmark3D(
            name="pelvis",
            xyz_world=(1, 2, 3),
            quality=Quality(state="observed", uncertainty=0.1),
        )
    )
    source.samples = [s for s in source.samples if not 0.4 < s.global_seconds < 0.7]
    result = view(source, ground(source))
    missing = result.query(0.2)
    assert missing.root is not None and missing.root.xy_ground is None
    assert missing.root.quality.state == "unknown"
    fallback = result.query(0.22).root
    assert (
        fallback is not None and fallback.xy_ground == (1, 2) and fallback.z_ground == 3
    )
    assert fallback.reasons == ["pelvis_fallback"]
    assert len(result.root_path_indices) == 3
    assert result.query(0.5).reasons == ["native_time_gap"]
    assert result.query(-0.1).reasons == ["outside_execution"]
    assert result.query(1.1).root is None
    for t in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            result.query(t)
    bad = result.model_dump()
    bad["root_path_indices"] = [list(range(len(source.samples)))]
    with pytest.raises(ValueError, match="gaps"):
        GroundViewSeries.model_validate(bad)


def test_unavailable_ground_and_scale_normalization_are_explicit() -> None:
    source = motion(metric=False, scale=7)
    result = view(source, ground(source, metric=False))
    assert result.world_unit == "arbitrary" and result.physical is not None
    assert result.root_trajectory[0].z_ground == 7
    assert result.physical.placements.normalization == shape(7).measurements[0]
    measures = [m for r in result.physical.placements.relations for m in r.measurements]
    assert measures and not any(m.unit == "m" for m in measures)
    assert any(m.unit == "body_ratio" for m in measures)
    cal = calibration(False)
    cal.ground_status, cal.ground_z, cal.ground_frame = "unresolved", None, None
    unavailable = derive_ground_view(source, cal)
    assert unavailable.ground_status == "unavailable"
    assert unavailable.scene_bounds is None and unavailable.physical is None
    assert len(unavailable.root_trajectory) == len(source.samples)
    assert all(
        p.xy_ground is None and p.z_ground is None for p in unavailable.root_trajectory
    )
    assert unavailable.query(0.4).reasons == ["ground_unavailable"]
    with pytest.raises(ValueError, match="unavailable ground"):
        derive_ground_view(source, cal, result.physical, ground_id="invalid")


def test_publication_roundtrip_lineage_and_parser_independent_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    source, _ = sequence()
    source.samples[10].root_xyz_world = None
    motion_handle, cal = persist(store, source), persist(store, calibration())
    contacts = publish_contacts(store, motion_handle, cal)
    placements = publish_footprints(store, motion_handle, contacts, cal)
    pivots = publish_pivots(store, placements)
    before = pivots.metadata.model_dump_json()
    output = publish_ground_view(store, motion_handle, cal, pivots)
    result = load_ground_view(output)
    assert pivots.metadata.model_dump_json() == before
    assert isinstance(output.metadata, Ground)
    assert isinstance(pivots.metadata, Ground)
    assert output.metadata.samples == pivots.metadata.samples
    for name in (
        "contact_evidence_json",
        "footprint_evidence_json",
        "pivot_evidence_json",
    ):
        assert np.array_equal(output.read_array(name), pivots.read_array(name))
    root = output.read_array("ground_view_root")
    mask = output.read_array("ground_view_root_missing")
    assert root.shape == mask.shape == (len(source.samples), 4)
    assert root[0].tolist() == [0, 0, 0, 1]
    assert mask[10].tolist() == [False, True, True, True]
    assert not mask[0].any()
    assert load_ground_view(output) == result
    changed = publish_ground_view(
        store, motion_handle, cal, pivots, GroundViewConfig(max_gap_seconds=0.1)
    )
    assert changed.path != output.path
    altered = source.model_copy(deep=True)
    altered.samples[0].root_xyz_world = (99, 0, 1)
    with pytest.raises(ValueError, match="revisions"):
        publish_ground_view(store, persist(store, altered), cal, pivots)

    def no_derivation(*args: object, **kwargs: object) -> None:
        raise AssertionError("parser rerun must not regenerate ground-view data")

    monkeypatch.setattr(
        "reconstruction.ground_view.artifact.derive_ground_view", no_derivation
    )
    # Persist two parser revisions referencing the same physical artifact. They
    # must not cause a ground re-derivation or change its physical bytes/IDs.
    semantic = Semantics(
        kind="semantics",
        id="parser-v1",
        schema_version="1.0.0",
        provenance=source.provenance.model_copy(
            update={"producer": "synthetic.parser"}
        ),
        reconstruction_id=source.id,
        ground_id=output.metadata.id,
        execution=Interval(start=0, end=1),
        steps=[],
        stances=[],
        actions=[],
        phases=[],
        keyframes=[],
        relations=[],
    )
    first_parser = persist(store, semantic)
    second_parser = persist(store, semantic.model_copy(update={"id": "parser-v2"}))
    assert first_parser.path != second_parser.path
    assert publish_ground_view(store, motion_handle, cal, pivots).path == output.path
    assert load_ground_view(output) == result
    with pytest.raises(ValueError, match="ground-view artifact"):
        load_ground_view(pivots)


def test_mismatched_times_and_physical_identity_fail_closed() -> None:
    source = motion()
    cal = calibration()
    physical = derive_pivots(derive_footprints(source, ground(source), cal))
    for field, value in (("reconstruction_id", "other"), ("calibration_id", "other")):
        bad = physical.model_copy(deep=True)
        setattr(bad.placements, field, value)
        with pytest.raises(ValueError, match="identities"):
            derive_ground_view(source, cal, bad, ground_id="pivots")
    shifted = source.model_copy(deep=True)
    shifted.samples[0].global_seconds = -0.02
    with pytest.raises(ValueError, match="contact links"):
        derive_ground_view(shifted, cal, physical, ground_id="pivots")


def test_temporal_root_quality_survives_ground_publication(tmp_path: Path) -> None:
    from reconstruction.temporal import (
        TemporalConfig,
        load_temporal_motion,
        publish_temporal_motion,
    )
    from tests.test_articulated import participant, raw_handle

    store = ArtifactStore(StorageRoot(tmp_path))
    raw_source = participant()
    raw_source.provenance.producer = "reconstruction.triangulation"
    for i, sample in enumerate(raw_source.samples):
        sample.global_seconds = i * 0.02
        sample.quality = Quality(
            state="observed", uncertainty=0.02, source_ids=[f"root-native:{i}"]
        )
    temporal = publish_temporal_motion(
        store,
        raw_handle(store, raw_source),
        TemporalConfig(max_position_uncertainty=0.15),
    ).motion
    assert isinstance(temporal.metadata, Reconstruction)
    cal = persist(store, calibration())
    contacts = publish_contacts(store, temporal, cal)
    placements = publish_footprints(store, temporal, contacts, cal)
    pivots = publish_pivots(store, placements)
    expected = [
        Quality.model_validate(q)
        for q in load_temporal_motion(temporal)["root_translation_quality"]
    ]
    assert expected[0].uncertainty == 0.02
    assert expected[0].state == "observed"
    assert expected[0].source_ids == ["root-native:0"]
    assert temporal.metadata.samples[0].quality != expected[0]
    output = publish_ground_view(store, temporal, cal, pivots)
    summary = load_ground_view(output)
    for i, point in enumerate(summary.root_trajectory):
        assert point.quality == expected[i]
        snapshot = summary.query(point.global_seconds)
        assert snapshot.root is not None and snapshot.root.quality == expected[i]
        assert point.reasons == ["root"]
    assert publish_ground_view(store, temporal, cal, pivots).path == output.path
    assert summary.algorithm_revision == "ground-view-v2"
    # Metadata of a real temporal artifact cannot silently replace its root evidence.
    assert summary.physical is not None
    with pytest.raises(ValueError, match="explicit root translation quality"):
        derive_ground_view(
            temporal.metadata,
            calibration(),
            summary.physical,
            ground_id=pivots.metadata.id,
        )


def test_explicit_root_quality_alignment_unknown_and_pelvis_fallback() -> None:
    source = motion()
    qualities = [
        Quality(state="observed", uncertainty=0.02, source_ids=[f"root:{i}"])
        for i in range(len(source.samples))
    ]
    qualities[10] = Quality(state="unknown", source_ids=["root-hidden:10"])
    source.samples[10].root_xyz_world = None
    source.samples[11].root_xyz_world = None
    qualities[11] = Quality(state="unknown")
    pelvis_quality = Quality(
        state="interpolated", uncertainty=0.07, source_ids=["pelvis:11"]
    )
    source.samples[11].landmarks.append(
        Landmark3D(name="pelvis", xyz_world=(1, 2, 3), quality=pelvis_quality)
    )
    physical = view(source, ground(source)).physical
    result = derive_ground_view(
        source,
        calibration(),
        physical,
        ground_id="pivots",
        root_translation_quality=qualities,
    )
    assert result.root_trajectory[0].quality == qualities[0]
    assert result.root_trajectory[10].quality == qualities[10]
    assert result.root_trajectory[10].xy_ground is None
    assert result.root_trajectory[11].quality == pelvis_quality
    assert result.root_trajectory[11].reasons == ["pelvis_fallback"]
    with pytest.raises(ValueError, match="align with native"):
        derive_ground_view(
            source,
            calibration(),
            physical,
            ground_id="pivots",
            root_translation_quality=qualities[:-1],
        )
    assert (
        "metadata_only_root_quality"
        in view(source, ground(source)).root_trajectory[0].reasons
    )
