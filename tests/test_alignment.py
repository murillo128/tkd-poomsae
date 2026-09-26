"""Native-time joins, masks and independent alignment cache revisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from contracts.models import (
    DenseArray,
    FrameTime,
    Interval,
    Landmark,
    Landmark2D,
    Observation,
    Provenance,
    Quality,
    RawScore,
    RegionalGeometry2D,
    SubjectCandidateEvidence,
    SubjectSelection,
    Synchronization,
    SyncOffset,
    ViewRegionQuality,
    validate_artifact,
)
from pose.observation_run import ARRAY_ID as WINDOW_ARRAY_ID
from pose.observation_run import load_receipt, load_window
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, StorageRoot, hash_config
from sync.alignment import (
    CameraQuery,
    JoinConfig,
    ObservationJoin,
    load_alignment,
    publish_alignment,
)

PROV = Provenance(producer="synthetic-join", config_digest="0" * 64)
NAMES: tuple[Landmark, ...] = ("left_wrist", "left_index_tip", "left_heel", "nose")


def observation(
    source: str,
    pts: int,
    den: int = 1000,
    *,
    physical_offset: float = 0,
    track: str = "practitioner",
    missing: tuple[Landmark, ...] = (),
) -> Observation:
    time = pts / den
    points = [
        Landmark2D(
            name=name,
            xy_px=None if name in missing else (time + physical_offset, i + 1.0),
            raw_score=RawScore(
                value=1.1, range_min=None, range_max=None, domain="simcc"
            ),
            raw_visibility=1,
            quality=Quality(
                state="unknown" if name in missing else "observed",
                score=None if name in missing else 0.8,
            ),
        )
        for i, name in enumerate(NAMES)
    ]
    return Observation(
        kind="observation",
        id=f"{source}:{pts}",
        schema_version="1.0.0",
        provenance=PROV,
        frame=FrameTime(
            source_id=source,
            camera_id=f"camera-{source}",
            frame_index=pts,
            pts=pts,
            time_base_num=1,
            time_base_den=den,
            source_seconds=time,
            offset_seconds=0,
            global_seconds=time,
        ),
        landmarks=points,
        wholebody_landmarks=[p.model_copy(deep=True) for p in points],
        refined_landmarks=[points[1].model_copy(deep=True)],
        subject_selection=SubjectSelection(
            state="selected",
            track_id=track,
            candidate_index=0,
            method="initial",
            candidates=[
                SubjectCandidateEvidence(
                    index=0,
                    bbox_xyxy_px=(0, 0, 10, 10),
                    detector_score=RawScore(value=0.9, range_min=0, range_max=1),
                )
            ],
        ),
        regional_geometry=[
            RegionalGeometry2D(
                part="left_foot",
                availability="missing" if "left_heel" in missing else "complete",
                provider="synthetic-foot",
                supporting_landmarks=["left_heel"],
                orientation_state="unavailable"
                if "left_heel" in missing
                else "available",
                axis_start_px=None
                if "left_heel" in missing
                else (time + physical_offset, 3),
                axis_end_px=None
                if "left_heel" in missing
                else (time + physical_offset + 1, 3),
                orientation_rad=None if "left_heel" in missing else 0,
            )
        ],
    )


def synchronization(
    a: float = 0, b: float = 0, *, revision: str = "sync"
) -> Synchronization:
    return Synchronization(
        kind="synchronization",
        id=revision,
        schema_version="1.0.0",
        provenance=PROV,
        reference_source_id="a",
        offsets=[
            SyncOffset(
                source_id=source,
                automatic_seconds=offset,
                quality=Quality(state="observed", score=0.9),
            )
            for source, offset in (("a", a), ("b", b))
        ],
    )


def point(camera: CameraQuery, name: str = "left_wrist") -> Landmark2D:
    return next(p for p in camera.landmarks if p.name == name)


def test_mixed_rates_nonzero_pts_fractional_offsets_and_original_evidence() -> None:
    offsets = (0.0, -1.025)
    frames = [observation("a", i, 30) for i in range(60, 67)] + [
        observation("b", i, 60, physical_offset=offsets[1]) for i in range(180, 195)
    ]
    before = [obs.model_dump(mode="json") for obs in frames]
    join = ObservationJoin(frames, synchronization(*offsets))
    query = join.query(2.05)
    assert query.available_sources == ["a", "b"]
    assert query.landmark_sources["left_index_tip"] == ["a", "b"]
    for camera in query.cameras:
        assert camera.state == "bracket"
        assert camera.source_seconds == pytest.approx(
            2.05 - (0 if camera.source_id == "a" else offsets[1])
        )
        assert camera.weights == pytest.approx([0.5, 0.5])
        assert point(camera).xy_px == pytest.approx((2.05, 1))
        assert point(camera).quality.state == "interpolated"
        assert point(camera).raw_score is None
        assert len(camera.refined_landmarks) == 1
        assert camera.regional_geometry[0].axis_start_px == pytest.approx((2.05, 3))
        assert camera.endpoints[0].landmarks[0].raw_score is not None
        assert camera.endpoints[0].landmarks[0].raw_score.value == 1.1
        assert camera.endpoints[0].frame.offset_seconds == 0
    assert [endpoint.frame.pts for endpoint in query.cameras[0].endpoints] == [61, 62]
    assert [endpoint.frame.pts for endpoint in query.cameras[1].endpoints] == [184, 185]
    assert [obs.model_dump(mode="json") for obs in frames] == before
    exact = join.query(2.0).cameras[0]
    assert exact.state == "exact" and exact.weights == [1]
    assert exact.endpoints[0].frame.pts == 60
    assert point(exact).quality.state == "observed"
    assert point(exact).raw_score == frames[0].landmarks[0].raw_score
    assert join.query(2.05).cameras[0].max_bracket_seconds == pytest.approx(2 / 30)


@pytest.mark.parametrize(
    "failure", ["missing", "occluded", "identity", "ambiguous", "untracked"]
)
def test_invalid_endpoints_retain_unknown(failure: str) -> None:
    frames = [observation("a", i) for i in (0, 100, 200)]
    if failure == "missing":
        frames[1] = observation("a", 100, missing=("left_index_tip",))
    elif failure == "occluded":
        frames[1].region_quality = [
            ViewRegionQuality(part="left_hand", usable=False, reasons=["occluded"])
        ]
    elif failure == "identity":
        assert frames[1].subject_selection is not None
        frames[1].subject_selection.track_id = "other-person"
    elif failure == "ambiguous":
        frames[1].subject_selection = SubjectSelection(state="ambiguous")
    else:
        frames[1].subject_selection = None
    camera = ObservationJoin(frames, synchronization()).query(0.05).cameras[0]
    assert camera.state == "bracket" and len(camera.endpoints) == 2
    assert point(camera, "left_index_tip").xy_px is None
    assert point(camera, "left_index_tip").quality.state == "unknown"
    assert camera.missing_masks["refined_landmarks"] == [True]
    if failure in {"missing", "occluded"}:
        assert point(camera).xy_px is not None
    else:
        assert all(camera.missing_masks["landmarks"])
        assert "identity_unavailable_or_changed" in camera.reasons


def test_sparse_landmarks_are_union_and_have_independent_view_sets() -> None:
    frames = [observation(source, i) for source in ("a", "b") for i in (0, 100, 200)]
    frames[1].landmarks = [p for p in frames[1].landmarks if p.name != "left_index_tip"]
    frames[4] = observation("b", 100, missing=("nose",))
    query = ObservationJoin(frames, synchronization()).query(0.05)
    assert len(query.cameras[0].landmarks) == len(NAMES)
    assert query.landmark_sources["left_index_tip"] == ["b"]
    assert query.landmark_sources["nose"] == ["a"]
    assert query.available_sources == ["a", "b"]


def test_local_gap_bounds_coverage_and_no_extrapolation() -> None:
    frames = [observation("a", i) for i in (0, 100, 200, 350, 450, 1500, 1600)]
    frames += [observation("b", i) for i in (100, 200, 300)]
    join = ObservationJoin(frames, synchronization())
    assert point(join.query(0.275).cameras[0]).quality.state == "interpolated"
    long = join.query(0.7).cameras[0]
    assert long.local_median_seconds == pytest.approx(0.1)
    assert long.max_bracket_seconds == pytest.approx(0.2)
    assert "long_gap" in long.reasons and point(long).xy_px is None
    assert join.query(-0.01).available_sources == []
    assert len(join.query(-0.01).cameras[0].landmarks) == len(NAMES)
    assert all(join.query(-0.01).cameras[0].missing_masks["landmarks"])
    assert join.query(1.7).available_sources == []
    assert join.query(0.05).available_sources == ["a"]
    assert join.query(0.25).available_sources == ["a", "b"]
    assert join.query(0.4).available_sources == ["a"]
    coverage = join.coverage()
    assert [(span.start, span.end) for span in coverage[0].usable] == [
        (0, 0.45),
        (1.5, 1.6),
    ]
    assert [(span.start, span.end) for span in coverage[1].usable] == [(0.1, 0.3)]
    bounded = ObservationJoin(
        frames, synchronization(), JoinConfig(max_bracket_seconds=0.12)
    )
    assert point(bounded.query(0.275).cameras[0]).xy_px is None
    assert point(join.query(0.2).cameras[0]).quality.state == "observed"


def test_offset_revision_and_reference_clock_changes() -> None:
    frames = [
        observation(source, i) for source in ("a", "b") for i in (2000, 2100, 2200)
    ]
    original = ObservationJoin(frames, synchronization(0, -0.025)).query(2.05)
    shifted_sync = synchronization(-0.5, -0.525, revision="new-reference")
    shifted_sync.reference_source_id = "b"
    shifted = ObservationJoin(frames, shifted_sync).query(1.55)
    for a, b in zip(original.cameras, shifted.cameras):
        assert a.source_seconds == pytest.approx(b.source_seconds)
        assert a.weights == pytest.approx(b.weights)
        assert [obs.id for obs in a.endpoints] == [obs.id for obs in b.endpoints]
        assert point(a).xy_px == pytest.approx(point(b).xy_px)
    revised = synchronization()
    revised.offsets[1].manual_seconds = 0.05
    revised.offsets[1].manual_author = "operator"
    revised.offsets[1].manual_reason = "test correction"
    camera = ObservationJoin(frames, revised).query(2.05).cameras[1]
    assert camera.state == "exact" and camera.offset_seconds == 0.05
    assert camera.endpoints[0].frame.pts == 2000


def test_excluded_empty_and_synchronized_coverage() -> None:
    frames = [observation("a", i) for i in (0, 100, 200)]
    sync = synchronization()
    sync.offsets[0].source_interval = Interval(start=0.05, end=0.15)
    join = ObservationJoin(frames, sync)
    assert join.query(0).available_sources == []
    assert join.query(0.1).available_sources == ["a"]
    assert join.query(0.1).cameras[1].reasons == ["no_observations"]
    assert [(s.start, s.end) for s in join.coverage()[0].usable] == [(0.05, 0.15)]
    sync.offsets[0].retained = False
    sync.offsets[0].automatic_seconds = None
    sync.offsets[0].exclusion_reason = "bad camera"
    excluded = ObservationJoin(frames, sync)
    assert excluded.query(0.1).cameras[0].source_seconds is None
    assert excluded.coverage()[0].reason == "bad camera"


def test_snapshot_inputs_and_outputs_cannot_change_later_queries() -> None:
    frames = [observation("a", i) for i in (0, 100, 200)]
    sync = synchronization()
    join = ObservationJoin(frames, sync)
    result = join.query(0.05)
    frames[0].landmarks.clear()
    sync.offsets[0].automatic_seconds = 99
    result.cameras[0].endpoints[0].landmarks.clear()
    result.cameras[0].regional_geometry[0].supporting_landmarks.clear()
    again = join.query(0.05)
    assert len(again.cameras[0].landmarks) == len(NAMES)
    assert again.cameras[0].endpoints[0].landmarks
    assert again.cameras[0].regional_geometry[0].supporting_landmarks == ["left_heel"]


@pytest.mark.parametrize("invalid", ["duplicate", "wrong_source", "camera", "no_pts"])
def test_reject_invalid_native_inputs(invalid: str) -> None:
    frames = [observation("a", i) for i in (0, 100, 200)]
    if invalid == "duplicate":
        frames.append(frames[0])
    elif invalid == "wrong_source":
        frames[0].frame.source_id = "unknown"
    elif invalid == "camera":
        frames[0].frame.camera_id = "different"
    else:
        frames[0].frame.pts = None
        frames[0].frame.time_base_num = None
        frames[0].frame.time_base_den = None
    with pytest.raises(ValueError):
        ObservationJoin(frames, synchronization())


def test_invalid_time_and_configuration() -> None:
    join = ObservationJoin([observation("a", 0)], synchronization())
    for invalid in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            join.query(invalid)
    assert join.coverage()[0].usable[0].start == join.coverage()[0].usable[0].end == 0
    for settings in (
        {"max_bracket_factor": 0},
        {"local_radius": 0},
        {"exact_tolerance_seconds": 0.1},
        {"max_bracket_seconds": -1},
    ):
        with pytest.raises(ValidationError):
            JoinConfig.model_validate(settings)


def window(store: ArtifactStore, frames: list[Observation]) -> ArtifactHandle:
    records = [
        {
            "observation": obs.model_dump(mode="json"),
            "model_identity": {},
            "inference_settings": {},
            "hand_observations": {},
            "hand_roi_to_source": {},
            "source_orientation": {},
            "tracker_state": {},
        }
        for obs in frames
    ]
    array = np.frombuffer(
        json.dumps({"version": 1, "records": records}).encode(), dtype=np.uint8
    )
    key = ArtifactKey(
        layer="observation",
        inputs={"source": hash_config(records)},
        schema_version="1.0.0",
        algorithm_revision="test",
        config_digest=PROV.config_digest,
    )
    metadata = frames[0].model_copy(
        update={
            "arrays": [
                DenseArray(
                    id=WINDOW_ARRAY_ID,
                    dtype="uint8",
                    shape=[len(array)],
                    axes=["json_byte"],
                )
            ]
        }
    )
    return store.get_or_create(key, lambda: (metadata, {WINDOW_ARRAY_ID: array}))


def sync_handle(store: ArtifactStore, sync: Synchronization) -> ArtifactHandle:
    key = ArtifactKey(
        layer="synchronization",
        inputs={"sync": hash_config(sync.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision="test",
        config_digest=PROV.config_digest,
    )
    return store.get_or_create(key, lambda: (sync, {}))


def test_persistence_cache_and_offset_changes_reuse_native_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    frames = [
        observation("a", i, missing=("left_index_tip",) if i == 100 else ())
        for i in (0, 100, 200)
    ]
    native = window(store, frames)
    initial_files = {p: p.read_bytes() for p in native.path.rglob("*") if p.is_file()}
    sync = sync_handle(store, synchronization())
    handle = publish_alignment(store, [native], sync, [0.05, 1])
    assert (
        validate_artifact(handle.metadata.model_dump(mode="json")).kind == "alignment"
    )
    queries, coverage = load_alignment(handle)
    assert point(queries[0].cameras[0]).quality.state == "interpolated"
    assert point(queries[0].cameras[0], "left_index_tip").quality.state == "unknown"
    assert queries[0].cameras[0].weights == [0.5, 0.5]
    assert queries[0].cameras[0].endpoints[1].frame.pts == 100
    assert queries[1].available_sources == []
    assert coverage
    # A cache hit cannot even reload windows, much less run inference.
    import sync.alignment as module

    with monkeypatch.context() as patch:
        patch.setattr(
            module, "load_window", lambda *_: pytest.fail("cache reloaded input")
        )
        assert publish_alignment(store, [native], sync, [0.05, 1]).path == handle.path
    new_sync = sync_handle(store, synchronization(0.025, 0, revision="offset-revision"))
    revised = publish_alignment(store, [native], new_sync, [0.05, 1])
    assert revised.path != handle.path
    assert load_alignment(revised)[0][0].cameras[0].weights == pytest.approx(
        [0.75, 0.25]
    )
    assert publish_alignment(store, [native], sync, [0.1]).path != handle.path
    assert (
        publish_alignment(
            store, [native], sync, [0.05, 1], JoinConfig(max_bracket_factor=1)
        ).path
        != handle.path
    )
    assert {p: p.read_bytes() for p in initial_files} == initial_files
    assert load_window(native) == frames
    with pytest.raises(ValueError, match="nonempty"):
        publish_alignment(store, [native], sync, [])


@pytest.mark.local_data
def test_local_native_observations_are_read_only() -> None:
    store = ArtifactStore()
    receipts = store.root.namespace("derived") / "observation-runs"
    paths = sorted(receipts.glob("*.json"))
    path = next(
        (
            p
            for p in paths
            if json.loads(p.read_text())["identity"]["selection"] == "smoke-short"
        ),
        None,
    )
    if path is None:
        pytest.skip("registered smoke-short native observations unavailable")
    before = path.read_bytes()
    receipt, observations = load_receipt(path, store)
    native_paths = [
        store.get(ArtifactKey(**entry["key"])).path for entry in receipt["windows"]
    ]
    import hashlib

    native_hashes = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in native_paths
        for p in directory.rglob("*")
        if p.is_file()
    }
    sources = sorted({obs.frame.source_id for obs in observations})
    assert len(sources) >= 2
    sync = Synchronization(
        kind="synchronization",
        id="offline-functional-offsets",
        schema_version="1.0.0",
        provenance=PROV,
        reference_source_id=sources[0],
        offsets=[
            SyncOffset(
                source_id=source,
                automatic_seconds=i * 0.0125,
                quality=Quality(state="unknown"),
            )
            for i, source in enumerate(sources)
        ],
    )
    join = ObservationJoin(observations, sync)
    time = observations[len(observations) // 4].frame.source_seconds
    result = join.query(time)
    assert len(result.cameras) >= 2
    assert any(camera.endpoints for camera in result.cameras)
    assert any(len(camera.wholebody_landmarks) > 20 for camera in result.cameras)
    assert path.read_bytes() == before
    assert receipt["frame_count"] == len(observations)
    assert all(obs.frame.offset_seconds == 0 for obs in observations)

    assert {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in native_hashes
    } == native_hashes


def test_higher_rate_grid_retains_native_frequency_and_geometry_quality() -> None:
    frames = [observation("a", i, 30) for i in range(7)]
    queries = ObservationJoin(frames, synchronization()).query_many(
        [i / 60 for i in range(13)]
    )
    assert (
        sum(point(query.cameras[0]).quality.state == "observed" for query in queries)
        == 7
    )
    assert (
        sum(
            point(query.cameras[0]).quality.state == "interpolated" for query in queries
        )
        == 6
    )
    for i, query in enumerate(queries):
        camera = query.cameras[0]
        assert camera.regional_quality["left_foot"].state == (
            "observed" if i % 2 == 0 else "interpolated"
        )


def test_regional_axes_do_not_bridge_missing_support_or_provider_changes() -> None:
    a, b = observation("a", 0), observation("a", 100)
    b.regional_geometry[0].provider = "another-foot-provider"
    join = ObservationJoin([a, b], synchronization())
    camera = join.query(0.05).cameras[0]
    assert point(camera, "left_heel").quality.state == "interpolated"
    assert camera.regional_geometry[0].availability == "missing"
    assert camera.regional_quality["left_foot"].state == "unknown"
    assert [obs.regional_geometry[0].provider for obs in camera.endpoints] == [
        "synthetic-foot",
        "another-foot-provider",
    ]
    b.regional_geometry[0].provider = a.regional_geometry[0].provider
    b.regional_geometry[0].axis_end_px = (-0.9, 3)
    b.regional_geometry[0].orientation_rad = 3.141592653589793
    degenerate = ObservationJoin([a, b], synchronization()).query(0.05).cameras[0]
    assert degenerate.regional_geometry[0].orientation_state == "degenerate"
    assert degenerate.regional_geometry[0].orientation_rad is None
    assert degenerate.regional_quality["left_foot"].state == "interpolated"
