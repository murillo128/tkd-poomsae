"""Synthetic native-time identity and regional evidence sequences."""

from __future__ import annotations

from dataclasses import replace

from contracts.models import FrameTime, Observation, Provenance, RawScore
from pose.providers.mmpose.adapter import (
    NamedPoint,
    PersonCandidate,
    PoseFrame,
    canonical_landmarks,
)
from pose.providers.mmpose.geometry import PixelTransform
from pose.providers.mmpose.hand import HandObservation, HandROI, RefinedPoint
from pose.providers.mmpose.mapping import NAMES
from pose.regions import WholebodyRegionalProvider
from pose.stream import PractitionerTracker

PROVENANCE = Provenance(producer="test", config_digest="0" * 64)


def candidate(
    index: int,
    x: float,
    *,
    size: float = 80,
    missing: set[str] | None = None,
    crossing: bool = False,
    hand_outside: bool = False,
) -> PersonCandidate:
    missing = missing or set()
    positions = {
        "left_shoulder": (x + 25, 30),
        "right_shoulder": (x + 55, 30),
        "left_hip": (x + 30, 60),
        "right_hip": (x + 50, 60),
        "left_wrist": (x + (60 if crossing else 20), 45),
        "right_wrist": (x + (20 if crossing else 60), 45),
        "left_heel": (x + 30, 85),
        "left_big_toe": (x + 35, 92),
        "left_small_toe": (x + 25, 92),
        "right_heel": (x + 50, 85),
        "right_big_toe": (x + 55, 92),
        "right_small_toe": (x + 45, 92),
        "left_eye": (x + 35, 15),
        "right_eye": (x + 45, 15),
        "nose": (x + 40, 22),
    }
    points = tuple(
        NamedPoint(
            name,
            (x + 20, 45)
            if hand_outside and name.startswith("left_hand_")
            else positions.get(name, (x + 40, 50)),
            RawScore(value=0.01 if name in missing else 0.91, range_min=0, range_max=1),
            0.0 if name in missing else 1.0,
        )
        for name in NAMES
    )
    hand = HandObservation(
        "left",
        points[91:112],
        tuple(
            RefinedPoint(name, None, 0.78, 0.0, "unknown", "out_of_frame")
            for name in NAMES[91:112]
        ),
        None,
        "skipped",
        False,
    )
    raw = PersonCandidate(
        index,
        (x, 0, x + size, size + 20),
        RawScore(value=0.55 if index == 0 else 0.99, range_min=0, range_max=1),
        points,
        {},
        {},
        {"left": hand} if hand_outside else {},
    )
    return PersonCandidate(
        **{
            **raw.__dict__,
            "regional_geometry": WholebodyRegionalProvider()(canonical_landmarks(raw)),
        }
    )


def frame(ordinal: int, *candidates: PersonCandidate, camera: str = "cam") -> PoseFrame:
    seconds = ordinal / 30
    return PoseFrame(
        frame=FrameTime(
            source_id="source",
            camera_id=camera,
            frame_index=ordinal,
            pts=ordinal,
            time_base_num=1,
            time_base_den=30,
            source_seconds=seconds,
            offset_seconds=0,
            global_seconds=seconds,
        ),
        ordinal=ordinal,
        image_size=(400, 100),
        candidates=tuple(candidates),
        model_identity={},
        inference_settings={},
    )


def observe(
    tracker: PractitionerTracker, item: PoseFrame, *, operator: int | None = None
) -> Observation:
    return tracker.observe(
        item,
        artifact_id=f"obs-{item.ordinal}",
        provenance=PROVENANCE,
        operator_candidate_index=operator,
    )


def test_distractor_loss_and_explicit_recovery() -> None:
    tracker = PractitionerTracker()
    first = observe(tracker, frame(0, candidate(0, 50), candidate(1, 250, size=35)))
    assert first.subject_selection is not None
    assert first.subject_selection.candidate_index == 0
    assert first.subject_selection.method == "initial"
    # A higher detector score and changing candidate index cannot steal the track.
    second = observe(tracker, frame(1, candidate(0, 250, size=35), candidate(1, 53)))
    assert second.subject_selection is not None
    assert second.subject_selection.candidate_index == 1
    assert second.subject_selection.method == "temporal"
    assert second.subject_selection.candidates[0].detector_score.value == 0.55
    absent = observe(tracker, frame(2))
    assert absent.subject_selection is not None
    assert absent.subject_selection.state == "missing"
    assert all(point.xy_px is None for point in absent.landmarks)
    recovered = observe(tracker, frame(3, candidate(0, 56), candidate(1, 250, size=35)))
    assert recovered.subject_selection is not None
    assert recovered.subject_selection.candidate_index == 0
    long_gap = observe(tracker, frame(40, candidate(0, 60)))
    assert long_gap.subject_selection is not None
    assert long_gap.subject_selection.state == "ambiguous"
    assert "identity_gap" in long_gap.subject_selection.reasons
    manual = observe(tracker, frame(41, candidate(0, 60)), operator=0)
    assert manual.subject_selection is not None
    assert manual.subject_selection.method == "operator"
    assert manual.subject_selection.track_id == first.subject_selection.track_id


def test_crossing_missing_feet_and_hand_quality_survive_json() -> None:
    tracker = PractitionerTracker()
    observe(tracker, frame(0, candidate(0, 50)))
    crossing = observe(
        tracker,
        frame(
            1,
            candidate(
                0,
                50,
                crossing=True,
                hand_outside=True,
                missing={"right_heel", "right_big_toe", "right_small_toe"},
            ),
        ),
    )
    restored = Observation.model_validate_json(crossing.model_dump_json())
    regions = {item.part: item for item in restored.region_quality}
    assert "anatomical_side_ambiguous" in regions["left_hand"].reasons
    assert "missing_geometry" in regions["right_foot"].reasons
    assert regions["left_foot"].usable
    assert restored.subject_selection is not None
    assert restored.subject_selection.candidates[0].detector_score.value == 0.55
    original = {point.name: point for point in restored.wholebody_landmarks}
    selected = {point.name: point for point in restored.landmarks}
    assert original["right_heel"].raw_score is not None
    assert selected["right_heel"].raw_score is not None
    assert original["right_heel"].raw_score.value == 0.01
    assert original["right_heel"].raw_visibility == 0.0
    assert selected["right_heel"].xy_px is None
    assert selected["right_heel"].raw_score.value == 0.01
    assert any(
        point.raw_score is not None and point.raw_score.value == 0.78
        for point in restored.refined_landmarks
    )
    assert selected["left_index_tip"].xy_px is None


def test_fast_limb_motion_keeps_body_identity_and_camera_isolation() -> None:
    tracker = PractitionerTracker()
    observe(tracker, frame(0, candidate(0, 50)))
    moved = candidate(0, 52)
    points = tuple(
        NamedPoint(point.name, (300, 5), point.raw_score, point.raw_visibility)
        if point.name == "left_wrist"
        else point
        for point in moved.landmarks
    )
    moved = PersonCandidate(**{**moved.__dict__, "landmarks": points})
    result = observe(tracker, frame(1, moved))
    assert result.subject_selection is not None
    assert result.subject_selection.state == "selected"
    assert {point.name: point for point in result.landmarks}["left_wrist"].xy_px == (
        300,
        5,
    )
    try:
        observe(tracker, frame(2, moved, camera="other"))
    except ValueError as error:
        assert "one camera" in str(error)
    else:
        raise AssertionError("mixed-camera track was accepted")


def test_ambiguous_initial_candidates_and_torso_jump_are_explicit() -> None:
    tracker = PractitionerTracker()
    ambiguous = observe(tracker, frame(0, candidate(0, 40), candidate(1, 230, size=78)))
    assert ambiguous.subject_selection is not None
    assert ambiguous.subject_selection.state == "ambiguous"
    assert ambiguous.subject_selection.candidate_index is None
    observe(tracker, frame(1, candidate(0, 40)), operator=0)
    jumped = candidate(0, 41)
    points = tuple(
        NamedPoint(point.name, (point.xy_px[0] + 75, point.xy_px[1]), point.raw_score)
        if point.name == "left_shoulder"
        else point
        for point in jumped.landmarks
    )
    jumped = PersonCandidate(**{**jumped.__dict__, "landmarks": points})
    flagged = observe(tracker, frame(2, jumped))
    body = next(item for item in flagged.region_quality if item.part == "body")
    assert not body.usable
    assert "implausible_torso_jump" in body.reasons
    assert flagged.wholebody_landmarks


def test_invisible_foot_stays_unknown_despite_high_raw_scores() -> None:
    source = candidate(0, 50)
    foot_names = {"left_heel", "left_big_toe", "left_small_toe"}
    points = tuple(
        replace(point, raw_visibility=0.0) if point.name in foot_names else point
        for point in source.landmarks
    )
    source = replace(
        source,
        landmarks=points,
        regional_geometry=WholebodyRegionalProvider()(
            canonical_landmarks(replace(source, landmarks=points))
        ),
    )
    restored = Observation.model_validate_json(
        observe(PractitionerTracker(), frame(0, source)).model_dump_json()
    )
    quality = {part.part: part for part in restored.region_quality}
    assert not quality["left_foot"].usable
    assert "low_raw_visibility" in quality["left_foot"].reasons
    assert "missing_landmarks" in quality["left_foot"].reasons
    assert quality["right_foot"].usable
    assert restored.regional_geometry[0].availability == "missing"
    selected = {point.name: point for point in restored.landmarks}
    coarse = {point.name: point for point in restored.wholebody_landmarks}
    for name in ("left_heel", "left_forefoot", "left_foot_outer"):
        assert selected[name].xy_px is None
        assert selected[name].quality.state == "unknown"
        assert selected[name].raw_score == coarse[name].raw_score
        assert selected[name].raw_visibility == coarse[name].raw_visibility == 0.0


def test_skipped_hand_is_unusable_without_crossing() -> None:
    source = candidate(0, 50, hand_outside=True)
    restored = Observation.model_validate_json(
        observe(PractitionerTracker(), frame(0, source)).model_dump_json()
    )
    quality = {part.part: part for part in restored.region_quality}
    assert not quality["left_hand"].usable
    assert "tiny_roi" in quality["left_hand"].reasons
    assert "refinement_skipped" in quality["left_hand"].reasons
    assert "anatomical_side_ambiguous" not in quality["left_hand"].reasons
    selected = {point.name: point for point in restored.landmarks}
    coarse = {point.name: point for point in restored.wholebody_landmarks}
    assert selected["left_index_tip"].xy_px is None
    assert selected["left_index_tip"].quality.state == "unknown"
    assert coarse["left_index_tip"].xy_px is not None
    assert restored.refined_landmarks


def test_good_refined_hand_remains_usable_when_foot_is_missing() -> None:
    source = candidate(
        0, 50, missing={"right_heel", "right_big_toe", "right_small_toe"}
    )
    roi = HandROI(
        "left",
        (0, 0, 100, 100),
        (0, 0, 100, 100),
        PixelTransform(crop_width=100, crop_height=100),
        1.0,
        30.0,
    )
    hand = HandObservation(
        "left",
        source.landmarks[91:112],
        tuple(
            RefinedPoint(name, (75.0, 45.0), 0.8, 1.0, "observed")
            for name in NAMES[91:112]
        ),
        roi,
        "refined",
        False,
    )
    source = replace(source, hand_observations={"left": hand})
    result = observe(PractitionerTracker(), frame(0, source))
    quality = {part.part: part for part in result.region_quality}
    assert quality["left_hand"].usable
    assert not quality["right_foot"].usable
    selected = {point.name: point for point in result.landmarks}
    coarse = {point.name: point for point in result.wholebody_landmarks}
    assert selected["left_index_tip"].xy_px == (75, 45)
    assert selected["left_index_tip"].raw_score is not None
    assert coarse["left_index_tip"].raw_score is not None
    assert selected["left_index_tip"].raw_score.value == 0.8
    assert coarse["left_index_tip"].raw_score.value == 0.91


def test_crossed_foot_sides_are_ambiguous_without_relabeling_raw_points() -> None:
    first = candidate(0, 50)
    tracker = PractitionerTracker()
    observe(tracker, frame(0, first))
    source = candidate(0, 50)
    before = {point.name: point for point in source.landmarks}
    points = tuple(
        replace(
            point,
            xy_px=before[point.name.replace("left_", "right_", 1)].xy_px,
        )
        if point.name in {"left_heel", "left_big_toe", "left_small_toe"}
        else replace(
            point,
            xy_px=before[point.name.replace("right_", "left_", 1)].xy_px,
        )
        if point.name in {"right_heel", "right_big_toe", "right_small_toe"}
        else point
        for point in source.landmarks
    )
    source = replace(
        source,
        landmarks=points,
        regional_geometry=WholebodyRegionalProvider()(
            canonical_landmarks(replace(source, landmarks=points))
        ),
    )
    restored = Observation.model_validate_json(
        observe(tracker, frame(1, source)).model_dump_json()
    )
    selected = {point.name: point for point in restored.landmarks}
    raw = {point.name: point for point in restored.wholebody_landmarks}
    quality = {part.part: part for part in restored.region_quality}
    assert restored.subject_selection is not None
    assert restored.subject_selection.state == "selected"
    for part in ("left_foot", "right_foot"):
        assert not quality[part].usable
        assert "anatomical_side_ambiguous" in quality[part].reasons
    assert selected["left_heel"].xy_px is None
    assert selected["right_heel"].xy_px is None
    assert raw["left_heel"].xy_px == before["right_heel"].xy_px
    assert raw["right_heel"].xy_px == before["left_heel"].xy_px
    assert {item.part: item.availability for item in restored.regional_geometry}[
        "left_foot"
    ] == "missing"
    repeated = observe(tracker, frame(2, source))
    repeated_quality = {part.part: part for part in repeated.region_quality}
    repeated_points = {point.name: point for point in repeated.landmarks}
    assert repeated.subject_selection is not None
    assert repeated.subject_selection.state == "selected"
    assert not repeated_quality["left_foot"].usable
    assert not repeated_quality["right_foot"].usable
    assert "anatomical_side_ambiguous" in repeated_quality["left_foot"].reasons
    assert "anatomical_side_ambiguous" in repeated_quality["right_foot"].reasons
    assert repeated_points["left_heel"].xy_px is None
    assert repeated_points["right_heel"].xy_px is None
    recovered = observe(tracker, frame(3, candidate(0, 50)))
    recovered_quality = {part.part: part for part in recovered.region_quality}
    assert recovered_quality["left_foot"].usable
    assert recovered_quality["right_foot"].usable


def test_out_of_frame_foot_geometry_is_retained_only_as_source() -> None:
    source = candidate(0, 50)
    names = {"left_heel", "left_big_toe", "left_small_toe"}
    points = tuple(
        replace(point, xy_px=(480.0, point.xy_px[1])) if point.name in names else point
        for point in source.landmarks
    )
    source = replace(
        source,
        landmarks=points,
        regional_geometry=WholebodyRegionalProvider()(
            canonical_landmarks(replace(source, landmarks=points))
        ),
    )
    restored = Observation.model_validate_json(
        observe(PractitionerTracker(), frame(0, source)).model_dump_json()
    )
    quality = {part.part: part for part in restored.region_quality}
    derived = {item.part: item for item in restored.regional_geometry}
    source_geometry = {item.part: item for item in restored.source_regional_geometry}
    assert not quality["left_foot"].usable
    assert "out_of_frame" in quality["left_foot"].reasons
    assert derived["left_foot"].availability == "missing"
    assert derived["left_foot"].axis_start_px is None
    assert source_geometry["left_foot"].availability == "complete"
    assert source_geometry["left_foot"].axis_start_px == (480, 85)
    assert quality["right_foot"].usable
