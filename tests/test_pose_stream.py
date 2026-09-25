"""Synthetic native-time identity and regional evidence sequences."""

from __future__ import annotations

from contracts.models import FrameTime, Observation, Provenance, RawScore
from pose.providers.mmpose.adapter import (
    NamedPoint,
    PersonCandidate,
    PoseFrame,
    canonical_landmarks,
)
from pose.providers.mmpose.hand import HandObservation, RefinedPoint
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


def test_fast_hand_motion_keeps_body_identity_and_camera_isolation() -> None:
    tracker = PractitionerTracker()
    observe(tracker, frame(0, candidate(0, 50)))
    moved = candidate(0, 52)
    points = tuple(
        NamedPoint(point.name, (300, 5), point.raw_score, point.raw_visibility)
        if point.name == "left_hand_index_4"
        else point
        for point in moved.landmarks
    )
    moved = PersonCandidate(**{**moved.__dict__, "landmarks": points})
    result = observe(tracker, frame(1, moved))
    assert result.subject_selection is not None
    assert result.subject_selection.state == "selected"
    assert {point.name: point for point in result.landmarks}[
        "left_index_tip"
    ].xy_px == (300, 5)
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
