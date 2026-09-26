"""Functional wholebody topology and view-space regional evidence."""

from __future__ import annotations

import pytest

from contracts.models import (
    FrameTime,
    Observation,
    Provenance,
    RawScore,
    RegionalGeometry2D,
)
from pose.providers.mmpose.adapter import (
    NamedPoint,
    PersonCandidate,
    canonical_landmarks,
)
from pose.providers.mmpose.geometry import PixelTransform
from pose.providers.mmpose.mapping import NAMES
from pose.regions import WholebodyRegionalProvider


def candidate(
    locations: dict[str, tuple[float, float]],
    *,
    missing: set[str] | None = None,
) -> PersonCandidate:
    missing = missing or set()
    points = tuple(
        NamedPoint(
            name=name,
            xy_px=locations.get(name, (1.0, 1.0)),
            raw_score=RawScore(
                value=0.0 if name in missing else 0.9, range_min=0, range_max=1
            ),
        )
        for name in NAMES
    )
    return PersonCandidate(
        0,
        (0, 0, 100, 100),
        RawScore(value=0.9, range_min=0, range_max=1),
        points,
        {},
        {},
    )


def locations() -> dict[str, tuple[float, float]]:
    return {
        "left_heel": (20, 50),
        "left_big_toe": (30, 30),
        "left_small_toe": (40, 30),
        "right_heel": (70, 50),
        "right_big_toe": (60, 30),
        "right_small_toe": (80, 30),
        "nose": (50, 25),
        "left_eye": (45, 15),
        "right_eye": (55, 15),
        "left_ear": (40, 17),
        "right_ear": (60, 17),
        "left_shoulder": (40, 60),
        "right_shoulder": (60, 60),
    }


def test_foot_identity_anatomical_sides_and_head_support() -> None:
    observed = candidate(locations())
    canonical = canonical_landmarks(observed)
    by_name = {point.name: point for point in canonical}
    assert by_name["left_heel"].xy_px == (20, 50)
    assert by_name["left_forefoot"].xy_px == (30, 30)
    assert by_name["left_foot_outer"].xy_px == (40, 30)
    assert by_name["right_heel"].xy_px == (70, 50)
    for name in (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
    ):
        assert name in by_name
        assert by_name[name].raw_score is not None

    regions = {region.part: region for region in WholebodyRegionalProvider()(canonical)}
    assert all(region.availability == "complete" for region in regions.values())
    assert all(region.orientation_state == "available" for region in regions.values())
    assert regions["left_foot"].axis_start_px == (20, 50)
    assert regions["left_foot"].axis_end_px == (35, 30)
    assert regions["left_foot"].supporting_landmarks == [
        "left_heel",
        "left_forefoot",
        "left_foot_outer",
    ]
    assert regions["right_foot"].axis_start_px == (70, 50)
    assert regions["right_foot"].axis_end_px == (70, 30)
    assert regions["head"].axis_start_px == (50, 15)
    assert regions["head"].axis_end_px == (50, 25)


def test_missing_collinear_and_foreshortened_are_independent() -> None:
    positions = locations()
    positions["left_big_toe"] = (30, 40)
    positions["left_small_toe"] = (40, 30)
    positions["nose"] = (50, 15)
    observed = candidate(positions, missing={"right_heel"})
    regions = {
        region.part: region
        for region in WholebodyRegionalProvider()(canonical_landmarks(observed))
    }
    assert regions["left_foot"].availability == "complete"
    assert regions["left_foot"].orientation_state == "degenerate"
    assert regions["left_foot"].axis_start_px == (20, 50)
    assert regions["left_foot"].orientation_rad is None
    assert regions["right_foot"].availability == "partial"
    assert regions["right_foot"].orientation_state == "unavailable"
    assert regions["right_foot"].axis_start_px is None
    assert regions["head"].availability == "complete"
    assert regions["head"].orientation_state == "degenerate"
    assert regions["head"].orientation_rad is None

    positions = locations()
    positions["left_big_toe"] = (20.5, 50)
    positions["left_small_toe"] = (20.5, 50.5)
    region = WholebodyRegionalProvider()(canonical_landmarks(candidate(positions)))[0]
    assert region.orientation_state == "degenerate"
    assert region.orientation_rad is None

    partial = WholebodyRegionalProvider()(
        canonical_landmarks(candidate(locations(), missing={"left_small_toe"}))
    )[0]
    assert partial.availability == "partial"
    assert partial.orientation_state == "unavailable"
    assert partial.axis_start_px == (20, 50)
    assert partial.axis_end_px == (30, 30)

    absent = WholebodyRegionalProvider()(
        canonical_landmarks(
            candidate(
                locations(),
                missing={
                    "left_heel",
                    "left_big_toe",
                    "left_small_toe",
                    "nose",
                    "left_eye",
                    "right_eye",
                    "left_shoulder",
                    "right_shoulder",
                },
            )
        )
    )
    assert absent[0].availability == "missing"
    assert absent[0].axis_start_px is None
    assert absent[1].orientation_state == "available"
    assert absent[2].availability == "missing"


def test_original_pixel_transform_and_observation_round_trip() -> None:
    transform = PixelTransform(
        crop_x=20,
        crop_y=10,
        crop_width=100,
        crop_height=80,
        clockwise=90,
        scale_x=2,
        scale_y=2,
        pad_x=8,
        pad_y=4,
    )
    original = locations()
    source_points = {
        name: transform.to_source(transform.to_model(xy))
        for name, xy in original.items()
    }
    observed = candidate(source_points)
    landmarks = canonical_landmarks(observed)
    regions = WholebodyRegionalProvider()(landmarks)
    frame = FrameTime(
        source_id="source",
        camera_id="camera",
        pts=30,
        time_base_num=1,
        time_base_den=30,
        source_seconds=1,
        offset_seconds=0,
        global_seconds=1,
    )
    artifact = Observation(
        kind="observation",
        id="obs",
        schema_version="1.0.0",
        provenance=Provenance(
            producer="pose",
            model="wholebody",
            model_version="fixture",
            config_digest="0" * 64,
        ),
        frame=frame,
        landmarks=landmarks,
        regional_geometry=list(regions),
    )
    recovered = Observation.model_validate_json(artifact.model_dump_json())
    by_name = {point.name: point for point in recovered.landmarks}
    assert by_name["left_heel"].xy_px == pytest.approx((20, 50))
    assert by_name["left_heel"].raw_score is not None
    assert by_name["left_heel"].raw_score.value == 0.9
    assert recovered.frame.source_id == "source"
    assert recovered.provenance.model == "wholebody"
    assert {point.name for point in recovered.landmarks} >= {
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
    }
    assert recovered.regional_geometry == list(regions)


def test_region_contract_rejects_orientation_on_missing_geometry() -> None:
    region = WholebodyRegionalProvider()(canonical_landmarks(candidate(locations())))[0]
    with pytest.raises(ValueError, match="unavailable region"):
        RegionalGeometry2D.model_validate(
            region.model_dump() | {"orientation_state": "degenerate"}
        )
