"""Native source-pixel hand refinement geometry and evidence regression."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from contracts.models import RawScore
from pose.providers.mmpose.adapter import NamedPoint
from pose.providers.mmpose.geometry import PixelTransform
from pose.providers.mmpose.hand import (
    HandRefinementCache,
    HandROIConfig,
    crop_original,
    identity_ambiguous,
    localize_hand,
    map_refinement,
)
from pose.providers.mmpose.mapping import NAMES


def _wholebody(
    left: tuple[float, float] = (20, 50),
    right: tuple[float, float] = (80, 50),
    *,
    flat: bool = False,
) -> tuple[NamedPoint, ...]:
    locations = np.full((133, 2), 50.0)
    for start, center in ((91, left), (112, right)):
        for i in range(21):
            offset = 0 if flat else i - 10
            locations[start + i] = (center[0] + offset, center[1] + offset / 2)
    locations[9], locations[10] = left, right
    locations[7] = (left[0], left[1] - 25)
    locations[8] = (right[0], right[1] - 25)
    score = RawScore(value=0.9, range_min=0, range_max=1)
    return tuple(
        NamedPoint(name, (float(locations[i, 0]), float(locations[i, 1])), score, 1.0)
        for i, name in enumerate(NAMES)
    )


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_edge_crop_padding_rotation_mirror_round_trip(
    rotation: Any, mirror: bool
) -> None:
    config = HandROIConfig(
        clockwise=rotation, mirror_left=mirror, min_visible_fraction=0.3
    )
    roi = localize_hand(_wholebody(left=(4, 4)), "left", (100, 100), config)
    assert roi is not None
    assert roi.xyxy_px[0] < 0 and roi.xyxy_px[1] < 0
    assert roi.padding_px[0] > 0 and roi.padding_px[1] > 0
    source = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    crop = crop_original(source, roi)
    assert crop.shape[:2] == (
        roi.xyxy_px[3] - roi.xyxy_px[1],
        roi.xyxy_px[2] - roi.xyxy_px[0],
    )
    for xy in ((0.0, 0.0), (4.5, 6.0), (12.0, 14.0)):
        mapped = roi.transform.to_model(xy)
        assert roi.transform.to_source(mapped) == pytest.approx(xy)
        affine = roi.roi_to_source_affine
        assert (
            affine[0][0] * mapped[0] + affine[0][1] * mapped[1] + affine[0][2],
            affine[1][0] * mapped[0] + affine[1][1] * mapped[1] + affine[1][2],
        ) == pytest.approx(xy)
        x, y = map(int, mapped)
        if xy == (0.0, 0.0):
            np.testing.assert_array_equal(crop[y, x], source[0, 0])
    outside = roi.transform.to_model((roi.xyxy_px[0], roi.xyxy_px[1]))
    assert (crop[int(outside[1]), int(outside[0])] == 0).all()


def test_per_finger_identity_missing_and_implausible_refinement() -> None:
    config = HandROIConfig()
    roi = localize_hand(_wholebody(), "left", (100, 100), config)
    assert roi is not None
    # Distinct model coordinates survive the crop-to-source inverse by name.
    xy = np.array(
        [
            [
                roi.transform.to_model((20 + i / 4, 50))[0],
                roi.transform.to_model((20 + i / 4, 50))[1],
            ]
            for i in range(21)
        ]
    )
    scores = np.full(21, 0.9)
    scores[7] = 0.05
    scores[8] = 1.0008281469345093
    visible = np.ones(21)
    visible[11] = 0
    result = map_refinement(
        "left", xy, scores, visible, roi, (100, 100), config, (10, 45)
    )
    assert len({point.name for point in result}) == 21
    assert result[4].name == "left_hand_thumb_4"
    assert result[8].name == "left_hand_index_4"
    assert result[7].state == "unknown" and result[7].xy_px is None
    assert result[7].raw_score == 0.05
    assert result[11].reason == "low_model_evidence"
    assert result[8].xy_px == pytest.approx((22, 50))
    assert result[8].raw_score == 1.0008281469345093
    off_frame = xy.copy()
    off_frame[20] = roi.transform.to_model((-5, 50))
    truncated = map_refinement(
        "left", off_frame, scores, visible, roi, (100, 100), config, (10, 45)
    )
    assert truncated[20].reason == "outside_source"
    bad = map_refinement(
        "left", xy + 500, scores, visible, roi, (100, 100), config, (10, 45)
    )
    assert all(point.reason == "implausible_wrist" for point in bad)
    assert bad[0].raw_score == 0.9


def test_sparse_coarse_support_does_not_promote_model_points() -> None:
    points = list(_wholebody())
    for index in range(91, 112):
        if index not in (91, 101, 111):
            points[index] = replace(
                points[index], raw_score=RawScore(value=0.01, range_min=0, range_max=1)
            )
    roi = localize_hand(tuple(points), "left", (100, 100), HandROIConfig())
    assert roi is not None and roi.coarse_support_fraction < 0.5
    xy = np.tile(np.array(roi.transform.to_model((20, 50))), (21, 1))
    refined = map_refinement(
        "left",
        xy,
        np.ones(21),
        None,
        roi,
        (100, 100),
        HandROIConfig(),
        None,
    )
    assert all(point.state == "inferred" for point in refined)


def test_one_occluded_finger_stays_uncertain_with_high_model_score() -> None:
    points = list(_wholebody())
    thumb_tip = 91 + 4
    points[thumb_tip] = replace(
        points[thumb_tip],
        raw_score=RawScore(value=0.01, range_min=0, range_max=1),
        raw_visibility=0.0,
    )
    coarse_hand = tuple(points[91:112])
    config = HandROIConfig()
    roi = localize_hand(tuple(points), "left", (100, 100), config)
    assert roi is not None and roi.coarse_support_fraction == pytest.approx(20 / 21)
    xy = np.tile(np.asarray(roi.transform.to_model((20, 50))), (21, 1))
    refined = map_refinement(
        "left",
        xy,
        np.full(21, 0.9),
        None,
        roi,
        (100, 100),
        config,
        coarse_hand[0].xy_px,
        coarse_hand,
    )
    assert len(refined) == 21
    assert refined[4].name == "left_hand_thumb_4"
    assert refined[4].state == "inferred"
    assert refined[4].reason == "unsupported_coarse_point"
    assert refined[4].raw_score == 0.9
    assert refined[4].xy_px == pytest.approx((20, 50))
    assert all(point.state == "observed" for i, point in enumerate(refined) if i != 4)


def test_tiny_hand_and_crossing_identity() -> None:
    config = HandROIConfig()
    assert localize_hand(_wholebody(flat=True), "left", (100, 100), config) is None
    assert (
        localize_hand(_wholebody(left=(-80, -80)), "left", (100, 100), config) is None
    )
    left = localize_hand(
        _wholebody(left=(50, 50), right=(54, 50)), "left", (100, 100), config
    )
    right = localize_hand(
        _wholebody(left=(50, 50), right=(54, 50)), "right", (100, 100), config
    )
    assert identity_ambiguous(left, right)
    assert left is not None and right is not None
    assert left.side == "left" and right.side == "right"


def test_crop_and_model_output_cache_is_immutable(tmp_path: Path) -> None:
    root = tmp_path
    cache = HandRefinementCache(root)
    crop = np.arange(30, dtype=np.uint8).reshape(2, 5, 3)
    calls = 0

    def run() -> tuple[np.ndarray, np.ndarray, None]:
        nonlocal calls
        calls += 1
        return np.zeros((21, 2)), np.ones(21), None

    identity: dict[str, object] = {
        "source": "a",
        "model": "b",
        "roi": asdict(HandROIConfig()),
    }
    first_key, first = cache.get_or_compute(identity, crop, run)
    second_key, second = cache.get_or_compute(identity, crop, run)
    assert calls == 1 and first_key == second_key
    assert first is not None and second is not None
    np.testing.assert_array_equal(first[0], second[0])
    assert (root / first_key / "crop.npy").is_file()
    other_key, _ = cache.get_or_compute({**identity, "model": "c"}, crop, run)
    assert other_key != first_key and calls == 2
    with pytest.raises(ValueError, match="crop identity"):
        cache.get_or_compute(identity, crop + 1, run)


def test_pixel_transform_non_square_mirror_and_padding() -> None:
    transform = PixelTransform(
        crop_x=-4,
        crop_y=8,
        crop_width=80,
        crop_height=40,
        clockwise=90,
        mirror_x=True,
        scale_x=1.5,
        scale_y=0.75,
        pad_x=9,
        pad_y=11,
    )
    for source in ((-4.0, 8.0), (30.5, 24.25), (75.0, 47.0)):
        assert transform.to_source(transform.to_model(source)) == pytest.approx(source)
