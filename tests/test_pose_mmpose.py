"""Recorded-shape model doubles; no model download or remote inference."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from contracts.models import FrameTime
from media.reader import DecodedFrame, FrameRef
from pose.providers.mmpose.adapter import (
    MMPoseAdapter,
    PersonCandidate,
    _points,
    canonical_landmarks,
)
from pose.providers.mmpose.geometry import PixelTransform
from pose.providers.mmpose.mapping import NAMES
from tkd_poomsae.vision.device import DeviceCancelled


def _sample(count: int, *, bad: bool = False) -> Any:
    xy = np.stack((np.arange(count), np.arange(count) + 0.5), axis=-1)
    if bad:
        xy = xy[:-1]
    return SimpleNamespace(
        pred_instances=SimpleNamespace(
            keypoints=xy[None, :, :],
            keypoint_scores=np.full((1, count), 0.75),
            keypoints_visible=np.ones((1, count)),
        )
    )


def test_full_topology_and_canonical_projection() -> None:
    points = _points(_sample(133), NAMES)
    assert len(points) == len({point.name for point in points}) == 133
    assert points[17].name == "left_big_toe"
    assert points[91].name == "left_hand_wrist"
    assert points[132].name == "right_hand_pinky_4"
    candidate = cast(PersonCandidate, SimpleNamespace(landmarks=points))
    canonical = canonical_landmarks(candidate)
    by_name = {point.name: point for point in canonical}
    assert len(canonical) == len(by_name) == 63
    assert by_name["left_forefoot"].xy_px == (17.0, 17.5)
    assert by_name["right_pinky_tip"].raw_score is not None
    assert by_name["right_pinky_tip"].raw_score.value == 0.75
    assert by_name["left_wrist"].quality.score is None
    with pytest.raises(ValueError, match="topology"):
        _points(_sample(133, bad=True), NAMES)
    invalid_score = _sample(133)
    invalid_score.pred_instances.keypoint_scores[0, 20] = 1.1
    with pytest.raises(ValueError, match="score outside"):
        _points(invalid_score, NAMES)
    invalid_visibility = _sample(133)
    invalid_visibility.pred_instances.keypoints_visible = np.ones((1, 132))
    with pytest.raises(ValueError, match="visibility topology"):
        _points(invalid_visibility, NAMES)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_resize_letterbox_crop_rotation_round_trip(rotation: Any) -> None:
    transform = PixelTransform(
        crop_x=37,
        crop_y=18,
        crop_width=160,
        crop_height=90,
        clockwise=rotation,
        scale_x=1.3,
        scale_y=0.8,
        pad_x=12,
        pad_y=22,
    )
    for source in ((37.0, 18.0), (81.25, 50.5), (196.0, 107.0)):
        assert transform.to_source(transform.to_model(source)) == pytest.approx(source)


class _Backend:
    def __init__(self, _device: str) -> None:
        self.empty = False
        self.seen: list[np.ndarray] = []
        self.hand_calls = 0
        self.crossed = False
        self.empty_hand = False
        self.occluded_thumb_tip = False
        self.framework_versions = {"mmpose": "test", "mmdet": "test"}

    def detect(self, image: np.ndarray) -> Any:
        self.seen.append(image)
        scores = [] if self.empty else [0.9, 0.7]
        return SimpleNamespace(
            labels=_Cpu(np.array([0] * len(scores))),
            scores=_Cpu(np.array(scores)),
            bboxes=_Cpu(
                np.array([[1, 2, 100, 80], [3, 4, 90, 70]][: len(scores)]).reshape(
                    -1, 4
                )
            ),
        )

    def pose(
        self, image: np.ndarray, boxes: np.ndarray, *, hand: bool = False
    ) -> list[Any]:
        if hand:
            self.hand_calls += 1
            if self.empty_hand:
                return []
            sample = _sample(21)
            sample.pred_instances.keypoints[0, :, 0] = image.shape[1] / 2
            sample.pred_instances.keypoints[0, :, 1] = image.shape[0] / 2
            return [sample for _ in boxes]
        sample = _sample(133)
        right_x = 38 if self.crossed else 85
        for start, center_x in ((91, 35), (112, right_x)):
            sample.pred_instances.keypoints[0, start : start + 21, 0] = (
                center_x + np.linspace(-9, 9, 21)
            )
            sample.pred_instances.keypoints[0, start : start + 21, 1] = (
                50 + np.linspace(-9, 9, 21)
            )
        sample.pred_instances.keypoints[0, 9] = [35, 50]
        sample.pred_instances.keypoints[0, 10] = [right_x, 50]
        sample.pred_instances.keypoints[0, 7] = [35, 25]
        sample.pred_instances.keypoints[0, 8] = [right_x, 25]
        if self.occluded_thumb_tip:
            sample.pred_instances.keypoint_scores[0, 95] = 0.01
            sample.pred_instances.keypoints_visible[0, 95] = 0
        return [sample for _ in boxes]


class _Cpu:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def cpu(self) -> np.ndarray:
        return self.array


def _frame() -> tuple[Any, DecodedFrame]:
    ref = FrameRef(0, 45, 1, 30, True)
    time = FrameTime(
        source_id="source",
        camera_id="cam",
        pts=45,
        time_base_num=1,
        time_base_den=30,
        source_seconds=1.5,
        offset_seconds=0,
        global_seconds=1.5,
    )
    recording = SimpleNamespace(
        source=SimpleNamespace(width_px=128, height_px=96),
        frame_time=lambda _ref: time,
    )
    return recording, DecodedFrame(ref, np.zeros((96, 128, 3), dtype=np.uint8))


def test_native_frames_multiple_candidates_empty_and_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pose.providers.mmpose.adapter as module

    monkeypatch.setattr(module, "inference_job", lambda *_a, **_kw: nullcontext())
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(COLOR_RGB2BGR=1, cvtColor=lambda image, _code: image),
    )
    backend = _Backend("cpu")
    adapter = MMPoseAdapter(max_people=2, _backend_factory=lambda _device: backend)
    recording, decoded = _frame()
    result = list(adapter.infer(recording, [decoded]))
    assert len(result) == 1
    assert result[0].frame.pts == 45
    assert result[0].frame.camera_id == "cam"
    assert len(result[0].candidates) == 2
    assert len(result[0].candidates[0].landmarks) == 133
    assert {region.part for region in result[0].candidates[0].regional_geometry} == {
        "left_foot",
        "right_foot",
        "head",
    }
    assert result[0].model_identity["wholebody"]["framework"] == "mmpose"
    assert len(result[0].candidates[0].refined_hands["left"]) == 21
    assert "left" in result[0].candidates[0].refined_hand_boxes
    alternate = MMPoseAdapter(
        _backend_factory=lambda _device: backend,
        regional_provider=lambda _landmarks: (),
    )
    alternate_frame = list(alternate.infer(recording, [decoded]))[0]
    assert alternate_frame.candidates[0].regional_geometry == ()
    backend.empty = True
    assert list(adapter.infer(recording, [decoded]))[0].candidates == ()
    with pytest.raises(ValueError, match="max_frames"):
        list(
            MMPoseAdapter(max_frames=1, _backend_factory=lambda _device: backend).infer(
                recording, [decoded, decoded]
            )
        )
    with pytest.raises(DeviceCancelled):
        list(
            MMPoseAdapter(
                cancelled=lambda: True, _backend_factory=lambda _device: backend
            ).infer(recording, [decoded])
        )


def test_adapter_reuses_native_crop_and_refinement_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pose.providers.mmpose.adapter as module

    monkeypatch.setattr(module, "inference_job", lambda *_a, **_kw: nullcontext())
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(COLOR_RGB2BGR=1, cvtColor=lambda image, _code: image),
    )
    backend = _Backend("cpu")
    recording, decoded = _frame()
    recording.sha256 = "a" * 64
    adapter = MMPoseAdapter(
        max_people=1,
        hand_cache_root=tmp_path,
        _backend_factory=lambda _device: backend,
    )
    first = list(adapter.infer(recording, [decoded]))[0].candidates[0]
    second = list(adapter.infer(recording, [decoded]))[0].candidates[0]
    assert backend.hand_calls == 2
    assert first.hand_observations["left"].cache_identity == (
        second.hand_observations["left"].cache_identity
    )
    assert first.landmarks == second.landmarks  # Immutable coarse observations.
    assert len(first.hand_observations["left"].refined) == 21


def test_overlapping_hands_are_flagged_and_not_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pose.providers.mmpose.adapter as module

    monkeypatch.setattr(module, "inference_job", lambda *_a, **_kw: nullcontext())
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(COLOR_RGB2BGR=1, cvtColor=lambda image, _code: image),
    )
    backend = _Backend("cpu")
    backend.crossed = True
    recording, decoded = _frame()
    candidate = list(
        MMPoseAdapter(max_people=1, _backend_factory=lambda _device: backend).infer(
            recording, [decoded]
        )
    )[0].candidates[0]
    assert all(
        item.handedness_ambiguous for item in candidate.hand_observations.values()
    )
    assert all(not values for values in candidate.refined_hands.values())
    assert all(
        point.state != "observed"
        for item in candidate.hand_observations.values()
        for point in item.refined
    )


def test_empty_hand_model_result_keeps_explicit_missing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pose.providers.mmpose.adapter as module

    monkeypatch.setattr(module, "inference_job", lambda *_a, **_kw: nullcontext())
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(COLOR_RGB2BGR=1, cvtColor=lambda image, _code: image),
    )
    backend = _Backend("cpu")
    backend.empty_hand = True
    recording, decoded = _frame()
    candidate = list(
        MMPoseAdapter(max_people=1, _backend_factory=lambda _device: backend).infer(
            recording, [decoded]
        )
    )[0].candidates[0]
    assert candidate.hand_observations["left"].status == "empty"
    assert len(candidate.hand_observations["left"].refined) == 21
    assert all(
        point.reason == "empty_model_result"
        for point in candidate.hand_observations["left"].refined
    )


def test_adapter_does_not_promote_one_occluded_thumb_tip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pose.providers.mmpose.adapter as module

    monkeypatch.setattr(module, "inference_job", lambda *_a, **_kw: nullcontext())
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(COLOR_RGB2BGR=1, cvtColor=lambda image, _code: image),
    )
    backend = _Backend("cpu")
    backend.occluded_thumb_tip = True
    recording, decoded = _frame()
    candidate = list(
        MMPoseAdapter(max_people=1, _backend_factory=lambda _device: backend).infer(
            recording, [decoded]
        )
    )[0].candidates[0]
    left = candidate.hand_observations["left"]
    assert left.roi is not None
    assert left.roi.coarse_support_fraction == pytest.approx(20 / 21)
    assert left.refined[4].state == "inferred"
    assert left.refined[4].raw_score == 0.75
    assert left.refined[4].name == "left_hand_thumb_4"
    assert "left_hand_thumb_4" not in {
        point.name for point in candidate.refined_hands["left"]
    }
