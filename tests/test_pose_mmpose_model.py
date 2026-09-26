"""Explicit offline check of the provisioned CPU model runtime."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import pytest

from pose.providers.mmpose.adapter import _OpenMMLab, _points
from pose.providers.mmpose.geometry import PixelTransform
from pose.providers.mmpose.hand import (
    HandROI,
    HandROIConfig,
    crop_original,
    map_refinement,
)
from pose.providers.mmpose.mapping import NAMES
from tkd_poomsae.vision.device import inference_job


@pytest.mark.model_required
def test_installed_models_infer_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("model inference attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    backend = _OpenMMLab("cpu")
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    detected = backend.detect(image)
    assert detected.bboxes.shape[-1] == 4
    box = np.array([[0, 0, 64, 64]], dtype=np.float32)
    wholebody = backend.pose(image, box)
    assert len(_points(wholebody[0], NAMES)) == 133
    hand = backend.pose(image, box, hand=True)
    assert len(_points(hand[0], NAMES[91:112])) == 21


@pytest.mark.model_required
def test_pinned_hand_model_on_native_video_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded offline call on source pixels; no accuracy claim."""
    video = os.environ.get("TKD_HAND_TEST_VIDEO")
    if not video:
        pytest.skip("set TKD_HAND_TEST_VIDEO to a locally registered source video")
    assert Path(video).is_file()
    import cv2

    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("model inference attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    reader = cv2.VideoCapture(video)
    try:
        success, frame = reader.read()
    finally:
        reader.release()
    assert success and frame is not None
    height, width = frame.shape[:2]
    assert width >= 512 and height >= 512
    # The crop copies source pixels at native scale before topdown preprocessing.
    x0, y0 = width // 2 - 128, height // 2 - 128
    roi = HandROI(
        "left",
        (x0, y0, x0 + 256, y0 + 256),
        (x0, y0, x0 + 256, y0 + 256),
        PixelTransform(crop_x=x0, crop_y=y0, crop_width=256, crop_height=256),
        1.0,
        256.0,
    )
    crop = crop_original(frame, roi)
    np.testing.assert_array_equal(crop, frame[y0 : y0 + 256, x0 : x0 + 256])
    with inference_job("cpu"):
        backend = _OpenMMLab("cpu")
        samples = backend.pose(
            crop, np.array([[0, 0, 256, 256]], dtype=np.float32), hand=True
        )
    assert len(samples) == 1
    instances = samples[0].pred_instances
    mapped = map_refinement(
        "left",
        np.asarray(instances.keypoints)[0],
        np.asarray(instances.keypoint_scores)[0],
        None,
        roi,
        (width, height),
        HandROIConfig(),
        None,
    )
    assert len(mapped) == 21
    assert mapped[0].name == "left_hand_wrist"
    assert mapped[-1].name == "left_hand_pinky_4"
