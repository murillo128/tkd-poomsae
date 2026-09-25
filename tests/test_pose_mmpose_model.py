"""Explicit offline check of the provisioned CPU model runtime."""

from __future__ import annotations

import socket

import numpy as np
import pytest

from pose.providers.mmpose.adapter import _OpenMMLab, _points
from pose.providers.mmpose.mapping import NAMES


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
