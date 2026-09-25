"""Regression checks for empty detections and physical GPU identity."""

from __future__ import annotations

import ctypes
import os
import subprocess
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tkd_poomsae.vision import device
from tkd_poomsae.vision.inference import _pose_boxes

GPU_A = "GPU-11111111-1111-1111-1111-111111111111"
GPU_B = "GPU-22222222-2222-2222-2222-222222222222"
MIG_UUID = "GPU-33333333-3333-3333-3333-333333333333"


def test_empty_detection_only_uses_full_frame_in_smoke_mode() -> None:
    empty = np.empty((0, 4), dtype=np.float32)
    assert _pose_boxes(empty, 128, 96, smoke=False) is None
    np.testing.assert_array_equal(
        _pose_boxes(empty, 128, 96, smoke=True),
        np.array([[0, 0, 128, 96]], dtype=np.float32),
    )
    detected = np.array([[10, 20, 30, 40]], dtype=np.float32)
    assert _pose_boxes(detected, 128, 96, smoke=False) is detected


class _FakeFunction:
    def __init__(self, implementation: Callable[..., int]) -> None:
        self.implementation = implementation
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *args: Any) -> int:
        return self.implementation(*args)


def test_cuda_ordinals_share_physical_lease_across_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cuda_device_get(output: Any, ordinal: int) -> int:
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int)).contents.value = ordinal
        return 0

    def cuda_uuid_get(output: Any, ordinal: int) -> int:
        assert ordinal == 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        selected = (
            GPU_A if visible == "0" else MIG_UUID if visible == "MIG-test" else GPU_B
        )
        ctypes.memmove(output, bytes.fromhex(selected[4:].replace("-", "")), 16)
        return 0

    driver = SimpleNamespace(
        cuInit=_FakeFunction(lambda _flags: 0),
        cuDeviceGet=_FakeFunction(cuda_device_get),
        cuDeviceGetUuid_v2=_FakeFunction(cuda_uuid_get),
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda _library: driver)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=f"{GPU_A}\n{GPU_B}\n"),
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    # CUDA 0 is GPU B here even though nvidia-smi lists GPU A first.
    assert device.physical_gpu_uuid(0) == GPU_B
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert device.physical_gpu_uuid(0) == GPU_B
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert device.physical_gpu_uuid(0) == GPU_A
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-test")
    with pytest.raises(ValueError, match="physical GPU UUID"):
        device.physical_gpu_uuid(0)
