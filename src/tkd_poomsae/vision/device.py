"""Bounded inference resources and physical-GPU leases."""

from __future__ import annotations

import ctypes
import fcntl
import re
import subprocess
import time
import uuid as uuidlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from tkd_poomsae.vision.assets import models_root


class DeviceBusy(RuntimeError):
    """A GPU is leased by another process beyond the wait deadline."""


class DeviceCancelled(RuntimeError):
    """Inference was cancelled before a device lease was acquired."""


def _cuda_device_uuid(cuda_index: int) -> str:
    """Ask the CUDA driver for the device at this process's visible ordinal."""
    if cuda_index < 0:
        raise ValueError("CUDA index must be nonnegative")
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise ValueError("CUDA driver library is unavailable") from error
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None)
    if get_uuid is None:
        get_uuid = getattr(driver, "cuDeviceGetUuid", None)
    if get_uuid is None:
        raise ValueError("CUDA driver cannot report a device UUID")
    get_uuid.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
    get_uuid.restype = ctypes.c_int
    device = ctypes.c_int()
    raw = (ctypes.c_ubyte * 16)()
    if (
        driver.cuInit(0) != 0
        or driver.cuDeviceGet(ctypes.byref(device), cuda_index) != 0
    ):
        raise ValueError(f"CUDA device {cuda_index} is unavailable")
    if get_uuid(raw, device.value) != 0:
        raise ValueError(f"CUDA device {cuda_index} UUID lookup failed")
    return f"GPU-{uuidlib.UUID(bytes=bytes(raw))}"


def physical_gpu_uuid(cuda_index: int) -> str:
    """Resolve actual CUDA ordinal to a physical UUID, independent of GPU order."""
    queried = _cuda_device_uuid(cuda_index)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Cannot verify CUDA UUID against physical GPUs") from error
    physical = [value.strip().casefold() for value in result.stdout.splitlines()]
    if physical.count(queried.casefold()) != 1:
        # A MIG UUID is not a physical-GPU UUID. Never lease it independently.
        raise ValueError(f"CUDA device {cuda_index} has no unique physical GPU UUID")
    return queried


@contextmanager
def gpu_lease(
    uuid: str,
    *,
    timeout: float = 60,
    cancelled: Callable[[], bool] | None = None,
) -> Iterator[Path]:
    """One inference process per physical GPU, shared through TKD_DATA_ROOT."""
    if not re.fullmatch(r"GPU-[A-Fa-f0-9-]+", uuid):
        raise ValueError("Invalid physical GPU UUID")
    directory = models_root() / ".leases"
    if directory.is_symlink():
        raise ValueError(f"Symlink in GPU lease directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid}.lock"
    if path.is_symlink():
        raise ValueError(f"Symlink in GPU lease lock: {path}")
    with path.open("a+b") as lock:
        deadline = time.monotonic() + timeout
        while True:
            if cancelled is not None and cancelled():
                raise DeviceCancelled("Cancelled while waiting for GPU lease")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise DeviceBusy(
                        f"GPU {uuid} is leased by another inference process"
                    ) from None
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        try:
            yield path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def inference_job(
    device: str = "cpu",
    *,
    threads: int = 2,
    timeout: float = 60,
    cancelled: Callable[[], bool] | None = None,
) -> Iterator[None]:
    """Bound threads and hold the GPU lease through model teardown."""
    if not 1 <= threads <= 8:
        raise ValueError("CPU threads must be between 1 and 8")
    import cv2
    import torch  # type: ignore[import-not-found]

    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    cv2.setNumThreads(threads)
    if cancelled is not None and cancelled():
        raise DeviceCancelled("Inference cancelled")
    if device == "cpu":
        yield
        return
    if not re.fullmatch(r"cuda:\d+", device) or not torch.cuda.is_available():
        raise ValueError(f"Unavailable or unsupported inference device: {device}")
    index = int(device.partition(":")[2])
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} is not visible")
    with gpu_lease(physical_gpu_uuid(index), timeout=timeout, cancelled=cancelled):
        try:
            yield
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA out of memory on {device}; retry with batch size 1, "
                "fewer simultaneous jobs, or CPU."
            ) from error
