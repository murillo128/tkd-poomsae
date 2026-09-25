"""Bounded inference resources and physical-GPU leases."""

from __future__ import annotations

import fcntl
import os
import re
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from tkd_poomsae.vision.assets import models_root


class DeviceBusy(RuntimeError):
    """A GPU is leased by another process beyond the wait deadline."""


class DeviceCancelled(RuntimeError):
    """Inference was cancelled before a device lease was acquired."""


def physical_gpu_uuid(cuda_index: int) -> str:
    """Map a CUDA-visible index to an NVIDIA physical GPU UUID, or fail closed."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    physical = {}
    for line in result.stdout.splitlines():
        index, uuid = (part.strip() for part in line.split(",", 1))
        physical[int(index)] = uuid
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entries = [part.strip() for part in visible.split(",")]
        if cuda_index >= len(entries):
            raise ValueError(f"CUDA device {cuda_index} is not visible")
        entry = entries[cuda_index]
        if entry.startswith("MIG-"):
            raise ValueError("MIG CUDA visibility needs explicit parent-GPU mapping")
        uuid = physical[int(entry)] if entry.isdigit() else entry
    else:
        uuid = physical[cuda_index]
    if uuid not in physical.values() or not re.fullmatch(r"GPU-[A-Fa-f0-9-]+", uuid):
        raise ValueError(f"Cannot establish physical GPU UUID for cuda:{cuda_index}")
    return uuid


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
    import torch  # type: ignore[import-not-found]

    torch.set_num_threads(threads)
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
