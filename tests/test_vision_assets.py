"""Offline asset publication and cross-process GPU lease checks."""

from __future__ import annotations

import hashlib
import io
import multiprocessing as mp
import os
import time
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any

import pytest

from tkd_poomsae.vision import assets
from tkd_poomsae.vision.device import DeviceBusy, DeviceCancelled, gpu_lease


def _holder(root: str, ready: Event) -> None:
    os.environ["TKD_DATA_ROOT"] = root
    with gpu_lease("GPU-1234", timeout=2):
        ready.set()
        time.sleep(30)


def test_bootstrap_verifies_and_reuses_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"official pinned test asset"
    record = {
        "path": "mmpose/configs/test.py",
        "url": "https://example.invalid/fixed/test.py",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "kind": "config",
    }
    monkeypatch.setenv("TKD_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(assets, "registry", lambda: {"assets": [record]})
    with pytest.raises(assets.ModelAssetError, match="models bootstrap"):
        assets.verified_paths()
    assert assets.bootstrap(opener=lambda _url: io.BytesIO(payload)) == [record["path"]]

    def no_network(_url: str) -> io.BytesIO:
        raise AssertionError("cache hit attempted network access")

    assert assets.bootstrap(opener=no_network) == []
    assert assets.verified_paths()[record["path"]].read_bytes() == payload
    assets.verified_paths()[record["path"]].write_bytes(b"tampered")
    with pytest.raises(assets.ModelAssetError, match="changed"):
        assets.bootstrap(opener=no_network)


def test_bootstrap_rejects_bad_download_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TKD_DATA_ROOT", str(tmp_path))
    record: dict[str, Any] = {
        "path": "checkpoints/test.pth",
        "url": "https://example.invalid/fixed/test.pth",
        "sha256": "0" * 64,
        "kind": "checkpoint",
    }
    monkeypatch.setattr(assets, "registry", lambda: {"assets": [record]})
    with pytest.raises(assets.ModelAssetError, match="hash mismatch"):
        assets.bootstrap(opener=lambda _url: io.BytesIO(b"wrong"))
    assert not (tmp_path / "models/checkpoints/test.pth").exists()
    assert not list((tmp_path / "models/checkpoints").glob(".download-*"))


def test_gpu_lease_contention_cancel_and_owner_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TKD_DATA_ROOT", str(tmp_path))
    context = mp.get_context("fork")
    ready = context.Event()
    owner = context.Process(target=_holder, args=(str(tmp_path), ready))
    owner.start()
    try:
        assert ready.wait(5)
        with pytest.raises(DeviceBusy):
            with gpu_lease("GPU-1234", timeout=0.15):
                pytest.fail("contender acquired held GPU")
        with pytest.raises(DeviceCancelled):
            with gpu_lease("GPU-1234", cancelled=lambda: True):
                pytest.fail("cancelled contender acquired GPU")
    finally:
        owner.terminate()
        owner.join(5)
    assert owner.exitcode is not None
    with gpu_lease("GPU-1234", timeout=2):
        pass
