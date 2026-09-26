"""Shared interpreter discovery must not provision or depend on the caller cwd."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tkd_poomsae.vision import runtime


def test_delegate_uses_shared_interpreter_from_another_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "shared"
    python = root / "runtime/vision/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setenv("TKD_DATA_ROOT", str(root))
    monkeypatch.delenv("TKD_VISION_PYTHON", raising=False)
    monkeypatch.delenv("TKD_VISION_ACTIVE", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    observed: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        observed.update(command=command, **kwargs)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(subprocess, "run", run)
    assert runtime.delegate(["observations", "smoke-short"]) == 7
    assert observed["command"][0] == str(python)
    assert observed["command"][-2:] == ["observations", "smoke-short"]
    assert observed["env"]["TKD_VISION_ACTIVE"] == "1"
    assert str(runtime.repository_root()) in observed["env"]["PYTHONPATH"]
    monkeypatch.setenv("TKD_VISION_ACTIVE", "1")
    assert runtime.delegate(["observations", "smoke-short"]) is None


def test_missing_shared_runtime_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TKD_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("TKD_VISION_PYTHON", raising=False)
    monkeypatch.delenv("TKD_VISION_ACTIVE", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="runtime-bootstrap"):
        runtime.delegate(["observations", "smoke-short"])


def test_provisioned_runtime_receipt_reuses_without_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TKD_DATA_ROOT", str(tmp_path))
    shared = tmp_path / "runtime"
    python = shared / "vision/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    (shared / "vision-receipt.json").write_text(
        json.dumps({"revision": runtime.runtime_revision()})
    )

    def deny(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a verified runtime cache hit must not run the installer")

    monkeypatch.setattr(subprocess, "run", deny)
    assert runtime.bootstrap_runtime()["cache_hit"] is True
