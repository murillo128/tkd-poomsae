"""Explicit persistent vision provisioning and offline interpreter discovery."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from storage import StorageRoot, hash_config, hash_file


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / "vision/requirements.lock").is_file():
        raise RuntimeError("vision provisioning requires the repository checkout")
    return root


def runtime_python() -> Path:
    override = os.environ.get("TKD_VISION_PYTHON")
    if override:
        path = Path(override)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                "TKD_VISION_PYTHON must name an existing absolute interpreter"
            )
        return path
    return StorageRoot.from_env().path / "runtime/vision/bin/python"


def runtime_revision() -> str:
    root = repository_root()
    return hash_config(
        {
            "vision_lock": hash_file(root / "vision/requirements.lock"),
            "media_lock": hash_file(root / "vision/media-requirements.lock"),
        }
    )


def delegate(arguments: list[str]) -> int | None:
    """Use the configured local interpreter; never provision on an ordinary run."""
    if os.environ.get("TKD_VISION_ACTIVE") == "1" or all(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "mmpose", "mmdet", "av")
    ):
        return None
    python = runtime_python()
    if not python.is_file():
        raise RuntimeError(
            f"Missing shared vision interpreter {python}; run "
            "`tkd-poomsae models runtime-bootstrap` once on this host"
        )
    root = repository_root()
    environment = dict(os.environ)
    environment["TKD_VISION_ACTIVE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    return subprocess.run(
        [str(python), "-m", "tkd_poomsae.cli", *arguments],
        env=environment,
        check=False,
    ).returncode


def bootstrap_runtime() -> dict[str, Any]:
    """Build one hash-locked CPU interpreter outside all implementation worktrees."""
    root = repository_root()
    shared = StorageRoot.from_env().path / "runtime"
    if shared.is_symlink():
        raise ValueError("shared runtime directory must not be a symlink")
    shared.mkdir(parents=True, exist_ok=True)
    target = shared / "vision"
    if target.is_symlink() or root == target or root in target.parents:
        raise ValueError("vision runtime must be outside the implementation worktree")
    receipt_path = shared / "vision-receipt.json"
    revision = runtime_revision()
    python = target / "bin/python"
    with (shared / "vision-bootstrap.lock").open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if receipt_path.is_file() and python.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("revision") == revision:
                return {"python": str(python), "revision": revision, "cache_hit": True}
        if not python.is_file():
            subprocess.run(["uv", "venv", "--python", "3.11", str(target)], check=True)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "pip==26.2.1",
                "setuptools==80.10.2",
                "wheel",
            ],
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "sync",
                "--python",
                str(python),
                str(root / "vision/requirements.lock"),
                str(root / "vision/media-requirements.lock"),
                "--require-hashes",
                "--no-build-isolation",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-c",
                "import av,torch,mmcv,mmengine,mmdet,mmpose; "
                "from mmcv.ops import nms; assert av.__version__ == '16.1.0'",
            ],
            check=True,
        )
        receipt = {"python": str(python), "revision": revision, "cache_hit": False}
        temporary = shared / ".vision-receipt.json"
        temporary.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        temporary.replace(receipt_path)
        return receipt
