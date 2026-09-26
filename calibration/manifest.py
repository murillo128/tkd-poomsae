"""Explicit optional calibration-capture input, independent of dataset metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from calibration.cameras import IntrinsicProfile
from calibration.target import (
    BoardSpec,
    Capture,
    Detection,
    estimate_calibration,
    observe_capture,
    persist_calibration,
)
from contracts.models import Calibration, Intrinsics
from storage import ArtifactHandle, ArtifactStore, StorageRoot


def load_manifest(
    path: Path,
) -> tuple[BoardSpec, list[tuple[Path, Capture]], dict[str, IntrinsicProfile]]:
    """Read a versioned JSON manifest with explicit camera/capture identities."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("calibration manifest version must be 1")
    if set(data) != {"version", "board", "captures", "profiles"}:
        raise ValueError("manifest requires version, board, captures and profiles")
    board = BoardSpec(**data["board"])
    captures: list[tuple[Path, Capture]] = []
    for record in data["captures"]:
        if not isinstance(record, dict) or "image_file" not in record:
            raise ValueError("capture requires image_file")
        image_file = Path(record["image_file"])
        if not image_file.is_absolute():
            image_file = path.parent / image_file
        fields = {key: value for key, value in record.items() if key != "image_file"}
        capture = Capture(**fields)
        captures.append((image_file, capture))
    profiles: dict[str, IntrinsicProfile] = {}
    for camera_id, record in data["profiles"].items():
        if not isinstance(record, dict):
            raise ValueError("profile must be an object")
        profile = IntrinsicProfile(
            camera_id=camera_id, width_px=record["width_px"],
            height_px=record["height_px"],
            crop_xywh=tuple(record["crop_xywh"]),
            rotation_cw=record["rotation_cw"],
            intrinsics=Intrinsics.model_validate(record["intrinsics"]),
        )
        profiles[camera_id] = profile
    return board, captures, profiles


def calibrate_from_manifest(
    path: Path, data_root: Path,
) -> tuple[Calibration, ArtifactHandle]:
    board, captures, profiles = load_manifest(path)
    observations: list[Detection] = [
        observe_capture(image_file, capture, board)
        for image_file, capture in captures
    ]
    candidate = estimate_calibration(board, observations, profiles)
    handle = persist_calibration(ArtifactStore(StorageRoot(data_root)),
                                 candidate, observations)
    return candidate, handle
