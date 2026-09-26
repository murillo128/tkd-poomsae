"""Resolve a persisted natural-scene candidate using explicit evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calibration.ground import (
    GroundEvidence,
    SizeEvidence,
    persist_scene_calibration,
    resolve_scene,
)
from calibration.natural import SceneCandidate
from calibration.natural_cli import load_synchronization_artifact
from calibration.quality import QualityThresholds, require_synchronization
from storage import ArtifactStore, StorageRoot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sync-artifact", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path)
    args = parser.parse_args()
    data: dict[str, Any] = json.loads(args.candidate.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("route") != "natural_scene_v1":
        raise ValueError("unsupported natural-scene candidate")
    candidate = SceneCandidate(
        status=data["status"],
        reasons=data["reasons"],
        cameras=data["cameras"],
        static_points=data["static_points"],
        evidence=data["evidence"],
    )
    require_synchronization(
        candidate,
        load_synchronization_artifact(args.sync_artifact),
    )
    supplied = (
        json.loads(args.evidence.read_text(encoding="utf-8")) if args.evidence else {}
    )
    if set(supplied) - {"ground", "size"}:
        raise ValueError("evidence may contain only ground and size")
    ground = GroundEvidence(**supplied["ground"]) if "ground" in supplied else None
    size = SizeEvidence(**supplied["size"]) if "size" in supplied else None
    thresholds = (
        QualityThresholds(**json.loads(args.thresholds.read_text(encoding="utf-8")))
        if args.thresholds
        else QualityThresholds()
    )
    calibration = resolve_scene(candidate, ground, size, thresholds)
    handle = persist_scene_calibration(
        ArtifactStore(StorageRoot(args.data_root.resolve())),
        candidate,
        calibration,
    )
    print(
        json.dumps(
            {
                "path": str(handle.path),
                "id": calibration.id,
                "publication_status": calibration.publication_status,
                "camera_status": calibration.camera_status,
                "ground_status": calibration.ground_status,
                "scale_status": calibration.scale_status,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
