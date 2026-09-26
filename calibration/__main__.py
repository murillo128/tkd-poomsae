"""Run target calibration from separate, explicitly described capture frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration.manifest import calibrate_from_manifest, load_manifest
from calibration.target import calibration_key
from pipeline.runner import _write_json, key_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate cameras from ChArUco captures"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--key-output", type=Path, help="Export a key for tkd-poomsae run"
    )
    args = parser.parse_args()
    candidate, handle = calibrate_from_manifest(args.manifest, args.data_root)
    _, captures, _ = load_manifest(args.manifest)
    key = calibration_key(candidate, [capture for _, capture in captures])
    if args.key_output is not None:
        _write_json(args.key_output, key_data(key))
    print(
        json.dumps(
            {
                "artifact": str(handle.path),
                "artifact_key": key_data(key),
                "calibration_id": candidate.id,
                "cameras": [
                    {
                        "camera_id": c.camera_id,
                        "intrinsic_source": c.intrinsic_source,
                        "rms_reprojection_px": c.rms_reprojection_px,
                    }
                    for c in candidate.cameras
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
