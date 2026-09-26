"""Run target calibration from separate, explicitly described capture frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration.manifest import calibrate_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate cameras from ChArUco captures"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    candidate, handle = calibrate_from_manifest(args.manifest, args.data_root)
    print(json.dumps({
        "artifact": str(handle.path),
        "calibration_id": candidate.id,
        "cameras": [{
            "camera_id": c.camera_id,
            "intrinsic_source": c.intrinsic_source,
            "rms_reprojection_px": c.rms_reprojection_px,
        } for c in candidate.cameras],
    }, indent=2))


if __name__ == "__main__":
    main()
