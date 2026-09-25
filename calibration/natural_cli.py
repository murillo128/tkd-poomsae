"""Attempt the natural-scene route on registered native video selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration.natural import SceneView, estimate_scene, persist_scene_candidate
from contracts.models import Intrinsics
from media import MediaReader, index_recording
from tkd_poomsae.selections import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default="smoke-short")
    parser.add_argument(
        "--profiles", type=Path, help="JSON map of camera IDs to measured Intrinsics"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    profiles = {}
    if args.profiles:
        payload = json.loads(args.profiles.read_text(encoding="utf-8"))
        profiles = {
            key: Intrinsics.model_validate(value) for key, value in payload.items()
        }
    windows = resolve(args.selection)
    executions = {window.execution_id for window in windows}
    if len(executions) != 1:
        parser.error("one execution is required per calibration attempt")
    if set(profiles) - {window.camera_id for window in windows}:
        parser.error("profile names absent camera")
    views = []
    for window in windows:
        recording = index_recording(window.camera_id, window.source_path)
        if recording.sha256 != window.source_sha256:
            raise ValueError(f"registered source bytes changed: {window.camera_id}")
        reader = MediaReader(recording)
        duration = window.end_seconds - window.start_seconds
        frames = []
        frame_seconds = []
        frame_native_ids = []
        for index in range(args.samples):
            target = window.start_seconds + duration * (index + 0.5) / args.samples
            ref = min(recording.frames, key=lambda frame: abs(frame.seconds - target))
            frames.append(reader.frame(ref).rgb)
            frame_seconds.append(ref.seconds)
            frame_native_ids.append(
                f"{ref.pts}:{ref.time_base_num}/{ref.time_base_den}"
            )
        views.append(
            SceneView(
                window.camera_id,
                recording.source_id,
                tuple(frames),
                profiles.get(window.camera_id),
                window.source_sha256,
                tuple(frame_seconds),
                tuple(frame_native_ids),
            )
        )
    candidate = estimate_scene(views)
    path = persist_scene_candidate(candidate, args.output_dir)
    print(
        json.dumps(
            {
                "candidate": str(path),
                "status": candidate.status,
                "reasons": candidate.reasons,
                "overlap_edges": candidate.evidence.get("overlap_edges", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
