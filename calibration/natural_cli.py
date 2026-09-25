"""Attempt the natural-scene route on registered native video selections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from calibration.natural import (
    SceneCandidate,
    SceneView,
    estimate_scene,
    persist_scene_candidate,
)
from contracts.models import Intrinsics, Synchronization
from media import FrameRef, MediaReader, index_recording
from tkd_poomsae.selections import resolve


def load_synchronization_artifact(directory: Path) -> Synchronization:
    """Read a complete immutable storage artifact, including its metadata hash."""
    if directory.is_symlink() or directory.parent.name != "synchronization":
        raise ValueError("expected derived/synchronization artifact directory")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    if (
        manifest["version"] != 1
        or manifest["key"] != directory.name
        or metadata["path"] != "metadata.json"
        or (directory / "metadata.json").is_symlink()
    ):
        raise ValueError("invalid synchronization artifact manifest")
    raw = (directory / "metadata.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("synchronization artifact metadata hash mismatch")
    return Synchronization.model_validate_json(raw)


def nearest_global_refs(
    frames: tuple[FrameRef, ...],
    targets: list[float],
    offset: float,
) -> list[FrameRef]:
    """Choose native frames nearest *global* targets with bounded timing error."""
    if not frames:
        raise ValueError("no native frames to sample")
    selected = [
        min(frames, key=lambda ref: abs(ref.seconds + offset - target))
        for target in targets
    ]
    if any(
        abs(ref.seconds + offset - target) > 0.05
        for ref, target in zip(selected, targets, strict=True)
    ):
        raise ValueError("no native frame within 50 ms of a global sample")
    return selected


def report_candidate(candidate: SceneCandidate, directory: Path) -> None:
    path = persist_scene_candidate(candidate, directory)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default="smoke-short")
    parser.add_argument(
        "--profiles", type=Path, help="JSON map of camera IDs to measured Intrinsics"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--sync-artifact",
        type=Path,
        help="validated derived/synchronization artifact directory",
    )
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
    candidate: SceneCandidate
    if args.sync_artifact is None:
        candidate = SceneCandidate(
            "unavailable",
            ["verified synchronization artifact is required"],
            evidence={
                "selection": args.selection,
                "source_sha256": {
                    window.camera_id: window.source_sha256 for window in windows
                },
            },
        )
        report_candidate(candidate, args.output_dir)
        return
    synchronization = load_synchronization_artifact(args.sync_artifact)
    by_source = {row.source_id: row for row in synchronization.offsets}
    if synchronization.common_interval is None or any(
        f"source:{window.source_sha256}" not in by_source
        or not by_source[f"source:{window.source_sha256}"].retained
        for window in windows
    ):
        raise ValueError("synchronization does not retain selected exact sources")
    start = max(
        [synchronization.common_interval.start]
        + [
            window.start_seconds
            + by_source[f"source:{window.source_sha256}"].effective_seconds
            for window in windows
        ]
    )
    end = min(
        [synchronization.common_interval.end]
        + [
            window.end_seconds
            + by_source[f"source:{window.source_sha256}"].effective_seconds
            for window in windows
        ]
    )
    if end <= start:
        candidate = SceneCandidate(
            "unavailable",
            ["selected windows have no common global interval"],
            evidence={
                "selection": args.selection,
                "synchronization_id": synchronization.id,
                "source_offsets_seconds": {
                    window.camera_id: by_source[
                        f"source:{window.source_sha256}"
                    ].effective_seconds
                    for window in windows
                },
            },
        )
        report_candidate(candidate, args.output_dir)
        return
    targets = [
        start + (end - start) * (index + 0.5) / args.samples
        for index in range(args.samples)
    ]
    views = []
    for window in windows:
        recording = index_recording(window.camera_id, window.source_path)
        if recording.sha256 != window.source_sha256:
            raise ValueError(f"registered source bytes changed: {window.camera_id}")
        reader = MediaReader(recording)
        offset = by_source[recording.source_id].effective_seconds
        window_refs = tuple(
            ref
            for ref in recording.frames
            if window.start_seconds <= ref.seconds <= window.end_seconds
        )
        if not window_refs:
            raise ValueError(f"no native frames in selection: {window.camera_id}")
        selected_refs = nearest_global_refs(window_refs, targets, offset)
        frames = []
        frame_seconds = []
        frame_native_ids = []
        for ref in selected_refs:
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
    candidate = estimate_scene(views, synchronization)
    report_candidate(candidate, args.output_dir)


if __name__ == "__main__":
    main()
