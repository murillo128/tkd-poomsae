"""Exercise a real registered selection offline and emit a compact receipt."""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pose.observation_run import load_window_records, run_selection, verify_receipt
from pose.providers.mmpose.adapter import _OpenMMLab
from storage import ArtifactKey, ArtifactStore, hash_config, hash_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", choices=["smoke-short", "demo-full"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calls: Counter[str] = Counter()

    def deny(*_args: Any, **_kwargs: Any) -> Any:
        calls["network"] += 1
        raise AssertionError("acceptance inference must remain offline")

    socket.socket.connect = deny  # type: ignore[method-assign]
    socket.create_connection = deny
    original_init = _OpenMMLab.__init__
    original_detect = _OpenMMLab.detect
    original_pose = _OpenMMLab.pose

    def initialize(self: _OpenMMLab, device: str) -> None:
        calls["model_loads"] += 1
        original_init(self, device)

    def detect(self: _OpenMMLab, bgr: Any) -> Any:
        calls["detector"] += 1
        return original_detect(self, bgr)

    def pose(self: _OpenMMLab, bgr: Any, boxes: Any, *, hand: bool = False) -> Any:
        calls["hand" if hand else "wholebody"] += 1
        return original_pose(self, bgr, boxes, hand=hand)

    _OpenMMLab.__init__ = initialize  # type: ignore[method-assign]
    _OpenMMLab.detect = detect  # type: ignore[method-assign]
    _OpenMMLab.pose = pose  # type: ignore[method-assign]
    started = time.monotonic()
    run = run_selection(args.selection)
    elapsed = time.monotonic() - started
    path = Path(run["receipt_path"])
    store = ArtifactStore()
    receipt = verify_receipt(path, store)
    regions: Counter[str] = Counter()
    scores = 0
    hand_states: Counter[str] = Counter()
    for entry in receipt["windows"]:
        for record in load_window_records(store.get(ArtifactKey(**entry["key"]))):
            regions.update(
                region.part
                for region in record.observation.region_quality
                if region.usable
            )
            scores += sum(
                point.raw_score is not None
                for point in record.observation.wholebody_landmarks
            )
            hand_states.update(
                hand["status"] for hand in record.hand_observations.values()
            )
    before = dict(calls)
    previous_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="tkd-observation-repeat-") as elsewhere:
            os.chdir(elsewhere)
            repeated = run_selection(args.selection)
    finally:
        os.chdir(previous_cwd)
    assert repeated["cached"] and dict(calls) == before
    assert calls["network"] == 0
    report = {
        "version": 1,
        "selection": args.selection,
        "device": "cpu",
        "receipt_path": str(path),
        "receipt_sha256": hash_file(path),
        "identity_sha256": hash_config(receipt["identity"]),
        "identity": receipt["identity"],
        "frame_count": receipt["frame_count"],
        "selected_frame_count": receipt["selected_frame_count"],
        "usable_by_view": receipt["usable_by_view"],
        "window_count": len(receipt["windows"]),
        "usable_region_frames": dict(regions),
        "persisted_raw_scores": scores,
        "hand_states": dict(hand_states),
        "first_run_calls": before,
        "elapsed_seconds": round(elapsed, 3),
        "offline_reload": True,
        "repeat_from_another_cwd": {
            "cached": True,
            "inference_calls": 0,
            "network_calls": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
