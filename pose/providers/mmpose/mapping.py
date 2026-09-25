"""Explicit COCO WholeBody 133 topology and canonical-name mapping."""

from __future__ import annotations

BODY = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
FEET = (
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HAND = ("wrist",) + tuple(
    f"{finger}_{joint}" for finger in FINGERS for joint in range(1, 5)
)
NAMES = (
    BODY
    + FEET
    + tuple(f"face_{i}" for i in range(68))
    + tuple(f"{side}_hand_{name}" for side in ("left", "right") for name in HAND)
)
assert len(NAMES) == len(set(NAMES)) == 133

CANONICAL = {name: name for name in BODY}
CANONICAL.update(
    {
        "left_big_toe": "left_forefoot",
        "left_small_toe": "left_foot_outer",
        "right_big_toe": "right_forefoot",
        "right_small_toe": "right_foot_outer",
        "left_heel": "left_heel",
        "right_heel": "right_heel",
    }
)
for side in ("left", "right"):
    for finger in FINGERS:
        joints = (
            ("cmc", "mcp", "ip", "tip")
            if finger == "thumb"
            else ("mcp", "pip", "dip", "tip")
        )
        for index, joint in enumerate(joints, start=1):
            CANONICAL[f"{side}_hand_{finger}_{index}"] = f"{side}_{finger}_{joint}"
