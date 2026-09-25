"""Local MMPose wholebody adapter."""

from pose.providers.mmpose.adapter import (
    MMPoseAdapter,
    NamedPoint,
    PersonCandidate,
    PoseFrame,
    canonical_landmarks,
)

__all__ = [
    "MMPoseAdapter",
    "NamedPoint",
    "PersonCandidate",
    "PoseFrame",
    "canonical_landmarks",
]
