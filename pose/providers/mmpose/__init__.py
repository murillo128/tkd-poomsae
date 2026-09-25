"""Local MMPose wholebody adapter."""

from pose.providers.mmpose.adapter import (
    MMPoseAdapter,
    NamedPoint,
    PersonCandidate,
    PoseFrame,
    canonical_landmarks,
)
from pose.providers.mmpose.hand import (
    HandObservation,
    HandROI,
    HandROIConfig,
    RefinedPoint,
)

__all__ = [
    "MMPoseAdapter",
    "NamedPoint",
    "PersonCandidate",
    "PoseFrame",
    "HandObservation",
    "HandROI",
    "HandROIConfig",
    "RefinedPoint",
    "canonical_landmarks",
]
