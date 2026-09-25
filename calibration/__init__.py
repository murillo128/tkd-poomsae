"""Target-based camera calibration and reusable camera geometry."""

from calibration.cameras import CameraModel, IntrinsicProfile, back_project_ray
from calibration.natural import (
    SceneCandidate,
    SceneView,
    estimate_scene,
    persist_scene_candidate,
)
from calibration.target import (
    BoardSpec,
    CalibrationFailure,
    Capture,
    Detection,
    estimate_calibration,
    observe_capture,
    persist_calibration,
)

__all__ = [
    "BoardSpec",
    "CalibrationFailure",
    "CameraModel",
    "Capture",
    "Detection",
    "IntrinsicProfile",
    "back_project_ray",
    "estimate_calibration",
    "observe_capture",
    "persist_calibration",
    "SceneCandidate",
    "SceneView",
    "estimate_scene",
    "persist_scene_candidate",
]
