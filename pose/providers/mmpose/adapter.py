"""Raw, per-view MMPose candidates from already decoded native-time frames."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from contracts.models import FrameTime, Landmark2D, Quality, RawScore
from pose.providers.mmpose.mapping import CANONICAL, NAMES
from tkd_poomsae.vision.assets import registry, verified_paths
from tkd_poomsae.vision.device import DeviceCancelled, inference_job

if TYPE_CHECKING:
    from media.reader import DecodedFrame, Recording


@dataclass(frozen=True)
class NamedPoint:
    name: str
    xy_px: tuple[float, float]
    raw_score: RawScore
    raw_visibility: float | None = None


@dataclass(frozen=True)
class PersonCandidate:
    """A detector candidate; index is local to this frame, not a track ID."""

    index: int
    bbox_xyxy_px: tuple[float, float, float, float]
    detector_score: RawScore
    landmarks: tuple[NamedPoint, ...]
    refined_hands: dict[str, tuple[NamedPoint, ...]]
    refined_hand_boxes: dict[str, tuple[float, float, float, float]]


@dataclass(frozen=True)
class PoseFrame:
    frame: FrameTime
    ordinal: int
    image_size: tuple[int, int]
    candidates: tuple[PersonCandidate, ...]
    model_identity: dict[str, Any]
    inference_settings: dict[str, Any]


def _score(value: Any) -> RawScore:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"model score outside declared [0, 1] range: {numeric}")
    return RawScore(value=numeric, range_min=0, range_max=1)


def _points(sample: Any, names: tuple[str, ...]) -> tuple[NamedPoint, ...]:
    instances = sample.pred_instances
    xy = np.asarray(instances.keypoints)
    scores = np.asarray(instances.keypoint_scores)
    if xy.shape != (1, len(names), 2) or scores.shape != (1, len(names)):
        raise ValueError(f"malformed {len(names)}-point model topology")
    visibility = getattr(instances, "keypoints_visible", None)
    if visibility is not None:
        visibility = np.asarray(visibility)
        if visibility.shape != scores.shape:
            raise ValueError("malformed model visibility topology")
    points = []
    for i, name in enumerate(names):
        coords = (float(xy[0, i, 0]), float(xy[0, i, 1]))
        if not all(map(math.isfinite, coords)):
            raise ValueError(f"nonfinite model coordinate for {name}")
        raw_visibility = None if visibility is None else float(visibility[0, i])
        if raw_visibility is not None and not math.isfinite(raw_visibility):
            raise ValueError(f"nonfinite visibility for {name}")
        points.append(NamedPoint(name, coords, _score(scores[0, i]), raw_visibility))
    return tuple(points)


def canonical_landmarks(candidate: PersonCandidate) -> list[Landmark2D]:
    """Project supported points without treating raw confidence as quality."""
    return [
        Landmark2D(
            name=CANONICAL[point.name],  # type: ignore[arg-type]
            xy_px=point.xy_px,
            raw_score=point.raw_score,
            quality=Quality(state="observed"),
        )
        for point in candidate.landmarks
        if point.name in CANONICAL
    ]


class _OpenMMLab:
    """Load only verified local assets; topdown returns original-image pixels."""

    def __init__(self, device: str) -> None:
        paths = verified_paths()
        specs = registry()["models"]
        import mmdet  # type: ignore[import-not-found]
        import mmpose  # type: ignore[import-not-found]
        from mmdet.apis import init_detector  # type: ignore[import-not-found]
        from mmengine.config import Config  # type: ignore[import-not-found]
        from mmpose.apis import init_model  # type: ignore[import-not-found]

        self.framework_versions = {
            "mmdet": mmdet.__version__,
            "mmpose": mmpose.__version__,
        }

        def config(name: str) -> Any:
            cfg = Config.fromfile(paths[specs[name]["config"]])
            # Training configs may reference remote pretrained backbones.
            if "backbone" in cfg.model:
                cfg.model.backbone.init_cfg = None
            cfg.model.init_cfg = None
            return cfg

        self.detector = init_detector(
            config("detector"),
            str(paths[specs["detector"]["checkpoint"]]),
            device=device,
        )
        self.wholebody = init_model(
            config("wholebody"),
            str(paths[specs["wholebody"]["checkpoint"]]),
            device=device,
        )
        self.hand = init_model(
            config("hand"),
            str(paths[specs["hand"]["checkpoint"]]),
            device=device,
        )

    def detect(self, bgr: np.ndarray) -> Any:
        from mmdet.apis import inference_detector
        from mmengine.registry import DefaultScope  # type: ignore[import-not-found]

        with DefaultScope.overwrite_default_scope("mmdet"):
            return inference_detector(self.detector, bgr).pred_instances

    def pose(self, bgr: np.ndarray, boxes: np.ndarray, *, hand: bool = False) -> Any:
        from mmengine.registry import DefaultScope
        from mmpose.apis import inference_topdown

        with DefaultScope.overwrite_default_scope("mmpose"):
            return inference_topdown(
                self.hand if hand else self.wholebody, bgr, bboxes=boxes
            )


class MMPoseAdapter:
    def __init__(
        self,
        *,
        device: str = "cpu",
        max_people: int = 4,
        detector_threshold: float = 0.3,
        max_frames: int = 32,
        cancelled: Callable[[], bool] | None = None,
        _backend_factory: Callable[[str], Any] = _OpenMMLab,
    ) -> None:
        if max_people < 1 or max_frames < 1 or not 0 <= detector_threshold <= 1:
            raise ValueError("invalid inference bounds or detector threshold")
        self.device = device
        self.max_people = max_people
        self.detector_threshold = detector_threshold
        self.max_frames = max_frames
        self.cancelled = cancelled
        self._backend_factory = _backend_factory

    def infer(
        self, recording: Recording, frames: Iterable[DecodedFrame]
    ) -> Iterator[PoseFrame]:
        """Consume at most one bounded window; errors and cancellation propagate."""
        import cv2
        import torch  # type: ignore[import-not-found]

        with (
            inference_job(self.device, cancelled=self.cancelled),
            torch.inference_mode(),
        ):
            backend = self._backend_factory(self.device)
            for count, decoded in enumerate(frames):
                if count >= self.max_frames:
                    raise ValueError("pose input exceeds max_frames window")
                if self.cancelled is not None and self.cancelled():
                    raise DeviceCancelled("Inference cancelled before frame")
                if decoded.rgb.dtype != np.uint8 or decoded.rgb.shape != (
                    recording.source.height_px,
                    recording.source.width_px,
                    3,
                ):
                    raise ValueError("decoded RGB geometry differs from source")
                frame_time = recording.frame_time(decoded.ref)
                bgr = cv2.cvtColor(decoded.rgb, cv2.COLOR_RGB2BGR)
                detections = backend.detect(bgr)
                labels = np.asarray(detections.labels.cpu())
                scores = np.asarray(detections.scores.cpu())
                boxes = np.asarray(detections.bboxes.cpu())
                if (
                    boxes.ndim != 2
                    or boxes.shape[1] != 4
                    or len(labels) != len(boxes)
                    or len(scores) != len(boxes)
                ):
                    raise ValueError("malformed detector output")
                selected = [
                    i
                    for i in range(len(boxes))
                    if int(labels[i]) == 0
                    and _score(scores[i]).value >= self.detector_threshold
                ]
                selected.sort(key=lambda i: float(scores[i]), reverse=True)
                selected = selected[: self.max_people]
                candidates: list[PersonCandidate] = []
                if selected:
                    poses = backend.pose(bgr, boxes[selected].astype(np.float32))
                    if len(poses) != len(selected):
                        raise ValueError("pose count differs from selected detections")
                    for index, (det_index, pose) in enumerate(zip(selected, poses)):
                        if self.cancelled is not None and self.cancelled():
                            raise DeviceCancelled("Inference cancelled during poses")
                        bbox = (
                            float(boxes[det_index, 0]),
                            float(boxes[det_index, 1]),
                            float(boxes[det_index, 2]),
                            float(boxes[det_index, 3]),
                        )
                        if (
                            not all(map(math.isfinite, bbox))
                            or bbox[2] <= bbox[0]
                            or bbox[3] <= bbox[1]
                        ):
                            raise ValueError("malformed detector box")
                        points = _points(pose, NAMES)
                        refined: dict[str, tuple[NamedPoint, ...]] = {}
                        refined_boxes: dict[str, tuple[float, float, float, float]] = {}
                        for side, start in (("left", 91), ("right", 112)):
                            visible = np.array(
                                [
                                    p.xy_px
                                    for p in points[start : start + 21]
                                    if p.raw_score.value >= 0.2
                                ]
                            )
                            if len(visible) < 2:
                                continue
                            lo, hi = visible.min(axis=0), visible.max(axis=0)
                            center = (lo + hi) / 2
                            span = max(float(max(hi - lo)) * 1.5, 16.0)
                            width, height = (
                                recording.source.width_px,
                                recording.source.height_px,
                            )
                            hand_box = np.array(
                                [
                                    [
                                        max(0.0, center[0] - span / 2),
                                        max(0.0, center[1] - span / 2),
                                        min(float(width), center[0] + span / 2),
                                        min(float(height), center[1] + span / 2),
                                    ]
                                ],
                                dtype=np.float32,
                            )
                            if (
                                hand_box[0, 2] <= hand_box[0, 0]
                                or hand_box[0, 3] <= hand_box[0, 1]
                            ):
                                continue
                            hand_samples = backend.pose(bgr, hand_box, hand=True)
                            if len(hand_samples) != 1:
                                raise ValueError(
                                    "hand refinement returned wrong sample count"
                                )
                            hand_names = NAMES[start : start + 21]
                            refined[side] = _points(hand_samples[0], hand_names)
                            refined_boxes[side] = (
                                float(hand_box[0, 0]),
                                float(hand_box[0, 1]),
                                float(hand_box[0, 2]),
                                float(hand_box[0, 3]),
                            )
                        candidates.append(
                            PersonCandidate(
                                index,
                                bbox,
                                _score(scores[det_index]),
                                points,
                                refined,
                                refined_boxes,
                            )
                        )
                manifest = registry()
                specs = manifest["models"]
                hashes = {
                    asset["path"]: asset["sha256"] for asset in manifest["assets"]
                }
                yield PoseFrame(
                    frame=frame_time,
                    ordinal=decoded.ref.ordinal,
                    image_size=(recording.source.width_px, recording.source.height_px),
                    candidates=tuple(candidates),
                    model_identity={
                        name: {
                            "framework": "mmpose" if name != "detector" else "mmdet",
                            "framework_version": backend.framework_versions[
                                "mmpose" if name != "detector" else "mmdet"
                            ],
                            **spec,
                            "config_sha256": hashes[spec["config"]],
                            "checkpoint_sha256": hashes[spec["checkpoint"]],
                        }
                        for name, spec in specs.items()
                    },
                    inference_settings={
                        "device": self.device,
                        "max_people": self.max_people,
                        "detector_threshold": self.detector_threshold,
                        "max_frames": self.max_frames,
                        "hand_threshold": 0.2,
                        "coordinates": "display-oriented original pixels",
                    },
                )
