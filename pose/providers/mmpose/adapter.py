"""Raw, per-view MMPose candidates from already decoded native-time frames."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from contracts.models import (
    FrameTime,
    Landmark2D,
    Quality,
    RawScore,
    RegionalGeometry2D,
)
from pose.providers.mmpose.hand import (
    SIDES,
    HandObservation,
    HandRefinementCache,
    HandROIConfig,
    crop_original,
    identity_ambiguous,
    localize_hand,
    map_refinement,
    missing_points,
)
from pose.providers.mmpose.mapping import CANONICAL, NAMES
from pose.regions import RegionalProvider, WholebodyRegionalProvider
from storage.store import StorageRoot, hash_config
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
    hand_observations: dict[str, HandObservation] = field(default_factory=dict)
    regional_geometry: tuple[RegionalGeometry2D, ...] = ()


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
        hand_roi: HandROIConfig | None = None,
        hand_cache_root: Path | None = None,
        cancelled: Callable[[], bool] | None = None,
        _backend_factory: Callable[[str], Any] = _OpenMMLab,
        regional_provider: RegionalProvider | None = None,
    ) -> None:
        if max_people < 1 or max_frames < 1 or not 0 <= detector_threshold <= 1:
            raise ValueError("invalid inference bounds or detector threshold")
        self.device = device
        self.max_people = max_people
        self.detector_threshold = detector_threshold
        self.max_frames = max_frames
        self.hand_roi = hand_roi or HandROIConfig()
        self.hand_cache_root = hand_cache_root
        self.cancelled = cancelled
        self._backend_factory = _backend_factory
        self.regional_provider = regional_provider or WholebodyRegionalProvider()

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
                        width, height = (
                            recording.source.width_px,
                            recording.source.height_px,
                        )
                        rois = {
                            side: localize_hand(
                                points, side, (width, height), self.hand_roi
                            )
                            for side in SIDES
                        }
                        ambiguous = identity_ambiguous(rois["left"], rois["right"])
                        observations: dict[str, HandObservation] = {}
                        manifest = registry()
                        hand_spec = manifest["models"]["hand"]
                        asset_hashes = {
                            asset["path"]: asset["sha256"]
                            for asset in manifest["assets"]
                        }
                        model_hash = hash_config(
                            {
                                "checkpoint_sha256": asset_hashes[
                                    hand_spec["checkpoint"]
                                ],
                                "config_sha256": asset_hashes[hand_spec["config"]],
                                "framework_version": backend.framework_versions[
                                    "mmpose"
                                ],
                            }
                        )
                        for side in SIDES:
                            start = 91 if side == "left" else 112
                            roi = rois[side]
                            coarse = points[start : start + 21]
                            if roi is None:
                                observations[side] = HandObservation(
                                    side,
                                    coarse,
                                    missing_points(side, "insufficient_source_detail"),
                                    None,
                                    "skipped",
                                    ambiguous,
                                )
                                continue
                            crop = crop_original(bgr, roi)

                            def run_hand() -> (
                                tuple[np.ndarray, np.ndarray, np.ndarray | None] | None
                            ):
                                sample = backend.pose(
                                    crop,
                                    np.array(
                                        [[0, 0, crop.shape[1], crop.shape[0]]],
                                        dtype=np.float32,
                                    ),
                                    hand=True,
                                )
                                if not sample:
                                    return None
                                if len(sample) != 1:
                                    raise ValueError(
                                        "hand refinement returned wrong sample count"
                                    )
                                instances = sample[0].pred_instances
                                xy = np.asarray(instances.keypoints)
                                scores = np.asarray(instances.keypoint_scores)
                                visible = getattr(instances, "keypoints_visible", None)
                                if xy.shape != (1, 21, 2) or scores.shape != (1, 21):
                                    raise ValueError("malformed 21-point hand topology")
                                if visible is not None:
                                    visible = np.asarray(visible)
                                    if visible.shape != (1, 21):
                                        raise ValueError(
                                            "malformed hand visibility topology"
                                        )
                                return (
                                    xy[0],
                                    scores[0],
                                    None if visible is None else visible[0],
                                )

                            cache_identity = None
                            cache_root = self.hand_cache_root
                            if cache_root is None and getattr(
                                recording, "sha256", None
                            ):
                                cache_root = (
                                    StorageRoot.from_env().namespace("derived")
                                    / "hand-refinement-cache"
                                )
                            if cache_root is not None:
                                identity = {
                                    "source_sha256": getattr(recording, "sha256", None),
                                    "frame_ordinal": decoded.ref.ordinal,
                                    "frame_pts": decoded.ref.pts,
                                    "side": side,
                                    "crop_sha256": hashlib.sha256(
                                        crop.tobytes()
                                    ).hexdigest(),
                                    "roi": asdict(roi),
                                    "config": asdict(self.hand_roi),
                                    "model_sha256": model_hash,
                                }
                                cache_identity, output = HandRefinementCache(
                                    cache_root
                                ).get_or_compute(identity, crop, run_hand)
                            else:
                                output = run_hand()
                            if output is None:
                                mapped = missing_points(side, "empty_model_result")
                                status = "empty"
                            else:
                                wrist = (
                                    coarse[0].xy_px
                                    if coarse[0].raw_score.value
                                    >= self.hand_roi.coarse_threshold
                                    else None
                                )
                                mapped = map_refinement(
                                    side,
                                    *output,
                                    roi,
                                    (width, height),
                                    self.hand_roi,
                                    wrist,
                                )
                                if ambiguous:
                                    mapped = tuple(
                                        replace(point, state="inferred")
                                        if point.state == "observed"
                                        else point
                                        for point in mapped
                                    )
                                status = (
                                    "refined"
                                    if any(p.state == "observed" for p in mapped)
                                    else "low_evidence"
                                )
                            observations[side] = HandObservation(
                                side,
                                coarse,
                                mapped,
                                roi,
                                status,
                                ambiguous,
                                model_hash,
                                cache_identity,
                            )
                            refined[side] = tuple(
                                NamedPoint(
                                    p.name,
                                    p.xy_px,
                                    _score(p.raw_score),
                                    p.raw_visibility,
                                )
                                for p in mapped
                                if p.state == "observed"
                                and p.xy_px is not None
                                and p.raw_score is not None
                            )
                            refined_boxes[side] = (
                                float(roi.xyxy_px[0]),
                                float(roi.xyxy_px[1]),
                                float(roi.xyxy_px[2]),
                                float(roi.xyxy_px[3]),
                            )
                        candidate = PersonCandidate(
                            index,
                            bbox,
                            _score(scores[det_index]),
                            points,
                            refined,
                            refined_boxes,
                            observations,
                        )
                        candidates.append(
                            replace(
                                candidate,
                                regional_geometry=self.regional_provider(
                                    canonical_landmarks(candidate)
                                ),
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
                        "hand_roi": asdict(self.hand_roi),
                        "coordinates": "display-oriented original pixels",
                    },
                )
