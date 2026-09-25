"""Local-only RTMDet, whole-body RTMPose, and original-frame hand inference."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from tkd_poomsae.vision.assets import registry, verified_paths
from tkd_poomsae.vision.device import DeviceCancelled, inference_job


def _hand_box(
    points: Any, scores: Any, start: int, width: int, height: int
) -> list[float] | None:
    import numpy as np

    selected = points[start : start + 21][scores[start : start + 21] >= 0.2]
    if len(selected) < 2:
        return None
    lo = np.min(selected, axis=0)
    hi = np.max(selected, axis=0)
    center = (lo + hi) / 2
    side = max(float(np.max(hi - lo)) * 1.5, 16.0)
    return [
        max(0.0, float(center[0] - side / 2)),
        max(0.0, float(center[1] - side / 2)),
        min(float(width), float(center[0] + side / 2)),
        min(float(height), float(center[1] + side / 2)),
    ]


def _pose_boxes(
    boxes: Any, width: int, height: int, *, smoke: bool
) -> Any | None:
    """Never let MMPose turn an empty detection into a whole-frame pose."""
    if len(boxes):
        return boxes
    if not smoke:
        return None
    import numpy as np

    return np.array([[0, 0, width, height]], dtype=np.float32)


def infer_image(
    image_path: Path,
    *,
    device: str = "cpu",
    smoke: bool = False,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the three pinned models with local paths and batch size one.

    Smoke mode also feeds whole-frame boxes to both pose models when the detector
    finds no person; it proves execution, not prediction quality.
    """
    paths = verified_paths()
    specs = registry()["models"]
    import cv2  # type: ignore[import-not-found]
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from mmdet.apis import (  # type: ignore[import-not-found]
        inference_detector,
        init_detector,
    )
    from mmengine.config import Config  # type: ignore[import-not-found]
    from mmpose.apis import (  # type: ignore[import-not-found]
        inference_topdown,
        init_model,
    )

    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode local image: {image_path}")
    height, width = image.shape[:2]
    with inference_job(device, cancelled=cancelled), torch.inference_mode():
        det_spec = specs["detector"]
        detector = init_detector(
            str(paths[det_spec["config"]]),
            str(paths[det_spec["checkpoint"]]),
            device=device,
        )
        detected = inference_detector(detector, image).pred_instances
        person = (detected.labels == 0) & (detected.scores >= 0.3)
        boxes = detected.bboxes[person].cpu().numpy()
        if len(boxes) > 1:
            boxes = boxes[:1]  # bounded single-person inference
        del detector
        if cancelled is not None and cancelled():
            raise DeviceCancelled("Inference cancelled after detection")
        pose_boxes = _pose_boxes(boxes, width, height, smoke=smoke)
        if pose_boxes is None:
            return {
                "device": device,
                "image_size": [width, height],
                "detected_people": 0,
                "wholebody_samples": 0,
                "refined_hands": 0,
                "smoke": False,
            }

        whole_spec = specs["wholebody"]
        whole_cfg = Config.fromfile(paths[whole_spec["config"]])
        # Official training configs contain remote pretraining URLs. Checkpoint
        # inference must never resolve them, even if upstream API behavior changes.
        whole_cfg.model.backbone.init_cfg = None
        whole = init_model(
            whole_cfg, str(paths[whole_spec["checkpoint"]]), device=device
        )
        poses = inference_topdown(whole, image, bboxes=pose_boxes)
        if smoke and not poses:
            raise RuntimeError("Whole-body smoke inference returned no sample")
        for pose in poses:
            if pose.pred_instances.keypoints.shape[-2] != 133:
                raise RuntimeError("Whole-body topology differs from pinned 133 points")
        del whole
        if cancelled is not None and cancelled():
            raise DeviceCancelled("Inference cancelled after whole-body pose")

        hand_spec = specs["hand"]
        hand_cfg = Config.fromfile(paths[hand_spec["config"]])
        hand_cfg.model.backbone.init_cfg = None
        hand = init_model(hand_cfg, str(paths[hand_spec["checkpoint"]]), device=device)
        hand_boxes: list[list[float]] = []
        if poses:
            points = poses[0].pred_instances.keypoints[0]
            scores = poses[0].pred_instances.keypoint_scores[0]
            for start in (91, 112):
                box = _hand_box(points, scores, start, width, height)
                if box is not None:
                    hand_boxes.append(box)
        if smoke and not hand_boxes:
            hand_boxes = [[0, 0, width, height]]
        hand_count = 0
        for box in hand_boxes[:2]:
            # The API crops the supplied bounding box from the untouched source image.
            refined = inference_topdown(
                hand, image, bboxes=np.array([box], dtype=np.float32)
            )
            for sample in refined:
                if sample.pred_instances.keypoints.shape[-2] != 21:
                    raise RuntimeError("Hand topology differs from pinned 21 points")
                hand_count += 1
        if smoke and not hand_count:
            raise RuntimeError("Hand smoke inference returned no sample")
        del hand
    return {
        "device": device,
        "image_size": [width, height],
        "detected_people": int(len(boxes)),
        "wholebody_samples": len(poses),
        "refined_hands": hand_count,
        "smoke": smoke,
    }
