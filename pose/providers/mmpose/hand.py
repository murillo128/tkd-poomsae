"""Native-pixel hand ROIs and conservative per-landmark refinement evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from contracts.models import RawScore
from pose.providers.mmpose.geometry import PixelTransform
from pose.providers.mmpose.mapping import HAND

Side = Literal["left", "right"]
HandState = Literal["observed", "inferred", "unknown"]
SIDES: tuple[Side, Side] = ("left", "right")


class CoarsePoint(Protocol):
    @property
    def xy_px(self) -> tuple[float, float]: ...

    @property
    def raw_score(self) -> RawScore: ...

    @property
    def raw_visibility(self) -> float | None: ...


@dataclass(frozen=True)
class HandROIConfig:
    margin: float = 0.35
    min_source_span_px: float = 12.0
    max_span_px: float = 512.0
    min_hand_points: int = 3
    coarse_threshold: float = 0.2
    refined_threshold: float = 0.2
    min_visible_fraction: float = 0.4
    clockwise: Literal[0, 90, 180, 270] = 0
    mirror_left: bool = False
    mirror_right: bool = False

    def __post_init__(self) -> None:
        if self.clockwise not in (0, 90, 180, 270):
            raise ValueError("unsupported hand ROI rotation")
        if not all(
            math.isfinite(value)
            for value in (
                self.margin,
                self.min_source_span_px,
                self.max_span_px,
                self.coarse_threshold,
                self.refined_threshold,
                self.min_visible_fraction,
            )
        ):
            raise ValueError("nonfinite hand ROI setting")
        if (
            not 0 <= self.margin <= 2
            or not 0 < self.min_source_span_px <= self.max_span_px
        ):
            raise ValueError("invalid hand ROI extent")
        if not 1 <= self.min_hand_points <= 21:
            raise ValueError("invalid hand point count")
        if not all(
            0 <= value <= 1
            for value in (
                self.coarse_threshold,
                self.refined_threshold,
                self.min_visible_fraction,
            )
        ):
            raise ValueError("invalid hand evidence threshold")


@dataclass(frozen=True)
class HandROI:
    side: Side
    xyxy_px: tuple[
        int, int, int, int
    ]  # Virtual source bounds; may cross the frame edge.
    source_overlap_px: tuple[int, int, int, int]
    transform: PixelTransform
    visible_fraction: float
    source_span_px: float
    coarse_support_fraction: float = 1.0

    @property
    def padding_px(self) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.xyxy_px
        ix0, iy0, ix1, iy1 = self.source_overlap_px
        return ix0 - x0, iy0 - y0, x1 - ix1, y1 - iy1

    @property
    def roi_to_source_affine(self) -> tuple[tuple[float, float, float], ...]:
        origin = self.transform.to_source((0, 0))
        ex = self.transform.to_source((1, 0))
        ey = self.transform.to_source((0, 1))
        return (
            (ex[0] - origin[0], ey[0] - origin[0], origin[0]),
            (ex[1] - origin[1], ey[1] - origin[1], origin[1]),
        )


@dataclass(frozen=True)
class RefinedPoint:
    name: str
    xy_px: tuple[float, float] | None
    raw_score: float | None
    raw_visibility: float | None
    state: HandState
    reason: str | None = None


@dataclass(frozen=True)
class HandObservation:
    side: Side
    coarse: tuple[CoarsePoint, ...]
    refined: tuple[RefinedPoint, ...]
    roi: HandROI | None
    status: str
    handedness_ambiguous: bool
    model_identity: str | None = None
    cache_identity: str | None = None


def missing_points(side: Side, reason: str) -> tuple[RefinedPoint, ...]:
    return tuple(
        RefinedPoint(f"{side}_hand_{name}", None, None, None, "unknown", reason)
        for name in HAND
    )


def localize_hand(
    points: tuple[CoarsePoint, ...],
    side: Side,
    image_size: tuple[int, int],
    config: HandROIConfig,
) -> HandROI | None:
    """Use wholebody hand and adjacent forearm geometry, never model-size pixels."""
    width, height = image_size
    start = 91 if side == "left" else 112
    hand = points[start : start + 21]
    supported = [
        p.xy_px
        for p in hand
        if p.raw_score.value >= config.coarse_threshold
        and (p.raw_visibility is None or p.raw_visibility > 0)
        and -width <= p.xy_px[0] < 2 * width
        and -height <= p.xy_px[1] < 2 * height
    ]
    support_fraction = len(supported) / 21
    if len(supported) < config.min_hand_points:
        return None
    body_wrist = points[9 if side == "left" else 10]
    elbow = points[7 if side == "left" else 8]
    if body_wrist.raw_score.value >= config.coarse_threshold:
        supported.append(body_wrist.xy_px)
    xy = np.asarray(supported, dtype=np.float64)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    source_span = float(max(hi - lo))
    if not config.min_source_span_px <= source_span <= config.max_span_px:
        return None  # Interpolation cannot restore missing image detail.
    forearm = 0.0
    if (
        elbow.raw_score.value >= config.coarse_threshold
        and body_wrist.raw_score.value >= config.coarse_threshold
    ):
        forearm = float(
            np.linalg.norm(np.asarray(body_wrist.xy_px) - np.asarray(elbow.xy_px))
        )
    span = min(
        max(source_span * (1 + 2 * config.margin), forearm * 0.8), config.max_span_px
    )
    span = max(span, source_span)
    center = (lo + hi) / 2
    x0, y0 = (math.floor(float(c - span / 2)) for c in center)
    side_px = max(1, math.ceil(span))
    x1, y1 = x0 + side_px, y0 + side_px
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(width, x1), min(height, y1)
    overlap = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    fraction = overlap / (side_px * side_px)
    if overlap == 0 or fraction < config.min_visible_fraction:
        return None
    transform = PixelTransform(
        crop_x=x0,
        crop_y=y0,
        crop_width=side_px,
        crop_height=side_px,
        clockwise=config.clockwise,
        mirror_x=config.mirror_left if side == "left" else config.mirror_right,
    )
    return HandROI(
        side,
        (x0, y0, x1, y1),
        (ix0, iy0, ix1, iy1),
        transform,
        fraction,
        source_span,
        support_fraction,
    )


def crop_original(image: np.ndarray, roi: HandROI) -> np.ndarray:
    """Black-pad a native-resolution crop; rotation/mirror are reversible."""
    x0, y0, x1, y1 = roi.xyxy_px
    ix0, iy0, ix1, iy1 = roi.source_overlap_px
    crop = np.zeros((y1 - y0, x1 - x0, image.shape[2]), dtype=image.dtype)
    crop[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = image[iy0:iy1, ix0:ix1]
    if roi.transform.clockwise:
        crop = np.rot90(crop, -(roi.transform.clockwise // 90))
    if roi.transform.mirror_x:
        crop = np.flip(crop, axis=1)
    return np.ascontiguousarray(crop)


def map_refinement(
    side: Side,
    xy: np.ndarray,
    scores: np.ndarray,
    visibility: np.ndarray | None,
    roi: HandROI,
    image_size: tuple[int, int],
    config: HandROIConfig,
    coarse_wrist: tuple[float, float] | None,
    coarse_hand: tuple[CoarsePoint, ...] | None = None,
) -> tuple[RefinedPoint, ...]:
    if (
        xy.shape != (21, 2)
        or scores.shape != (21,)
        or (visibility is not None and visibility.shape != (21,))
    ):
        raise ValueError("malformed 21-point hand topology")
    if coarse_hand is not None and len(coarse_hand) != 21:
        raise ValueError("malformed coarse hand topology")
    width, height = image_size
    mapped = [roi.transform.to_source((float(p[0]), float(p[1]))) for p in xy]
    implausible = coarse_wrist is not None and math.dist(mapped[0], coarse_wrist) > max(
        roi.source_span_px, (roi.xyxy_px[2] - roi.xyxy_px[0]) * 0.6
    )
    result = []
    for index, name in enumerate(HAND):
        point = mapped[index]
        score = float(scores[index])
        visible = None if visibility is None else float(visibility[index])
        if not math.isfinite(score):
            raise ValueError("invalid hand model score")
        if visible is not None and not math.isfinite(visible):
            raise ValueError("invalid hand model visibility")
        reason = None
        if implausible:
            reason = "implausible_wrist"
        elif not all(map(math.isfinite, point)):
            reason = "nonfinite_coordinate"
        elif score < config.refined_threshold or (visible is not None and visible <= 0):
            reason = "low_model_evidence"
        elif not (0 <= point[0] < width and 0 <= point[1] < height):
            reason = "outside_source"
        elif not (
            roi.xyxy_px[0] <= point[0] < roi.xyxy_px[2]
            and roi.xyxy_px[1] <= point[1] < roi.xyxy_px[3]
        ):
            reason = "outside_roi"
        coarse_point = None if coarse_hand is None else coarse_hand[index]
        coarse_visibility = (
            None if coarse_point is None else coarse_point.raw_visibility
        )
        unsupported_coarse = coarse_point is not None and (
            coarse_point.raw_score.value < config.coarse_threshold
            or (coarse_visibility is not None and coarse_visibility <= 0)
        )
        state: HandState = (
            "unknown"
            if reason
            else (
                "inferred"
                if (
                    roi.visible_fraction < 0.8
                    or roi.coarse_support_fraction < 0.5
                    or unsupported_coarse
                )
                else "observed"
            )
        )
        if state == "inferred" and unsupported_coarse:
            reason = "unsupported_coarse_point"
        result.append(
            RefinedPoint(
                f"{side}_hand_{name}",
                None if state == "unknown" else point,
                score,
                visible,
                state,
                reason,
            )
        )
    return tuple(result)


def identity_ambiguous(left: HandROI | None, right: HandROI | None) -> bool:
    if left is None or right is None:
        return False
    a, b = left.source_overlap_px, right.source_overlap_px
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return area > 0 and overlap / area > 0.25


class HandRefinementCache:
    """Immutable crop and raw model output, keyed by source, model and ROI settings."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get_or_compute(
        self,
        identity: dict[str, object],
        crop: np.ndarray,
        compute: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray | None] | None],
    ) -> tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray | None] | None]:
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        path = self.root / digest
        lock_dir = self.root / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / f"{digest}.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if path.exists():
                    meta = json.loads((path / "result.json").read_text())
                    cached_crop = np.load(path / "crop.npy", allow_pickle=False)
                    if not np.array_equal(cached_crop, crop):
                        raise ValueError("hand cache crop identity mismatch")
                    if meta["identity"] != json.loads(payload):
                        raise ValueError("hand cache metadata identity mismatch")
                    if meta["result"] is None:
                        return digest, None
                    result = meta["result"]
                    return digest, (
                        np.asarray(result["xy"], dtype=float),
                        np.asarray(result["scores"], dtype=float),
                        None
                        if result["visibility"] is None
                        else np.asarray(result["visibility"], dtype=float),
                    )
                output = compute()
                result = (
                    None
                    if output is None
                    else {
                        "xy": output[0].tolist(),
                        "scores": output[1].tolist(),
                        "visibility": None if output[2] is None else output[2].tolist(),
                    }
                )
                self.root.mkdir(parents=True, exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=self.root))
                try:
                    with (stage / "crop.npy").open("wb") as stream:
                        np.save(stream, crop, allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    with (stage / "result.json").open("w") as stream:
                        json.dump(
                            {"identity": identity, "result": result},
                            stream,
                            sort_keys=True,
                            allow_nan=False,
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                    stage.rename(path)
                finally:
                    if stage.exists():
                        shutil.rmtree(stage)
                return digest, output
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
