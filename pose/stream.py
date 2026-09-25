"""Native-time practitioner selection and regional observation assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from contracts.models import (
    Landmark2D,
    Observation,
    Provenance,
    Quality,
    RawScore,
    RegionalGeometry2D,
    SubjectCandidateEvidence,
    SubjectSelection,
    ViewRegionQuality,
)
from pose.providers.mmpose.adapter import (
    PersonCandidate,
    PoseFrame,
    canonical_landmarks,
)
from pose.providers.mmpose.mapping import CANONICAL
from pose.regions import WholebodyRegionalProvider

_PARTS = ("body", "left_hand", "right_hand", "left_foot", "right_foot", "head")


@dataclass(frozen=True)
class TrackingConfig:
    min_raw_score: float = 0.2
    min_raw_visibility: float = 0.5
    max_gap_seconds: float = 0.5
    max_match_cost: float = 1.5
    ambiguity_margin: float = 0.2
    min_initial_area_ratio: float = 1.2
    min_roi_span_px: float = 12.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_raw_score <= 1 or not 0 <= self.min_raw_visibility <= 1:
            raise ValueError("invalid raw evidence threshold")
        if any(
            value <= 0 or not math.isfinite(value)
            for value in (
                self.max_gap_seconds,
                self.max_match_cost,
                self.ambiguity_margin,
                self.min_initial_area_ratio,
                self.min_roi_span_px,
            )
        ):
            raise ValueError("invalid tracking threshold")


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _match_cost(previous: PersonCandidate, current: PersonCandidate) -> float:
    old, new = previous.bbox_xyxy_px, current.bbox_xyxy_px
    scale = max(math.sqrt(_area(old)), 1.0)
    center = (
        math.dist(
            ((old[0] + old[2]) / 2, (old[1] + old[3]) / 2),
            ((new[0] + new[2]) / 2, (new[1] + new[3]) / 2),
        )
        / scale
    )
    size = abs(math.log(max(_area(new), 1.0) / max(_area(old), 1.0)))
    prior = {point.name: point for point in previous.landmarks}
    torso = []
    for point in current.landmarks:
        if point.name not in (
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        ):
            continue
        old_point = prior.get(point.name)
        if old_point and min(old_point.raw_score.value, point.raw_score.value) >= 0.2:
            torso.append(math.dist(old_point.xy_px, point.xy_px) / scale)
    keypoint = sum(torso) / len(torso) if torso else center
    return 0.45 * center + 0.35 * keypoint + 0.2 * size


def _part(name: str) -> str:
    if name.startswith("left_") and any(
        item in name for item in ("thumb", "index", "middle", "ring", "pinky")
    ):
        return "left_hand"
    if name.startswith("right_") and any(
        item in name for item in ("thumb", "index", "middle", "ring", "pinky")
    ):
        return "right_hand"
    if name.startswith("left_") and any(
        item in name for item in ("heel", "forefoot", "foot_outer")
    ):
        return "left_foot"
    if name.startswith("right_") and any(
        item in name for item in ("heel", "forefoot", "foot_outer")
    ):
        return "right_foot"
    if name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
        return "head"
    return "body"


def _side_ambiguous(
    previous: PersonCandidate | None, current: PersonCandidate, side: str
) -> bool:
    hand = current.hand_observations.get(side)
    if hand is not None and hand.handedness_ambiguous:
        return True
    if previous is None:
        return False
    before = {point.name: point for point in previous.landmarks}
    after = {point.name: point for point in current.landmarks}
    left, right = (f"{label}_wrist" for label in ("left", "right"))
    if not all(name in before and name in after for name in (left, right)):
        return False
    if (
        min(
            *(before[name].raw_score.value for name in (left, right)),
            *(after[name].raw_score.value for name in (left, right)),
        )
        < 0.2
    ):
        return False
    same = sum(
        math.dist(before[name].xy_px, after[name].xy_px) for name in (left, right)
    )
    swapped = math.dist(before[left].xy_px, after[right].xy_px) + math.dist(
        before[right].xy_px, after[left].xy_px
    )
    return swapped + 0.05 * math.sqrt(_area(current.bbox_xyxy_px)) < same


def _foot_side_ambiguous(
    previous: PersonCandidate | None, current: PersonCandidate
) -> bool:
    if previous is None:
        return False
    before = {point.name: point for point in previous.landmarks}
    after = {point.name: point for point in current.landmarks}
    for suffix in ("heel", "big_toe", "small_toe"):
        left, right = f"left_{suffix}", f"right_{suffix}"
        if not all(name in before and name in after for name in (left, right)):
            continue
        if any(
            point.raw_score.value < 0.2
            or (point.raw_visibility is not None and point.raw_visibility < 0.5)
            for point in (before[left], before[right], after[left], after[right])
        ):
            continue
        same = math.dist(before[left].xy_px, after[left].xy_px) + math.dist(
            before[right].xy_px, after[right].xy_px
        )
        swapped = math.dist(before[left].xy_px, after[right].xy_px) + math.dist(
            before[right].xy_px, after[left].xy_px
        )
        if swapped + 0.05 * math.sqrt(_area(current.bbox_xyxy_px)) < same:
            return True
    return False


def _reconcile_geometry(
    source: tuple[RegionalGeometry2D, ...],
    landmarks: dict[str, Landmark2D],
    image_size: tuple[int, int],
    reasons: dict[str, set[str]],
    config: TrackingConfig,
) -> list[RegionalGeometry2D]:
    projected = {
        item.part: item
        for item in WholebodyRegionalProvider(
            min_raw_score=config.min_raw_score,
            min_raw_visibility=config.min_raw_visibility,
        )(list(landmarks.values()))
    }
    width, height = image_size

    def in_frame(xy: tuple[float, float] | None) -> bool:
        return xy is None or 0 <= xy[0] < width and 0 <= xy[1] < height

    result = []
    for part in ("left_foot", "right_foot", "head"):
        original = next((item for item in source if item.part == part), None)
        if original is None:
            result.append(projected[part])
        elif original.availability == "missing":
            result.append(original)
        elif (
            original.supporting_landmarks
            and "anatomical_side_ambiguous" not in reasons[part]
            and all(
                landmarks[name].xy_px is not None
                for name in original.supporting_landmarks
                if name in landmarks
            )
            and all(name in landmarks for name in original.supporting_landmarks)
            and in_frame(original.axis_start_px)
            and in_frame(original.axis_end_px)
            and (
                original.availability != "complete"
                or projected[part].availability == "complete"
            )
        ):
            result.append(original)
        else:
            result.append(projected[part])
    return result


class PractitionerTracker:
    """One camera/source track; uncertainty never advances the identity anchor."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._anchor: PersonCandidate | None = None
        self._anchor_time: float | None = None
        self._camera: str | None = None
        self._source: str | None = None
        self._last_ordinal: int | None = None
        self._last_time: float | None = None

    def observe(
        self,
        frame: PoseFrame,
        *,
        artifact_id: str,
        provenance: Provenance,
        operator_candidate_index: int | None = None,
    ) -> Observation:
        if self._camera is None:
            self._camera = frame.frame.camera_id
            self._source = frame.frame.source_id
        if (frame.frame.camera_id, frame.frame.source_id) != (
            self._camera,
            self._source,
        ):
            raise ValueError("tracker accepts one camera and source")
        if self._last_ordinal is not None and (
            frame.ordinal <= self._last_ordinal
            or frame.frame.source_seconds < (self._last_time or 0)
        ):
            raise ValueError("frames must advance in native order")
        self._last_ordinal = frame.ordinal
        self._last_time = frame.frame.source_seconds
        candidates = frame.candidates
        indices = {item.index: item for item in candidates}
        if len(indices) != len(candidates):
            raise ValueError("duplicate candidate index")
        costs = {
            item.index: _match_cost(self._anchor, item)
            for item in candidates
            if self._anchor is not None
        }
        selected: PersonCandidate | None = None
        method: Literal["initial", "temporal", "operator"] | None = None
        reasons: list[str] = []
        if operator_candidate_index is not None:
            if operator_candidate_index not in indices:
                raise ValueError("operator candidate is absent from frame")
            selected = indices[operator_candidate_index]
            method = "operator"
            reasons.append("operator_selection")
        elif not candidates:
            reasons.append("no_candidate")
        elif self._anchor is None:
            ranked = sorted(
                candidates, key=lambda item: _area(item.bbox_xyxy_px), reverse=True
            )
            if len(ranked) > 1 and _area(ranked[0].bbox_xyxy_px) < (
                self.config.min_initial_area_ratio * _area(ranked[1].bbox_xyxy_px)
            ):
                reasons.append("initial_candidates_ambiguous")
            else:
                selected = ranked[0]
                method = "initial"
        elif self._anchor_time is not None and (
            frame.frame.source_seconds - self._anchor_time > self.config.max_gap_seconds
        ):
            reasons.append("identity_gap")
        else:
            ranked = sorted(candidates, key=lambda item: costs[item.index])
            best = ranked[0]
            if costs[best.index] > self.config.max_match_cost:
                reasons.append("no_consistent_candidate")
            elif len(ranked) > 1 and (
                costs[ranked[1].index] - costs[best.index]
                < self.config.ambiguity_margin
            ):
                reasons.append("candidate_match_ambiguous")
            else:
                selected = best
                method = "temporal"
        if selected is not None:
            previous = self._anchor
            self._anchor = selected
            self._anchor_time = frame.frame.source_seconds
        else:
            previous = None
        evidence = [
            SubjectCandidateEvidence(
                index=item.index,
                bbox_xyxy_px=item.bbox_xyxy_px,
                detector_score=item.detector_score,
                match_cost=costs.get(item.index),
            )
            for item in candidates
        ]
        selection = SubjectSelection(
            state="selected"
            if selected is not None
            else "ambiguous"
            if candidates
            else "missing",
            track_id=f"{self._camera}:practitioner:0",
            candidate_index=selected.index if selected is not None else None,
            method=method,
            reasons=reasons,
            candidates=evidence,
        )
        return _assemble(
            frame, selected, previous, selection, artifact_id, provenance, self.config
        )


def _assemble(
    frame: PoseFrame,
    selected: PersonCandidate | None,
    previous: PersonCandidate | None,
    selection: SubjectSelection,
    artifact_id: str,
    provenance: Provenance,
    config: TrackingConfig,
) -> Observation:
    raw: list[Landmark2D] = []
    refined: list[Landmark2D] = []
    geometry: list[RegionalGeometry2D] = []
    combined: dict[str, Landmark2D] = {}
    reasons: dict[str, set[str]] = {part: set() for part in _PARTS}
    if selected is None:
        for part in _PARTS:
            reasons[part].add("subject_unavailable")
        combined = {
            name: Landmark2D(
                name=name,  # type: ignore[arg-type]
                xy_px=None,
                quality=Quality(state="unknown"),
            )
            for name in dict.fromkeys(CANONICAL.values())
        }
    else:
        raw = canonical_landmarks(selected)
        width, height = frame.image_size
        x0, y0, x1, y1 = selected.bbox_xyxy_px
        if x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height:
            reasons["body"].add("truncated_box")
        if previous is not None:
            old = {point.name: point for point in previous.landmarks}
            new = {point.name: point for point in selected.landmarks}
            old_box = previous.bbox_xyxy_px
            box_motion = (
                (x0 + x1 - old_box[0] - old_box[2]) / 2,
                (y0 + y1 - old_box[1] - old_box[3]) / 2,
            )
            scale = max(math.sqrt(_area(selected.bbox_xyxy_px)), 1.0)
            for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
                if (
                    name in old
                    and name in new
                    and min(old[name].raw_score.value, new[name].raw_score.value)
                    >= config.min_raw_score
                ):
                    motion = (
                        new[name].xy_px[0] - old[name].xy_px[0] - box_motion[0],
                        new[name].xy_px[1] - old[name].xy_px[1] - box_motion[1],
                    )
                    if math.hypot(*motion) > 0.6 * scale:
                        reasons["body"].add("implausible_torso_jump")
                        break
        for point in raw:
            part = _part(point.name)
            score = point.raw_score.value if point.raw_score is not None else 0
            x, y = point.xy_px or (float("nan"), float("nan"))
            if score < config.min_raw_score:
                reasons[part].add("low_raw_score")
            elif (
                point.raw_visibility is not None
                and point.raw_visibility < config.min_raw_visibility
            ):
                reasons[part].add("low_raw_visibility")
            elif not 0 <= x < width or not 0 <= y < height:
                reasons[part].add("out_of_frame")
            else:
                combined[point.name] = point
                continue
            combined[point.name] = point.model_copy(
                update={"xy_px": None, "quality": Quality(state="unknown")}
            )
        for side in ("left", "right"):
            part = f"{side}_hand"
            hand = selected.hand_observations.get(side)
            # Coarse wholebody fingers are source evidence, not detailed hand
            # observations. Only independently supported refinement can populate
            # the derived hand region.
            for name, point in tuple(combined.items()):
                if _part(name) == part:
                    combined[name] = point.model_copy(
                        update={"xy_px": None, "quality": Quality(state="unknown")}
                    )
            if hand is None:
                reasons[part].add("no_refinement")
            else:
                roi_usable = True
                if hand.roi is None:
                    reasons[part].add("tiny_roi")
                    roi_usable = False
                elif hand.roi.source_span_px < config.min_roi_span_px:
                    reasons[part].add("tiny_roi")
                    roi_usable = False
                elif hand.roi.visible_fraction < 0.5:
                    reasons[part].add("truncated_roi")
                    roi_usable = False
                if hand.status != "refined":
                    reasons[part].add("refinement_" + hand.status)
                for refined_point in hand.refined:
                    canonical_name = CANONICAL.get(refined_point.name)
                    if canonical_name is None:
                        continue
                    raw_score = (
                        RawScore(
                            value=refined_point.raw_score, range_min=0, range_max=1
                        )
                        if refined_point.raw_score is not None
                        else None
                    )
                    observed = (
                        refined_point.state == "observed"
                        and refined_point.xy_px is not None
                    )
                    item = Landmark2D(
                        name=canonical_name,  # type: ignore[arg-type]
                        xy_px=refined_point.xy_px if observed else None,
                        raw_score=raw_score,
                        raw_visibility=refined_point.raw_visibility,
                        quality=Quality(state="observed" if observed else "unknown"),
                    )
                    refined.append(item)
                    if (
                        observed
                        and roi_usable
                        and hand.status == "refined"
                        and (
                            refined_point.raw_visibility is None
                            or refined_point.raw_visibility >= config.min_raw_visibility
                        )
                    ):
                        combined[canonical_name] = item
                    elif (
                        refined_point.raw_visibility is not None
                        and refined_point.raw_visibility < config.min_raw_visibility
                    ):
                        reasons[part].add("low_raw_visibility")
            if _side_ambiguous(previous, selected, side):
                reasons[part].add("anatomical_side_ambiguous")
                for name, point in tuple(combined.items()):
                    if _part(name) == part:
                        combined[name] = point.model_copy(
                            update={"xy_px": None, "quality": Quality(state="unknown")}
                        )
        if _foot_side_ambiguous(previous, selected):
            for part in ("left_foot", "right_foot"):
                reasons[part].add("anatomical_side_ambiguous")
            for name, point in tuple(combined.items()):
                if _part(name) in ("left_foot", "right_foot"):
                    combined[name] = point.model_copy(
                        update={"xy_px": None, "quality": Quality(state="unknown")}
                    )
        geometry = _reconcile_geometry(
            selected.regional_geometry, combined, frame.image_size, reasons, config
        )
        for side in ("left", "right"):
            part = f"{side}_foot"
            foot_geometry = next((g for g in geometry if g.part == part), None)
            if foot_geometry is None or foot_geometry.availability == "missing":
                reasons[part].add("missing_geometry")
            elif foot_geometry.availability == "partial":
                reasons[part].add("partial_geometry")
            elif foot_geometry.orientation_state != "available":
                reasons[part].add("degenerate_geometry")
        head = next((g for g in geometry if g.part == "head"), None)
        if head is None or head.availability == "missing":
            reasons["head"].add("missing_geometry")
        elif head.orientation_state != "available":
            reasons["head"].add("incomplete_geometry")
    for part in _PARTS:
        if selected is not None and not any(
            _part(name) == part and item.xy_px is not None
            for name, item in combined.items()
        ):
            reasons[part].add("missing_landmarks")
    quality = [
        ViewRegionQuality(
            part=part,  # type: ignore[arg-type]
            usable=selected is not None
            and any(
                _part(name) == part and item.xy_px is not None
                for name, item in combined.items()
            )
            and "anatomical_side_ambiguous" not in reasons[part]
            and "implausible_torso_jump" not in reasons[part],
            reasons=sorted(reasons[part]),
        )
        for part in _PARTS
    ]
    return Observation(
        kind="observation",
        id=artifact_id,
        schema_version="1.0.0",
        provenance=provenance,
        frame=frame.frame,
        landmarks=list(combined.values()),
        wholebody_landmarks=raw,
        refined_landmarks=refined,
        regional_geometry=geometry,
        source_regional_geometry=list(selected.regional_geometry) if selected else [],
        subject_selection=selection,
        region_quality=quality,
    )
