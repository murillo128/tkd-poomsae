"""Replaceable view-space regional geometry from named 2D landmarks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Protocol, cast

from contracts.models import Landmark, Landmark2D, RegionalGeometry2D


class RegionalProvider(Protocol):
    def __call__(
        self, landmarks: Sequence[Landmark2D]
    ) -> tuple[RegionalGeometry2D, ...]: ...


class WholebodyRegionalProvider:
    """Use wholebody toes, heels and sparse head/torso points only.

    The score cutoff selects usable geometry; it is not a calibrated confidence
    or a claim that a point is anatomically accurate. Original points and their
    raw scores remain available even when this projection marks a region missing.
    """

    def __init__(
        self,
        *,
        min_raw_score: float = 0.2,
        min_raw_visibility: float = 0.5,
        min_span_px: float = 2.0,
    ) -> None:
        if (
            not 0 <= min_raw_score <= 1
            or not 0 <= min_raw_visibility <= 1
            or min_span_px <= 0
        ):
            raise ValueError("invalid regional geometry thresholds")
        self.min_raw_score = min_raw_score
        self.min_raw_visibility = min_raw_visibility
        self.min_span_px = min_span_px

    def __call__(
        self, landmarks: Sequence[Landmark2D]
    ) -> tuple[RegionalGeometry2D, ...]:
        by_name: dict[str, Landmark2D] = {point.name: point for point in landmarks}

        def point(name: str) -> tuple[float, float] | None:
            item = by_name.get(name)
            if (
                item is None
                or item.xy_px is None
                or item.raw_score is None
                or not (
                    item.raw_score.domain == "rtmpose_simcc_response"
                    or (item.raw_score.range_min == 0 and item.raw_score.range_max == 1)
                )
                or item.raw_score.value < self.min_raw_score
                or (
                    item.raw_visibility is not None
                    and item.raw_visibility < self.min_raw_visibility
                )
            ):
                return None
            return item.xy_px

        def midpoint(
            a: tuple[float, float], b: tuple[float, float]
        ) -> tuple[float, float]:
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

        def span(a: tuple[float, float], b: tuple[float, float]) -> float:
            return math.dist(a, b)

        def transverse(
            a: tuple[float, float],
            b: tuple[float, float],
            start: tuple[float, float],
            end: tuple[float, float],
        ) -> float:
            length = span(start, end)
            if length == 0:
                return 0.0
            return (
                abs(
                    (b[0] - a[0]) * (end[1] - start[1])
                    - (b[1] - a[1]) * (end[0] - start[0])
                )
                / length
            )

        regions: list[RegionalGeometry2D] = []
        for side in ("left", "right"):
            heel_name = f"{side}_heel"
            medial_name = f"{side}_forefoot"
            lateral_name = f"{side}_foot_outer"
            heel = point(heel_name)
            medial = point(medial_name)
            lateral = point(lateral_name)
            support: list[Landmark] = [
                cast(Landmark, name)
                for name, value in (
                    (heel_name, heel),
                    (medial_name, medial),
                    (lateral_name, lateral),
                )
                if value is not None
            ]
            toes = [value for value in (medial, lateral) if value is not None]
            end = (
                midpoint(toes[0], toes[1])
                if len(toes) == 2
                else (toes[0] if toes else None)
            )
            availability: Literal["complete", "partial", "missing"] = (
                "complete" if len(support) == 3 else "partial" if support else "missing"
            )
            orientation_state: Literal["available", "degenerate", "unavailable"]
            if heel is None or end is None:
                orientation_state = "unavailable"
                start = None
                end = None
            elif span(heel, end) < self.min_span_px:
                orientation_state = (
                    "degenerate" if availability == "complete" else "unavailable"
                )
                start = heel
            else:
                start = heel
                orientation_state = (
                    "available"
                    if availability == "complete"
                    and medial is not None
                    and lateral is not None
                    and transverse(medial, lateral, heel, end) >= self.min_span_px
                    else "degenerate"
                    if availability == "complete"
                    else "unavailable"
                )
            regions.append(
                RegionalGeometry2D(
                    part=cast(Literal["left_foot", "right_foot"], f"{side}_foot"),
                    availability=availability,
                    orientation_state=orientation_state,
                    provider="mmpose/wholebody",
                    supporting_landmarks=support,
                    axis_start_px=start,
                    axis_end_px=end,
                    orientation_rad=math.atan2(end[1] - start[1], end[0] - start[0])
                    if orientation_state == "available"
                    and start is not None
                    and end is not None
                    else None,
                )
            )

        names = ("nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder")
        values = {name: point(name) for name in names}
        support = [cast(Landmark, name) for name in names if values[name] is not None]
        availability = (
            "complete"
            if len(support) == len(names)
            else "partial"
            if support
            else "missing"
        )
        nose = values["nose"]
        left_eye = values["left_eye"]
        right_eye = values["right_eye"]
        left_shoulder = values["left_shoulder"]
        right_shoulder = values["right_shoulder"]
        if nose is not None and left_eye is not None and right_eye is not None:
            eye_mid = midpoint(left_eye, right_eye)
            axis_valid = span(eye_mid, nose) >= self.min_span_px
            orientation_state = (
                "available"
                if availability == "complete"
                and axis_valid
                and left_shoulder is not None
                and right_shoulder is not None
                and span(left_shoulder, right_shoulder) >= self.min_span_px
                and transverse(left_eye, right_eye, eye_mid, nose) >= self.min_span_px
                else "degenerate"
                if availability == "complete"
                else "unavailable"
            )
        else:
            eye_mid = None
            axis_valid = False
            orientation_state = (
                "degenerate" if availability == "complete" else "unavailable"
            )
        regions.append(
            RegionalGeometry2D(
                part="head",
                availability=availability,
                orientation_state=orientation_state,
                provider="mmpose/wholebody",
                supporting_landmarks=support,
                axis_start_px=eye_mid if axis_valid else None,
                axis_end_px=nose if axis_valid else None,
                orientation_rad=(
                    math.atan2(nose[1] - eye_mid[1], nose[0] - eye_mid[0])
                    if orientation_state == "available"
                    and nose is not None
                    and eye_mid is not None
                    else None
                ),
            )
        )
        return tuple(regions)
