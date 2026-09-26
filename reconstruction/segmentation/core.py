"""Inspectable offline coarse segmentation; no formal technique classification."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from contracts.models import Interval, Quality, SequenceStep, StrictModel, Track
from reconstruction.features import TRACKS, FeatureSample, FeatureSeries

REVISION: Literal["execution-segmentation-v1"] = "execution-segmentation-v1"
LIMBS = set(TRACKS[:4])


class SegmentationConfig(StrictModel):
    # Physical speed thresholds/units come from the qualified feature input.
    min_activity_seconds: float = Field(default=0.12, gt=0)
    edge_quiet_seconds: float = Field(default=0.4, gt=0)
    terminal_hold_seconds: float = Field(default=1.0, gt=0)
    step_quiet_seconds: float = Field(default=0.35, gt=0)
    max_gap_seconds: float = Field(default=0.15, gt=0)
    max_unknown_seconds: float = Field(default=0.2, ge=0)
    min_coverage: float = Field(default=0.8, gt=0, le=1)
    min_confidence: float = Field(default=0.65, gt=0, le=1)
    min_active_tracks: int = Field(default=2, ge=1, le=5)


class Activity(StrictModel):
    global_seconds: float
    motion_sample_index: int = Field(ge=0)
    known_tracks: list[Track]
    active_tracks: list[Track]


class Boundary(StrictModel):
    global_seconds: float
    motion_sample_index: int = Field(ge=0)
    reason: str
    event_ids: list[str]
    tolerance_seconds: float = Field(ge=0)


class StepEvidence(StrictModel):
    step_id: str
    # Inclusive source links; a boundary sample can belong to both neighbors.
    motion_sample_indices: list[int]
    event_ids: list[str]
    active_tracks: list[Track]


class SegmentationResult(StrictModel):
    version: Literal[1] = 1
    algorithm_revision: Literal["execution-segmentation-v1"] = REVISION
    reconstruction_id: str
    ground_id: str
    config: SegmentationConfig
    execution: Interval | None
    steps: list[SequenceStep]
    quality: Quality
    diagnostics: list[str]
    activity: list[Activity]
    boundaries: list[Boundary]
    step_evidence: list[StepEvidence]

    @model_validator(mode="after")
    def topology(self) -> SegmentationResult:
        times = [a.global_seconds for a in self.activity]
        if not times or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("increasing absolute activity times required")
        for i, row in enumerate(self.activity):
            if row.motion_sample_index != i or not set(row.active_tracks) <= set(
                row.known_tracks
            ):
                raise ValueError("activity source links disagree")
        if self.execution is None:
            if (
                self.steps
                or self.boundaries
                or self.step_evidence
                or (self.quality.state != "unknown")
            ):
                raise ValueError("indeterminate output cannot fabricate boundaries")
            return self
        if (
            self.quality.state != "inferred"
            or not self.steps
            or (
                len(self.boundaries) != len(self.steps) + 1
                or len(self.step_evidence) != len(self.steps)
            )
        ):
            raise ValueError("resolved output requires complete boundary evidence")
        for boundary in self.boundaries:
            i = boundary.motion_sample_index
            if i >= len(times) or times[i] != boundary.global_seconds:
                raise ValueError("boundary must link exact native time")
        if (self.boundaries[0].global_seconds, self.boundaries[-1].global_seconds) != (
            self.execution.start,
            self.execution.end,
        ):
            raise ValueError("execution and boundaries disagree")
        if len({s.id for s in self.steps}) != len(self.steps):
            raise ValueError("unique coarse step IDs required")
        for i, (step, evidence) in enumerate(zip(self.steps, self.step_evidence)):
            left, right = self.boundaries[i : i + 2]
            if (
                step.interval
                != Interval(start=left.global_seconds, end=right.global_seconds)
                or step.action_ids
                or evidence.step_id != step.id
                or evidence.motion_sample_indices
                != list(range(left.motion_sample_index, right.motion_sample_index + 1))
            ):
                raise ValueError("coarse containers must retain all dense source links")
        return self


def _state(row: FeatureSample, features: FeatureSeries) -> tuple[bool, bool]:
    known, moving = False, False
    for derivative, threshold in (
        (row.linear, features.config.enter_speed),
        (row.angular, features.config.enter_angular_speed_rad_s),
    ):
        q = derivative.velocity_quality
        if (
            derivative.velocity is None
            or q.state in ("unknown", "interpolated")
            or q.uncertainty is None
            or not q.source_ids
            or features.config.uncertainty_multiplier * q.uncertainty >= threshold
        ):
            continue
        known |= derivative is row.linear
        speed = math.sqrt(sum(v * v for v in derivative.velocity))
        moving |= (
            speed - features.config.uncertainty_multiplier * q.uncertainty >= threshold
        )
    return known or moving, moving


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            output.append((start, i - 1))
            start = None
    return output


def segment_execution(
    features: FeatureSeries, config: SegmentationConfig | None = None
) -> SegmentationResult:
    """Propose coarse units from complete offline features on the original clock."""
    config = config or SegmentationConfig()
    activity = []
    for i in range(len(features.trajectory) // len(TRACKS)):
        rows = features.trajectory[i * len(TRACKS) : (i + 1) * len(TRACKS)]
        states = [(r.track, *_state(r, features)) for r in rows]
        activity.append(
            Activity(
                global_seconds=rows[0].global_seconds,
                motion_sample_index=i,
                known_tracks=[t for t, known, _ in states if known],
                active_tracks=[t for t, _, moving in states if moving],
            )
        )
    result = SegmentationResult(
        reconstruction_id=features.reconstruction_id,
        ground_id=features.ground_id,
        config=config,
        execution=None,
        steps=[],
        quality=Quality(state="unknown", source_ids=[features.reconstruction_id]),
        diagnostics=[],
        activity=activity,
        boundaries=[],
        step_evidence=[],
    )

    def unknown(reason: str) -> SegmentationResult:
        result.diagnostics = [reason]
        return result

    times = [a.global_seconds for a in activity]
    if any(b - a > config.max_gap_seconds for a, b in zip(times, times[1:])):
        return unknown("native_sampling_gap")
    bouts = [
        (a, b)
        for a, b in _runs([bool(set(row.active_tracks) - {"head"}) for row in activity])
        if times[b] - times[a] >= config.min_activity_seconds
    ]
    if not bouts:
        return unknown("no_sustained_activity")
    first, last = bouts[0][0], bouts[-1][1]
    tracks = set().union(
        *(set(a.active_tracks) - {"head"} for a in activity[first : last + 1])
    )
    if len(tracks) < config.min_active_tracks:
        return unknown("insufficient_multi_track_activity")
    # Evidence after the final movement must include the whole configured hold
    # plus a separate post-roll bracket. Never shorten the hold to fit the video.
    end = next(
        (
            i
            for i in range(last + 1, len(times))
            if times[i] >= times[last] + config.terminal_hold_seconds
        ),
        None,
    )
    if end is None or times[-1] - times[end] < config.edge_quiet_seconds:
        return unknown("terminal_hold_or_post_roll_not_observed")
    if times[first] - times[0] < config.edge_quiet_seconds:
        return unknown("pre_roll_not_observed")
    quiet = [LIMBS <= set(a.known_tracks) and not a.active_tracks for a in activity]
    pre = [
        i for i in range(first) if times[i] >= times[first] - config.edge_quiet_seconds
    ]
    post = [
        i
        for i in range(last + 1, len(times))
        if times[i] <= times[end] + config.edge_quiet_seconds
    ]
    if not pre or not post or not all(quiet[i] for i in [*pre, *post]):
        return unknown("edge_context_unknown_or_active")
    covered = [LIMBS <= set(a.known_tracks) for a in activity[first : end + 1]]
    duration = times[end] - times[first]
    coverage = (
        sum(
            times[i + 1] - times[i]
            for i in range(first, end)
            if covered[i - first] and covered[i + 1 - first]
        )
        / duration
    )
    unknown_runs = _runs([not c for c in covered])
    if coverage < config.min_coverage or any(
        times[min(end, first + b + 1)] - times[max(first, first + a - 1)]
        > config.max_unknown_seconds
        for a, b in unknown_runs
    ):
        return unknown("insufficient_continuous_evidence")
    # This score is a reproducible evidence strength, not a calibrated probability.
    score = coverage * min(1.0, len(tracks) / config.min_active_tracks)
    if score < config.min_confidence:
        return unknown("confidence_below_threshold")

    cuts = [first]
    for (_, previous_end), (next_start, _) in zip(bouts, bouts[1:]):
        if times[next_start] - times[previous_end] < config.step_quiet_seconds:
            continue
        # Candidate evidence windows are support for measurements, not action
        # durations. Retain spanning references without forcing coarse bounds.
        # Physical lift/pivot state can keep recovery within one progression.
        awaiting_placement = False
        for track in ("left_leg", "right_leg"):
            contact_events = [
                e
                for e in features.events
                if e.track == track
                and e.kind
                in (
                    "lift_off",
                    "first_contact",
                    "stable_placement",
                    "pivot_start",
                    "pivot_end",
                )
                and times[cuts[-1]] <= e.global_seconds < times[next_start]
            ]
            if contact_events and contact_events[-1].kind in (
                "lift_off",
                "pivot_start",
            ):
                awaiting_placement = True
        if not awaiting_placement and all(quiet[previous_end + 1 : next_start]):
            cuts.append(next_start)
    cuts.append(end)
    result.execution = Interval(start=times[first], end=times[end])
    result.quality = Quality(
        state="inferred",
        score=score,
        uncertainty=max(b - a for a, b in zip(times, times[1:])),
        source_ids=[features.reconstruction_id, features.ground_id],
    )
    result.diagnostics = [
        "heuristic_confidence_not_probability",
        "terminal_hold_is_configured_allowance",
    ]
    for j, index in enumerate(cuts):
        result.boundaries.append(
            Boundary(
                global_seconds=times[index],
                motion_sample_index=index,
                reason=(
                    "sustained_activity_after_pre_roll"
                    if j == 0
                    else "configured_terminal_hold"
                    if j == len(cuts) - 1
                    else "renewed_activity_after_coordinated_quiet"
                ),
                event_ids=[
                    e.id
                    for e in features.events
                    if abs(e.global_seconds - times[index]) <= e.tolerance_seconds
                ],
                tolerance_seconds=result.quality.uncertainty or 0,
            )
        )
    for j, (a, b) in enumerate(zip(cuts, cuts[1:])):
        step = SequenceStep(
            id=f"sequence-step:{j}",
            interval=Interval(start=times[a], end=times[b]),
            action_ids=[],
        )
        result.steps.append(step)
        result.step_evidence.append(
            StepEvidence(
                step_id=step.id,
                motion_sample_indices=list(range(a, b + 1)),
                event_ids=[
                    e.id
                    for e in features.events
                    if e.evidence_start_seconds <= times[b]
                    and e.evidence_end_seconds >= times[a]
                ],
                active_tracks=[
                    t
                    for t in TRACKS
                    if any(t in r.active_tracks for r in activity[a : b + 1])
                ],
            )
        )
    return SegmentationResult.model_validate(result.model_dump())
