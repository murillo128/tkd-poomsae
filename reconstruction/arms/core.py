"""Offline independent arm proposals; heuristic geometry, never correctness."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from contracts.models import Interval, Quality, StrictModel
from reconstruction.detailed import DetailedSample
from reconstruction.detailed.core import Hand, Relation
from reconstruction.features import TRACKS, EventCandidate, FeatureSample, FeatureSeries
from reconstruction.segmentation import SegmentationResult

REVISION: Literal["arm-actions-v1"] = "arm-actions-v1"
Arm = Literal["left_arm", "right_arm"]
ARMS: tuple[Arm, Arm] = ("left_arm", "right_arm")


class ArmConfig(StrictModel):
    min_activity_seconds: float = Field(default=0.12, gt=0)
    quiet_seconds: float = Field(default=0.12, gt=0)
    max_gap_seconds: float = Field(default=0.15, gt=0)
    coordination_overlap: float = Field(default=0.7, gt=0, le=1)
    mirror_tolerance_ratio: float = Field(default=0.15, gt=0, lt=1)
    center_ratio: float = Field(default=0.45, gt=0, lt=1)


class PhaseCandidate(StrictModel):
    track: Arm
    name: Literal["preparation", "chamber", "extension", "retraction"]
    interval: Interval
    event_ids: list[str]
    motion_sample_indices: list[int]
    quality: Quality
    reasons: list[str]


class ArmSnapshot(StrictModel):
    motion_sample_index: int = Field(ge=0)
    global_seconds: float
    hand: Hand | None  # None is missing hand evidence, never a closed/open hand.
    relations: list[Relation]  # Includes unknown crossing and unknown depth order.
    relation_evidence_quality: Quality


class TrackSpan(StrictModel):
    track: Arm
    interval: Interval
    motion_sample_indices: list[int]
    quality: Quality
    snapshots: list[ArmSnapshot]


class ArmProposal(StrictModel):
    id: str
    interval: Interval
    category: Literal["arm", "special"]
    role: Literal["unknown", "special"]
    step_ids: list[str]  # Associations, not clipping/containment requirements.
    tracks: list[TrackSpan]
    quality: Quality
    reasons: list[str]
    events: list[EventCandidate]
    phases: list[PhaseCandidate]


class ArmResult(StrictModel):
    version: Literal[1] = 1
    algorithm_revision: Literal["arm-actions-v1"] = REVISION
    reconstruction_id: str
    ground_id: str
    config: ArmConfig
    execution: Interval | None
    motion_times: list[float]
    proposals: list[ArmProposal]
    diagnostics: list[str]

    @model_validator(mode="after")
    def topology(self) -> ArmResult:
        times = self.motion_times
        if not times or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("increasing absolute motion times required")
        if self.execution is None and self.proposals:
            raise ValueError("indeterminate execution cannot fabricate arm actions")
        if len({a.id for a in self.proposals}) != len(self.proposals):
            raise ValueError("unique arm proposal IDs required")
        for action in self.proposals:
            if not action.tracks or len({t.track for t in action.tracks}) != len(
                action.tracks
            ):
                raise ValueError("unique nonempty arm tracks required")
            if (action.category, action.role) not in (
                ("arm", "unknown"),
                ("special", "special"),
            ) or (action.category == "special" and len(action.tracks) != 2):
                raise ValueError("special actions require both tracks")
            if action.interval != Interval(
                start=min(t.interval.start for t in action.tracks),
                end=max(t.interval.end for t in action.tracks),
            ):
                raise ValueError("action must span its independent track intervals")
            if self.execution is None or not (
                self.execution.start
                <= action.interval.start
                <= action.interval.end
                <= self.execution.end
            ):
                raise ValueError("action outside execution")
            for track in action.tracks:
                indices = track.motion_sample_indices
                if (
                    not indices
                    or indices != list(range(indices[0], indices[-1] + 1))
                    or indices[0] < 0
                    or indices[-1] >= len(times)
                    or track.interval
                    != Interval(start=times[indices[0]], end=times[indices[-1]])
                    or [
                        (s.motion_sample_index, s.global_seconds)
                        for s in track.snapshots
                    ]
                    != [(i, times[i]) for i in indices]
                ):
                    raise ValueError("dense track links or snapshots disagree")
            if len({e.id for e in action.events}) != len(action.events):
                raise ValueError("unique source event IDs required")
            track_map = {t.track: t for t in action.tracks}
            for event in action.events:
                indices = event.motion_sample_indices
                if (
                    event.track not in track_map
                    or not indices
                    or indices != sorted(set(indices))
                    or indices[0] < 0
                    or indices[-1] >= len(times)
                    or event.global_seconds not in [times[i] for i in indices]
                    or event.evidence_start_seconds != times[indices[0]]
                    or event.evidence_end_seconds != times[indices[-1]]
                    or event.duration_seconds != times[indices[-1]] - times[indices[0]]
                ):
                    raise ValueError("source event evidence disagrees")
            for phase in action.phases:
                indices = phase.motion_sample_indices
                if (
                    phase.track not in track_map
                    or not indices
                    or indices != sorted(set(indices))
                    or indices[0] < 0
                    or indices[-1] >= len(times)
                    or phase.interval
                    != Interval(start=times[indices[0]], end=times[indices[-1]])
                    or not action.interval.start
                    <= phase.interval.start
                    <= phase.interval.end
                    <= action.interval.end
                    or not set(indices)
                    <= set(track_map[phase.track].motion_sample_indices)
                    or not set(phase.event_ids) <= {e.id for e in action.events}
                ):
                    raise ValueError("phase candidate evidence disagrees")
        return self


def _qualified(q: Quality, limit: float) -> bool:
    return (
        q.state not in ("unknown", "interpolated")
        and bool(q.source_ids)
        and q.uncertainty is not None
        and q.uncertainty <= limit
    )


def _movement(row: FeatureSample, features: FeatureSeries) -> int:
    """1 moving, 0 demonstrably quiet, -1 unknown (including threshold band)."""
    quiet = []
    for derivative, enter, leave in (
        (row.linear, features.config.enter_speed, features.config.leave_speed),
        (
            row.angular,
            features.config.enter_angular_speed_rad_s,
            features.config.leave_angular_speed_rad_s,
        ),
    ):
        if derivative.velocity is None or not _qualified(
            derivative.velocity_quality, leave / features.config.uncertainty_multiplier
        ):
            quiet.append(False)
            continue
        sigma = features.config.uncertainty_multiplier * float(
            derivative.velocity_quality.uncertainty or 0
        )
        speed = math.sqrt(sum(v * v for v in derivative.velocity))
        if speed - sigma >= enter:
            return 1
        quiet.append(speed + sigma <= leave)
    return 0 if all(quiet) else -1


def _spans(
    rows: list[FeatureSample], features: FeatureSeries, config: ArmConfig
) -> list[tuple[int, int]]:
    output = []
    start: int | None = None
    last: int | None = None
    quiet_start: int | None = None

    def finish() -> None:
        nonlocal start, last, quiet_start
        if (
            start is not None
            and last is not None
            and (
                rows[last].global_seconds - rows[start].global_seconds
                >= config.min_activity_seconds
            )
        ):
            output.append((start, last))
        start = last = quiet_start = None

    for i, row in enumerate(rows):
        if (
            i
            and row.global_seconds - rows[i - 1].global_seconds > config.max_gap_seconds
        ):
            finish()
        state = _movement(row, features)
        if state == -1:
            finish()
            continue
        if state == 1:
            if start is None:
                start = i
            last = i
            quiet_start = None
        elif state == 0 and start is not None:
            if quiet_start is None:
                quiet_start = i
            if (
                row.global_seconds - rows[quiet_start].global_seconds
                >= config.quiet_seconds
            ):
                finish()
    finish()
    return output


def _quality(rows: list[FeatureSample], features: FeatureSeries) -> Quality:
    sources = {features.reconstruction_id}
    scores = []
    for row in rows:
        sources.update(row.linear.velocity_quality.source_ids)
        sources.update(row.angular.velocity_quality.source_ids)
        for q in (row.linear.velocity_quality, row.angular.velocity_quality):
            if q.score is not None and q.state not in ("unknown", "interpolated"):
                scores.append(q.score)
    return Quality(
        state="inferred",
        score=min(scores) if scores else None,
        uncertainty=max(
            (b.global_seconds - a.global_seconds for a, b in zip(rows, rows[1:])),
            default=0,
        ),
        source_ids=sorted(sources),
    )


def _centering(
    left: TrackSpan,
    right: TrackSpan,
    rows: dict[Arm, list[FeatureSample]],
    features: FeatureSeries,
    config: ArmConfig,
) -> bool:
    overlap = min(left.interval.end, right.interval.end) - max(
        left.interval.start, right.interval.start
    )
    duration = max(
        left.interval.end - left.interval.start,
        right.interval.end - right.interval.start,
    )
    if duration <= 0 or overlap / duration < config.coordination_overlap:
        return False
    indices = sorted(set(left.motion_sample_indices) & set(right.motion_sample_indices))
    if len(indices) < 3:
        return False
    points = []
    for i in indices:
        left_position, right_position = (
            rows["left_arm"][i].position_local,
            rows["right_arm"][i].position_local,
        )
        if (
            left_position.value is None
            or right_position.value is None
            or not all(
                _qualified(p.quality, features.config.max_position_uncertainty)
                for p in (left_position, right_position)
            )
        ):
            return False
        points.append(
            (
                left_position.value,
                right_position.value,
                float(left_position.quality.uncertainty or 0)
                + float(right_position.quality.uncertainty or 0),
            )
        )
    first_l, first_r, _ = points[0]
    radius = max(abs(first_l[0]), abs(first_r[0]))
    # Body frame X is lateral. Supported pattern: mirrored hands converge
    # toward the sagittal plane, with equal front/height paths throughout.
    if first_l[0] >= 0 or first_r[0] <= 0 or radius <= 0:
        return False
    for left_point, right_point, sigma in points:
        error = math.sqrt(
            (left_point[0] + right_point[0]) ** 2
            + (left_point[1] - right_point[1]) ** 2
            + (left_point[2] - right_point[2]) ** 2
        )
        if error + features.config.uncertainty_multiplier * sigma > (
            radius * config.mirror_tolerance_ratio
        ):
            return False
    final_l, final_r, sigma = points[-1]
    for first, final in ((first_l, final_l), (first_r, final_r)):
        if (
            abs(final[0]) + features.config.uncertainty_multiplier * sigma
            > (abs(first[0]) * config.center_ratio)
            or abs(first[0]) - abs(final[0]) <= features.config.min_excursion
        ):
            return False
    return True


def _retractions(
    span: TrackSpan,
    rows: list[FeatureSample],
    features: FeatureSeries,
) -> list[PhaseCandidate]:
    output: list[PhaseCandidate] = []
    run: list[int] = []

    def finish() -> None:
        if len(run) < 2:
            return
        first, last = rows[run[0]], rows[run[-1]]
        assert first.extension.value is not None and last.extension.value is not None
        sigma = features.config.uncertainty_multiplier * (
            float(first.extension.quality.uncertainty or 0)
            + float(last.extension.quality.uncertainty or 0)
        )
        if (
            first.extension.value - last.extension.value
            <= max(sigma, features.config.extension_prominence_ratio)
            or last.global_seconds - first.global_seconds
            < features.config.min_duration_seconds
        ):
            return
        output.append(
            PhaseCandidate(
                track=span.track,
                name="retraction",
                interval=Interval(start=first.global_seconds, end=last.global_seconds),
                event_ids=[],
                motion_sample_indices=list(run),
                quality=Quality(
                    state="inferred",
                    uncertainty=sigma,
                    source_ids=sorted(
                        {s for i in run for s in rows[i].extension.quality.source_ids}
                    ),
                ),
                reasons=[
                    "provisional_physical_phase",
                    "prominent_decreasing_chain_extension",
                ],
            )
        )

    for i in span.motion_sample_indices:
        extension = rows[i].extension
        if extension.value is None or not _qualified(
            extension.quality, features.config.extension_prominence_ratio
        ):
            finish()
            run = []
            continue
        previous = rows[run[-1]].extension.value if run else None
        if previous is not None and extension.value >= previous:
            finish()
            run = []
        run.append(i)
    finish()
    return output


def parse_arm_actions(
    features: FeatureSeries,
    segmentation: SegmentationResult,
    geometry: list[DetailedSample],
    config: ArmConfig | None = None,
) -> ArmResult:
    """Consume qualified physical products; no labels, vision or model execution."""
    config = config or ArmConfig()
    times = [r.global_seconds for r in features.trajectory[:: len(TRACKS)]]
    if (features.reconstruction_id, features.ground_id) != (
        segmentation.reconstruction_id,
        segmentation.ground_id,
    ) or [(a.motion_sample_index, a.global_seconds) for a in segmentation.activity] != (
        list(enumerate(times))
    ):
        raise ValueError("coarse proposals and features must share exact motion input")
    if [g.global_seconds for g in geometry] != times or any(
        g.reconstruction_id != features.reconstruction_id for g in geometry
    ):
        raise ValueError("hand geometry must match reconstruction and native time")
    result = ArmResult(
        reconstruction_id=features.reconstruction_id,
        ground_id=features.ground_id,
        config=config,
        execution=segmentation.execution,
        motion_times=times,
        proposals=[],
        diagnostics=["heuristic_evidence_score_not_probability"],
    )
    if result.execution is None:
        result.diagnostics.append("execution_indeterminate")
        return result
    rows: dict[Arm, list[FeatureSample]] = {
        track: features.trajectory[j :: len(TRACKS)] for j, track in enumerate(ARMS)
    }
    spans: list[TrackSpan] = []
    for track in ARMS:
        for a, b in _spans(rows[track], features, config):
            # Execution gates whole proposals; no clipping at execution/step cuts.
            if (
                not result.execution.start
                <= times[a]
                <= times[b]
                <= result.execution.end
            ):
                continue
            indices = list(range(a, b + 1))
            spans.append(
                TrackSpan(
                    track=track,
                    interval=Interval(start=times[a], end=times[b]),
                    motion_sample_indices=indices,
                    quality=_quality(rows[track][a : b + 1], features),
                    snapshots=[
                        ArmSnapshot(
                            motion_sample_index=i,
                            global_seconds=times[i],
                            hand=geometry[i].hands.get(track.split("_")[0]),
                            relations=rows[track][i].relations,
                            relation_evidence_quality=rows[track][
                                i
                            ].relation_evidence_quality,
                        )
                        for i in indices
                    ],
                )
            )
    used: set[int] = set()
    groups: list[list[TrackSpan]] = []
    for i, left in enumerate(spans):
        if i in used:
            continue
        partner = next(
            (
                j
                for j, right in enumerate(spans)
                if j not in used
                and left.track == "left_arm"
                and right.track == "right_arm"
                and _centering(left, right, rows, features, config)
            ),
            None,
        )
        if partner is not None:
            used.add(partner)
            groups.append([left, spans[partner]])
        else:
            groups.append([left])
        used.add(i)
    groups.sort(key=lambda g: (min(t.interval.start for t in g), g[0].track))
    for j, group in enumerate(groups):
        interval = Interval(
            start=min(t.interval.start for t in group),
            end=max(t.interval.end for t in group),
        )
        events = [
            e
            for e in features.events
            if any(
                e.track == t.track
                and e.evidence_start_seconds <= t.interval.end
                and e.evidence_end_seconds >= t.interval.start
                for t in group
            )
        ]
        phases = []
        for span in group:
            track_events = [e for e in events if e.track == span.track]
            for event in track_events:
                name: Literal["preparation", "chamber", "extension", "retraction"]
                if event.kind == "preparation_candidate":
                    name = "preparation"
                elif event.kind == "extension_minimum":
                    name = "chamber"
                elif event.kind in (
                    "extension_start",
                    "extension_end",
                    "extension_maximum",
                ):
                    name = "extension"
                else:
                    continue
                indices = sorted(
                    set(event.motion_sample_indices) & set(span.motion_sample_indices)
                )
                if indices:
                    phases.append(
                        PhaseCandidate(
                            track=span.track,
                            name=name,
                            interval=Interval(
                                start=times[indices[0]], end=times[indices[-1]]
                            ),
                            event_ids=[event.id],
                            motion_sample_indices=indices,
                            quality=event.quality,
                            reasons=["provisional_physical_phase", *event.reasons],
                        )
                    )
            phases.extend(_retractions(span, rows[span.track], features))
        special = len(group) == 2
        result.proposals.append(
            ArmProposal(
                id=f"arm-action:{j}",
                interval=interval,
                category="special" if special else "arm",
                role="special" if special else "unknown",
                step_ids=[
                    s.id
                    for s in segmentation.steps
                    if s.interval.start <= interval.end
                    and s.interval.end >= interval.start
                ],
                tracks=group,
                quality=_quality(
                    [rows[t.track][i] for t in group for i in t.motion_sample_indices],
                    features,
                ),
                reasons=[
                    "mirrored_two_arm_centering"
                    if special
                    else "independent_qualified_arm_activity"
                ],
                events=events,
                phases=phases,
            )
        )
    return ArmResult.model_validate(result.model_dump())
