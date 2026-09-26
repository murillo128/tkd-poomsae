"""Conservative offline lower-body rules on native times, without formal names."""

from __future__ import annotations

import math
from typing import Literal, cast

from pydantic import Field, model_validator

from contracts.models import Ground, Interval, Quality, SequenceStep, StrictModel
from reconstruction.features import FeatureSample, FeatureSeries
from reconstruction.segmentation import SegmentationResult

REVISION: Literal["lower-body-rules-v2"] = "lower-body-rules-v2"
Leg = Literal["left_leg", "right_leg"]
Class = Literal["kick", "placement", "pivot", "unknown"]


class LowerBodyConfig(StrictModel):
    max_gap_seconds: float = Field(default=0.15, gt=0)
    min_phase_seconds: float = Field(default=0.04, gt=0)
    stance_seconds: float = Field(default=0.12, gt=0)
    chamber_max_ratio: float = Field(default=0.8, gt=0, lt=1)
    extension_excursion_ratio: float = Field(default=0.15, gt=0, lt=1)
    placement_distance: float = Field(default=0.04, gt=0)
    min_pivot_rad: float = Field(default=0.12, gt=0)
    max_position_uncertainty: float = Field(default=0.03, gt=0)
    max_extension_uncertainty: float = Field(default=0.03, gt=0)
    uncertainty_multiplier: float = Field(default=3, ge=1)


class Evidence(StrictModel):
    interval: Interval
    motion_sample_indices: list[int]
    ground_sample_indices: list[int]
    feature_event_ids: list[str]
    ground_event_ids: list[str]
    quality: Quality
    reasons: list[str] = Field(min_length=1)


class PhaseCandidate(Evidence):
    name: Literal["chamber", "extension", "retraction", "recovery", "placement"]


class ActionCandidate(Evidence):
    id: str
    track: Leg
    category: Class
    phases: list[PhaseCandidate]
    sequence_step_ids: list[str]
    previous_action_id: str | None = None


class StanceCandidate(Evidence):
    id: str
    label: Literal["stable_two_foot_configuration"] = "stable_two_foot_configuration"
    contributing_action_ids: list[str]
    # Physical placements, never a formal stance vocabulary.
    footprint_ids: list[str]


class LowerBodyResult(StrictModel):
    version: Literal[1] = 1
    algorithm_revision: Literal["lower-body-rules-v1", "lower-body-rules-v2"] = REVISION
    reconstruction_id: str
    ground_id: str
    config: LowerBodyConfig
    world_unit: Literal["m", "arbitrary"]
    execution_proposal: Interval | None
    sequence_steps: list[SequenceStep]
    # Full source clock including portions outside the execution proposal.
    native_times: list[float]
    actions: list[ActionCandidate]
    stances: list[StanceCandidate]
    diagnostics: list[str]

    @model_validator(mode="after")
    def links(self) -> LowerBodyResult:
        times = self.native_times
        if not times or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("increasing native times required")
        ids = [a.id for a in self.actions] + [s.id for s in self.stances]
        if len(ids) != len(set(ids)):
            raise ValueError("unique candidate IDs required")
        actions = {a.id: a for a in self.actions}
        for item in [*self.actions, *self.stances]:
            for evidence in [item, *getattr(item, "phases", [])]:
                indices = evidence.motion_sample_indices
                if (
                    len(indices) < 2
                    or indices != list(range(indices[0], indices[-1] + 1))
                    or indices[0] < 0
                    or indices[-1] >= len(times)
                    or evidence.interval
                    != Interval(start=times[indices[0]], end=times[indices[-1]])
                ):
                    raise ValueError("candidate must preserve dense native links")
                if evidence is not item and not (
                    item.interval.start
                    <= evidence.interval.start
                    < evidence.interval.end
                    <= item.interval.end
                ):
                    raise ValueError("phase outside action")
            if isinstance(item, ActionCandidate):
                if item.sequence_step_ids != [
                    s.id
                    for s in self.sequence_steps
                    if s.interval.start < item.interval.end
                    and s.interval.end > item.interval.start
                ]:
                    raise ValueError("coarse overlap associations disagree")
                if item.category == "unknown" and item.quality.state != "unknown":
                    raise ValueError("unknown class cannot carry confident quality")
                if item.category == "kick" and not {
                    "chamber",
                    "extension",
                    "retraction",
                } <= {p.name for p in item.phases if p.quality.state == "inferred"}:
                    raise ValueError("independent kick phase evidence required")
                if item.previous_action_id is not None:
                    previous = actions.get(item.previous_action_id)
                    if (
                        previous is None
                        or previous.track != item.track
                        or (previous.interval.end > item.interval.start)
                    ):
                        raise ValueError("invalid recovery predecessor")
            elif isinstance(item, StanceCandidate) and any(
                ref not in actions for ref in item.contributing_action_ids
            ):
                raise ValueError("stance contributing action missing")
        return self


def _usable(q: Quality, limit: float) -> bool:
    return (
        q.state in ("observed", "inferred")
        and bool(q.source_ids)
        and q.uncertainty is not None
        and q.uncertainty <= limit
    )


def _runs(flags: list[bool], times: list[float], gap: float) -> list[tuple[int, int]]:
    result = []
    start: int | None = None
    for i, flag in enumerate([*flags, False]):
        if start is not None and (not flag or times[i] - times[i - 1] > gap):
            if i - 1 > start:
                result.append((start, i - 1))
            start = None
        if flag and start is None:
            start = i
    return result


def parse_lower_body(
    features: FeatureSeries,
    ground: Ground,
    proposals: SegmentationResult,
    config: LowerBodyConfig | None = None,
) -> LowerBodyResult:
    """Classify each leg independently; contact/support remain upstream facts."""
    config = config or LowerBodyConfig()
    if (features.reconstruction_id, features.ground_id) != (
        ground.reconstruction_id,
        ground.id,
    ) or (proposals.reconstruction_id, proposals.ground_id) != (
        features.reconstruction_id,
        features.ground_id,
    ):
        raise ValueError("matching reconstruction/ground lineage required")
    times = [r.global_seconds for r in features.trajectory[::6]]
    if times != [a.global_seconds for a in proposals.activity]:
        raise ValueError("proposal and feature clocks disagree")
    if features.config.world_unit != ("m" if ground.scale == "metric" else "arbitrary"):
        raise ValueError("world threshold units disagree")
    floor = {s.global_seconds: (i, s) for i, s in enumerate(ground.samples)}
    channels = {
        t: [r for r in features.trajectory if r.track == t]
        for t in ("left_leg", "right_leg")
    }
    # Exact native contact association, with no nearest-neighbor support invention.
    for rows in channels.values():
        for row in rows:
            if row.ground_sample_index is not None:
                entry = floor.get(row.global_seconds)
                if (
                    entry is None
                    or entry[0] != row.ground_sample_index
                    or (
                        row.contact
                        != (
                            entry[1].left if row.track == "left_leg" else entry[1].right
                        )
                    )
                ):
                    raise ValueError("feature and ground contact links disagree")

    def contact(row: FeatureSample, state: str) -> bool:
        return bool(
            row.ground_sample_index is not None
            and row.contact is not None
            and row.contact.state == state
            and _usable(row.contact.quality, config.max_position_uncertainty)
        )

    def evidence(
        rows: list[FeatureSample],
        a: int,
        b: int,
        reasons: list[str],
        known: bool = True,
        ground_ids: list[str] | None = None,
    ) -> Evidence:
        events = [
            e
            for e in features.events
            if e.track == rows[a].track
            and e.evidence_start_seconds <= times[b]
            and e.evidence_end_seconds >= times[a]
        ]
        return Evidence(
            interval=Interval(start=times[a], end=times[b]),
            motion_sample_indices=list(range(a, b + 1)),
            ground_sample_indices=sorted(
                {
                    r.ground_sample_index
                    for r in rows[a : b + 1]
                    if r.ground_sample_index is not None
                }
            ),
            feature_event_ids=[e.id for e in events],
            ground_event_ids=ground_ids or [],
            quality=Quality(
                state="inferred" if known else "unknown",
                score=0.8 if known else None,
                uncertainty=max(times[i + 1] - times[i] for i in range(a, b)),
                source_ids=[features.reconstruction_id, ground.id],
            ),
            reasons=reasons,
        )

    actions: list[ActionCandidate] = []

    def add(
        rows: list[FeatureSample],
        a: int,
        b: int,
        category: Class,
        reasons: list[str],
        phases: list[PhaseCandidate] | None = None,
        previous: str | None = None,
        ground_ids: list[str] | None = None,
    ) -> ActionCandidate:
        item = ActionCandidate(
            **evidence(
                rows, a, b, reasons, category != "unknown", ground_ids
            ).model_dump(),
            id=f"lower-body:{rows[a].track}:{category}:{a}:{b}",
            track=cast(Leg, rows[a].track),
            category=category,
            phases=phases or [],
            sequence_step_ids=[
                s.id
                for s in proposals.steps
                if s.interval.start < times[b] and s.interval.end > times[a]
            ],
            previous_action_id=previous,
        )
        actions.append(item)
        return item

    def phase(
        rows: list[FeatureSample],
        a: int,
        b: int,
        name: Literal["chamber", "extension", "retraction", "recovery", "placement"],
    ) -> PhaseCandidate:
        return PhaseCandidate(
            **evidence(rows, a, b, [f"{name}_chain_evidence"]).model_dump(), name=name
        )

    for track, rows in channels.items():
        covered: set[int] = set()
        # Supported physical pivots require rotation AND the supporting foot.
        for pivot in ground.pivots:
            if pivot.foot != track.split("_")[0]:
                continue
            indices = [
                i
                for i, t in enumerate(times)
                if pivot.interval.start <= t <= pivot.interval.end
            ]
            if len(indices) < 2:
                continue
            a, b = indices[0], indices[-1]
            valid = (
                times[a] == pivot.interval.start
                and times[b] == pivot.interval.end
                and pivot.rotation_rad is not None
                and abs(pivot.rotation_rad) >= config.min_pivot_rad
                and _usable(pivot.quality, config.min_pivot_rad / 3)
                and all(contact(r, "contact") for r in rows[a : b + 1])
                and all(
                    times[i + 1] - times[i] <= config.max_gap_seconds
                    for i in range(a, b)
                )
            )
            add(
                rows,
                a,
                b,
                "pivot" if valid else "unknown",
                [
                    "supported_physical_foot_rotation"
                    if valid
                    else "pivot_support_or_rotation_unavailable"
                ],
                ground_ids=[pivot.id],
            )
            covered.update(indices)

        # Airborne bouts are not automatically kicks or steps.
        for a, b in _runs(
            [contact(r, "no_contact") for r in rows], times, config.max_gap_seconds
        ):
            values = [r.extension.value for r in rows[a : b + 1]]
            qualified = all(
                v is not None
                and _usable(r.extension.quality, config.max_extension_uncertainty)
                for v, r in zip(values, rows[a : b + 1])
            )
            kick: ActionCandidate | None = None
            recovery_end = a
            kick_like = False
            partial_phases: list[PhaseCandidate] = []
            missing_chamber = False
            if qualified:
                ext = [float(cast(float, v)) for v in values]
                peak = a + max(range(len(ext)), key=ext.__getitem__)
                chamber = min(range(a, peak + 1), key=lambda i: ext[i - a])
                recovery = min(range(peak, b + 1), key=lambda i: ext[i - a])
                error = (
                    config.uncertainty_multiplier
                    * 2
                    * max(
                        cast(float, r.extension.quality.uncertainty)
                        for r in rows[a : b + 1]
                    )
                )
                excursion = max(config.extension_excursion_ratio, error)
                kick_like = (
                    ext[chamber - a] < config.chamber_max_ratio
                    and ext[peak - a] - ext[chamber - a] > excursion
                    and times[peak] - times[chamber] >= config.min_phase_seconds
                )
                if kick_like:
                    partial_phases.append(phase(rows, chamber, peak, "extension"))
                    # A chamber phase needs its own native sample bracket. An
                    # already-flexed onset followed immediately by the peak does
                    # not substantiate an interval; do not invent one.
                    chamber_end = (
                        chamber
                        if times[chamber] - times[a] + 1e-9 >= config.min_phase_seconds
                        else next(
                            (
                                i
                                for i in range(a + 1, peak)
                                if times[i] - times[a] + 1e-9
                                >= config.min_phase_seconds
                                and max(ext[: i - a + 1]) < config.chamber_max_ratio
                            ),
                            None,
                        )
                    )
                    chamber_observed = chamber_end is not None
                    missing_chamber = not chamber_observed
                    if chamber_end is not None:
                        partial_phases.insert(0, phase(rows, a, chamber_end, "chamber"))
                    retraction_observed = (
                        ext[peak - a] - ext[recovery - a] > excursion
                        and times[recovery] - times[peak] >= config.min_phase_seconds
                    )
                    if retraction_observed:
                        partial_phases.append(phase(rows, peak, recovery, "retraction"))
                    if chamber_observed and retraction_observed:
                        kick = add(
                            rows,
                            a,
                            recovery,
                            "kick",
                            ["airborne_chamber_extension_retraction"],
                            partial_phases,
                        )
                        recovery_end = recovery
                        covered.update(range(a, recovery + 1))
            # Placement requires a bracketed lift/landing and observed displacement.
            start, end = a - 1, b + 1
            landed = (
                start >= 0
                and end < len(rows)
                and contact(rows[start], "contact")
                and contact(rows[end], "contact")
                and times[a] - times[start] <= config.max_gap_seconds
                and times[end] - times[b] <= config.max_gap_seconds
            )
            displaced = False
            if landed:
                first, last = rows[start].position_world, rows[end].position_world
                if (
                    first.value is not None
                    and last.value is not None
                    and all(
                        _usable(p.quality, config.max_position_uncertainty)
                        for p in (first, last)
                    )
                ):
                    delta = math.dist(first.value[:2], last.value[:2])
                    error = config.uncertainty_multiplier * sum(
                        cast(float, p.quality.uncertainty) for p in (first, last)
                    )
                    displaced = delta > max(config.placement_distance, error)
            if landed and displaced:
                begin = recovery_end if kick else start
                phases = [
                    *(partial_phases if not kick else []),
                    phase(rows, begin, end, "recovery" if kick else "placement"),
                ]
                add(
                    rows,
                    begin,
                    end,
                    "placement" if kick or (qualified and not kick_like) else "unknown",
                    ["observed_lift_landing_and_planar_displacement"]
                    + (
                        []
                        if kick or (qualified and not kick_like)
                        else ["kick_phase_evidence_incomplete"]
                    )
                    + (["chamber_interval_unavailable"] if missing_chamber else []),
                    phases,
                    previous=kick.id if kick else None,
                    ground_ids=[
                        f.id
                        for f in ground.footprints
                        if f.foot == track.split("_")[0]
                        and f.interval.start <= times[end] <= f.interval.end
                    ],
                )
                covered.update(range(begin, end + 1))
            elif not kick:
                add(
                    rows,
                    a,
                    b,
                    "unknown",
                    ["airborne_motion_without_class_evidence"]
                    + (["chamber_interval_unavailable"] if missing_chamber else []),
                    partial_phases,
                )
                covered.update(range(a, b + 1))
            elif recovery_end < b:
                add(
                    rows,
                    recovery_end,
                    b,
                    "unknown",
                    ["recovery_without_observed_placement"],
                    previous=kick.id,
                )
                covered.update(range(recovery_end, b + 1))

        # Preserve otherwise unexplained motion, including missing-support turns.
        moving = []
        for i, row in enumerate(rows):
            active = False
            for d, threshold in (
                (row.linear, features.config.enter_speed),
                (row.angular, features.config.enter_angular_speed_rad_s),
            ):
                if d.velocity is not None and _usable(
                    d.velocity_quality, threshold / 3
                ):
                    active |= math.sqrt(sum(v * v for v in d.velocity)) > threshold
            moving.append(active and i not in covered)
        for a, b in _runs(moving, times, config.max_gap_seconds):
            add(rows, a, b, "unknown", ["motion_without_supported_class_evidence"])

    # Stances are stable configurations, never actions. Unknown/airborne feet
    # interrupt state intervals; geometry is retained in the source features.
    stationary = []
    for i in range(len(times)):
        stable = True
        for rows in channels.values():
            row = rows[i]
            previous = rows[max(0, i - 1)].position_world
            current = row.position_world
            stable &= bool(
                contact(row, "contact")
                and previous.value is not None
                and current.value is not None
                and _usable(previous.quality, config.max_position_uncertainty)
                and _usable(current.quality, config.max_position_uncertainty)
                and (i == 0 or times[i] - times[i - 1] <= config.max_gap_seconds)
                and math.dist(
                    previous.value,
                    current.value,
                )
                <= features.config.leave_speed * (times[i] - times[max(0, i - 1)])
                and not any(
                    x.track == row.track
                    and x.interval.start < times[i] < x.interval.end
                    for x in actions
                )
            )
        stationary.append(stable)
    stances = []
    for a, b in _runs(stationary, times, config.max_gap_seconds):
        if times[b] - times[a] < config.stance_seconds:
            continue
        e = evidence(channels["left_leg"], a, b, ["both_feet_observed_stationary"])
        stances.append(
            StanceCandidate(
                **e.model_dump(),
                id=f"stance:{a}:{b}",
                contributing_action_ids=[
                    x.id
                    for x in actions
                    if 0 <= times[a] - x.interval.end <= config.max_gap_seconds
                ],
                footprint_ids=[
                    f.id
                    for f in ground.footprints
                    if f.interval.start <= times[b] and f.interval.end >= times[a]
                ],
            )
        )
    return LowerBodyResult(
        reconstruction_id=features.reconstruction_id,
        ground_id=ground.id,
        config=config,
        world_unit=features.config.world_unit,
        execution_proposal=proposals.execution,
        sequence_steps=proposals.steps,
        native_times=times,
        actions=sorted(actions, key=lambda x: (x.interval.start, x.track)),
        stances=stances,
        diagnostics=[
            "heuristic_confidence_not_probability",
            "unsupported_distinctions_remain_unknown",
        ],
    )
