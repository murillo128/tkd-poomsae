"""Deterministic assembly on the original clock, without upstream inference."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import Field

from contracts.models import (
    Action,
    BodyEntity,
    Interval,
    Keyframe,
    Phase,
    Provenance,
    Quality,
    SemanticMotionLink,
    Semantics,
    SpatialRelation,
    StanceState,
    StrictModel,
    Track,
)
from reconstruction.arms import ArmResult
from reconstruction.detailed.core import Relation
from reconstruction.features import TRACKS, EventCandidate, FeatureSeries
from reconstruction.lower_body import LowerBodyResult
from reconstruction.segmentation import SegmentationResult
from storage import hash_config

REVISION = "semantic-assembly-v1"


class AssemblyConfig(StrictModel):
    max_gap_seconds: float = Field(default=0.15, gt=0)


def _id(kind: str, *parts: object) -> str:
    return f"{kind}:{hash_config(parts)}"


def assemble_semantics(
    features: FeatureSeries,
    coarse: SegmentationResult,
    arms: ArmResult,
    lower: LowerBodyResult,
    config: AssemblyConfig | None = None,
) -> Semantics:
    """One action per proposal; steps reference every overlapping action."""
    config = config or AssemblyConfig()
    # Revalidate caller-owned models: assignment/model_copy can bypass validation.
    features = FeatureSeries.model_validate(features.model_dump())
    coarse = SegmentationResult.model_validate(coarse.model_dump())
    arms = ArmResult.model_validate(arms.model_dump())
    lower = LowerBodyResult.model_validate(lower.model_dump())
    lineage = (features.reconstruction_id, features.ground_id)
    times = [r.global_seconds for r in features.trajectory[::6]]
    if any(
        (x.reconstruction_id, x.ground_id) != lineage for x in (coarse, arms, lower)
    ):
        raise ValueError("matching reconstruction/ground lineage required")
    if (
        times != arms.motion_times
        or times != lower.native_times
        or times != [a.global_seconds for a in coarse.activity]
    ):
        raise ValueError("semantic input clocks disagree")
    if (
        arms.execution != coarse.execution
        or lower.execution_proposal != coarse.execution
    ):
        raise ValueError("semantic execution proposals disagree")
    if lower.sequence_steps != coarse.steps:
        raise ValueError("semantic coarse steps disagree")
    identity = _id(
        "semantics",
        REVISION,
        config.model_dump(mode="json"),
        *[x.model_dump(mode="json") for x in (features, coarse, arms, lower)],
    )
    result = Semantics(
        kind="semantics",
        id=identity,
        schema_version="1.0.0",
        provenance=Provenance(
            producer="reconstruction.semantics",
            model=REVISION,
            config_digest=hash_config(config.model_dump(mode="json")),
        ),
        reconstruction_id=lineage[0],
        ground_id=lineage[1],
        execution=None,
        quality=Quality(state="unknown"),
        steps=[],
        stances=[],
        actions=[],
        phases=[],
        keyframes=[],
        relations=[],
    )
    if coarse.execution is None:
        return result
    result.quality = coarse.quality

    def overlaps(a: Interval, b: Interval) -> bool:
        return a.start < b.end and b.start < a.end

    # Keep complete connected actions at execution edges, including recovery.
    selected_lower = [
        a for a in lower.actions if overlaps(a.interval, coarse.execution)
    ]
    extent = coarse.execution.model_copy(deep=True)
    while True:
        expanded = [
            a
            for a in lower.actions
            if overlaps(a.interval, extent)
            or any(a.previous_action_id == b.id for b in selected_lower)
        ]
        edges = [extent, *(a.interval for a in expanded)]
        updated = Interval(
            start=min(i.start for i in edges), end=max(i.end for i in edges)
        )
        selected_lower = expanded
        if updated == extent:
            break
        extent = updated
    result.execution = extent
    if extent != coarse.execution:
        result.quality = Quality(state="unknown", source_ids=list(lineage))
    result.motion_sample_indices = [
        i for i, t in enumerate(times) if extent.start <= t <= extent.end
    ]
    result.steps = [s.model_copy(deep=True) for s in coarse.steps]
    result.steps[0].interval = Interval(
        start=extent.start, end=result.steps[0].interval.end
    )
    result.steps[-1].interval = Interval(
        start=result.steps[-1].interval.start, end=extent.end
    )
    for step in result.steps:
        step.motion_sample_indices = [
            i
            for i, t in enumerate(times)
            if step.interval.start <= t <= step.interval.end
        ]
    candidate_ids = {a.id: _id("action", *lineage, a.id) for a in selected_lower}
    candidate_ids.update({a.id: _id("action", *lineage, a.id) for a in arms.proposals})

    def owner(interval: Interval) -> str:
        return next(
            s.id
            for s in result.steps
            if s.interval.start <= interval.start < s.interval.end
        )

    def link(track: Track, interval: Interval, quality: Quality) -> SemanticMotionLink:
        indices = [
            i for i, t in enumerate(times) if interval.start <= t <= interval.end
        ]
        if (
            not indices
            or times[indices[0]] != interval.start
            or times[indices[-1]] != interval.end
        ):
            raise ValueError("semantic intervals must use exact native sample times")
        return SemanticMotionLink(
            track=track,
            interval=interval,
            motion_sample_indices=indices,
            quality=quality,
        )

    def add_phase(
        action: Action, track: Track, interval: Interval, name: str, quality: Quality
    ) -> None:
        phase = Phase(
            id=_id("phase", action.id, track, name, interval.model_dump()),
            action_id=action.id,
            interval=interval,
            name=name,
            quality=quality,
            motion_links=[link(track, interval, quality)],
        )
        result.phases.append(phase)
        for suffix, time in (("start", interval.start), ("end", interval.end)):
            result.keyframes.append(
                Keyframe(
                    id=_id("keyframe", phase.id, suffix),
                    action_id=action.id,
                    phase_id=phase.id,
                    global_seconds=time,
                    event=f"{name}_{suffix}",
                    track=track,
                    quality=quality,
                    motion_sample_indices=[times.index(time)],
                )
            )

    for proposal in arms.proposals:
        action = Action(
            id=candidate_ids[proposal.id],
            interval=proposal.interval,
            step_id=owner(proposal.interval),
            tracks=[t.track for t in proposal.tracks],
            category=proposal.category,
            role=proposal.role,
            quality=proposal.quality,
            source_candidate_id=proposal.id,
            motion_links=[
                link(t.track, t.interval, t.quality) for t in proposal.tracks
            ],
        )
        result.actions.append(action)
        for arm_phase in proposal.phases:
            add_phase(
                action,
                arm_phase.track,
                arm_phase.interval,
                arm_phase.name,
                arm_phase.quality,
            )
    for proposal_leg in selected_lower:
        category: Literal["placement", "pivot", "kick", "transition"] = (
            "transition"
            if proposal_leg.category == "unknown"
            else proposal_leg.category
        )
        action = Action(
            id=candidate_ids[proposal_leg.id],
            interval=proposal_leg.interval,
            step_id=owner(proposal_leg.interval),
            tracks=[proposal_leg.track],
            category=category,
            quality=proposal_leg.quality,
            source_candidate_id=proposal_leg.id,
            previous_action_id=candidate_ids.get(proposal_leg.previous_action_id or ""),
            motion_links=[
                link(proposal_leg.track, proposal_leg.interval, proposal_leg.quality)
            ],
        )
        result.actions.append(action)
        for leg_phase in proposal_leg.phases:
            add_phase(
                action,
                proposal_leg.track,
                leg_phase.interval,
                leg_phase.name,
                leg_phase.quality,
            )
        if not proposal_leg.phases:
            add_phase(
                action,
                proposal_leg.track,
                proposal_leg.interval,
                "unknown_interval" if proposal_leg.category == "unknown" else category,
                proposal_leg.quality,
            )

    # Partial phase evidence must not make the rest of a candidate disappear.
    for action in result.actions:
        for span in action.motion_links:
            known = sorted(
                (
                    p.interval
                    for p in result.phases
                    if p.action_id == action.id
                    and any(
                        span_link.track == span.track for span_link in p.motion_links
                    )
                ),
                key=lambda i: (i.start, i.end),
            )
            cursor = span.interval.start
            for interval in known:
                if cursor < interval.start:
                    add_phase(
                        action,
                        span.track,
                        Interval(start=cursor, end=interval.start),
                        "unknown_interval",
                        Quality(state="unknown"),
                    )
                cursor = max(cursor, interval.end)
            if cursor < span.interval.end:
                add_phase(
                    action,
                    span.track,
                    Interval(start=cursor, end=span.interval.end),
                    "unknown_interval",
                    Quality(state="unknown"),
                )

    # The complement of each track is retained. No guess that a quiet/unknown
    # interval was a preparation, weight transfer, or correct technique.
    for track in TRACKS:
        spans = sorted(
            (
                span_link.interval
                for a in result.actions
                for span_link in a.motion_links
                if span_link.track == track
            ),
            key=lambda i: (i.start, i.end),
        )
        cursor = extent.start
        gaps: list[Interval] = []
        for covered_interval in spans:
            if cursor < covered_interval.start:
                gaps.append(Interval(start=cursor, end=covered_interval.start))
            cursor = max(cursor, covered_interval.end)
        if cursor < extent.end:
            gaps.append(Interval(start=cursor, end=extent.end))
        for gap in gaps:
            unknown = Quality(state="unknown", source_ids=list(lineage))
            action = Action(
                id=_id("transition", *lineage, track, gap.model_dump()),
                interval=gap,
                step_id=owner(gap),
                tracks=[track],
                category="transition",
                quality=unknown,
                motion_links=[link(track, gap, unknown)],
            )
            result.actions.append(action)
            add_phase(action, track, gap, "unknown_interval", unknown)

    # Only physical candidates and phase boundaries become keyframes. Extension
    # extrema remain extrema; they never become claims of impact or force.
    events: dict[str, EventCandidate] = {e.id: e for e in features.events}
    for proposal in arms.proposals:
        for event in proposal.events:
            if event.id in events and events[event.id] != event:
                raise ValueError("conflicting physical event identity")
            events[event.id] = event
    for action in result.actions:
        for event in sorted(events.values(), key=lambda e: (e.global_seconds, e.id)):
            if not any(
                span_link.track == event.track
                and span_link.interval.start
                <= event.global_seconds
                <= span_link.interval.end
                for span_link in action.motion_links
            ):
                continue
            phase = next(
                (
                    p
                    for p in result.phases
                    if p.action_id == action.id
                    and any(
                        span_link.track == event.track for span_link in p.motion_links
                    )
                    and p.interval.start <= event.global_seconds <= p.interval.end
                ),
                None,
            )
            result.keyframes.append(
                Keyframe(
                    id=_id("keyframe", action.id, event.id),
                    action_id=action.id,
                    phase_id=phase.id if phase else None,
                    global_seconds=event.global_seconds,
                    event=event.kind,
                    track=event.track,
                    quality=event.quality,
                    source_event_ids=[event.id],
                    motion_sample_indices=event.motion_sample_indices,
                )
            )
    for stance in lower.stances:
        if overlaps(stance.interval, extent):
            # A stance is a state, observed here only for the analyzed interval;
            # its complete source interval remains in the immutable evidence.
            interval = Interval(
                start=max(extent.start, stance.interval.start),
                end=min(extent.end, stance.interval.end),
            )
            result.stances.append(
                StanceState(
                    id=_id("stance", *lineage, stance.id),
                    interval=interval,
                    label=stance.label,
                    quality=stance.quality,
                    source_candidate_id=stance.id,
                    motion_sample_indices=[
                        i
                        for i, t in enumerate(times)
                        if interval.start <= t <= interval.end
                    ],
                )
            )
    _relations(result, features, arms, times, config)
    result.actions.sort(key=lambda a: (a.interval.start, a.id))
    result.phases.sort(key=lambda p: (p.interval.start, p.id))
    result.keyframes.sort(key=lambda k: (k.global_seconds, k.id))
    for step in result.steps:
        step.action_ids = [
            a.id for a in result.actions if overlaps(a.interval, step.interval)
        ]
    return Semantics.model_validate(result.model_dump())


def _relations(
    result: Semantics,
    features: FeatureSeries,
    arms: ArmResult,
    times: list[float],
    config: AssemblyConfig,
) -> None:
    assert result.execution is not None
    samples: dict[str, tuple[Relation, set[int]]] = {}

    def collect(index: int, relations: list[Relation]) -> None:
        if (
            not result.execution
            or not result.execution.start <= times[index] <= result.execution.end
        ):
            return
        for relation in relations:
            if relation.value == "not_crossed":
                continue  # Exact negative geometry remains in physical evidence.
            signature = hash_config(relation.model_dump(mode="json"))
            if signature not in samples:
                samples[signature] = (relation, set())
            samples[signature][1].add(index)

    for row in features.trajectory:
        collect(row.motion_sample_index, row.relations)
    for proposal in arms.proposals:
        for span in proposal.tracks:
            for snapshot in span.snapshots:
                collect(snapshot.motion_sample_index, snapshot.relations)
    for signature, (relation, indices) in sorted(samples.items()):
        runs: list[list[int]] = []
        for index in sorted(indices):
            if (
                not runs
                or index != runs[-1][-1] + 1
                or times[index] - times[runs[-1][-1]] > config.max_gap_seconds
            ):
                runs.append([])
            runs[-1].append(index)
        for run in runs:
            interval = (
                Interval(start=times[run[0]], end=times[run[-1]])
                if len(run) > 1
                else None
            )
            result.relations.append(
                SpatialRelation(
                    id=_id("relation", signature, times[run[0]], times[run[-1]]),
                    subject=cast(BodyEntity, relation.subject),
                    object=cast(BodyEntity, relation.object),
                    relation=cast(
                        Literal[
                            "in_front_of",
                            "behind",
                            "above",
                            "below",
                            "left_of",
                            "right_of",
                            "crossed",
                            "unknown",
                        ],
                        relation.value,
                    ),
                    interval=interval,
                    global_seconds=times[run[0]] if interval is None else None,
                    quality=relation.quality,
                    front_entity=cast(BodyEntity, relation.front_entity)
                    if relation.front_quality.state in ("observed", "inferred")
                    else None,
                    front_quality=relation.front_quality,
                    reference_frame=relation.reference_frame,
                )
            )
