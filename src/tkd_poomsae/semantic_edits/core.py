"""Pure, annotation-only semantic editing; never invokes a producer."""

from bisect import bisect_left, bisect_right

from contracts.models import (
    Action,
    BoundaryEdit,
    Interval,
    KeyframeAdd,
    KeyframeMove,
    KeyframeRemove,
    Phase,
    Quality,
    SemanticEditOperation,
    Semantics,
)


class EditError(ValueError):
    """Invalid target, interval, hierarchy, or edit operation."""


class StaleRevision(EditError):
    """The caller's optimistic revision does not match the session head."""


class IncompatibleAutomaticBase(EditError):
    """A session is pinned to another exact automatic parser artifact."""


def sample_indices(times: list[float], start: float, end: float) -> list[int]:
    if not times or start < times[0] or end > times[-1] or end < start:
        raise EditError("time outside the automatic dense clock")
    indices = list(range(bisect_left(times, start), bisect_right(times, end)))
    if not indices:
        raise EditError("edited interval must contain a native motion sample")
    return indices


def _boundary_links(
    item: Action | Phase, interval: Interval, times: list[float]
) -> None:
    old = item.interval
    for link in item.motion_links:
        start = (
            interval.start
            if link.interval.start == old.start
            else max(interval.start, link.interval.start)
        )
        end = (
            interval.end
            if link.interval.end == old.end
            else min(interval.end, link.interval.end)
        )
        link.interval = Interval(start=start, end=end)
        link.motion_sample_indices = sample_indices(times, start, end)
        link.quality = Quality(state="unknown")
    item.interval = interval
    item.quality = Quality(state="unknown")


def apply_operations(
    original: Semantics, times: list[float], operations: list[SemanticEditOperation]
) -> Semantics:
    """Validate batches atomically, permitting coordinated adjacent edge changes.

    Dense references select native samples within absolute edited bounds. Independent
    links retain asymmetric spans; only owner-edge links follow an adjusted edge.
    Children are never silently clipped to make an invalid hierarchy fit.
    """
    view = original.model_copy(deep=True)
    for operation in operations:
        entities = {
            item.id: item
            for group in (view.steps, view.actions, view.phases, view.keyframes)
            for item in group
        }
        if isinstance(operation, BoundaryEdit):
            interval = operation.interval.model_copy(deep=True)
            indices = sample_indices(times, interval.start, interval.end)
            if operation.target_id == original.id:
                view.execution = interval
                view.motion_sample_indices = indices
                view.quality = Quality(state="unknown")
            else:
                target = entities.get(operation.target_id)
                if target is None or not hasattr(target, "interval"):
                    raise EditError("unknown boundary target")
                if isinstance(target, (Action, Phase)):
                    _boundary_links(target, interval, times)
                else:
                    target.interval = interval
                    target.motion_sample_indices = indices
        elif isinstance(operation, KeyframeAdd):
            frame = operation.keyframe.model_copy(deep=True)
            all_ids = {original.id} | {
                item.id
                for group in (
                    view.steps,
                    view.stances,
                    view.actions,
                    view.phases,
                    view.keyframes,
                    view.relations,
                )
                for item in group
            }
            if not frame.id or frame.id in all_ids:
                raise EditError("duplicate or empty keyframe ID")
            frame.quality = Quality(state="unknown")
            frame.source_event_ids = []
            frame.motion_sample_indices = _frame_indices(times, frame.global_seconds)
            view.keyframes.append(frame)
        elif isinstance(operation, (KeyframeMove, KeyframeRemove)):
            target_frame = next(
                (f for f in view.keyframes if f.id == operation.target_id), None
            )
            if target_frame is None:
                raise EditError("deleted or unknown keyframe target")
            if isinstance(operation, KeyframeRemove):
                view.keyframes.remove(target_frame)
            else:
                target_frame.global_seconds = operation.global_seconds
                target_frame.motion_sample_indices = _frame_indices(
                    times, operation.global_seconds
                )
                target_frame.quality = Quality(state="unknown")
                target_frame.source_event_ids = []
    if view.execution is not None:
        edges = [view.execution.start]
        for step in view.steps:
            if step.interval.start != edges[-1]:
                raise EditError("steps must be ordered and tile execution")
            edges.append(step.interval.end)
            step.action_ids = [
                action.id
                for action in view.actions
                if action.interval.start < step.interval.end
                and step.interval.start < action.interval.end
            ]
        if edges[-1] != view.execution.end:
            raise EditError("steps must cover execution")
        for action in view.actions:
            owner = next(
                (
                    step
                    for step in view.steps
                    if step.interval.start <= action.interval.start < step.interval.end
                ),
                None,
            )
            if owner is None:
                raise EditError("action onset has no step owner")
            action.step_id = owner.id
    # Preserve original order of unchanged entities, including automatic keyframes.
    return Semantics.model_validate(view.model_dump())


def _frame_indices(times: list[float], seconds: float) -> list[int]:
    if seconds < times[0] or seconds > times[-1]:
        raise EditError("keyframe outside automatic dense clock")
    i = bisect_left(times, seconds)
    return [i] if times[i] == seconds or i == 0 else [i - 1, i]
