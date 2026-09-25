"""Consensus constant-offset alignment of timestamped native-time cues."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from contracts.models import (
    Interval,
    Provenance,
    Quality,
    Synchronization,
    SyncOffset,
    SyncPairEstimate,
)
from media import IngestManifest
from storage import ArtifactHandle, ArtifactKey, ArtifactStore

from .cues import CueSet, CueStream


class TimelineFailure(ValueError):
    """Available cues cannot establish two reliable overlapping views."""


@dataclass(frozen=True)
class SyncSource:
    source_id: str
    start_seconds: float
    end_seconds: float
    cues: CueSet

    def __post_init__(self) -> None:
        if not self.source_id or self.end_seconds <= self.start_seconds:
            raise ValueError("source needs an ID and ordered native interval")


def sources_from_manifest(
    manifest: IngestManifest,
    cues: Mapping[str, CueSet],
) -> dict[str, SyncSource]:
    """Bind extracted cues to the exact source bytes in an ingest manifest."""
    recordings = {item.source_id: item for item in manifest}
    if len(recordings) != len(manifest) or set(recordings) != set(cues):
        raise ValueError("cue sources must match ingest manifest exactly")
    result: dict[str, SyncSource] = {}
    for source_id, recording in recordings.items():
        if cues[source_id].source_sha256 != recording.sha256:
            raise ValueError(f"stale cues for {source_id}")
        if len(recording.frames) < 2:
            raise ValueError(f"source {source_id} has no usable interval")
        result[source_id] = SyncSource(
            source_id,
            recording.frames[0].seconds,
            recording.frames[-1].seconds,
            cues[source_id],
        )
    return result


@dataclass(frozen=True)
class PairEstimate:
    first: str
    second: str
    shift_seconds: float | None  # offset(second) - offset(first)
    score: float
    peak_separation: float
    overlap_seconds: float
    window_scores: tuple[float, ...]
    cue_kinds: tuple[str, ...]
    reliable: bool
    diagnostics: tuple[str, ...]


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 8 or np.std(a) < 0.03 or np.std(b) < 0.03:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _series(stream: CueStream) -> tuple[np.ndarray, np.ndarray]:
    times = np.array([(s.start_seconds + s.end_seconds) / 2 for s in stream.samples])
    values = np.array([s.normalized for s in stream.samples])
    return times, values


def _pair(a: SyncSource, b: SyncSource) -> PairEstimate:
    candidates: list[tuple[float, float, float, str, tuple[float, ...]]] = []
    diagnostics: list[str] = []
    for kind in ("audio", "motion"):
        left: CueStream = getattr(a.cues, kind)
        right: CueStream = getattr(b.cues, kind)
        if not left.available or not right.available:
            diagnostics.append(f"{kind}_unavailable")
            continue
        ta, va = _series(left)
        tb, vb = _series(right)
        if len(ta) < 8 or len(tb) < 8:
            diagnostics.append(f"{kind}_too_short")
            continue
        step = max(left.sample_interval_seconds, right.sample_interval_seconds)
        # Candidate shifts align right native time = left native time - shift.
        lower = a.start_seconds - b.end_seconds
        upper = a.end_seconds - b.start_seconds
        shifts = np.arange(lower, upper + step / 2, step)
        scores: list[tuple[float, float, float, tuple[float, ...]]] = []
        for shift in shifts:
            start = max(a.start_seconds, b.start_seconds + shift, ta[0], tb[0] + shift)
            end = min(a.end_seconds, b.end_seconds + shift, ta[-1], tb[-1] + shift)
            if end - start < max(0.4, 8 * step):
                continue
            grid = np.arange(start, end, step)
            if len(grid) < 8:
                continue
            x = np.interp(grid, ta, va)
            y = np.interp(grid - shift, tb, vb)
            # Require agreement in more than one useful part of the overlap.
            windows = tuple(
                _correlation(x[part], y[part])
                for part in np.array_split(np.arange(len(grid)), 3)
            )
            useful = [v for v in windows if v > 0.2]
            if len(useful) < 2:
                continue
            score = _correlation(x, y) * min(1.0, len(useful) / 3)
            score *= min(1.0, (end - start) / 2)
            scores.append((score, float(shift), end - start, windows))
        if not scores:
            diagnostics.append(f"{kind}_no_useful_overlap")
            continue
        scores.sort(reverse=True)
        best = scores[0]
        alternatives = [row for row in scores[1:] if abs(row[1] - best[1]) > 2 * step]
        separation = best[0] - (alternatives[0][0] if alternatives else 0.0)
        candidates.append((best[0], best[1], separation, kind, best[3]))
    if not candidates:
        return PairEstimate(
            a.source_id,
            b.source_id,
            None,
            0,
            0,
            0,
            (),
            (),
            False,
            tuple(diagnostics or ["no_shared_cues"]),
        )
    candidates.sort(reverse=True)
    best_score, best_shift, separation, kind, windows = candidates[0]
    # An independent cue may break a periodic peak tie, but must agree in time.
    agreeing = [
        c
        for c in candidates
        if abs(c[1] - best_shift)
        <= 2
        * max(
            getattr(a.cues, c[3]).sample_interval_seconds,
            getattr(b.cues, c[3]).sample_interval_seconds,
        )
    ]
    if len(candidates) > 1 and len(agreeing) != len(candidates):
        diagnostics.append("cue_disagreement")
    if separation < 0.08:
        diagnostics.append("ambiguous_peak")
    if best_score < 0.55:
        diagnostics.append("weak_correlation")
    if sum(w > 0.2 for w in windows) < 2:
        diagnostics.append("window_disagreement")
    reliable = not any(
        d in diagnostics
        for d in (
            "cue_disagreement",
            "ambiguous_peak",
            "weak_correlation",
            "window_disagreement",
        )
    )
    overlap = min(a.end_seconds, b.end_seconds + best_shift) - max(
        a.start_seconds, b.start_seconds + best_shift
    )
    return PairEstimate(
        a.source_id,
        b.source_id,
        best_shift,
        best_score,
        separation,
        overlap,
        tuple(float(w) for w in windows),
        tuple(c[3] for c in agreeing),
        reliable,
        tuple(diagnostics),
    )


def solve_offsets(
    sources: Mapping[str, SyncSource],
    *,
    reference: str | None = None,
    overrides: Mapping[str, Mapping[str, object]] | None = None,
    artifact_id: str = "synchronization",
    config_digest: str = "0" * 64,
) -> Synchronization:
    """Solve a connected reliable graph, excluding inconsistent cameras when safe.

    Offset signs obey global = native + offset. Resolution is bounded by the
    slower cue sampling interval; estimates and diagnostics remain inspectable.
    """
    if len(sources) < 2 or any(key != item.source_id for key, item in sources.items()):
        raise ValueError("at least two uniquely identified sources are required")
    ids = sorted(sources)
    pairs = [_pair(sources[i], sources[j]) for i, j in combinations(ids, 2)]
    usable = [p for p in pairs if p.reliable and p.shift_seconds is not None]
    revisions = overrides or {}
    if set(revisions) - set(ids):
        raise ValueError("manual offset names an unknown source")
    manual_values: dict[str, float] = {}
    for name, revision in revisions.items():
        value = revision.get("offset_seconds")
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("manual offset must be finite")
        if any(
            not isinstance(revision.get(field), str) or not str(revision[field]).strip()
            for field in ("author", "source", "reason")
        ):
            raise ValueError("manual offset needs author, source, and reason")
        manual_values[name] = float(value)
    # Prefer the largest connected, cycle-consistent subset. A contradictory
    # camera is removed only if two other cameras still establish a timeline.
    ranked: list[tuple[int, float, tuple[str, ...], dict[str, float]]] = []
    from itertools import combinations as choose

    for size in range(len(ids), 1, -1):
        for subset in choose(ids, size):
            edges = [p for p in usable if p.first in subset and p.second in subset]
            if len(edges) < size - 1:
                continue
            anchor = subset[0]
            columns = {name: i for i, name in enumerate(subset[1:])}
            matrix = np.zeros((len(edges), size - 1))
            target = np.zeros(len(edges))
            for row, edge in enumerate(edges):
                if edge.first != anchor:
                    matrix[row, columns[edge.first]] = -1
                if edge.second != anchor:
                    matrix[row, columns[edge.second]] = 1
                assert edge.shift_seconds is not None
                target[row] = edge.shift_seconds
            solution, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
            if rank != size - 1:
                continue
            tolerance = max(
                0.05,
                2
                * max(
                    max(
                        sources[n].cues.audio.sample_interval_seconds,
                        sources[n].cues.motion.sample_interval_seconds,
                    )
                    for n in subset
                ),
            )
            residual = float(np.max(np.abs(matrix @ solution - target)))
            if residual > tolerance:
                continue
            offsets = {
                anchor: 0.0,
                **{name: float(solution[index]) for name, index in columns.items()},
            }
            intervals = [
                (
                    sources[n].start_seconds + offsets[n],
                    sources[n].end_seconds + offsets[n],
                )
                for n in subset
            ]
            overlap = min(end for _, end in intervals) - max(
                start for start, _ in intervals
            )
            if overlap <= 0:
                continue
            ranked.append((size, sum(p.score for p in edges), subset, offsets))
        if ranked:
            break
    manual_timeline = not ranked
    if ranked:
        _, _, retained, offsets = max(ranked, key=lambda row: (row[0], row[1]))
        ref = reference or retained[0]
        if ref not in retained:
            raise TimelineFailure("requested timing reference is not retained")
        origin = offsets[ref]
        offsets = {name: value - origin for name, value in offsets.items()}
    else:
        # A manual offset can recover an ambiguous pair, but contributes no
        # automatic confidence. One source must still define global zero.
        candidate_reference = reference or next(
            (n for n in ids if n not in revisions), None
        )
        candidate_reference = candidate_reference or next(
            (n for n in ids if manual_values.get(n) == 0), None
        )
        if candidate_reference is None or candidate_reference not in sources:
            raise TimelineFailure("manual timeline needs a zero-offset reference")
        ref = candidate_reference
        retained = tuple(sorted({ref, *revisions}))
        offsets = {}
        if len(retained) < 2:
            raise TimelineFailure("manual timeline needs two attributed views")
    if ref in manual_values and manual_values[ref] != 0:
        raise ValueError("timing reference offset must remain zero")
    rows: list[SyncOffset] = []
    for name in ids:
        source = sources[name]
        auto = offsets.get(name)
        manual_revision = revisions.get(name)
        manual_value = manual_values.get(name)
        correction = (
            manual_value - auto
            if manual_value is not None and auto is not None
            else None
        )
        keep = name in retained or manual_revision is not None
        effective = manual_value if manual_value is not None else auto
        if manual_timeline and name == ref and effective is None:
            effective = 0.0
        if keep:
            assert effective is not None
            global_interval = Interval(
                start=source.start_seconds + effective,
                end=source.end_seconds + effective,
            )
        else:
            global_interval = None
        quality = (
            Quality(
                state="observed",
                score=min(
                    1.0,
                    max(
                        (p.score for p in usable if name in (p.first, p.second)),
                        default=0.0,
                    ),
                ),
            )
            if name in offsets
            else Quality(state="unknown")
        )
        rows.append(
            SyncOffset(
                source_id=name,
                automatic_seconds=auto,
                manual_correction_seconds=correction,
                manual_seconds=manual_value,
                manual_author=str(manual_revision["author"])
                if manual_revision
                else None,
                manual_source=str(manual_revision["source"])
                if manual_revision
                else None,
                manual_reason=str(manual_revision["reason"])
                if manual_revision
                else None,
                timing_reference=name == ref,
                retained=keep,
                exclusion_reason=None
                if keep
                else "weak, ambiguous, or inconsistent cues",
                source_interval=Interval(
                    start=source.start_seconds, end=source.end_seconds
                ),
                global_interval=global_interval,
                quality=quality,
            )
        )
    kept = [r for r in rows if r.retained]
    common_start = max(r.global_interval.start for r in kept if r.global_interval)
    common_end = min(r.global_interval.end for r in kept if r.global_interval)
    if len(kept) < 2 or common_end <= common_start:
        raise TimelineFailure("manual offsets leave fewer than two overlapping views")
    return Synchronization(
        id=artifact_id,
        kind="synchronization",
        schema_version="1.0.0",
        provenance=Provenance(producer="sync.solver.v1", config_digest=config_digest),
        offsets=rows,
        reference_source_id=ref,
        common_interval=Interval(start=common_start, end=common_end),
        pair_estimates=[SyncPairEstimate(**asdict(pair)) for pair in pairs],
        diagnostics=(
            ["manual_timeline_without_reliable_pair"] if manual_timeline else []
        )
        + [f"excluded:{row.source_id}" for row in rows if not row.retained],
    )


def publish_offsets(
    store: ArtifactStore,
    key: ArtifactKey,
    sources: Mapping[str, SyncSource],
    *,
    reference: str | None = None,
    overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> ArtifactHandle:
    """Publish one immutable, validated synchronization artifact."""
    if key.layer != "synchronization":
        raise ValueError("synchronization artifact key required")
    return store.get_or_create(
        key,
        lambda: (
            solve_offsets(
                sources,
                reference=reference,
                overrides=overrides,
                artifact_id=key.digest,
                config_digest=key.config_digest,
            ),
            {},
        ),
    )
