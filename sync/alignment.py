"""Read-only global-time queries over immutable native observation windows."""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections.abc import Sequence
from statistics import median
from typing import Literal

import numpy as np
from pydantic import Field

from contracts.models import (
    Alignment,
    DenseArray,
    Landmark,
    Landmark2D,
    Observation,
    Provenance,
    Quality,
    RegionalGeometry2D,
    StrictModel,
    Synchronization,
    SyncOffset,
)
from pose.observation_run import load_window
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

REVISION = "native-time-alignment-v1"
ARRAY_ID = "alignment_queries_json"
Channel = Literal["landmarks", "wholebody_landmarks", "refined_landmarks"]
CHANNELS: tuple[Channel, ...] = (
    "landmarks",
    "wholebody_landmarks",
    "refined_landmarks",
)


class JoinConfig(StrictModel):
    max_bracket_factor: float = Field(default=2.0, gt=0)
    local_radius: int = Field(default=5, ge=1)
    max_bracket_seconds: float | None = Field(default=None, gt=0)
    # A grid is requested explicitly via query_many, never inferred from fps.
    exact_tolerance_seconds: float = Field(default=1e-9, ge=0, le=1e-6)


class TimeSpan(StrictModel):
    # Closed intervals permit isolated exact observations.
    start: float
    end: float


class CameraCoverage(StrictModel):
    source_id: str
    camera_id: str | None
    retained: bool
    usable: list[TimeSpan]
    landmarks: dict[str, list[TimeSpan]]
    reason: str | None = None


class CameraQuery(StrictModel):
    source_id: str
    camera_id: str | None
    source_seconds: float | None
    offset_seconds: float | None
    synchronization_quality: Quality
    state: Literal["exact", "bracket", "unknown"]
    # Unmodified evidence: frame identities, scores, ROIs, all geometry/providers.
    endpoints: list[Observation]
    weights: list[float]
    local_median_seconds: float | None = None
    max_bracket_seconds: float | None = None
    landmarks: list[Landmark2D]
    wholebody_landmarks: list[Landmark2D]
    refined_landmarks: list[Landmark2D]
    missing_masks: dict[str, list[bool]]
    regional_geometry: list[RegionalGeometry2D]
    regional_quality: dict[str, Quality]
    reasons: list[str]


class TimeQuery(StrictModel):
    global_seconds: float
    cameras: list[CameraQuery]
    available_sources: list[str]
    landmark_sources: dict[str, list[str]]


def _part(name: str) -> str:
    if any(
        token in name for token in ("thumb_", "index_", "middle_", "ring_", "pinky_")
    ):
        return "left_hand" if name.startswith("left_") else "right_hand"
    if any(token in name for token in ("heel", "forefoot", "foot_outer")):
        return "left_foot" if name.startswith("left_") else "right_foot"
    if name in {
        "head",
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
    }:
        return "head"
    return "body"


def _usable(observation: Observation, point: Landmark2D) -> bool:
    subject = observation.subject_selection
    if subject is not None and subject.state != "selected":
        return False
    regions: dict[str, bool] = {
        region.part: region.usable for region in observation.region_quality
    }
    return (
        point.xy_px is not None
        and point.quality.state == "observed"
        and (point.raw_visibility is None or point.raw_visibility > 0)
        and regions.get(_part(point.name), True)
    )


def _same_track(a: Observation, b: Observation) -> bool:
    x, y = a.subject_selection, b.subject_selection
    return (
        x is not None
        and y is not None
        and x.state == y.state == "selected"
        and bool(x.track_id)
        and x.track_id == y.track_id
    )


def _mix(
    a: tuple[float, float], b: tuple[float, float], w: float
) -> tuple[float, float]:
    return (a[0] * (1 - w) + b[0] * w, a[1] * (1 - w) + b[1] * w)


class ObservationJoin:
    """Snapshot inputs, index PTS once, then query without inference or media I/O."""

    def __init__(
        self,
        observations: Sequence[Observation],
        synchronization: Synchronization,
        config: JoinConfig | None = None,
    ) -> None:
        self.config = (config or JoinConfig()).model_copy(deep=True)
        self.synchronization = synchronization.model_copy(deep=True)
        self._streams: dict[str, list[Observation]] = {}
        self._times: dict[str, list[float]] = {}
        self._names: dict[str, dict[Channel, list[Landmark]]] = {}
        offsets = {item.source_id for item in self.synchronization.offsets}
        for item in observations:
            obs = Observation.model_validate(item.model_dump(mode="json"))
            if obs.frame.pts is None:
                raise ValueError("alignment requires native presentation timestamps")
            if obs.frame.source_id not in offsets:
                raise ValueError("observation source is absent from synchronization")
            self._streams.setdefault(obs.frame.source_id, []).append(obs)
        if not self._streams:
            raise ValueError("alignment needs native observations")
        for source, stream in self._streams.items():
            stream.sort(key=lambda obs: obs.frame.source_seconds)
            times = [obs.frame.source_seconds for obs in stream]
            if len(set(times)) != len(times):
                raise ValueError("duplicate native presentation timestamps")
            if len({obs.frame.camera_id for obs in stream}) != 1:
                raise ValueError("source maps to multiple cameras")
            if len({obs.id for obs in stream}) != len(stream):
                raise ValueError("duplicate native observation identities")
            self._times[source] = times
            self._names[source] = {
                channel: list(
                    dict.fromkeys(
                        point.name for obs in stream for point in getattr(obs, channel)
                    )
                )
                for channel in CHANNELS
            }
        self._limits: dict[str, list[tuple[float, float]]] = {}
        for source, times in self._times.items():
            deltas = [b - a for a, b in zip(times, times[1:])]
            limits = []
            for i, delta in enumerate(deltas):
                # Exclude the tested gap so a dropout cannot inflate its own bound.
                neighbors = [
                    deltas[j]
                    for j in range(
                        max(0, i - self.config.local_radius),
                        min(len(deltas), i + self.config.local_radius + 1),
                    )
                    if j != i
                ]
                interval = median(neighbors) if neighbors else delta
                bound = self.config.max_bracket_factor * interval
                if self.config.max_bracket_seconds is not None:
                    bound = min(bound, self.config.max_bracket_seconds)
                limits.append((interval, bound))
            self._limits[source] = limits

    @classmethod
    def from_windows(
        cls,
        windows: Sequence[ArtifactHandle],
        synchronization: Synchronization,
        config: JoinConfig | None = None,
    ) -> ObservationJoin:
        return cls(
            [obs for window in windows for obs in load_window(window)],
            synchronization,
            config,
        )

    def _channels(
        self,
        endpoints: list[Observation],
        weight: float,
        valid: bool,
        names: dict[Channel, list[Landmark]],
    ) -> dict[str, list[Landmark2D]]:
        result: dict[str, list[Landmark2D]] = {}
        for channel in CHANNELS:
            groups = [{p.name: p for p in getattr(obs, channel)} for obs in endpoints]
            canonical = [{p.name: p for p in obs.landmarks} for obs in endpoints]
            points: list[Landmark2D] = []
            for name in names[channel]:
                items = [group.get(name) for group in groups]
                usable = valid and all(
                    point is not None
                    and _usable(obs, point)
                    # Raw provider geometry is evidence; it cannot undo canonical masks.
                    and name in accepted
                    and _usable(obs, accepted[name])
                    for obs, point, accepted in zip(endpoints, items, canonical)
                )
                if not usable:
                    points.append(
                        Landmark2D(
                            name=name, xy_px=None, quality=Quality(state="unknown")
                        )
                    )
                elif len(items) == 1:
                    assert items[0] is not None
                    points.append(items[0].model_copy(deep=True))
                else:
                    a, b = items
                    assert a is not None and b is not None
                    assert a.xy_px is not None and b.xy_px is not None
                    scores = [p.quality.score for p in (a, b)]
                    uncertainty = [p.quality.uncertainty for p in (a, b)]
                    points.append(
                        Landmark2D(
                            name=name,
                            xy_px=_mix(a.xy_px, b.xy_px, weight),
                            quality=Quality(
                                state="interpolated",
                                score=min(scores)
                                if all(s is not None for s in scores)
                                else None,
                                uncertainty=max(uncertainty)
                                if all(s is not None for s in uncertainty)
                                else None,
                                source_ids=list(
                                    dict.fromkeys(
                                        a.quality.source_ids + b.quality.source_ids
                                    )
                                ),
                            ),
                        )
                    )
            result[channel] = points
        return result

    def _geometry(
        self,
        endpoints: list[Observation],
        points: list[Landmark2D],
        weight: float,
        valid: bool,
    ) -> list[RegionalGeometry2D]:
        if not endpoints:
            return []
        usable = {p.name for p in points if p.xy_px is not None}
        result = []
        groups = [{g.part: g for g in obs.regional_geometry} for obs in endpoints]
        for part in dict.fromkeys(part for group in groups for part in group):
            items = [group.get(part) for group in groups]
            a = next(g for g in items if g is not None)
            supported = valid and all(
                g is not None and set(g.supporting_landmarks) <= usable for g in items
            )
            if supported and len(items) == 1:
                result.append(a.model_copy(deep=True))
            elif supported and all(
                g is not None
                and g.provider == a.provider
                and g.supporting_landmarks == a.supporting_landmarks
                and g.orientation_state == "available"
                for g in items
            ):
                b = items[1]
                assert b is not None
                assert a.axis_start_px is not None and b.axis_start_px is not None
                assert a.axis_end_px is not None and b.axis_end_px is not None
                start = _mix(a.axis_start_px, b.axis_start_px, weight)
                end = _mix(a.axis_end_px, b.axis_end_px, weight)
                degenerate = math.dist(start, end) <= 1e-9
                result.append(
                    RegionalGeometry2D(
                        part=a.part,
                        provider=a.provider,
                        availability="complete",
                        supporting_landmarks=a.supporting_landmarks,
                        axis_start_px=start,
                        axis_end_px=end,
                        orientation_state="degenerate" if degenerate else "available",
                        orientation_rad=None
                        if degenerate
                        else math.atan2(end[1] - start[1], end[0] - start[0]),
                    )
                )
            else:
                result.append(
                    RegionalGeometry2D(
                        part=a.part,
                        provider=a.provider,
                        availability="missing",
                        orientation_state="unavailable",
                        supporting_landmarks=a.supporting_landmarks,
                    )
                )
        return result

    def _camera(self, offset: SyncOffset, global_seconds: float) -> CameraQuery:
        stream = self._streams.get(offset.source_id, [])
        times = self._times.get(offset.source_id, [])
        camera = stream[0].frame.camera_id if stream else None
        effective = offset.effective_seconds if offset.retained else None
        source_time = global_seconds - effective if effective is not None else None
        endpoints: list[Observation] = []
        weights: list[float] = []
        reasons: list[str] = []
        interval = bound = None
        state: Literal["exact", "bracket", "unknown"] = "unknown"
        valid = False
        if not offset.retained:
            reasons.append(offset.exclusion_reason or "excluded_source")
        elif not stream:
            reasons.append("no_observations")
        else:
            assert source_time is not None
            i = bisect_left(times, source_time)
            nearest = min(
                (j for j in (i - 1, i) if 0 <= j < len(times)),
                key=lambda j: abs(times[j] - source_time),
            )
            if abs(times[nearest] - source_time) <= self.config.exact_tolerance_seconds:
                endpoints, weights, state, valid = (
                    [stream[nearest]],
                    [1.0],
                    "exact",
                    True,
                )
            elif i == 0 or i == len(times):
                reasons.append("outside_native_coverage")
            else:
                endpoints = [stream[i - 1], stream[i]]
                weight = (source_time - times[i - 1]) / (times[i] - times[i - 1])
                weights = [1 - weight, weight]
                state = "bracket"
                interval, bound = self._limits[offset.source_id][i - 1]
                if times[i] - times[i - 1] > bound + 1e-12:
                    reasons.append("long_gap")
                if not _same_track(*endpoints):
                    reasons.append("identity_unavailable_or_changed")
                valid = not reasons
            # Respect the synchronized recording interval, even with a broader run.
            if offset.source_interval is not None and not (
                offset.source_interval.start
                <= source_time
                <= offset.source_interval.end
            ):
                valid = False
                reasons.append("outside_synchronized_coverage")
        channels = self._channels(
            endpoints,
            weights[-1] if weights else 0,
            valid,
            self._names.get(offset.source_id, {channel: [] for channel in CHANNELS}),
        )
        if endpoints and not any(p.xy_px is not None for p in channels["landmarks"]):
            reasons.append("no_usable_landmarks")
        geometry = self._geometry(
            endpoints, channels["landmarks"], weights[-1] if weights else 0, valid
        )
        return CameraQuery(
            source_id=offset.source_id,
            camera_id=camera,
            source_seconds=source_time,
            offset_seconds=effective,
            synchronization_quality=offset.quality.model_copy(deep=True),
            state=state,
            endpoints=[obs.model_copy(deep=True) for obs in endpoints],
            weights=weights,
            local_median_seconds=interval,
            max_bracket_seconds=bound,
            landmarks=channels["landmarks"],
            wholebody_landmarks=channels["wholebody_landmarks"],
            refined_landmarks=channels["refined_landmarks"],
            missing_masks={
                name: [p.xy_px is None for p in values]
                for name, values in channels.items()
            },
            regional_geometry=geometry,
            regional_quality={
                region.part: Quality(
                    state="unknown"
                    if region.availability == "missing"
                    else ("observed" if state == "exact" else "interpolated"),
                    source_ids=[obs.id for obs in endpoints],
                )
                for region in geometry
            },
            reasons=reasons,
        )

    def query(self, global_seconds: float) -> TimeQuery:
        if not math.isfinite(global_seconds):
            raise ValueError("query time must be finite")
        cameras = [
            self._camera(offset, global_seconds)
            for offset in self.synchronization.offsets
        ]
        sources: dict[str, list[str]] = {}
        for camera in cameras:
            for point in camera.landmarks:
                sources.setdefault(point.name, [])
                if point.xy_px is not None:
                    sources[point.name].append(camera.source_id)
        return TimeQuery(
            global_seconds=global_seconds,
            cameras=cameras,
            available_sources=[
                camera.source_id
                for camera in cameras
                if any(p.xy_px is not None for p in camera.landmarks)
            ],
            landmark_sources=sources,
        )

    def query_many(self, global_times: Sequence[float]) -> list[TimeQuery]:
        return [self.query(time) for time in global_times]

    def coverage(self) -> list[CameraCoverage]:
        result = []
        for offset in self.synchronization.offsets:
            stream = self._streams.get(offset.source_id, [])
            spans: dict[str, list[TimeSpan]] = {}
            if offset.retained:
                shift = offset.effective_seconds
                # Queries determine exactly the same masks/bounds used in coverage.
                for i, obs in enumerate(stream):
                    t = obs.frame.source_seconds + shift
                    candidates = [(t, t)]
                    if i + 1 < len(stream):
                        candidates.append(
                            (t, stream[i + 1].frame.source_seconds + shift)
                        )
                    for start, end in candidates:
                        if offset.source_interval is not None:
                            start = max(start, offset.source_interval.start + shift)
                            end = min(end, offset.source_interval.end + shift)
                        if start > end:
                            continue
                        query = self._camera(offset, (start + end) / 2)
                        for point in query.landmarks:
                            if point.xy_px is not None:
                                spans.setdefault(point.name, []).append(
                                    TimeSpan(start=start, end=end)
                                )

            def merge(values: list[TimeSpan]) -> list[TimeSpan]:
                merged: list[TimeSpan] = []
                for span in sorted(values, key=lambda value: value.start):
                    if merged and span.start <= merged[-1].end:
                        merged[-1].end = max(merged[-1].end, span.end)
                    else:
                        merged.append(span.model_copy())
                return merged

            result.append(
                CameraCoverage(
                    source_id=offset.source_id,
                    camera_id=stream[0].frame.camera_id if stream else None,
                    retained=offset.retained,
                    usable=merge(
                        [span for values in spans.values() for span in values]
                    ),
                    landmarks={name: merge(values) for name, values in spans.items()},
                    reason=offset.exclusion_reason
                    if not offset.retained
                    else ("no_usable_observations" if not spans else None),
                )
            )
        return result


def publish_alignment(
    store: ArtifactStore,
    windows: Sequence[ArtifactHandle],
    synchronization: ArtifactHandle,
    global_times: Sequence[float],
    config: JoinConfig | None = None,
) -> ArtifactHandle:
    """Cache derived queries only, bound to exact native/sync/config/grid identities."""
    if not isinstance(synchronization.metadata, Synchronization):
        raise ValueError("alignment requires a synchronization artifact")
    if not global_times:
        raise ValueError("alignment requires an explicit nonempty sampling grid")
    config = config or JoinConfig()
    settings = {
        "join": config.model_dump(mode="json"),
        "global_times": list(global_times),
    }
    digests = sorted(hash_file(window.path / "manifest.json") for window in windows)
    key = ArtifactKey(
        layer="alignment",
        inputs={f"window_{i}": digest for i, digest in enumerate(digests)},
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=hash_config(settings),
        sync_revision=hash_file(synchronization.path / "manifest.json"),
    )
    sync_metadata = synchronization.metadata

    def produce() -> tuple[Alignment, dict[str, np.ndarray]]:
        join = ObservationJoin.from_windows(windows, sync_metadata, config)
        payload = {
            "version": 1,
            "settings": settings,
            "coverage": [item.model_dump(mode="json") for item in join.coverage()],
            "queries": [
                item.model_dump(mode="json") for item in join.query_many(global_times)
            ],
        }
        array = np.frombuffer(
            json.dumps(
                payload, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode(),
            dtype=np.uint8,
        )
        return Alignment(
            kind="alignment",
            id=f"alignment:{key.digest}",
            schema_version="1.0.0",
            provenance=Provenance(
                producer="sync.alignment", config_digest=key.config_digest
            ),
            synchronization_id=sync_metadata.id,
            observation_digests=digests,
            query_count=len(global_times),
            arrays=[
                DenseArray(
                    id=ARRAY_ID, dtype="uint8", shape=[len(array)], axes=["json_byte"]
                )
            ],
        ), {ARRAY_ID: array}

    return store.get_or_create(key, produce)


def load_alignment(
    handle: ArtifactHandle,
) -> tuple[list[TimeQuery], list[CameraCoverage]]:
    if not isinstance(handle.metadata, Alignment):
        raise ValueError("not an alignment artifact")
    payload = json.loads(handle.read_array(ARRAY_ID).tobytes())
    if payload.get("version") != 1:
        raise ValueError("unsupported alignment payload version")
    queries = [TimeQuery.model_validate(item) for item in payload["queries"]]
    coverage = [CameraCoverage.model_validate(item) for item in payload["coverage"]]
    if len(queries) != handle.metadata.query_count:
        raise ValueError("alignment query count disagrees with metadata")
    return queries, coverage
