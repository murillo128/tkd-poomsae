"""End-to-end parser regression on constructed motion, no accuracy claims."""

from __future__ import annotations

import importlib
import socket
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from contracts.models import (
    Ground,
    Keyframe,
    KeyframeAdd,
    Quality,
    Reconstruction,
    Semantics,
)
from reconstruction.arms import parse_arm_actions, publish_arm_actions
from reconstruction.detailed import derive_sample
from reconstruction.features import (
    TRACKS,
    FeatureConfig,
    FeatureSeries,
    derive_features,
    load_feature_evidence,
    publish_features,
)
from reconstruction.lower_body import parse_lower_body, publish_lower_body
from reconstruction.segmentation import publish_segmentation, segment_execution
from reconstruction.semantics import (
    assemble_semantics,
    load_semantics,
    publish_semantics,
)
from storage import ArtifactHandle, ArtifactStore, StorageRoot, hash_file
from tests.fixtures.parser_motion import (
    Case,
    persist_inputs,
    pivot_motion,
    reconstructed,
)
from tkd_poomsae.semantic_edits import (
    IncompatibleAutomaticBase,
    SemanticEditor,
    StaleRevision,
)


def parse(source: Reconstruction, floor: Ground) -> tuple[Semantics, FeatureSeries]:
    geometry = [
        derive_sample(s, reconstruction_id=source.id, representation="regularized")
        for s in source.samples
    ]
    features = derive_features(source, floor, geometry=geometry)
    coarse = segment_execution(features)
    arms = parse_arm_actions(features, coarse, geometry)
    legs = parse_lower_body(features, floor, coarse)
    return assemble_semantics(features, coarse, arms, legs), features


def invariants(result: Semantics, features: FeatureSeries) -> None:
    assert result.execution is not None and result.steps and result.actions
    times = [r.global_seconds for r in features.trajectory[::6]]
    assert times[0] == 23 and times[-1] == 30
    assert result.execution.start > times[0] and result.execution.end < times[-1]
    expected = {
        i
        for i, t in enumerate(times)
        if result.execution.start <= t <= result.execution.end
    }
    assert set(result.motion_sample_indices) == expected
    assert result.steps[0].interval.start == result.execution.start
    assert result.steps[-1].interval.end == result.execution.end
    for left, right in zip(result.steps, result.steps[1:]):
        assert left.interval.end == right.interval.start
    for track in TRACKS:
        links = [
            link
            for a in result.actions
            for link in a.motion_links
            if link.track == track
        ]
        assert {i for link in links for i in link.motion_sample_indices} == expected
        for link in links:
            indices = link.motion_sample_indices
            assert indices == list(range(indices[0], indices[-1] + 1))
            assert link.interval.start == times[indices[0]]
            assert link.interval.end == times[indices[-1]]
    events = {e.id: e for e in features.events}
    for frame in result.keyframes:
        assert frame.motion_sample_indices
        for identifier in frame.source_event_ids:
            assert frame.event == events[identifier].kind
            assert frame.global_seconds == events[identifier].global_seconds
    expected_events = {
        e.id
        for e in features.events
        if any(
            link.track == e.track
            and link.interval.start <= e.global_seconds <= link.interval.end
            for a in result.actions
            for link in a.motion_links
        )
    }
    assert {
        identifier for k in result.keyframes for identifier in k.source_event_ids
    } == expected_events
    assert result.phases and result.keyframes
    assert not ({s.id for s in result.stances} & {a.id for a in result.actions})
    assert any(a.category == "transition" for a in result.actions)
    assert all(
        a.quality.state == "unknown"
        for a in result.actions
        if a.category == "transition"
    )
    assert Semantics.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("case", ["compound", "stationary_arms", "special", "crossing"])
def test_constructed_positive_scenarios(case: Case) -> None:
    source, floor = reconstructed(case)
    before = (source.model_dump_json(), floor.model_dump_json())
    result, features = parse(source, floor)
    invariants(result, features)
    assert before == (source.model_dump_json(), floor.model_dump_json())
    assert any(
        a.quality.state == "inferred" and a.category != "transition"
        for a in result.actions
    )
    if case == "compound":
        kick = next(a for a in result.actions if a.category == "kick")
        placement = next(a for a in result.actions if a.category == "placement")
        assert placement.previous_action_id == kick.id
        assert placement.interval.start == kick.interval.end
        assert {p.name for p in result.phases if p.action_id == kick.id} >= {
            "chamber",
            "extension",
            "retraction",
        }
        assert any(
            p.name == "recovery" and p.action_id == placement.id for p in result.phases
        )
        assert any(s.interval.start >= placement.interval.end for s in result.stances)
        arms = [a for a in result.actions if a.category == "arm"]
        assert len({(a.interval.start, a.interval.end) for a in arms}) > 1
        assert any(
            a.interval.start < kick.interval.end
            and kick.interval.start < a.interval.end
            for a in arms
        )
        assert any(
            k.event == "extension_maximum" and k.action_id == kick.id
            for k in result.keyframes
        )
        assert (
            len(
                {
                    round(b.global_seconds - a.global_seconds, 6)
                    for a, b in zip(result.keyframes, result.keyframes[1:])
                }
            )
            > 2
        )
    elif case == "stationary_arms":
        assert any(a.category == "arm" for a in result.actions)
        assert not any(
            a.category in ("kick", "placement", "pivot") for a in result.actions
        )
        assert result.stances
        assert any(
            len(
                [
                    a
                    for a in result.actions
                    if a.id in step.action_ids and a.category == "arm"
                ]
            )
            > 1
            for step in result.steps
        )
        assert all(a.role == "unknown" for a in result.actions if a.category == "arm")
        # Multiple extension candidates used to create duplicate semantic IDs.
        phases = [
            (p.action_id, p.name, p.interval.start, p.interval.end)
            for p in result.phases
        ]
        assert len(phases) == len(set(phases))
        for kind in ("extension_end", "extension_maximum"):
            assert any(k.event == kind for k in result.keyframes)
    elif case == "special":
        special = next(a for a in result.actions if a.category == "special")
        assert special.tracks == ["left_arm", "right_arm"] and special.role == "special"
        assert len(special.motion_links) == 2
        assert special.motion_links[0].interval != special.motion_links[1].interval
    else:
        crossings = [k for k in result.keyframes if k.event == "arm_crossing"]
        assert {k.track for k in crossings} == {"left_arm", "right_arm"}
        relations = [r for r in result.relations if r.relation == "crossed"]
        assert relations and any(r.front_entity == "left_forearm" for r in relations)
        assert any(p.name == "preparation" for p in result.phases)
        assert any(k.event == "preparation_candidate" for k in result.keyframes)
        assert all(a.role == "unknown" for a in result.actions if a.category == "arm")


def test_supported_pivot_uses_detected_ground_event() -> None:
    source, floor = pivot_motion()
    result, features = parse(source, floor)
    invariants(result, features)
    pivot = next(a for a in result.actions if a.category == "pivot")
    assert pivot.tracks == ["left_leg"] and pivot.quality.state == "inferred"
    frames = [k for k in result.keyframes if k.action_id == pivot.id]
    assert {k.event for k in frames} >= {"pivot_start", "pivot_end"}
    assert pivot.interval == floor.pivots[0].interval


@pytest.mark.parametrize(
    "defect",
    [
        "occlusion",
        "gap",
        "unsupported",
        "unsupported_pivot",
        "noise",
        "ambiguous_crossing",
    ],
)
def test_uncertain_evidence_does_not_become_confident_labels(defect: str) -> None:
    source, floor = (
        pivot_motion()
        if defect == "unsupported_pivot"
        else reconstructed(
            "crossing"
            if defect == "ambiguous_crossing"
            else "noise"
            if defect == "noise"
            else "compound"
        )
    )
    if defect == "occlusion":
        for sample in source.samples:
            if 24 <= sample.global_seconds <= 25:
                for p in sample.landmarks:
                    if p.name in (
                        "left_knee",
                        "left_ankle",
                        "left_wrist",
                        "left_elbow",
                    ):
                        p.xyz_world, p.quality = None, Quality(state="unknown")
    elif defect == "gap":
        source.samples = [
            s for s in source.samples if not 24.2 < s.global_seconds < 24.8
        ]
        floor.samples = [s for s in floor.samples if not 24.2 < s.global_seconds < 24.8]
    elif defect in ("unsupported", "unsupported_pivot"):
        for ground_sample in floor.samples:
            ground_sample.left.quality = ground_sample.right.quality = Quality(
                state="unknown"
            )
            ground_sample.left.state = ground_sample.right.state = "unknown"
            ground_sample.left.region = ground_sample.right.region = None
            ground_sample.support = "unknown"
    elif defect == "ambiguous_crossing":
        for sample in source.samples:
            for point in sample.landmarks:
                if point.name in ("left_wrist", "left_elbow"):
                    assert point.xyz_world is not None
                    x, _, z = point.xyz_world
                    point.xyz_world = (x, 0, z)
    result, features = parse(source, floor)
    assert not any(a.category in ("kick", "placement", "pivot") for a in result.actions)
    if defect == "noise":
        assert not features.events and result.execution is None and not result.actions
    if defect == "occlusion":
        assert all(
            r.extension.value is None and r.linear.velocity is None
            for r in features.trajectory
            if r.track == "left_leg" and 24 <= r.global_seconds <= 25
        )
    if defect == "gap":
        assert not any(
            a.category == "arm" and a.interval.start < 24.2 and a.interval.end > 24.8
            for a in result.actions
        )
    if defect == "ambiguous_crossing":
        relations = [r for r in result.relations if r.relation == "crossed"]
        assert relations and all(
            r.front_entity is None and r.front_quality.state == "unknown"
            for r in relations
        )
        assert all(a.role == "unknown" for a in result.actions if a.category == "arm")


def hashes(handle: ArtifactHandle) -> dict[str, str]:
    return {str(p): hash_file(p) for p in handle.path.rglob("*") if p.is_file()}


def test_persisted_full_rerun_edits_and_zero_upstream_or_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(StorageRoot(tmp_path))
    source, floor = reconstructed("compound")
    motion, ground = persist_inputs(store, source, floor)
    original_inputs = {**hashes(motion), **hashes(ground)}
    calls: Counter[str] = Counter()

    def forbid(name: str) -> Callable[..., Any]:
        def forbidden(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            raise AssertionError(f"parser-only operation called {name}")

        return forbidden

    for module, name in (
        ("calibration.target", "estimate_calibration"),
        ("calibration.natural", "estimate_scene"),
        ("reconstruction.triangulation.artifact", "publish_triangulation"),
        ("reconstruction.articulated.artifact", "publish_articulated_fit"),
        ("reconstruction.temporal.artifact", "publish_temporal_motion"),
    ):
        monkeypatch.setattr(importlib.import_module(module), name, forbid(module))
    from pose.providers.mmpose.adapter import MMPoseAdapter

    monkeypatch.setattr(MMPoseAdapter, "infer", forbid("pose"))
    # Heavy runtime modules must not even be imported by the parser.
    import sys

    for module in ("torch", "tkd_poomsae.vision.inference"):
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setattr(socket.socket, "connect", forbid("network"))
    monkeypatch.setattr(socket.socket, "connect_ex", forbid("network"))
    monkeypatch.setattr(socket, "create_connection", forbid("network"))
    monkeypatch.setattr(socket, "getaddrinfo", forbid("network"))

    def count(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            return function(*args, **kwargs)

        return counted

    for module, name in (
        ("reconstruction.features.artifact", "derive_features"),
        ("reconstruction.segmentation.artifact", "segment_execution"),
        ("reconstruction.arms.artifact", "parse_arm_actions"),
        ("reconstruction.lower_body.artifact", "parse_lower_body"),
        ("reconstruction.semantics.artifact", "assemble_semantics"),
    ):
        adapter = importlib.import_module(module)
        monkeypatch.setattr(adapter, name, count(name, getattr(adapter, name)))

    def pipeline(
        config: FeatureConfig | None = None,
    ) -> tuple[ArtifactHandle, ArtifactHandle]:
        features = publish_features(store, motion, ground, config)
        coarse = publish_segmentation(store, features)
        arms = publish_arm_actions(store, features, coarse, motion)
        legs = publish_lower_body(store, features, ground, coarse)
        return publish_semantics(store, features, coarse, arms, legs), features

    automatic, features = pipeline()
    result = load_semantics(automatic)
    invariants(result, load_feature_evidence(features))
    assert any(a.category == "kick" for a in result.actions)
    assert all(
        calls[name] == 1
        for name in (
            "derive_features",
            "segment_execution",
            "parse_arm_actions",
            "parse_lower_body",
            "assemble_semantics",
        )
    )
    original_automatic = hashes(automatic)
    changed, _ = pipeline(FeatureConfig(min_excursion=0.03))
    assert changed.path != automatic.path
    assert any(a.category == "kick" for a in load_semantics(changed).actions)
    assert all(
        calls[name] == 2
        for name in (
            "derive_features",
            "segment_execution",
            "parse_arm_actions",
            "parse_lower_body",
            "assemble_semantics",
        )
    )
    snapshot = calls.copy()
    assert pipeline()[0].path == automatic.path
    assert calls == snapshot
    editor = SemanticEditor(store, "constructed-inspection")
    kick = next(a for a in result.actions if a.category == "kick")
    operation = KeyframeAdd(
        kind="keyframe_add",
        keyframe=Keyframe(
            id="manual-inspection",
            action_id=kick.id,
            track="left_leg",
            global_seconds=(kick.interval.start + kick.interval.end) / 2,
            event="manual-inspection",
        ),
    )
    provenance = dict(
        source="test", author="regression", reason="constructed inspection"
    )
    edited = editor.apply(automatic, 0, [operation], **provenance)
    assert edited.origin == "manual" and edited.validation_ground_truth is False
    assert SemanticEditor(store, "constructed-inspection").view(automatic) == edited
    with pytest.raises(StaleRevision):
        editor.undo(automatic, 0, **provenance)
    with pytest.raises(IncompatibleAutomaticBase):
        editor.view(changed)
    with pytest.raises(IncompatibleAutomaticBase):
        editor.apply(changed, 1, [operation], **provenance)
    assert editor.undo(automatic, 1, **provenance).semantics == result
    assert hashes(automatic) == original_automatic
    assert {**hashes(motion), **hashes(ground)} == original_inputs
    assert calls == snapshot
    assert calls["pose"] == calls["network"] == 0
    assert all(
        calls[module] == 0
        for module in (
            "calibration.target",
            "calibration.natural",
            "reconstruction.triangulation.artifact",
            "reconstruction.articulated.artifact",
            "reconstruction.temporal.artifact",
        )
    )
