"""Reversible annotation acceptance over offline synthetic parser artifacts."""

from __future__ import annotations

import json
import multiprocessing
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from contracts.models import (
    BoundaryEdit,
    Interval,
    Keyframe,
    KeyframeAdd,
    KeyframeMove,
    KeyframeRemove,
    ManualEdits,
    SemanticEditOperation,
    Semantics,
)
from contracts.schema import SCHEMA_PATH
from reconstruction.arms import publish_arm_actions
from reconstruction.lower_body import publish_lower_body
from reconstruction.segmentation import publish_segmentation
from reconstruction.semantics import AssemblyConfig, publish_semantics
from storage import (
    ArtifactHandle,
    ArtifactKey,
    ArtifactStore,
    StorageRoot,
    hash_config,
    hash_file,
)
from tests.test_arm_actions import persist_motion
from tests.test_segmentation import persist_features
from tests.test_semantic_assembly import compound
from tkd_poomsae.cli import main
from tkd_poomsae.semantic_edits import (
    EditError,
    IncompatibleAutomaticBase,
    SemanticEditor,
    StaleRevision,
)

PROVENANCE = dict(
    source="development-inspector", author="test-author", reason="manual inspection"
)


@pytest.fixture
def persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ArtifactStore, ArtifactHandle, ArtifactHandle]:
    physical, floor, geometry = compound()
    store = ArtifactStore(StorageRoot(tmp_path))
    features = persist_features(store, physical)
    coarse = publish_segmentation(store, features)
    motion = persist_motion(store, physical, geometry)
    arms = publish_arm_actions(store, features, coarse, motion)
    key = ArtifactKey(
        layer="ground",
        inputs={"fixture": hash_config(floor.model_dump(mode="json"))},
        schema_version="1.0.0",
        algorithm_revision="fixture",
        config_digest="0" * 64,
    )
    ground = store.get_or_create(key, lambda: (floor, {}))
    lower = publish_lower_body(store, features, ground, coarse)
    automatic = publish_semantics(store, features, coarse, arms, lower)
    rerun = publish_semantics(
        store, features, coarse, arms, lower, AssemblyConfig(max_gap_seconds=0.01)
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "edit tests must never invoke vision or upstream producers"
        )

    import reconstruction.arms.artifact as arms_publisher
    import reconstruction.lower_body.artifact as lower_publisher
    import reconstruction.segmentation.artifact as coarse_publisher
    import reconstruction.semantics.artifact as semantics_publisher

    monkeypatch.setattr(semantics_publisher, "assemble_semantics", forbidden)
    monkeypatch.setattr(arms_publisher, "parse_arm_actions", forbidden)
    monkeypatch.setattr(lower_publisher, "parse_lower_body", forbidden)
    monkeypatch.setattr(coarse_publisher, "segment_execution", forbidden)
    for module in (
        "cv2",
        "torch",
        "tkd_poomsae.vision.inference",
        "pose.providers.mmpose",
    ):
        monkeypatch.setitem(sys.modules, module, None)
    return store, automatic, rerun


def add_frame(
    automatic: ArtifactHandle, identifier: str = "manual-frame"
) -> KeyframeAdd:
    assert isinstance(automatic.metadata, Semantics)
    action = next(a for a in automatic.metadata.actions if a.category == "kick")
    return KeyframeAdd(
        kind="keyframe_add",
        keyframe=Keyframe(
            id=identifier,
            action_id=action.id,
            global_seconds=(action.interval.start + action.interval.end) / 2,
            event="manually_inspected",
            track=action.tracks[0],
        ),
    )


def test_apply_reload_undo_preserve_original_hash_and_dense_references(
    persisted: tuple[ArtifactStore, ArtifactHandle, ArtifactHandle],
) -> None:
    store, automatic, _ = persisted
    original_hashes = {
        str(p): hash_file(p) for p in automatic.path.rglob("*") if p.is_file()
    }
    editor = SemanticEditor(store, "inspection")
    first = editor.apply(automatic, 0, [add_frame(automatic)], **PROVENANCE)
    assert first.revision == 1 and first.origin == "manual"
    assert first.validation_ground_truth is False
    assert first.semantics.automatic_semantics_id == automatic.metadata.id
    assert first.semantics.manual_edits_id == first.manual_edits_id
    assert first.semantics.provenance.producer == "semantic_edits"
    assert isinstance(automatic.metadata, Semantics)
    for field in (
        "actions",
        "phases",
        "relations",
        "stances",
        "steps",
        "arrays",
        "motion_features_id",
    ):
        assert getattr(first.semantics, field) == getattr(automatic.metadata, field)
    frame = next(f for f in first.semantics.keyframes if f.id == "manual-frame")
    assert frame.quality.state == "unknown" and frame.quality.score is None
    assert frame.motion_sample_indices and not frame.source_event_ids
    jsonschema.validate(
        first.semantics.model_dump(mode="json"), json.loads(SCHEMA_PATH.read_text())
    )
    assert SemanticEditor(store, "inspection").view(automatic) == first
    record = editor.revision(1)
    assert (record.source, record.author, record.reason) == tuple(PROVENANCE.values())
    assert record.automatic_manifest_sha256 == hash_file(
        automatic.path / "manifest.json"
    )
    jsonschema.validate(
        record.model_dump(mode="json"), json.loads(SCHEMA_PATH.read_text())
    )
    restored = editor.undo(automatic, 1, **PROVENANCE)
    assert restored.origin == "automatic" and restored.revision == 2
    assert restored.semantics == automatic.metadata
    assert {p: hash_file(Path(p)) for p in original_hashes} == original_hashes
    assert editor.revision(1) == record


def test_move_remove_sequential_undo_reset_and_undo_reset(
    persisted: tuple[ArtifactStore, ArtifactHandle, ArtifactHandle],
) -> None:
    store, automatic, _ = persisted
    editor = SemanticEditor(store, "sequence")
    op = add_frame(automatic)
    editor.apply(automatic, 0, [op], **PROVENANCE)
    move = KeyframeMove(
        kind="keyframe_move",
        target_id=op.keyframe.id,
        global_seconds=op.keyframe.global_seconds + 0.005,
    )
    moved = editor.apply(automatic, 1, [move], **PROVENANCE)
    assert (
        next(
            f for f in moved.semantics.keyframes if f.id == op.keyframe.id
        ).global_seconds
        == move.global_seconds
    )
    removed = editor.apply(
        automatic,
        2,
        [KeyframeRemove(kind="keyframe_remove", target_id=op.keyframe.id)],
        **PROVENANCE,
    )
    assert op.keyframe.id not in {f.id for f in removed.semantics.keyframes}
    assert (
        editor.undo(automatic, 3, **PROVENANCE).semantics.keyframes
        == moved.semantics.keyframes
    )
    reset = editor.reset(automatic, 4, **PROVENANCE)
    assert reset.semantics == automatic.metadata
    assert (
        editor.undo(automatic, 5, **PROVENANCE).semantics.keyframes
        == moved.semantics.keyframes
    )
    editor.undo(automatic, 6, **PROVENANCE)
    assert editor.undo(automatic, 7, **PROVENANCE).semantics == automatic.metadata
    with pytest.raises(EditError, match="no active"):
        editor.undo(automatic, 8, **PROVENANCE)


@pytest.mark.parametrize(
    "operation",
    [
        lambda a: KeyframeMove(
            kind="keyframe_move", target_id=add_frame(a).keyframe.id, global_seconds=1e6
        ),
        lambda a: KeyframeRemove(kind="keyframe_remove", target_id="missing"),
        lambda a: BoundaryEdit(
            kind="boundary", target_id="missing", interval=Interval(start=24, end=25)
        ),
        lambda a: BoundaryEdit(
            kind="boundary",
            target_id=a.metadata.id,
            interval=Interval(start=-1, end=1e6),
        ),
        lambda a: KeyframeAdd(
            kind="keyframe_add",
            keyframe=add_frame(a).keyframe.model_copy(update={"track": "head"}),
        ),
        lambda a: KeyframeAdd(
            kind="keyframe_add",
            keyframe=add_frame(a).keyframe.model_copy(update={"action_id": "missing"}),
        ),
        lambda a: KeyframeAdd(
            kind="keyframe_add",
            keyframe=add_frame(a).keyframe.model_copy(update={"phase_id": "missing"}),
        ),
    ],
)
def test_invalid_batch_is_atomic(
    persisted: tuple[ArtifactStore, ArtifactHandle, ArtifactHandle],
    operation: Callable[[ArtifactHandle], SemanticEditOperation],
) -> None:
    store, automatic, _ = persisted
    editor = SemanticEditor(store, "invalid")
    with pytest.raises(ValueError):
        editor.apply(
            automatic, 0, [add_frame(automatic), operation(automatic)], **PROVENANCE
        )
    assert editor.view(automatic).revision == 0
    assert editor.view(automatic).semantics == automatic.metadata


def test_deleted_ids_duplicate_ids_stale_and_incompatible_base(
    persisted: tuple[ArtifactStore, ArtifactHandle, ArtifactHandle],
) -> None:
    store, automatic, rerun = persisted
    editor = SemanticEditor(store, "pinned")
    editor.apply(automatic, 0, [add_frame(automatic)], **PROVENANCE)
    with pytest.raises(StaleRevision):
        editor.reset(automatic, 0, **PROVENANCE)
    with pytest.raises(EditError, match="duplicate"):
        editor.apply(automatic, 1, [add_frame(automatic)], **PROVENANCE)
    editor.apply(
        automatic,
        1,
        [KeyframeRemove(kind="keyframe_remove", target_id="manual-frame")],
        **PROVENANCE,
    )
    with pytest.raises(EditError, match="deleted or unknown"):
        editor.apply(
            automatic,
            2,
            [
                KeyframeMove(
                    kind="keyframe_move", target_id="manual-frame", global_seconds=24
                )
            ],
            **PROVENANCE,
        )
    commands: list[Callable[[], object]] = [
        lambda: editor.view(rerun),
        lambda: editor.reset(rerun, 2, **PROVENANCE),
        lambda: editor.apply(rerun, 2, [add_frame(rerun)], **PROVENANCE),
    ]
    for command in commands:
        with pytest.raises(IncompatibleAutomaticBase):
            command()
    assert editor.view(automatic).revision == 2
    assert SemanticEditor(store, "rerun-session").view(rerun).revision == 0


def _writer(
    root: Path, automatic: ArtifactHandle, barrier: Any, queue: Any, identifier: str
) -> None:
    editor = SemanticEditor(ArtifactStore(StorageRoot(root)), "concurrent")
    barrier.wait(timeout=10)
    try:
        result = editor.apply(
            automatic, 0, [add_frame(automatic, identifier)], **PROVENANCE
        )
        queue.put(("success", result.revision))
    except StaleRevision:
        queue.put(("stale", 0))


def test_concurrent_process_writers_have_one_winner(
    persisted: tuple[ArtifactStore, ArtifactHandle, ArtifactHandle],
) -> None:
    store, automatic, _ = persisted
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    queue = context.Queue()
    processes = [
        context.Process(
            target=_writer, args=(store.root.path, automatic, barrier, queue, str(i))
        )
        for i in range(2)
    ]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=20) for _ in processes]
        assert sorted(results) == [("stale", 0), ("success", 1)]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    view = SemanticEditor(store, "concurrent").view(automatic)
    assert view.revision == 1
    assert len([f for f in view.semantics.keyframes if f.id in {"0", "1"}]) == 1


def test_cli_apply_view_undo_and_reset(
    persisted: tuple[ArtifactStore, ArtifactHandle, ArtifactHandle],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, automatic, _ = persisted
    # Reconstruct the publisher key from its evidence, as callers do when keeping
    # analysis handles in their project/session metadata.
    from reconstruction.semantics import load_semantic_evidence
    from reconstruction.semantics.core import REVISION

    payload = load_semantic_evidence(automatic)
    key = dict(
        layer="semantics",
        inputs=payload["input_revisions"],
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        config_digest=automatic.metadata.provenance.config_digest,
    )
    key_path = tmp_path / "automatic-key.json"
    key_path.write_text(json.dumps(key))
    operations = tmp_path / "operations.json"
    operations.write_text(json.dumps([add_frame(automatic).model_dump(mode="json")]))
    monkeypatch.setenv("TKD_DATA_ROOT", str(store.root.path))
    base = ["tkd-poomsae", "semantic-edits"]
    for command, expected in (("apply", 0), ("view", 1), ("undo", 1), ("reset", 2)):
        args = base + [command, "cli-session", "--automatic-key", str(key_path)]
        if command != "view":
            args += [
                "--expected-revision",
                str(expected),
                "--source",
                "cli",
                "--author",
                "tester",
                "--reason",
                "inspection",
            ]
        if command == "apply":
            args += ["--operations", str(operations)]
        monkeypatch.setattr(sys, "argv", args)
        assert main() == 0
        output = json.loads(capsys.readouterr().out)
        assert output["revision"] == (expected if command == "view" else expected + 1)
    monkeypatch.setattr(
        sys,
        "argv",
        base
        + [
            "undo",
            "cli-session",
            "--automatic-key",
            str(key_path),
            "--expected-revision",
            "0",
            "--source",
            "cli",
            "--author",
            "tester",
            "--reason",
            "stale",
        ],
    )
    assert main() == 1
    assert "stale" in capsys.readouterr().err


def test_provenance_required_and_operation_schema(
    persisted: tuple[ArtifactStore, ArtifactHandle, ArtifactHandle],
) -> None:
    store, automatic, _ = persisted
    editor = SemanticEditor(store, "provenance")
    for field in PROVENANCE:
        with pytest.raises(ValidationError):
            editor.apply(
                automatic, 0, [add_frame(automatic)], **(PROVENANCE | {field: ""})
            )
    with pytest.raises(ValidationError):
        KeyframeMove(kind="keyframe_move", target_id="x", global_seconds=float("nan"))
    with pytest.raises(ValidationError):
        Interval(start=2, end=1)
    assert editor.view(automatic).revision == 0
    assert (
        ManualEdits.model_validate(
            {
                "kind": "manual_edits",
                "id": "legacy",
                "schema_version": "1.0.0",
                "provenance": automatic.metadata.provenance,
                "automatic_semantics_id": automatic.metadata.id,
                "edits": [],
            }
        ).revision
        is None
    )


def test_execution_action_and_phase_boundary_adjustments_keep_overlaps(
    persisted: tuple[ArtifactStore, ArtifactHandle, ArtifactHandle],
) -> None:
    store, automatic, _ = persisted
    original = automatic.metadata
    assert isinstance(original, Semantics) and original.execution is not None
    arms = [a for a in original.actions if a.category == "arm"]
    assert len(arms) == 2
    arm = arms[0]
    phase = next(p for p in original.phases if p.action_id == arm.id)
    execution = Interval(
        start=original.execution.start - 0.01, end=original.execution.end
    )
    new_action = Interval(start=arm.interval.start, end=arm.interval.end + 0.01)
    new_phase = Interval(start=phase.interval.start, end=phase.interval.end + 0.005)
    operations: list[SemanticEditOperation] = [
        BoundaryEdit(kind="boundary", target_id=original.id, interval=execution),
        BoundaryEdit(
            kind="boundary",
            target_id=original.steps[0].id,
            interval=Interval(
                start=execution.start, end=original.steps[0].interval.end
            ),
        ),
        BoundaryEdit(kind="boundary", target_id=arm.id, interval=new_action),
        BoundaryEdit(kind="boundary", target_id=phase.id, interval=new_phase),
    ]
    editor = SemanticEditor(store, "bounds")
    view = editor.apply(automatic, 0, operations, **PROVENANCE)
    changed = next(a for a in view.semantics.actions if a.id == arm.id)
    untouched = next(a for a in view.semantics.actions if a.id == arms[1].id)
    assert changed.interval == new_action and changed.quality.state == "unknown"
    assert untouched == arms[1]
    assert changed.interval.end > untouched.interval.start
    assert changed.motion_links[0].interval == new_action
    assert (
        next(p for p in view.semantics.phases if p.id == phase.id).interval == new_phase
    )
    assert view.semantics.motion_features_id == original.motion_features_id
    assert view.semantics.arrays == original.arrays
    assert editor.reset(automatic, 1, **PROVENANCE).semantics == original


def test_shared_step_boundary_batch_and_hierarchy_rejection() -> None:
    from reconstruction.arms import parse_arm_actions
    from reconstruction.lower_body import parse_lower_body
    from reconstruction.segmentation import segment_execution
    from reconstruction.semantics import assemble_semantics
    from tests.test_semantic_assembly import split
    from tkd_poomsae.semantic_edits.core import apply_operations

    physical, floor, geometry = compound()
    coarse = split(segment_execution(physical))
    result = assemble_semantics(
        physical,
        coarse,
        parse_arm_actions(physical, coarse, geometry),
        parse_lower_body(physical, floor, coarse),
    )
    times = [row.global_seconds for row in physical.trajectory[::6]]
    first, second = result.steps
    left = BoundaryEdit(
        kind="boundary",
        target_id=first.id,
        interval=Interval(start=first.interval.start, end=24.35),
    )
    right = BoundaryEdit(
        kind="boundary",
        target_id=second.id,
        interval=Interval(start=24.35, end=second.interval.end),
    )
    with pytest.raises(EditError, match="tile"):
        apply_operations(result, times, [left])
    view = apply_operations(result, times, [left, right])
    assert view.actions == result.actions
    assert view.steps[0].interval.end == view.steps[1].interval.start == 24.35
    kick = next(a for a in view.actions if a.category == "kick")
    assert all(kick.id in step.action_ids for step in view.steps)
    with pytest.raises(ValueError, match="phase outside|keyframe outside"):
        apply_operations(
            result,
            times,
            [
                BoundaryEdit(
                    kind="boundary",
                    target_id=kick.id,
                    interval=Interval(start=24.1, end=24.4),
                )
            ],
        )


def test_mutated_automatic_handle_is_rejected(
    persisted: tuple[ArtifactStore, ArtifactHandle, ArtifactHandle],
) -> None:
    store, automatic, _ = persisted
    assert isinstance(automatic.metadata, Semantics)
    changed = automatic.metadata.model_copy(deep=True)
    changed.actions[0].category = "special"
    mutated = ArtifactHandle(automatic.path, changed, automatic.files)
    with pytest.raises(IncompatibleAutomaticBase, match="differs from stored"):
        SemanticEditor(store, "mutated").view(mutated)


def test_new_persisted_parser_revision_rejects_existing_session(
    persisted: tuple[ArtifactStore, ArtifactHandle, ArtifactHandle],
) -> None:
    from reconstruction.semantics import load_semantic_evidence

    store, automatic, _ = persisted
    editor = SemanticEditor(store, "parser-version")
    editor.apply(automatic, 0, [add_frame(automatic)], **PROVENANCE)
    payload = load_semantic_evidence(automatic)
    revision = "semantic-assembly-v3"
    key = ArtifactKey(
        layer="semantics",
        inputs=payload["input_revisions"],
        schema_version="1.0.0",
        algorithm_revision=revision,
        config_digest=automatic.metadata.provenance.config_digest,
    )
    metadata = automatic.metadata.model_copy(deep=True)
    assert isinstance(metadata, Semantics)
    metadata.id = f"semantics:{key.digest}"
    metadata.provenance.model = revision
    arrays = {a.id: automatic.read_array(a.id) for a in metadata.arrays}
    future = store.get_or_create(key, lambda: (metadata, arrays))
    with pytest.raises(IncompatibleAutomaticBase):
        editor.apply(future, 1, [add_frame(future)], **PROVENANCE)
    assert editor.view(automatic).revision == 1
    assert editor.revision(1).automatic_parser_revision == "semantic-assembly-v2"
