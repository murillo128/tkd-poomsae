# Offline manual semantic edits

`SemanticEditor` provides application-level `view`, `apply`, `undo`, `reset` and
`revision` functions over an existing MVP-37 automatic `ArtifactHandle`. It reads
only the automatic semantic evidence and edit history; it never runs observation,
calibration, reconstruction, inference or data acquisition.

```python
from contracts.models import KeyframeMove
from tkd_poomsae.semantic_edits import SemanticEditor

editor = SemanticEditor(store, "inspection-1")
view = editor.view(automatic_handle)  # revision 0 initially
edited = editor.apply(
    automatic_handle,
    expected_revision=view.revision,
    operations=[
        KeyframeMove(
            kind="keyframe_move",
            target_id="existing-keyframe-id",
            global_seconds=24.305,
        )
    ],
    source="development-inspector",
    author="reviewer",
    reason="inspect event timing",
)
restored = editor.undo(
    automatic_handle,
    expected_revision=edited.revision,
    source="development-inspector",
    author="reviewer",
    reason="revert inspection",
)
```

Each session is pinned to the automatic ID, complete manifest SHA-256 and parser
revision. A rerun must use a separate session; even a configuration-only artifact
change is rejected in an existing session with `IncompatibleAutomaticBase`.
Existing patches remain intact. There is no implicit migration or overwrite.

Typed operations are `boundary` (`target_id`, `interval`), `keyframe_add`
(`keyframe`), `keyframe_move` (`target_id`, `global_seconds`) and
`keyframe_remove` (`target_id`). The automatic semantic ID targets the execution;
step, action and phase IDs target their own boundaries. Operations in one apply
batch validate atomically, allowing both sides of a shared step edge to change.
Steps must remain ordered and tile the execution; actions may overlap freely,
including across steps. Step membership and onset ownership are rebuilt without
clipping actions. Invalid targets, tracks, phases, hierarchy, times, empty native
sample ranges, or stale revisions fail explicitly without advancing the session.
Children must be adjusted explicitly when their parent's new bounds exclude them.

Dense references keep the existing motion-feature identity and array descriptors.
Changed intervals select native samples within absolute bounds. Independent track
spans keep their asymmetry; links touching an owner's original edge follow that
edge, while interior spans are intersected with the new interval. Keyframes
between native samples reference both bracketing samples; they do not fabricate
new trajectory samples. Unchanged semantic entities and evidence remain intact.

Every accepted command produces a content-addressed `ManualEdits` artifact in
`derived/manual_edits`, with exact automatic identity, monotonic revision, base
revision, active-state ancestry, source/author/reason, and typed operations.
`runs/semantic-edits.sqlite3` indexes sessions and immutable revisions. SQLite
transactions serialize writers and compare the expected revision under the same
lock, including writers in separate processes. Publication precedes the index
commit: a crash can leave an unreferenced immutable artifact, never a head pointing
to an incomplete artifact. Artifacts and the session index should be backed up
together. The index is local to one shared store; no distributed locking is implied.

Undo walks applied state history; reset restores the complete original semantics
without deleting revisions. Undo immediately after reset restores the prior state.
Undo with no active edit fails explicitly. Reloading/resetting/undoing never alters
any byte of the original automatic artifact.

The `EffectiveSemanticView` envelope exposes `origin`, current revision, exact
automatic manifest hash and edit artifact ID. Effective edited `Semantics` also
has `automatic_semantics_id`, `manual_edits_id` and `semantic_edits` provenance.
Manual keyframes and modified boundaries have unknown quality, and changed
keyframes do not claim automatic event evidence. The complete automatic quality
and evidence remain available in the original artifact. An annotation is not
validation ground truth: `validation_ground_truth` is always false in this API.

The CLI takes a JSON serialization of the automatic `ArtifactKey`, so it opens a
verified, already-persisted artifact through the normal store:

```sh
tkd-poomsae semantic-edits view inspection-1 --automatic-key automatic-key.json
tkd-poomsae semantic-edits apply inspection-1 --automatic-key automatic-key.json \
  --expected-revision 0 --operations operations.json \
  --source development-inspector --author reviewer --reason 'inspect timing'
tkd-poomsae semantic-edits undo inspection-1 --automatic-key automatic-key.json \
  --expected-revision 1 --source development-inspector --author reviewer --reason revert
tkd-poomsae semantic-edits reset inspection-1 --automatic-key automatic-key.json \
  --expected-revision 2 --source development-inspector --author reviewer --reason reset
```

For example, `operations.json` is an array:

```json
[{"kind":"keyframe_move","target_id":"existing-keyframe-id","global_seconds":24.305}]
```

The CLI uses `TKD_DATA_ROOT`, emits the schema-valid effective envelope, and returns
a nonzero status for rejected edits. It adds no service or UI endpoint.

Run `uv run --frozen pytest tests/test_semantic_edits.py`. Synthetic tests disable
networking, vision imports and upstream parser calls during edit operations.
They establish edit behavior, not real-data parser accuracy.
