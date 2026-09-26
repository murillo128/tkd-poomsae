# Offline semantic assembly

`reconstruction.semantics` assembles persisted coarse segmentation, independent
arm/special actions, lower-body actions/states and physical event candidates into
canonical `Semantics`. It reads immutable inputs and runs independently of vision,
observations, reconstruction, ground estimation and upstream classification.
It neither reads dataset labels nor claims correctness, impact or force.

```python
from reconstruction.semantics import (
    AssemblyConfig,
    publish_semantics,
    load_semantics,
    load_semantic_evidence,
)

handle = publish_semantics(
    store,
    feature_handle,
    segmentation_handle,
    arm_actions_handle,
    lower_body_handle,
    AssemblyConfig(),
)
semantics = load_semantics(handle)
evidence = load_semantic_evidence(handle)
```

`assemble_semantics(features, coarse, arms, lower, config=None)` is the pure
entry point for inspection and synthetic acceptance. Exact reconstruction,
ground, feature and segmentation lineage, proposal intervals and shared native
clocks must agree. Malformed or inconsistent inputs fail closed. Indeterminate
execution produces a null execution with unknown quality and no invented
hierarchy; the persisted input candidates, diagnostics and full clock remain
available in the evidence payload.

Each physical proposal becomes one canonical Action. `Action.step_id` identifies
the step containing its onset; every overlapping `SequenceStep.action_ids` list
references that same Action ID. Steps tile the execution with half-open ownership
at a shared boundary, while dense references include both endpoint samples.
An action crossing a coarse boundary is never clipped or duplicated. Lower-body
proposals intersecting the execution and connected kick/recovery/placement chains
retain their complete intervals; only outer execution/step edges expand to
include them. The changed execution extent has unknown quality, and the original
coarse proposals remain evidence. Disjoint pre/post-roll candidates remain in
the payload. Stances are states, exposed over their intersection with the analyzed
execution, with their full source intervals retained in that payload.

`Action.motion_links` retain independent per-track spans and inclusive dense
sample indices, including different starts/ends inside a coordinated SpecialAction.
Steps, execution and stance states retain dense indices too. Phase links and
quality preserve chamber, extension, retraction, preparation, recovery and
placement evidence. Unclassified parts of actions and uncovered portions of all
six tracks become explicit unknown phases/transition actions. This includes quiet
or missing evidence: a retained interval does not imply that motion was observed.
Original feature trajectories preserve any weight transfer, relocation, local
velocity/acceleration and missing geometry through `motion_features_id`.

Keyframes come from physical candidates and phase boundaries on absolute global
seconds. Source event IDs, native evidence indices, track and quality remain
attached. No uniform sampling or equal event count is imposed. Extension extrema
remain extrema, never actual impact. Source relations retain their body reference
frame and independent crossing/front-order quality; an unknown front order never
becomes a chosen front arm. Single-sample relations use `global_seconds`; sustained
relations use an interval, split at native gaps exceeding `max_gap_seconds`
(default 0.15 s). Negative/unknown geometry also remains in source evidence.

`semantic-assembly-v1` uses stable content-derived IDs. Publication keys include
all four input manifest digests, schema version, algorithm revision and effective
config. Cache hits/reloads invoke no upstream producer. Config changes create a
new immutable semantic artifact. The evidence array retains original proposals,
physical candidates, full native time, dense ground links and input revisions;
large outputs belong in the existing external artifact store, never Git.

```sh
uv run --frozen pytest tests/test_semantic_assembly.py
```

Synthetic acceptance covers compound kick → recovery/placement, overlapping arms,
coordinated special actions with asymmetric spans, crossed-arm unknown transitions,
nonuniform meaningful keyframes, coarse/execution-edge reconciliation, unknown
contact, indeterminate output, lineage rejection, deterministic immutable reruns,
and cache/config invalidation without network or model inference. These checks
establish assembly behavior, not real-data parsing accuracy.
