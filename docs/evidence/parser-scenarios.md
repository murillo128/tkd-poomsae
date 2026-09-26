# Constructed parser regression

`tests/test_parser_scenarios.py` runs actual feature extraction, automatic
execution/SequenceStep proposals, arm and lower-body rules, and final assembly.
The fixture recipe in `tests/fixtures/parser_motion.py` constructs reconstructed
landmarks and physical contacts on a 50 Hz clock from 23 to 30 seconds. It never
supplies semantic actions, feature derivatives, event candidates or coarse cuts.
The pivot case also runs the physical footprint and pivot detectors. These are
software/algorithm regressions on constructed inputs, not real-world recognition
accuracy or Mendeley validation. No local-data inspection is part of this suite.

| Case | Expected synthetic semantics |
| --- | --- |
| compound | Left kick with chamber/extension/retraction, linked recovery/placement and subsequent stance; overlapping arms with different boundaries |
| stationary_arms | Several arm actions inside a coarse movement unit, no placement/kick/pivot, stance as state; ambiguous attack/defense roles remain unknown |
| special | One coordinated SpecialAction spanning both arms with different native intervals |
| crossing | Preparation/chamber evidence, crossing keyframes and left-forearm-front relation retained |
| supported_pivot | Left-foot physical supported rotation becomes pivot with ground-linked start/end events |
| occlusion | Unknown left-arm/leg evidence remains unknown; no confident lower-body label |
| gap | Missing native samples break evidence; no kick or arm action bridging the gap |
| unsupported | Unknown support never licenses a kick/placement/pivot |
| unsupported_pivot | Even a detected physical pivot does not license a semantic pivot with unknown support |
| noise | Deterministic 0.2 mm wrist jitter produces no execution or events |
| ambiguous_crossing | Crossing remains observable while forearm depth order and arm roles remain unknown |
| persisted rerun/edit | Every parser stage actually executes twice after configuration invalidation; a cache hit executes none; manual edit/reload/undo and stale revision/base rejection preserve all original bytes |

Positive cases require inferred broad semantics, so blanket unknown output cannot
pass. Shared invariants check pre/post-roll, step tiling, independent action
boundaries, retained transitions, event/phase keyframes, absolute times and dense
sample coverage on every track. Persistence checks hash every input and automatic
artifact file. The constructed temporal payload is a test input boundary, not a
claim that vision or reconstruction produced the fixture. It stores derived
geometry and landmarks only; production publishers generate all parser artifacts.

The persisted case instruments feature extraction, segmentation, arm rules,
lower-body rules and assembly. It rejects/counts pose inference, target/natural
calibration, triangulation, articulated fitting and temporal reconstruction calls;
heavy runtime imports and socket connection/DNS operations are disabled. Two
uncached full parser passes and all subsequent edits/cache reads must record zero
upstream and network calls. Both passes share the same persisted motion/ground
handles. No dataset or model assets are copied or downloaded.

## Revisions and reproduction

Fixture recipe: `constructed-parser-input-v1`. Parser revisions exercised:
`motion-features-v3`, `execution-segmentation-v1`, `arm-actions-v1`,
`lower-body-rules-v2`, `semantic-assembly-v2`. Manual-edit persistence also checks
that a future parser revision cannot be used with an existing pinned session.

Assembly v2 fixes the duplicate-phase failure exposed by real feature events:
`extension_end` and `extension_maximum` can propose the same named interval on
one track. One canonical phase and its boundary keyframes are retained, while
both physical source events remain separate keyframes. Its revision changes
artifact identity rather than reusing cached v1 semantics.

Run from the repository root with the locked Python 3.11 environment:

```sh
uv sync --frozen --group dev
uv run --frozen pytest tests/test_parser_scenarios.py -q
uv run --frozen pytest
uv run --frozen ruff check src tests contracts storage media sync calibration pose reconstruction
uv run --frozen mypy
```

The scenario suite contains 12 deterministic cases, with no capability skips.
The PR records the executed suite totals and exact published Git revision; bulk
arrays, temporary stores and logs stay outside Git.
