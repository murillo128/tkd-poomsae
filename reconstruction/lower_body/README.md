# Lower-body rules baseline

`parse_lower_body(features, ground, proposals, config=None)` consumes existing
`FeatureSeries`, canonical `Ground`, and `SegmentationResult` on their original
absolute clock. It does not invoke vision, reconstruction or any upstream
producer, read dataset CSV labels, or assign correctness, force or scores.
`lower-body-rules-v2` is a transparent offline heuristic, not a trained classifier
or fixed formal technique vocabulary.

Each leg is processed independently:

- A contiguous observed no-contact bout with qualified chain extension contains
  a kick when a flexed chamber precedes a significant extension and retraction.
  Chamber, extension and retraction are separately persisted phase candidates.
  A confident kick requires all three independently supported phase intervals;
  an already-flexed onset with a next-sample peak stays unknown when no chamber
  interval can be observed. Available extension, retraction and subsequent
  placement evidence remains intact.
- Placement requires observed contact on either side of the airborne interval,
  qualified planar displacement beyond the configured distance and uncertainty
  bound, and contiguous native timing. Following a kick, its recovery/placement
  interval begins at retraction completion and links its preceding kick.
- A physical pivot requires the upstream pivot's measured rotation and usable
  quality plus observed contact on the rotating foot throughout the interval.
  Airborne rotation, root rotation and missing supporting-foot evidence do not
  imply a pivot. Contact/support values are never redefined.
- Incomplete kick evidence does not become a generic confident step. Available
  extension and placement phases can remain inferred inside an unknown action.
  Remaining measured leg motion is retained as unknown candidates.
- Stances are stable two-foot configurations with qualified world positions and
  known contact. They are states, not actions, and link the contributing motion
  and available footprint IDs. Formal stance names remain unavailable.

Effective thresholds, world units (`m` or explicitly arbitrary), rule revision,
reason codes, heuristic class/phase quality, feature-event IDs, physical
placement/pivot IDs, native motion/ground indices and absolute intervals are
persisted. The default 0.8 inferred confidence is an evidence-strength convention,
not a calibrated probability. Missing evidence has unknown quality and no score.
The displacement threshold is in the declared feature world unit; arbitrary
scales require an explicitly appropriate configuration.

Candidates retain dense source ranges. Coarse SequenceSteps are associated by
interval overlap and do not trim, stretch or synchronize actions. Candidates
outside the proposed execution remain available, including when the execution
proposal is indeterminate. `native_times` covers the entire input and source
artifact references retain access to the original six-track dense trajectories.
This intermediate product is separate from the final shared `Semantics` hierarchy,
whose current single-step action containment rule cannot express these overlap
associations.

## Persistence

```python
from reconstruction.lower_body import publish_lower_body, load_lower_body

handle = publish_lower_body(store, feature_handle, ground_handle, proposal_handle)
result = load_lower_body(handle)
```

The additive `lower_body_parsing` artifact references the exact feature, ground,
reconstruction and segmentation IDs. Content addressing includes the three input
manifest digests, effective config and rule revision. Its versioned inspectable
JSON payload is stored as `lower_body_evidence_json` in the existing array store.
The v2 algorithm revision invalidates v1 cache keys for the chamber-evidence fix.
Cache hits and reloads do not rerun classification. Manual edits cannot masquerade
as this automatic product.

## Validation and limits

`tests/test_lower_body.py` covers kick → recovery/placement → stance, ordinary
steps, supported pivots, stationary stances, airborne/missing-support rotations,
concurrent legs, incomplete phases, native gaps, unknown/interpolated features,
coarse boundary overlap, lineage, dense links and cached immutable persistence.
A reconstructed synthetic chain also exercises real automatic feature extraction.
No models or dataset downloads are needed.

This deliberately conservative baseline requires observable airborne kick phases
and bracketed relocation. It may leave sliding placements, multiple kicks inside
one airborne bout, weakly observed chambers and partial recordings unknown.
It does not claim reliable formal technique naming or coverage of every kick.
