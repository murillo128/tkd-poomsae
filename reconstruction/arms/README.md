# Offline arm-action proposals

`reconstruction.arms` consumes physical offline features, coarse execution/step
proposals, and persisted reconstructed detailed hand geometry. It produces
independent arm actions and a supported coordinated SpecialAction pattern on the
original global clock. It does not execute vision, use poomsae names or label
files, judge correctness, infer force, assign scores, or evaluate a model against
a dataset.

```python
from reconstruction.arms import publish_arm_actions, load_arm_actions

handle = publish_arm_actions(store, feature_handle, segmentation_handle, motion_handle)
result = load_arm_actions(handle)
# result.proposals contains automatic candidates, not assembled final Semantics.
```

`parse_arm_actions(features, segmentation, geometry, config=None)` is the pure
entry point. Geometry must have the exact native times and reconstruction ID of
the features. Publication accepts final temporal motion (including its persisted
regularized detailed geometry) or a detailed reconstruction artifact. Features
and segmentation must refer to those exact inputs. Missing execution proposals
produce an empty result with an explicit `execution_indeterminate` diagnostic.

## Transparent baseline: arm-actions-v1

The baseline measures each arm independently using qualified linear and angular
velocity evidence. Movement requires speed minus the feature uncertainty bound
to exceed the corresponding feature enter threshold. Quiet requires both speed
channels plus their uncertainty bounds below the leave thresholds. Unknown,
interpolated, poorly constrained measurements, and threshold-band samples break
bouts conservatively; they do not become stationary observations. Qualified
angular motion alone can establish movement. Short observed quiet pauses remain
inside a bout; sustained quiet ends it. A boundary is the first/last qualified
active native sample, not an estimated impact or a regular-grid timestamp.

| Setting | Default | Meaning |
| --- | ---: | --- |
| min_activity_seconds | 0.12 s | Minimum retained arm bout |
| quiet_seconds | 0.12 s | Quiet duration separating bouts |
| max_gap_seconds | 0.15 s | Largest bridged native sample gap |
| coordination_overlap | 0.7 | Common interval / longer arm duration |
| mirror_tolerance_ratio | 0.15 | Reflection error bound / initial lateral radius |
| center_ratio | 0.45 | Final / initial lateral distance bound |

The supported SpecialAction pattern is **mirrored two-arm centering**. The arms
must overlap sufficiently and have at least three common samples. At every
common sample, both body-relative hand positions need native, qualified source
evidence. Left/right hands start on opposite sides of the body sagittal plane;
their lateral coordinates reflect each other and their front/height paths agree
within the configured reflection error, including the uncertainty bound. Both
hands end substantially closer to the sagittal plane, with excursion above the
feature input's configured spatial threshold. Spatial thresholds use the input's
explicit world units; reflection and centering tolerances are dimensionless.
Similar timing, divergent paths, missing geometry and uncertain geometry cannot
satisfy this detector.

A supported pair becomes one `category="special", role="special"` proposal.
Its union interval spans both tracks, while **each track keeps its original
start/end** and dense sample links. Other arm actions keep `role="unknown"`.
Geometry alone does not establish attack/defense intent. The detector is a narrow
inspectable heuristic: it does not identify every formal action, infer intent
from arbitrary bilateral motion, or establish real-data parsing accuracy.

## Evidence, candidates and downstream assembly

Every track retains inclusive motion sample indices and exact absolute-time hand
snapshots. Hand configuration, finger evidence, uncertainty and unavailable
states remain unchanged; a missing hand is `null`. Spatial relations come from
the qualified feature layer, including `unknown` crossing values and independent
`front_quality`. A known crossing may still have no established front forearm.
No ordering is guessed from 2D appearance or hand positions.

All relevant source feature events retain complete evidence windows and source
links, including windows extending outside an arm interval. Preparation,
chamber (extension-minimum evidence), and extension events create provisional
phase candidates within that track. Qualified contiguous decreasing chain
extension creates a retraction candidate only when its accumulated decrease
exceeds both feature prominence and uncertainty thresholds and lasts at least
the feature minimum duration. Phase candidates retain track, absolute interval,
source indices, event IDs, quality and reasons; they may overlap. They are not
claims about technique intent or final execution phases.

Proposal quality retains the minimum available upstream velocity score (or
`null` if scores are absent), contributors and native boundary sampling tolerance.
An upstream score is not a calibrated probability of semantic correctness.
The exact persisted feature reference provides dense trajectories and their
full velocity/acceleration and positional uncertainty. Coarse steps are linked
by interval intersection; actions spanning a coarse boundary are not clipped.
Final hierarchy assembly must resolve those associations under the final
`Semantics` containment contract.

The additive `arm_actions` artifact contains reconstruction/ground, exact feature
and segmentation IDs plus a versioned evidence byte-array descriptor. The storage
key binds all three input manifest hashes, schema, algorithm revision and effective
configuration. Reload validates track coverage, snapshots, phase/event links,
provenance and artifact identity without rerunning any producer. Automatic output
is immutable; manual correction remains a separate downstream edit overlay.

```sh
uv run --frozen pytest tests/test_arm_actions.py
```

Synthetic acceptance covers 25/50/100 Hz overlapping independent tracks, a positive
centering action with asymmetric boundaries, same-time unrelated motion,
preparation/chamber/crossing/extension/retraction candidates, uncertain depth and
missing hand evidence, missing measurements/native gaps, unknown execution,
source mismatch, persistence/cache/config invalidation, and an automatic positive
case fed by actual geometric feature extraction. These are synthetic behavior
checks, not dataset model evaluation.
