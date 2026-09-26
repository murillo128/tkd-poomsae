# Automatic execution interval and coarse SequenceSteps

`reconstruction.segmentation` consumes the complete persisted offline feature
series. It uses six physical tracks on their original global clock, qualified
linear/angular velocities, contact/pivot/placement candidate context, and dense
reconstruction/ground references. It never invokes vision or reads CSV labels,
a poomsae name, fixed step count, or canonical performance.

```python
from reconstruction.segmentation import (
    SegmentationConfig, publish_segmentation, load_segmentation,
)

handle = publish_segmentation(store, feature_handle, SegmentationConfig())
result = load_segmentation(handle)
# result.execution is None when the baseline cannot establish an interval.
# Otherwise result.steps are coarse SequenceStep proposals, with no actions yet.
```

`segment_execution(features, config)` is the pure inspection/synthetic entry
point. `execution-segmentation-v1` is an inspectable heuristic baseline, not a
trained parser or evidence of real-data accuracy. Mendeley remains functional
input only; its labels are not segmentation truth.

## Baseline and configuration

The baseline detects sustained activity using the feature input's explicit
world units and speed thresholds. A velocity must have non-interpolated,
non-unknown source evidence and uncertainty small enough to establish movement
above the speed threshold. Angular movement can establish activity even without
translation. Quiet context requires qualified linear measurements for all four
limbs; an unknown track cannot quietly become stationary. Head activity alone
cannot establish an execution.

| Setting | Default | Meaning |
| --- | ---: | --- |
| min_activity_seconds | 0.12 s | Minimum sustained non-head activity bout |
| edge_quiet_seconds | 0.4 s | Quiet pre/post-roll evidence bracket |
| terminal_hold_seconds | 1.0 s | Retained hold following the final activity |
| step_quiet_seconds | 0.35 s | Coordinated quiet before renewed progression |
| max_gap_seconds | 0.15 s | Maximum native sample separation |
| max_unknown_seconds | 0.2 s | Maximum unknown internal evidence bracket |
| min_coverage | 0.8 | Minimum time-weighted qualified limb coverage |
| min_confidence | 0.65 | Minimum heuristic evidence score |
| min_active_tracks | 2 | Distinct non-head tracks over the execution |

The execution spans the first through last sustained activity, including **all
internal pauses**, followed by the configured terminal hold. Quiet periods do
not produce separate executions or drop later activity. All of the terminal
hold and an additional quiet post-roll bracket must be observed; insufficient
recording length yields an indeterminate result instead of shortening the hold.
Missing/interpolated/excessively uncertain context also yields indeterminate
output, with a diagnostic and no fabricated steps or confidence score.

A static final pose alone cannot establish exactly when an intentional hold
ends and post-roll begins. The terminal hold is consequently a visible,
configurable baseline allowance, not a measured semantic end. Set it to cover
the expected hold when inspecting such recordings; longer unobservable holds
remain a limitation of this baseline. The reported score is time-weighted
coverage after multi-track and sustained-activity gates, **not a calibrated
probability**. Boundary evidence retains native sampling tolerance and the
explicit reason; the hold setting is part of persisted provenance.

Coarse boundaries are proposed only at renewed activity after coordinated
quiet. Overlapping/asymmetric per-track movement remains in the same progression.
Candidate evidence windows are measurement context, not action durations, and
do not prevent a cut by themselves. A lift-off awaiting contact/placement or an
unfinished supported pivot suppresses cuts inside their physical progressions.
No boundary is created merely because a foot lifts, contacts, or pivots. Entire
stationary upper-body units and compound units therefore remain possible.

## Dense links, artifacts and later assembly

The additive `segmentation` artifact kind separates coarse automatic proposals
from completed action semantics and manual edit overlays. It records the exact
feature artifact ID, reconstruction/ground IDs, nullable execution, proposed
canonical SequenceSteps, and inferred/unknown quality. The inspectable JSON array
retains config/revision, every native activity row, boundary reasons, event IDs,
and inclusive motion sample indices for each container. Event evidence that
intersects two containers remains linked from both, without clipping its source
interval. All pre/post-roll samples remain available through the feature artifact.

Per-track action intervals are **not** created or synchronized by this component.
Later assembly must consume original dense features and event intervals, and
associate spanning actions without clipping their motion to these coarse
proposals. [Semantic assembly](../semantics/README.md) assigns one onset owner and retains
references from all overlapping steps. Manual corrections belong to the edit component and never
replace the immutable automatic evidence here.

The storage key includes the exact feature manifest hash, schema version,
algorithm revision and effective segmentation configuration digest. Identical
inputs reuse the result. Configuration edits regenerate only segmentation;
loading needs no vision, model, media or upstream producer. Historical automatic
results stay immutable.

```sh
uv run --frozen pytest tests/test_segmentation.py
```

Synthetic acceptance covers 25/50/100 Hz, absolute timestamps, pre/post-roll,
stationary arm units, internal pauses, asymmetric arm timing, kick-to-placement,
spanning event links, terminal holds, insufficient activity/evidence, immutable
reload/cache/config invalidation, and integration with actual feature extraction.
