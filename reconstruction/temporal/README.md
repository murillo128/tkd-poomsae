# Temporally coherent reconstructed motion

`publish_temporal_motion(store, raw_handle, TemporalConfig(), FitConfig())` composes
the existing raw N-view triangulation artifact, participant morphology, bounded
articulated fit and detailed geometry functions into one immutable reconstructed
motion artifact. It runs offline and never invokes vision, contact inference or
semantic parsing. The return value exposes separate `morphology`, `fitted` and
`motion` handles. The existing pipeline raw reconstruction slot remains usable;
consumers explicitly choose this final motion for ground/semantic processing.

```python
from reconstruction.temporal import (
    TemporalConfig,
    load_temporal_motion,
    publish_temporal_motion,
    query_motion,
)

result = publish_temporal_motion(store, raw_handle, TemporalConfig())
evidence = load_temporal_motion(result.motion)
query = query_motion(result.motion, global_seconds=0.35)
# query.sample: geometry, source IDs and observed/inferred/interpolated/unknown state
# query.geometry: detailed frames, bends and body-relative relations at this time
```

## Filtering, quality and temporal bounds

Every native global timestamp remains in the result. For each available landmark
and root translation, a local quadratic polynomial minimizes weighted positional
residuals plus `regularization * sum(weights) * quadratic_coefficient**2`.
Time is centered on the output timestamp and scaled by half the filter window.
The estimate at its center is the constant coefficient. Only contiguous supported
samples within half `filter_window_seconds` contribute; searches use timestamp
bounds rather than scans of the complete recording. Windows with fewer than three
samples retain supported original geometry. `regularization=0` disables filtering,
while retaining the declared gap policy and downstream derivation.

Weights are inverse squared positional uncertainty with `uncertainty_floor`,
normalized to the largest window weight. Observed data gets weight 1, inferred
geometry gets `inferred_weight` and interpolation gets `interpolated_weight`.
Detector scores are never interpreted as probabilities. Finite uncertainty,
non-unknown state and contributing source IDs are required. Positional uncertainty
above `max_position_uncertainty` makes geometry unavailable. Filtered values are
`inferred`, with null scores and all window contributors. Their uncertainty is at
least the original margin, applied displacement and the sum of absolute estimator
weights times input margins. This retains correlated/systematic uncertainty rather
than dividing it away. Output exceeding the quality gate also becomes unknown.
These margins are numerical diagnostics, not calibrated accuracy.

Defaults are 60 ms filter window, regularization 0.001, 80 ms derivative window,
150 ms maximum adjacent interval, 40 ms short gap, 0.05 world-unit positional
uncertainty limit, inferred weight 0.5 and interpolated weight 0.25. Parameters
are explicit assumptions and can be changed for sampling rates and arbitrary
world scales. No population anatomy or standard participant size is imposed.

Missing geometry splits windows. An interior missing run may be linearly filled
only if its complete endpoint bracket is within `short_gap_seconds`, connected,
and supported. Its uncertainty is the larger endpoint margin plus half the endpoint
displacement; if this exceeds the gate, it remains unknown. It is `interpolated`
with both endpoints' source IDs. Missing ends never extrapolate. Long gaps stay
unknown. `break_times` explicitly splits edges with `a < break <= b`;
`max_interval_seconds` splits long timestamp jumps. Optional `max_step_world`
splits positional jumps larger than its threshold; genuine fast motions must not
be mistaken for discontinuities by selecting an inappropriate threshold. No
automatic detector can distinguish arbitrary genuine jumps from data discontinuities.

## Geometry, morphology and rotations

Morphology remains a separate immutable participant artifact. No normalization
replaces the participant's original world-scale geometry. Dense body, fingers,
feet, head and vertical root motion are filtered as separate channels; no floor
clamping, support/contact state or limb collapse is introduced. Independent channel
filtering has no new hard bone-length constraint: the participant's proportions
are retained in the morphology and preceding articulated fit, with any temporal
adjustment represented as uncertain inference rather than a fresh observation.

Detailed frames, digit bends, head relations and forearm front-order descriptors
are recomputed on final landmarks. Derived segment orientations use these final
frames rather than stale pre-filter triplets. Supplied root/segment rotations and
explicit parents survive; they are not replaced by geometric conventions.
Derived root rotation uses the final body's anatomical right/front/up frame.
Quaternion signs are made continuous within connected runs. Explicit rotations
are retained without a rotation smoothing prior to avoid erasing real pivots.
Rotations use active local-to-parent `[w,x,y,z]`; query interpolation uses
shortest-path SLERP, while geometric descriptors are recomputed at the query time.
Unsupported rotation brackets remain unavailable. Full revolutions and winding
cannot be inferred from aliased endpoint rotations.

## Derivatives, units and masks

`temporal_motion_json` contains versioned `kinematics` alongside `detailed` samples.
Each linear channel and root/segment angular channel has `velocity`, `acceleration`,
separate quality, explicit `velocity_valid`/`acceleration_valid` masks,
`support_times`, `parent`, base `unit` and `time_unit="s"`. Linear units are m/s
and m/s² for metric input, arbitrary/s and arbitrary/s² otherwise; angular units
are rad/s and rad/s². Null values have false masks and unknown quality.

Derivatives fit a quadratic in actual timestamps without a regularization penalty,
using the same evidence weights and half `derivative_window_seconds` on each side.
Velocity is coefficient 1 divided by the time radius; acceleration is twice
coefficient 2 divided by the radius squared. Conservative margins use the absolute
corresponding estimator coefficients. Endpoints use only a one-sided window
inside the same radius and connected run; fewer than three supported timestamps
makes both derivatives unknown. Windows never span unavailable geometry, declared
breaks or long gaps. Angular calculations use local SO(3) logs recentered at each
sample, expressed in the declared parent, rather than Euler-angle subtraction.
Near-pi log ambiguity and changes of parent invalidate affected windows.

## Persistence and queries

The final `Reconstruction` has schema version 1.0.0 and producer
`reconstruction.temporal`. Its version-1 `temporal_motion_json` payload records
algorithm/settings, exact raw/fit/morphology manifest hashes, separate morphology
ID, root component quality, derived geometry and kinematics. Every raw and fit
payload is copied unchanged for source inspection; the original artifacts remain
immutable. The payload's new `regularized` detailed representation is additive to
existing `raw` and `fitted` detailed representations.

Artifact keys include exact input revisions, algorithm and effective configuration.
Changing temporal/descriptor parameters creates a new final artifact while reusing
raw reconstruction, morphology and articulated fit. A semantic consumer can hash
the final manifest and recompute its own layer without vision. It must not treat
copied raw/fit diagnostics as descriptions of final geometry.

Queries are bounded by the first/last native times; nonfinite and out-of-domain
requests raise `ValueError`. Exact queries return a copy of the persisted sample.
Interior queries require the same short-gap and uncertainty policies, carry
`interpolated` state and bracket source IDs, and never add observations or dense
high-rate evidence. Long/unsupported brackets return unknown geometry at the
requested time. Detailed descriptors always derive from the queried geometry.
Store reload and repeated publication/query reproduce geometry and provenance.

## Synthetic acceptance

Run `uv run --frozen pytest tests/test_temporal_motion.py`. The deterministic
seed-28 noisy 5 ms trajectory uses a 40 ms filter window: extension/retraction
must reduce RMSE by at least 20%, peak amplitude error must be below 4%, peak delay
at most one sample (5 ms); direction-reversal timing error must be at most 10 ms
and amplitude error below 0.012 world units. A rapid 90-degree pivot uses 30 ms
filter/derivative windows: maximum angular geometry error below 0.02 rad,
midpoint delay at most 5 ms and angular-speed peak error below 15%.

Compact measured values and reproduction instructions are in
[`synthetic.json`](../../docs/evidence/temporal/synthetic.json). Tests additionally
cover irregular timestamps, one-sided endpoints, discontinuities, missing/weak
regions, bounded interpolation, quality weights, quaternion signs/wrap,
angular acceleration, detailed extremities, supplied rotations/parents, distinct
participant proportions, world units, actual N-view publisher input, immutable
reload/query and downstream-only cache invalidation. All evidence is synthetic;
there is no Mendeley tuning, dataset accuracy or generalization claim.
