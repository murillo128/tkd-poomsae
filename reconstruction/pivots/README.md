# Supported foot rotation and pivot regions

This component consumes the final native landmark/contact trajectory in a
`reconstruction.footprints` artifact. It derives physical rotation only, as owned
by `spec/footwork-ground.md`; semantic action assignment belongs to the parser.
It requires no dataset, model inference, or model download.

```python
from reconstruction.pivots import publish_pivots, load_pivot_evidence

pivots = publish_pivots(store, placements)
evidence = load_pivot_evidence(pivots)
```

For in-memory processing, use `derive_pivots(footprint_series, PivotConfig())`.
The canonical `Ground.pivots` contains the side, native interval, approximate
region, signed net rotation in radians and inferred quality. The dense
`pivot_evidence_json` preserves placement evidence, angular trajectories,
absolute angular travel, heel/forefoot/axis-midpoint translations, source indices,
relevant placement IDs and diagnostics. The midpoint is a geometric foot centre,
not an anatomical or pressure centre. Positions, displacement, path length and
excursions use the placement artifact's world unit (`m` or `arbitrary`); excursions
also retain foot-length ratios. Source contact and footprint arrays and canonical
samples/footprints are copied unchanged into the new immutable artifact. Cache
keys include the exact placement manifest, configuration and algorithm revision.

Axes are unwrapped using signed shortest angular differences; samples must resolve
rotation to less than a half-turn between queries. Near-half-turn differences are
marked ambiguous and split detection. Gaps, missing axes, reused landmark sources
and absent/indeterminate support split candidate intervals. Rotation and geometry
remain inspectable during flight or unknown support. Body/root orientation alone
cannot establish a foot rotation.

Detection uses configurable angular speed entry/exit thresholds and elapsed-time
quiet hysteresis. A candidate starts at the native sample before detected movement;
its end is the last moving sample. Brief quiet periods remain in the trajectory;
a sustained quiet period closes the interval. Angular excursion must exceed the
configured minimum plus landmark uncertainty. There is no temporal smoothing or
minimum duration that would erase a rapid pivot. Net rotation and total angular
travel distinguish reversals without inventing a revolution at angle wrap.

Heel/forefoot regions require that endpoint to stay within a configured fraction
of the minimum native foot-axis length, including positional uncertainty, and
that all interval contact regions support it (`heel`/`forefoot` or `flat`). Unknown
or conflicting contact regions retain `unknown`. Rotation with both endpoints and the midpoint
translating beyond that bound is classified `rotation_with_translation`; this
preserves possible sliding without claiming an observable pivot centre. An unknown
centre retains its rotation and translations. Pure slides and stationary jitter do not
create supported rotation events.

`region_confidence` is a conservative rule score: 0 when unknown, 0.8 when both
stationarity and contact-region evidence support the estimate. It is not an
empirically calibrated probability. `Pivot.quality.uncertainty` is the angular
uncertainty margin in radians; exact source qualities remain in the native
trajectory. An estimated region is never measured centre of pressure, load, force
or correctness. Occlusion, frame rate and reconstruction/contact uncertainty
limit what can be inferred.

Offline acceptance: `uv run --frozen pytest tests/test_pivots.py`.
