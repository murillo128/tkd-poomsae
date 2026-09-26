# Offline motion features and event candidates

`reconstruction.features` consumes existing temporal reconstruction and ground
artifacts. It runs with complete-sequence context and does not call pose,
calibration, reconstruction, or ground producers. It does not read the Mendeley
CSV or train/fit thresholds. `motion-features-v3` is a transparent heuristic
baseline, with versioned configuration and explicit evidence links.

```python
from reconstruction.features import publish_features, load_feature_evidence

# ArtifactHandles already loaded from the local ArtifactStore:
handle = publish_features(store, temporal_motion_handle, ground_handle)
features = load_feature_evidence(handle)
```

`derive_features(reconstruction, ground)` is the pure synthetic/inspection entry
point. Publication requires a final `reconstruction.temporal` artifact. Contact,
placement, and pivot ground products are supported. Stable placement candidates
require explicit `FootprintSeries` confirmation; a generic footprint is
insufficient because canonical footprints also represent moving placements.

The additive `motion_features` artifact kind is separate from `semantics`.
Consumers classify actions and group SequenceSteps later. Candidates are neither
verified strikes/impacts nor correctness judgements. In particular,
`preparation_candidate` means a prominent minimum of physical chain extension,
not a verified chamber or formal technique.

## Dense measurements and provenance

Every native reconstructed timestamp retains six ordered channels: left/right
arm, left/right leg, body/root, and head. No common event boundaries are imposed.
Each row links its original motion sample index and exact native ground sample
index where available; ground is never filled across a missing timestamp.

Rows retain world position, body-relative position, orientation with its explicit
parent, extension, local linear velocity/acceleration, angular
velocity/acceleration, spatial relations (including crossing/front-order quality),
and available contact state. Arms use wrists, legs use ankles, and head uses its
landmark; root uses the supplied root position with temporal translation quality.
The pure entry point accepts pelvis evidence for root only when positions agree.
Local frames use the geometric torso frame; root translations remain in world.
Distal hand/foot and head orientations are expressed in the torso frame. Missing
orientation evidence remains unknown even when endpoint positions are available.
Extension is endpoint distance divided by the sum of the two segment lengths,
a dimensionless straightness ratio of the current observed chain. This is a
geometric measurement, not normalization to a participant morphology model; it
does not overwrite participant geometry.

Derivatives reuse the upstream temporal component's quadratic, time-domain local
estimator with its propagated uncertainty and exact supporting times. They use
seconds, world-unit/s and world-unit/s², or rad/s and rad/s². The feature layer
performs no interpolation or smoothing of source trajectories. Unknown,
insufficient, or excessive-uncertainty measurements retain null values; supplied
interpolated values retain their quality and cannot establish candidate evidence.
Native landmark quality qualifies each derived torso, head, hand, and foot frame,
including frames loaded from persisted detailed geometry. Interpolation in a
reference frame propagates to body-relative measurements and relations. Every
derivative preserves interpolation state from its entire supporting window,
including at neighboring observed samples. Independently supplied root
orientations retain their own quality; torso-derived root orientations depend on
the native torso landmarks. Missing frame inputs leave dependent features unknown.

Every event carries its track, absolute time, evidence start/end and duration,
motion/contact indices, upstream placement/pivot IDs when used, source quality
IDs, reasons, and a timing tolerance. Per-measurement upstream quality remains in
the trajectory or linked artifact. Categorical candidate quality is inferred,
with uncertainty in **seconds** equal to its tolerance; it has no calibrated
probability score. Evidence duration is not an inferred semantic action duration.

## Heuristic defaults and units

World thresholds use `FeatureConfig.world_unit`: `m` (default) or `arbitrary`.
This must match reconstruction scale. Arbitrary-scale users explicitly configure
thresholds in that world's units; the baseline never silently assumes metres.

| Setting | Default | Meaning/unit |
| --- | ---: | --- |
| max_position_uncertainty | 0.05 | world units |
| max_angle_uncertainty_rad | 0.25 | radians |
| enter_speed / leave_speed | 0.15 / 0.05 | world units/s; hysteresis |
| min_excursion | 0.025 | world units |
| enter_angular_speed_rad_s / leave_angular_speed_rad_s | 0.3 / 0.1 | rad/s; hysteresis |
| min_angular_excursion_rad | 0.12 | radians |
| extension_prominence_ratio | 0.08 | endpoint/chain-length ratio |
| min_duration_seconds | 0.04 | sustained movement, quiet, crossing/contact dwell |
| refractory_seconds | 0.08 | minimum separation of same-kind candidates per track |
| max_gap_seconds | 0.15 | largest continuous native sampling interval |
| derivative_window_seconds | 0.12 | complete local estimation window |
| uncertainty_multiplier | 3 | conservative multiplier, no statistical calibration |

Motion onset requires a confirmed quiet-to-moving bout with excursion above both
its physical threshold and uncertainty bound, followed by sustained quiet.
A single near-zero derivative at a reversal does not start a new bout. One-sided
derivative endpoints cannot establish onset. Rotation can establish onset without
translation, including head/root motion. A bout still moving at a sequence edge
is conservatively left without onset; dense measurements are retained.

Prominent interior extension maxima/minima supply extension end/preparation and
direction-change candidates. Increasing extension inside a motion bout supplies
extension start only when chain measurements remain known and non-interpolated
from the bout onset through the first increase exceeding both the prominence
threshold and an uncertainty bound. That bound is `uncertainty_multiplier` times
the sum of baseline uncertainty and maximum uncertainty in the supporting prefix,
all in extension-ratio units. Candidate source references include this chain
evidence, including the elbow or knee. A chain gap before the confirmed increase
prevents an extension-start boundary while independently supported motion events
remain available. Local path extrema also supply direction changes when incoming
and outgoing displacements oppose each other and exceed the excursion bound.
Extrema use offline prominence on connected native evidence, selecting a single
plateau representative. Subthreshold jitter and same-kind nearby candidates are
suppressed; other meaningful candidates at the same time remain distinct.

Crossing requires observed not-crossed and crossed configurations plus sustained
crossing. Geometrically indeterminate but fully observed configurations remain in
a bounded timing bracket; missing/interpolated supporting points break it.
First contact/lift-off require known native states and dwell on **both** sides;
unknown intervals and one-frame contact flicker cannot create boundaries.
Supported pivot boundaries and stable placements are consumed from upstream
physical evidence without re-estimating ground state.

Event timing tolerance is the maximum native sample spacing in its evidence.
Onset/extension-start additionally include half the derivative window; crossing
includes any observed geometric ambiguity bracket. Synthetic acceptance checks
extrema/contact timing at 25, 50, and 100 Hz within one sample spacing. Absolute
time and durations are never normalized away.

## Persistence and regeneration

The immutable artifact key includes the exact temporal-motion and ground manifest
hashes, algorithm revision, schema version, and feature configuration digest.
Revision `motion-features-v2` invalidated extraction from before the native
frame/derivative quality correction. Revision `motion-features-v3` additionally
regenerates extraction with qualified extension-start evidence. Historical v1/v2
payloads remain readable with their original revision and evidence; publication
generates v3 results.
Identical inputs reuse extraction; changing motion, ground, or thresholds creates
only a new feature artifact. The inspectable versioned payload is stored as a
read-only uint8 array in the existing artifact store, preserving all dense rows
and candidate references. Original dense motion/ground artifacts remain available
and unchanged. `FeatureSeries` validates the shared-clock topology and candidate
source/time links on loading.

```sh
uv run --frozen pytest tests/test_motion_features.py
```

Synthetic coverage includes fast extension/retraction, independent arm timing,
chamber minima, crossing/front order, body translation, head rotation, leg
extension, contact transitions/flicker, stable placement, pivot boundaries,
flat/noisy/plateau/gapped/interpolated inputs, absolute timestamps, and immutable
publication/cache invalidation. These fixtures establish heuristic behavior, not
real-data segmentation accuracy.
