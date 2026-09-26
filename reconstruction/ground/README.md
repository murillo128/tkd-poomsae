# Physical foot contact and support

`derive_contacts(final_reconstruction, calibration, morphology=None, config=None)`
returns `ContactSeries` without editing or resampling its inputs. Each supplied
global time has canonical left/right `contact`, `no_contact`, or `unknown` and
`both`, `left`, `right`, `neither`, or `unknown` support. A foot with unknown
contact makes support unknown. These are geometric estimates; contact is always
`inferred` quality, never an observed force or detector-score probability.

The stage consumes final world landmarks, already aligned to ground z=0. It does
not apply `source_to_world` a second time. Heel, forefoot and outer-foot landmarks
provide the available ground-relative geometry; root coordinates remain in the
linked reconstruction and do not substitute for missing feet. Unknown, absent,
source-free, or uncertainty-free landmarks cannot establish contact or flight.
An inferred/interpolated landmark can contribute only with explicit positional
uncertainty, provenance and new native evidence for temporal confirmation.

## Versioned rules

Algorithm `foot-contact-v1` uses the following `ContactConfig` defaults. Values
are configurable and included in artifact identity; they are diagnostic choices
for synthetic acceptance, not a claim of calibrated contact accuracy.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `enter_height` | 0.025 | Maximum ground proximity for entering contact |
| `leave_height` | 0.05 | Proximity retained for established contact; height exceeded to enter flight |
| `enter_vertical_speed` | 0.15 | Maximum absolute vertical speed entering contact |
| `leave_vertical_speed` | 0.3 | Maximum absolute vertical speed retaining contact |
| `contact_seconds` | 0.08 | Continuous candidate evidence needed to confirm contact |
| `no_contact_seconds` | 0.04 | Continuous candidate evidence needed to confirm no-contact |
| `max_gap_seconds` | 0.15 | Maximum connected sampling interval |
| `uncertainty_multiplier` | 2 | Conservative margin applied to geometry and speed |

Metric mode uses metres and metres/second only with resolved metric calibration.
`units="body_normalized"` must be selected explicitly: the same numeric defaults
then mean fractions of the named morphology length and fractions/second.
The default length is `left_shank`; another measurement can be named explicitly.
There must be exactly one positive measurement in the reconstruction's world
units, with non-unknown quality, source IDs, and relative uncertainty at most
`max_morphology_relative_uncertainty` (default 0.1). No population height, opposite
limb length, or implicit unit conversion is substituted. Unresolved ground,
unknown calibration quality, unavailable metric scale or unreliable morphology
produces `evidence_unit="unavailable"`, unknown contact, and explicit reasons.

Position uncertainty, ground RMS residual and distance from the ground origin
times the sine of ground-normal uncertainty form a conservative height margin.
Body normalization adds the length uncertainty, including its denominator
margin. Adjacent-sample vertical velocity uses their actual elapsed seconds;
the sum of their height margins divided by that duration is its uncertainty.
Margins do not assume independent errors and are not confidence probabilities.
All input landmark quality, including its original evidence state, is retained.

Contact requires a heel or forefoot near ground with sufficiently low vertical
motion after margins. `flat` additionally requires all three landmarks near
ground; `heel` requires both distal landmarks raised; `forefoot` requires both
distal landmarks near ground and heel raised. Partial evidence can establish
contact while leaving region null. No-contact requires all three landmarks
confidently raised. Established no-contact is retained above `enter_height`;
established contact uses the wider leave thresholds. Geometry clearly below the
ground is unusable, and the uncertain band remains unknown. Horizontal motion
does not negate contact: this stage makes no friction, sliding, or pivot claim.

Candidates must persist for elapsed-time dwell, with no gap above the limit.
During an opposite candidate's confirmation the output is unknown; it does not
continue asserting contact through observed flight. Unusable/ambiguous evidence
immediately clears temporal certainty. Gaps reset history. No interpolation or
backdating is performed. Repeated source IDs cannot confirm a new state or
establish a fresh motion pair. Source IDs must identify native observations,
as in the reconstruction publishers; a render-query ID is not new evidence.
Rate equivalence is limited by native resolution: acceptance compares confirmation
times within two sample intervals plus the reference interval, for adequately
sampled motions below the gap limit. Sub-frame contact timing is not inferred.
Temporal smoothing can reuse the final native observation in its preceding
window; if the boundary sample introduces no new sources, its contact can be
unknown even when interior samples establish standing. This conservative edge
behavior is covered by an actual temporal-publisher roundtrip test.

## Independent publication and evidence

```python
from reconstruction.ground import publish_contacts, load_contact_evidence

ground = publish_contacts(
    store, final_motion_handle, calibration_handle, morphology_handle
)
evidence = load_contact_evidence(ground)
```

The publisher accepts `reconstruction.temporal` motion and verifies calibration
identity/scale and morphology participant. It creates an immutable canonical
`Ground` artifact with a separately versioned `contact_evidence_json` byte array.
The cache key includes exact motion/calibration/morphology manifest hashes,
algorithm revision and configuration. It never changes upstream motion or
reuses raw/fitted contacts after temporal reconstruction. A missing ground fails
publication explicitly because the canonical artifact graph requires resolved
ground. Resolved ground with unusable units can persist explicit unknown results.

The payload retains source identities, configuration, normalization quality,
native global times, per-landmark heights/speeds and uncertainty, motion brackets,
candidate reason codes, contributing confirmation times, and transition-onset
brackets. The canonical contact quality retains all confirmation-window source
IDs. Confirmation appears at the current native time; the onset bracket spans
the last sample before a candidate to its first sample. This is a geometric
candidate bracket, not a measured instant of force onset. Consumers use the
persisted contact/support states; viewers and semantic parsers do not redefine
them. The loader checks payload/sample/config consistency with the manifest.

No footprints, pivots, load, force distribution, centre of pressure, kick/step
classification, correctness scores or new pipeline orchestration are produced.

Run offline synthetic acceptance with:

```sh
uv run --frozen pytest tests/test_ground_contact.py
```

The cases cover standing, lift-off, flight, landing, all support combinations,
heel/forefoot/partial geometry, occlusion, uncertainty, noise, rate equivalence,
gaps, repeated native evidence, scale invariance, explicit unavailability and
immutable publication. No dataset downloads, contact-accuracy validation or
MMPose score comparisons are required or performed.
