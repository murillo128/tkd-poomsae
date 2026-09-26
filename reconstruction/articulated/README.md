# Participant-specific articulated fit

`publish_articulated_fit(store, raw_handle, FitConfig())` consumes the immutable
raw artifact produced by `publish_triangulation`. It returns an `ArticulatedFit`
with separate `morphology` and `pose` handles. It performs no model/media downloads
and does not change the default raw pipeline stage. Downstream callers explicitly
choose the fitted artifact; temporal filtering and support/contact analysis remain
separate consumers.

```python
from reconstruction.articulated import (
    FitConfig, load_fit_diagnostics, load_morphology_evidence,
    publish_articulated_fit,
)

result = publish_articulated_fit(store, raw_handle, FitConfig())
shape = result.morphology.metadata
motion = result.pose.metadata
shape_evidence = load_morphology_evidence(result.morphology)
fit_evidence = load_fit_diagnostics(result.pose)
```

## Shape and evidence

The explicit skeleton estimates independent left/right upper arms, forearms,
thighs and shanks plus shoulder/hip widths. Each estimate uses endpoint distances
from direct `observed` points with contributor IDs and finite uncertainty. It
requires at least five different contributor sets spanning 0.1 seconds. Reusing
the same observations cannot increase that count. Interpolated or previously
inferred points never become morphology evidence.

Each endpoint's conditional uncertainty must be at most 15% of the distance.
Median absolute deviation (MAD) rejects deviations beyond 3.5 times the scaled
MAD (1.4826), with a distance-relative 1e-6 numerical floor. At least five retained
samples and the minimum span must survive. Scaled MAD above 10% of the retained
median makes the parameter unavailable. Missing, repeated, weak or outlying samples are rejected. Insufficient
retained evidence, short spans or unstable geometry yield a null value and
`unknown` quality, with an explicit reason. Thresholds are configurable numerical assumptions, not learned
anthropometric accuracy.

Morphology is an inferred constant sequence estimate, never an individual
measurement. Its quality links the raw reconstruction. The version-1
`articulated_morphology_evidence_json` uint8 array retains eligible samples,
retained/rejected evidence, global times, observation IDs, length uncertainty,
settings, assumptions and the exact raw manifest digest. `Morphology.arrays` is
an optional additive descriptor field; old morphology metadata still loads.
Uncertainty is the maximum of scaled MAD and median endpoint uncertainty in
quadrature, without dividing by sample count. Correlated/systematic error is not
eliminated by repetition. No symmetry, standard body size or supplied population
prior fills missing proportions.

## Pose fit and limits

Each frame fits available directly observed skeletal endpoints to the participant's
constant lengths using SciPy bounded least squares. The objective sums squared
data residuals divided by point uncertainty, plus squared length residuals divided
by morphology uncertainty and weighted by `length_weight` (default 4). Both
denominators have a relative length floor of 1e-6. Every coordinate can move at
most `max_displacement_fraction * smallest_incident_length / sqrt(3)` (default
fraction 0.1), so each joint's Euclidean displacement has the same fraction bound.
There are at most 100 function evaluations by default. A failed/nonconverged fit
retains the original points and reports `fit_rejected`. Length constraints are
soft: bounded fits may retain residuals rather than forcing implausible corrections.
There are no contact, population, joint-angle, temporal or gap-filling priors.
The sequence-wide morphology supplies the constant articulated constraint;
rapid pose changes do not themselves alter shape.

Adjusted landmarks become `inferred` with null scores, contributor links and
uncertainty at least the original uncertainty, applied displacement and incident
morphology uncertainty. Unchanged, interpolated, inferred or absent input points
keep their original geometry/quality. Missing joints stay null. Detailed fingers,
foot points and head geometry are copied unchanged, even if a nearby coarse
wrist/ankle moves. This intentionally preserves independent refinement rather
than rigidly moving all detail with a coarse limb.

The fitted reconstruction carries its own ID/provenance and a version-1
`articulated_fit_diagnostics_json` array linking the separate morphology ID and
exact raw revision. It contains settings, optimizer status/cost, data and length
residuals, and explicit partial transforms. Every raw payload, including the
per-camera triangulation evidence, is copied unchanged. The immutable raw artifact
remains the geometry/evidence source of truth. Cache identity includes the raw
manifest, algorithm and settings; pose identity also includes morphology revision.
Refitting a fitted artifact is rejected.

## Frames and units

World root translation and supplied root/segment rotations survive. If root
translation is absent, only an available named pelvis can supply it. No midpoint
or standard body dimensions are used to invent its position. When root orientation
is absent, the labeled left hip/right hip/neck triplet defines a geometric frame.
Torso uses left shoulder/right shoulder/neck; head uses left ear/right ear/nose;
hands use wrist/index MCP/pinky MCP; feet use heel/forefoot/outer foot. The first
point anchors each segment, X points from the first to second, Z is the orthogonal
component toward the third, and Y = Z cross X. These are geometric conventions,
not calibrated anatomical axes. They use original source triplets, not adjusted
coarse joints. Triplets require the same direct evidence/uncertainty gate plus
nonzero axes and sine of their angle at least 0.1. Sparse, weak or collinear
triplets yield unknown orientation; a limb's endpoint pair never fixes axial twist.

Derived quaternions are active local-to-world `[w,x,y,z]`. Their conservative
conditional angular uncertainty is maximum endpoint uncertainty divided by the
shorter axis length and angle sine, in radians. A value above the configurable
0.25 radian default gate makes orientation unavailable even if the triplet is
noncollinear. Derived translations are in world
units. Transform entries declare parent, translation and rotation separately,
with separate quality. Supplied orientations retain their declared parents;
translations in other parent frames remain unknown because the input contract
does not provide them. Root transforms allow `world = R * local + translation`
when both components are available. Individual limb direction can be derived
from its endpoints even when full rotation is unavailable.

Metric input remains metres and arbitrary input remains arbitrary units. Scaling
input geometry and its uncertainty scales all physical output proportions equally.
There is no anatomy-normalized replacement or claim of inferred metric scale.

## Validation

`uv run --frozen pytest tests/test_articulated.py` exercises different participants
under the same changing poses, asymmetric/unstable geometry, length outliers,
bounded adjustments and reduced evidence, absent or weak parameters, failed
optimization, available/unknown orientations, preserved world root motion and
refined detail, immutable/cache round-trips and metric/arbitrary units. The input
path is also tested with the actual raw triangulation publisher. These are offline
software/numerical invariants and establish no real-world Mendeley accuracy.
