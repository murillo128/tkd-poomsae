# Raw N-view triangulation

`triangulate_point(TimeQuery, name, Calibration, config)` fuses canonical
landmarks from the accepted global-time join. Source observations remain intact;
refined finger/foot/head names retain their own geometry and camera subsets.
At least two different cameras with independent, sufficiently separated rays
must support a point. Unknown source/sync/calibration quality, source mismatches,
behind-camera hypotheses, narrow ray angles, ill conditioning and inconsistent
reprojection produce null geometry, never zero coordinates or a monocular guess.

Every useful pair proposes a hypothesis. All cameras vote in distorted pixel
space with an absolute reprojection gate (default 8 px). Equal-support conflicting
subsets are ambiguous and rejected. The largest consistent subset is refined
using weighted robust nonlinear least squares over **all** its observations;
cheirality, residuals and conditioning are checked again after refinement.
This is bounded deterministic pair enumeration, O(N²) hypotheses, for small
camera rigs. It cannot identify adversarially consistent false correspondences.

`publish_triangulation(store, calibration_handle, alignment_handle,
participant_id, config)` persists an immutable reconstruction artifact. Its ID
and diagnostics identify it as **raw triangulated motion**. Body fitting and
future temporal regularization must publish separate artifacts/revisions.
Cache identity includes calibration and alignment manifests (therefore source
observations, synchronization, sampling grid and join settings), algorithm,
participant and numerical settings. No media or model resources are fetched.
`load_diagnostics(handle)` reads the versioned uint8 JSON array; its per-point
records contain contributing/rejected camera and observation/frame references,
interpolation weights, source/model provenance, raw score, reasons, residuals,
ray angle, conditioning and covariance. Rejected views are retained even when
no point can be reconstructed. Unknown times and regions remain explicit.

The default pipeline reconstruction stage consumes `calibration` and an
`Alignment` artifact in the `attachment` slot. Upstream producers can be installed
via the existing stage registry. Set explicit reconstruction settings, e.g.
`{"participant_id":"practitioner","triangulation":{"pixel_sigma":2}}`.
The standalone join/publisher API works without installing media/inference stages.

## Conditional uncertainty

`Quality.score` is left null. `Quality.uncertainty` is the largest-axis local
standard deviation in calibration world units, from the robust weighted
reprojection Jacobian, inflated by unexplained residual variance (never deflated
below the assumed noise). The diagnostics contain the full 3×3 covariance.

Pixel noise starts at a 2 px assumed floor, weighted by source quality. Camera
RMS contributes a pixel-error proxy; absent RMS assumes 4 px. Timing uncertainty
in seconds is propagated with an assumed 1000 px/s image-speed bound; missing
timing uncertainty assumes 20 ms. Weak camera/sync quality inflates those terms.
Bracketed observations add half the bracket duration times that speed bound.
Raw model scores are recorded but do not set weights or accuracy probabilities.

These are configurable conservative noise assumptions, **not calibrated accuracy
bounds**. Covariance is conditional on fixed cameras, independent isotropic
pixel noise and correct identity/correspondence. Shared calibration/systematic
errors, nonlinear depth ambiguities, synchronization bias and motion faster than
the declared speed bound are not modeled. Calibration RMS does not supply
extrinsic covariance. Reprojection agreement is not a validation of MMPose.
Arbitrary-scale calibration produces arbitrary-unit uncertainty; it never
claims metres. No orientation or body-model fit is inferred from sparse points.

## Numerical evidence

`uv run --frozen pytest tests/test_triangulation.py` runs entirely offline.
Known distorted two-/three-/four-camera fixtures reconstruct all named body,
finger, foot and head points within 1e-7 world units, with nonzero root motion.
Tests exercise noisy extra-view influence, source weighting, outliers,
insufficient regions, near-parallel and behind-camera geometry, weak evidence,
interpolation lineage, uncertainty inflation, cache reuse and immutable output.
These synthetic tolerances do not establish dataset/model accuracy.
