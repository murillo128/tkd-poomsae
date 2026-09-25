# Camera calibration specification

## Responsibility

This component owns camera/world geometry required for multi-view reconstruction. It does not own temporal synchronization, pose detection, or semantic motion parsing.

## Inputs

Calibration consumes synchronized source imagery and may use static scene features, configured calibration targets/fiducials, or other reliable geometric evidence.

Camera count and placement are variable.

## Required outputs

For every retained camera, calibration provides the parameters required to project between world and image coordinates, including usable intrinsics and extrinsics.

The project calibration also establishes:

- a common world frame;
- a ground plane;
- world vertical;
- a consistent ground-axis orientation;
- metric scale when it can be resolved reliably;
- calibration confidence/quality diagnostics.

## Automatic camera placement

The normal MVP workflow MUST NOT require the operator to enter camera position or orientation manually.

The implementation may use structure-from-motion, calibration targets, scene landmarks, or another replaceable technique, provided it satisfies the output contract.

## Ground plane

Ground geometry is explicit because foot placement, step length, pivots, and body height are measured relative to it.

The component MUST expose the estimated plane and quality/uncertainty. Downstream code MUST NOT independently invent a ground plane from image coordinates.

## Metric scale

Multi-view reconstruction may be scale-ambiguous. The component MUST either resolve metric scale from reliable known-size evidence or mark scale as unresolved.

It MUST NOT fabricate centimetre values from an arbitrary reconstruction scale.

When scale is unresolved, angular and body-relative/normalized measurements may remain available; absolute metric distances must be reported as unavailable.

## Robustness

Calibration SHOULD prefer static evidence spanning the useful capture volume rather than depending on the moving practitioner alone when possible.

It MUST detect grossly inconsistent cameras and may exclude a bad camera if the minimum usable multi-view set remains.

## Persistence

Calibration is a persisted project artifact. Re-running pose/reconstruction on unchanged videos should not require recalibration unless calibration itself is intentionally rerun.

## Acceptance

Calibration is accepted when 3D/world points reproject consistently into retained views, the ground/world frame is available downstream, and weak geometry or unresolved metric scale is surfaced rather than hidden.
