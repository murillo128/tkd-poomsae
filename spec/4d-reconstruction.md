# 4D reconstruction specification

## Responsibility

This component fuses synchronized, calibrated multi-view observations into a temporally coherent 3D representation of the practitioner. It owns reconstructed kinematics and participant body fit, not semantic action boundaries.

“4D” means 3D state evolving over the shared global time axis.

## Inputs and output

Inputs are calibration and pose observations defined by their component contracts.

The output is the reconstructed motion layer defined in [motion-representation.md](motion-representation.md), queryable by global time.

## Multi-view fusion

The component MUST use all useful observations available for a landmark/time rather than assuming a fixed camera pair.

Different body regions may use different subsets of cameras at the same instant.

Every reconstructed quantity retains confidence/uncertainty and enough provenance to debug the contributing observations.

## Temporal coherence

The result is optimized/filtered as a motion sequence rather than as unrelated frames.

Temporal regularization should reject detector jitter and impossible discontinuities while preserving genuine high-speed movement. Filtering must not make rapid techniques artificially slow, smooth, or straight.

## Participant-specific body geometry

The reconstruction SHOULD fit an articulated participant-specific body representation such as SMPL-X or an equivalent model where useful.

Body shape/proportions and pose remain separate. The system retains the participant's estimated morphology rather than forcing all motion onto standard proportions.

A rendered body mesh is a derived visualization aid. Geometric/kinematic data remains the measurement source of truth.

## Detailed extremities

Refined hand and foot observations MUST survive reconstruction. Coarse whole-body fitting MUST NOT collapse them back to wrist/ankle-only geometry.

## Head

The reconstruction retains head/neck orientation needed to determine where the head points and how it moves relative to the torso. Facial expression is outside scope.

## Spatial relations

The reconstructed state or deterministic derived data MUST support the relation model from [motion-representation.md](motion-representation.md), including depth ordering at arm crossings.

This is necessary because two configurations can have similar joint positions while differing in which arm is physically in front.

## Root/body motion

The component retains root/pelvis translation, rotation, and vertical position over time. This feeds path, turn, and body-height analysis.

## Rendering boundary

A body mesh may be generated for context. Gaussian Splatting and photorealistic dynamic reconstruction are outside the MVP.

## Acceptance

The component is accepted when the practitioner can be replayed in a common 3D world with temporally stable body, hands, feet, head, and root motion; reprojection/source evidence can be inspected; and poorly constrained regions remain visibly uncertain.
