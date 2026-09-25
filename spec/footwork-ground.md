# Footwork and ground specification

## Responsibility

This component derives ground-relative physical facts from reconstructed motion. It owns contacts, support, footprints, pivots, planar foot geometry, root path, and ground measurements used by the top-down view.

It does not decide whether a movement is semantically a kick, step, or other poomsae action; [motion-parsing.md](motion-parsing.md) owns that interpretation.

## Inputs

The component consumes:

- calibrated ground/world frame;
- reconstructed foot landmarks and orientations;
- reconstructed root/pelvis motion;
- global time and uncertainty.

## Foot contact

For each foot over time, the component represents contact as known contact, known no-contact, or indeterminate.

Where evidence permits, contact may be refined into useful regions such as heel, forefoot, or broader/flat contact. The exact detailed taxonomy may evolve, but unknown must remain distinct from no-contact.

## Support state

The component derives support over time, including:

- both feet;
- left only;
- right only;
- neither;
- indeterminate.

When observable, support may identify the active region of the supporting foot.

## Footprints

A footprint event represents a stable or otherwise meaningful placement on the ground and includes:

- left/right identity;
- ground-plane position;
- foot orientation;
- relevant event/contact time or interval;
- geometry sufficient for top-down rendering;
- confidence.

Detector jitter MUST NOT create artificial extra footprints.

## Step geometry

The component derives geometric measurements between relevant placements, including:

- distance;
- longitudinal/lateral separation;
- relative alignment;
- relative foot orientation.

Metric units are emitted only when calibration scale is resolved. Body-relative/normalized forms may also be derived using participant morphology.

## Pivots

The component preserves and, where observable, derives:

- pivot interval;
- supporting foot;
- approximate pivot region such as forefoot or heel;
- foot rotation;
- translation of heel, forefoot, and foot center during the pivot.

Association of a pivot with a semantic poomsae action is left to the parser.

## Root path and body height

The component derives:

- ground-plane root/pelvis path;
- vertical root/pelvis trajectory.

These allow later inspection of path shape, lateral deviation, turns, and rise/fall while moving without assigning correctness.

## Top-down data product

The component exposes a deterministic 2D ground projection containing footprints, foot axes, root path, contacts/pivots, and available measurements.

The viewer owns presentation; this component owns the underlying geometry.

## Reference compatibility

A canonical/reference execution can be projected through the same representation so its path and placements can be overlaid with an observed execution.

This is geometric comparison only. No error label or score belongs in this component.

## Acceptance

The component is accepted when the operator can determine where each foot was placed, its orientation, how support changed, how the root travelled, and—where evidence permits—how a pivot occurred, all in one calibrated ground frame without correctness inference.
