# Pose observation specification

## Responsibility

This component extracts per-camera 2D observations of the practitioner. It owns detector output, confidence, and higher-resolution refinement of body regions. It does not triangulate 3D motion or interpret poomsae actions.

## Baseline

The MVP uses MMPose as the baseline body/whole-body pose framework. The persisted observation contract remains model-independent so detector implementations can change later.

## Body

For every useful camera/time sample, the component estimates the landmarks needed to reconstruct major joints and limb segments.

Detector confidence/visibility MUST be preserved alongside coordinates.

## Hands

Hands need more detail than a coarse wrist point.

The component MUST support a high-resolution hand observation path using regions localized from the whole-body estimate and the original-resolution source image.

The representation must be sufficient, when visible, to recover:

- finger configuration;
- palm/hand orientation;
- fist/open-hand configuration;
- wrist-to-hand alignment;
- relative configuration of both hands.

The component does not decide whether a martial hand shape is correct.

## Feet

Feet need more detail than a single ankle point.

The component MUST observe enough geometry to estimate at least:

- heel;
- forefoot/toe direction;
- longitudinal foot axis;
- foot orientation;
- useful inner/outer foot geometry when the detector supports it.

Individual toe reconstruction is not an MVP requirement unless evidence later shows it is necessary.

Ground contact is not decided here; it belongs to [footwork-ground.md](footwork-ground.md).

## Head and neck

Detailed face reconstruction is not required.

The observation set only needs sufficient head/face landmarks to support head orientation and neck motion relative to the torso.

## View-specific visibility

Different cameras may be best for different regions. Observations remain per-view rather than selecting one global “best camera”.

Occlusion and foreshortening reduce confidence; they MUST NOT silently produce fabricated high-confidence landmarks.

## Temporal processing

Temporal filtering may improve stability, but source observations remain traceable. Smoothing MUST NOT erase genuine fast motion such as kicks, punches, blocks, or pivots.

## Acceptance

The component is accepted when synchronized camera frames can be inspected with body, hand, foot, and head observations plus confidence, including explicit low-confidence/unknown states in difficult or occluded views.
