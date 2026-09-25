# Visualization specification

## Responsibility

This component provides the interactive web inspection interface for persisted project artifacts.

It visualizes and navigates results from the analysis pipeline. It MUST NOT independently invent reconstruction, contacts, semantic boundaries, or evaluation results.

## Delivery model

The MVP uses a web client backed by a local analysis/data service.

Heavy vision/ML processing remains outside the browser. Source videos and generated artifacts may stay local and be served to the client; the MVP does not require browser-based upload of large recordings.

The web client may later be wrapped as a desktop application without changing analysis contracts.

## Shared playback clock

The UI has one global time cursor.

Changing it updates:

- all visible source videos;
- 2D observation overlays;
- 3D reconstruction;
- top-down ground view;
- timeline/actions/keyframes;
- selected inspector values.

Navigation is by global time, not assumed common frame number.

Frame-level stepping MUST be possible even when display refresh cannot present every recorded frame in real-time playback.

## Multi-camera view

The camera area supports a variable number of recordings.

For each camera the operator can inspect:

- video;
- source/global time correspondence;
- 2D pose overlays;
- synchronization offset/confidence;
- per-view observation confidence.

Excluded or low-confidence views must be identifiable.

## 3D view

The 3D scene supports free inspection of the calibrated world and practitioner.

At minimum it provides switchable layers for:

- skeleton/joints;
- detailed hands;
- detailed feet;
- ground plane;
- root trajectory;
- selected landmark/action trajectories;
- camera positions/frusta;
- confidence/debug information where practical.

A fitted body mesh may be displayed for anatomical context. The kinematic representation remains measurement truth.

Gaussian Splatting is outside scope.

## Timeline

The timeline is a first-class analysis/development tool.

It displays global time and physical tracks. When parser output is available it also displays SequenceSteps, Actions, Phases, and Keyframes on the same clock.

Selecting a timeline entity highlights participating tracks and synchronizes every view to the relevant time.

Overlapping actions with distinct boundaries must remain visible rather than being flattened.

## Top-down view

The UI provides a ground-plane view driven exclusively by [footwork-ground.md](footwork-ground.md).

Two conceptual modes are required.

### Dynamic mode

Shows the current feet/root state and nearby contact/path context as time advances.

### Summary mode

Shows accumulated footprints/path for the whole execution and permits selection of placements/events.

Available overlays include:

- left/right footprints;
- foot axes/orientation;
- action/step start/end placements;
- step distances;
- root path;
- contact/support events;
- pivot paths/arcs where available;
- parser annotations/keyframes.

If metric scale is unresolved, the UI MUST NOT present metric values as known.

## Reference overlay

The UI MAY overlay a canonical/reference execution using the same representation, particularly in the top-down view.

The overlay is descriptive. Differences in position, angle, length, height, or trajectory MUST NOT automatically be labelled correct/incorrect in the MVP.

## Inspector and provenance

Selecting a joint, landmark, footprint, action, phase, or keyframe exposes relevant structured values and confidence.

The operator should be able to reach enough contributing camera evidence to debug why a reconstructed point or parser event exists.

## Development controls

The UI SHOULD support:

- analytical layer toggles;
- manual synchronization-offset adjustment;
- manual parser-boundary/keyframe inspection/editing;
- local trajectory windows around selected time;
- model/confidence/provenance display.

Manual changes are visibly distinguished from automatic results.

## Acceptance

The visualization is accepted when an operator can open a processed project and inspect the same instant across videos, 2D observations, 3D reconstruction, top-down path, and semantic timeline; navigate bidirectionally between actions/keyframes/footprints and views; and understand uncertainty rather than seeing false precision.
