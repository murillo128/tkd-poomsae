# General MVP specification

## Purpose

TKD Poomsae analyzes one Taekwondo poomsae execution recorded simultaneously by multiple cameras and converts it into a structured, inspectable representation of what the practitioner physically did.

The MVP is an **observation and motion-understanding system**. It reconstructs and decomposes the execution; it does not yet judge, score, or correct it.

## Primary use case

The operator provides two or more videos of one practitioner performing one poomsae. The cameras may start recording at different times, may have different viewpoints and recording characteristics, and their number and placement are not fixed.

The system processes the execution offline and produces:

1. a shared global timeline for all usable recordings;
2. a calibrated common 3D world and ground plane;
3. a time-varying 3D reconstruction of the practitioner, including detailed hands and feet;
4. ground-relative footwork and body-path products;
5. a structured temporal decomposition into poomsae steps, concurrent actions, phases, and keyframes;
6. a web-based synchronized inspection interface.

The continuous reconstructed motion is a durable output independent of semantic parsing so that parsing can be changed or improved without repeating the vision pipeline.

## MVP scope

The MVP MUST support a variable number of cameras, with at least two usable synchronized views.

The MVP MUST preserve both geometry and time. Local duration, velocity, acceleration, trajectory, ordering, and synchronization between body parts are observable properties even though overall poomsae rhythm or presentation are not yet evaluated.

The MVP MUST separate **observation** from **interpretation**. Coordinates, orientations, contacts, and trajectories describe physical evidence. Steps, kicks, arm actions, special actions, phases, and keyframes interpret that evidence. Correctness is a later layer.

The MVP MAY load a canonical/reference execution for geometric visualization or overlay. It MUST NOT convert those differences into deductions, scores, correctness labels, or coaching recommendations.

## Explicit non-goals

The MVP does not:

- assign competition scores;
- classify a movement as correct or incorrect;
- reproduce judge deductions;
- rank practitioners;
- provide coaching corrections;
- evaluate presentation, expressiveness, or whole-poomsae cadence;
- require real-time/online processing;
- require Gaussian Splatting or photorealistic reconstruction;
- require exact formal naming of every Taekwondo technique.

Those capabilities may be built later on top of the persisted motion representation.

## Functional architecture

The functional pipeline is:

**Ingest & Sync → Camera Calibration → Pose Observation → 4D Reconstruction → Footwork/Ground Derivation → Motion Parsing → Visualization**

Shared contracts between those stages are defined in [motion-representation.md](motion-representation.md).

Each stage MUST be replaceable without silently changing the semantic meaning of persisted upstream artifacts. In particular, semantic parsing MUST operate on reconstructed/derived motion rather than being embedded inside pose detection or triangulation.

## Core principles

### Reconstruct first, interpret second

The full motion is reconstructed continuously before semantic segmentation. The parser may inspect the complete execution, including future context, when deciding boundaries.

### Preserve continuous trajectories

Keyframes structure a movement; they do not replace it. Dense trajectories between keyframes remain available.

### Independent physical tracks, coordinated semantic actions

Left/right arms, left/right legs, body/root, and head evolve independently on one shared clock. A semantic action may involve one track or coordinate several tracks simultaneously.

### Preserve participant morphology

The system fits or estimates the participant's actual proportions. It MUST NOT erase those proportions by prematurely retargeting the motion onto a standard body. Later comparison may derive anatomy-normalized measurements while preserving the original reconstruction.

### Ground is first-class

Foot placement, support, orientation, pivots, step geometry, body path, and vertical root motion are central outputs rather than presentation-only overlays.

### Uncertainty is explicit

The system MUST distinguish “not reliably observed” from a confident observation. Confidence and provenance survive through reconstruction and derived products.

## Component ownership

Detailed requirements live in the component specs listed in [README.md](README.md). The general spec does not override a component's detailed contract unless it explicitly changes project scope.

## MVP completion criterion

The MVP is complete when an untrimmed multi-camera recording set can be processed into persisted artifacts that let an operator inspect the same physical instant across all usable cameras, the 3D reconstruction, the top-down ground view, and the parsed timeline; navigate steps/actions/keyframes; and inspect confidence/provenance for reconstructed measurements.

Success demonstrates faithful reconstruction and decomposition, not automatic technical correction.
