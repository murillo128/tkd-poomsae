# TKD Poomsae specifications

This directory contains the normative product and functional specifications for the project.

The files are split by ownership on purpose: a requirement should be defined in one place and referenced elsewhere rather than copied.

## Specification map

- [general.md](general.md) — MVP purpose, scope, system boundaries, pipeline, and system-level acceptance.
- [motion-representation.md](motion-representation.md) — shared time model, coordinate systems, data entities, provenance, confidence, and persistence contracts.
- [ingest-sync.md](ingest-sync.md) — source-video ingestion and automatic temporal alignment.
- [camera-calibration.md](camera-calibration.md) — camera geometry, ground plane, world frame, and scale.
- [pose-observation.md](pose-observation.md) — per-camera 2D body, hand, foot, and head observations.
- [4d-reconstruction.md](4d-reconstruction.md) — temporally coherent 3D reconstruction and participant-specific body geometry.
- [footwork-ground.md](footwork-ground.md) — contacts, support, footprints, pivots, root path, and ground-plane measurements.
- [motion-parsing.md](motion-parsing.md) — poomsae decomposition into steps, actions, phases, and keyframes.
- [visualization.md](visualization.md) — synchronized web inspection of videos, 3D reconstruction, timeline, and top-down path.

## Ownership rule

The general spec defines system intent and boundaries but does not redefine component behavior. Component specs own their functional requirements. The motion representation spec owns shared semantic/data definitions used by multiple components.

If two files appear to define the same fact, move the fact to the narrowest owning spec and replace the duplicate with a reference.
