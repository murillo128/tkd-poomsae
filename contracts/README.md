# Shared motion contracts (1.0.0)

`models.py` defines the versioned Python types and validators. `artifact.schema.json`
is the generated portable JSON Schema; `types.ts` exports the client types.
Regenerate the schema with `uv run --frozen python -m contracts.schema` after a
model change. Call `validate_artifact` for one artifact and `validate_bundle` to
check cross-artifact references and mask shapes. JSON round trips use
`model_dump(mode="json")` and `model_validate`/`validate_bundle`. Reject unknown
schema versions; a future incompatible representation needs a new version.
`config_digest` is the SHA-256 digest of the producer's effective configuration;
model identity/version are included when a model produced the artifact.

Time is float64 seconds for calculations. Each observation retains camera ID,
native frame index and PTS/time-base integers when available, source seconds,
and the constant offset used to calculate `global_seconds`. Frame numbers across
cameras never imply simultaneous instants. An automatic sync estimate and any
manual correction remain distinct.

World coordinates are right-handed, XY is ground and +Z is up. Image coordinates
are pixels from the upper-left corner, +x right and +y down; intrinsics are in
those pixels. `world_to_camera` is a row-major homogeneous transform from world
XYZ to camera XYZ. The root is pelvis anchored; `root_orientation` rotates root
axes into world axes. Segment quaternions `[w,x,y,z]` rotate local axes into the
named parent frame. Angles are radians unless a measurement explicitly says
`deg`. Distances carry their unit. `scale: arbitrary` means metres are unavailable,
and no implicit conversion to metres is allowed. Body ratios require an explicit
participant morphology reference in the consuming calculation.

Landmarks have stable names, including individual finger joints, heels,
forefeet, and head landmarks. Unavailable points are null with `unknown` quality.
Raw model scores retain their original declared range and are never interpreted
as calibrated accuracy probabilities. Derived quality is separate. Source IDs
on quality values identify contributing artifacts: synchronization, calibration,
and observations reference `Source`; reconstruction references `Observation`;
morphology and ground reference `Reconstruction`; semantics references
`Reconstruction` or `Ground`. Empty contributor lists are permitted when the
enclosing artifact or frame carries the only available provenance. Dense arrays
have typed shape/axis/mask descriptors; the storage format and location are owned
by producers. JSON must contain `null`, never NaN or Infinity.

Semantic intervals are absolute global seconds and permit overlapping actions.
The six physical tracks are fixed for this version. Spatial relations describe
geometry, including crossing front order, not correctness. Manual edits live in
a separate artifact linked to immutable automatic semantics.
