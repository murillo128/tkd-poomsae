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
Target calibration records each camera's intrinsic origin, capture IDs, corner
count, reprojection RMS in pixels, and planar-pose ambiguity gap. The calibration
producer/config digest distinguishes target estimation from imported profiles.
`Calibration.ground_status` and `scale_status` gate downstream use. A resolved
`ground_frame` owns the right-handed XY/Z-up transform, fitted plane diagnostics,
and floor/sign/axis evidence provenance; an unresolved ground has neither a frame
nor a ground Z. `scale_resolution` names the measured dimension, input unit,
conversion, and source revision. Manual recovery records author and reason in a
new calibration revision. Reconstruction and ground products reference the exact
calibration lineage; observations are immutable across scale revisions.

Landmarks have stable names, including individual finger joints, heels,
forefeet, and head landmarks. Unavailable points are null with `unknown` quality.
`Observation.regional_geometry` records independent per-view foot/head landmark
availability and orientation state, supporting landmark names, optional
pixel-space axis and projected orientation. A degenerate orientation retains
usable axis endpoints when available but has no angle;
these 2D values make no contact, 3D direction, or technique claim.
`source_regional_geometry` retains the regional provider's original output;
`regional_geometry` is reconciled against accepted in-frame derived landmarks.
Source axes that depend on masked landmarks remain evidence, not usable geometry.
`Observation.subject_selection` retains one per-camera practitioner track's
selected, ambiguous or missing state plus every candidate's box, detector score
and temporal match cost. `wholebody_landmarks` and `refined_landmarks` retain
independent source evidence; `landmarks` is the derived selected view with unknown
geometry explicitly null. Original visibility and scores remain separate from
derived `region_quality` usability and reasons for body, hands, feet and head.
Low raw visibility masks derived geometry without changing the source record;
unavailable detailed hand refinement likewise leaves coarse fingers only in
`wholebody_landmarks`.
Raw model scores retain their original declared range and are never interpreted
as calibrated accuracy probabilities. Unbounded model responses use null bounds
and an explicit score domain; RTMPose SimCC responses can exceed one and are
retained without clipping. Detector probabilities keep their bounded range.
Derived quality is separate. Source IDs
on quality values identify contributing artifacts: synchronization, calibration,
and observations reference `Source`; reconstruction references `Observation`;
morphology and ground reference `Reconstruction`; semantics references
`Reconstruction` or `Ground`. Empty contributor lists are permitted when the
enclosing artifact or frame carries the only available provenance. Dense arrays
have typed shape/axis/mask descriptors; the storage format and location are owned
by producers. JSON must contain `null`, never NaN or Infinity.

`Morphology` also permits optional dense array descriptors for separately
persisted shape evidence. Participant-specific skeletal fitting publishes
morphology and fitted pose independently; see
[`reconstruction/articulated/README.md`](../reconstruction/articulated/README.md).

Semantic intervals are absolute global seconds and permit overlapping actions.
The six physical tracks are fixed for this version. Spatial relations describe
geometry, including crossing front order, not correctness. Manual edits live in
a separate artifact linked to immutable automatic semantics.

`Alignment` is an additive derived artifact kind for global-time joins. It refers
to the exact synchronization artifact, immutable native observation manifest
digests, query count and versioned byte-payload descriptor. Query geometry and
native endpoint evidence are kept separately in the producer's payload; see
[`sync/README.md`](../sync/README.md#global-time-observation-queries).

`MotionFeatures` is an additive physical artifact kind before semantic actions.
It references exact reconstruction and ground artifacts and a versioned dense
feature/candidate payload. Shared-clock per-track measurements, candidate timing
and evidence links are described in
[`reconstruction/features/README.md`](../reconstruction/features/README.md).
These candidates do not assign technique classes, SequenceSteps, or correctness.

The additive `segmentation` artifact stores automatic coarse execution/SequenceStep
proposals before action assembly. A null execution plus unknown quality represents
indeterminate evidence. Resolved proposals tile the execution, retain empty action
lists, and link their exact motion-feature artifact plus dense boundary evidence.
See [`reconstruction/segmentation`](../reconstruction/segmentation/README.md).

The additive `arm_actions` artifact holds automatic upper-body proposals before
final semantic assembly. It binds reconstruction/ground, exact motion-feature
and coarse-segmentation IDs and a versioned evidence array. Independent track
intervals, SpecialAction candidates, provisional phases, complete crossing/depth
and hand uncertainty, and dense references are owned by
[`reconstruction/arms`](../reconstruction/arms/README.md).
