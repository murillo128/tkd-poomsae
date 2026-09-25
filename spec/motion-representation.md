# Motion representation specification

## Responsibility

This spec owns shared data semantics used across components. Other specs may reference these entities but MUST NOT redefine them.

The representation separates source observations, reconstructed physical state, derived ground state, and semantic motion interpretation so each layer can be regenerated independently.

## Time model

Every source frame retains its native identity:

- camera identifier;
- source frame index when available;
- source presentation timestamp when available.

After synchronization, observations also map to one **global execution time**, expressed in seconds.

The MVP synchronization model is one constant temporal offset per source recording. Clock drift is out of scope.

Consumers MUST use global time rather than assuming equal frame numbers represent equal physical instants. Different source frame rates are allowed.

## Coordinate systems

### World frame

A common 3D coordinate system shared by all cameras and the practitioner. Once calibrated, the XY plane is the ground plane and Z is vertical.

This frame owns poomsae path, absolute foot placement, root trajectory, and camera geometry.

### Body/root frame

A participant-relative frame anchored to the pelvis/root. It supports measurements relative to the body independently of where the practitioner is on the floor.

### Segment-local frames

Relevant segments such as torso, forearm, hand, foot, and head may expose local orientation frames. This allows anatomical/relative measurements that cannot be represented by point coordinates alone.

Transforms between available frames MUST be explicit.

## Source observation layer

A source observation records what a specific camera/model estimated at a specific source time. It includes:

- 2D landmark coordinates;
- landmark identity;
- confidence/visibility;
- detector/model provenance;
- optional region-of-interest metadata for refined hands, feet, or head.

Source observations are evidence for downstream reconstruction. Reconstructed landmarks SHOULD be traceable to the observations that contributed to them.

## Reconstructed motion layer

A reconstructed state at global time may contain:

- 3D joint positions;
- segment orientations;
- joint angles where derived;
- root/pelvis position and orientation;
- detailed hand landmarks/configuration;
- detailed foot landmarks/orientation;
- head/neck orientation;
- participant body-model parameters;
- confidence/uncertainty and provenance.

The canonical representation is geometric/kinematic. A rendered body mesh is derived and is not the measurement source of truth.

The representation MUST support querying/interpolating by global time rather than exposing only unrelated frames.

## Participant morphology

Participant-specific morphology is stored separately from pose. The representation MUST be able to retain estimated or supplied proportions relevant to later normalization, including limb lengths and shoulder/hip widths when available.

The original motion remains expressed for the real participant. Normalized values are derived data and MUST NOT overwrite original geometry.

## Ground/footwork layer

Ground-derived data is separate from raw reconstructed foot landmarks. It contains contact/support state and other ground-relative products defined in [footwork-ground.md](footwork-ground.md).

## Semantic motion layer

### PoomsaeExecution

The analyzed execution, including its detected start/end inside the longer synchronized recordings.

### SequenceStep

A coarse poomsae movement unit grouping actions belonging to the same overall progression. It is a semantic/temporal container and is not necessarily a literal walking step.

### Track

A physical motion channel. The initial set is:

- left arm;
- right arm;
- left leg;
- right leg;
- body/root;
- head.

Tracks do not imply semantic independence.

### Action

A semantic movement unit over a time interval. An action references one or more tracks and may overlap other actions.

### Phase

A meaningful sub-interval of an action.

### Keyframe

A significant temporal event within an action or phase. Keyframes are selected by motion meaning/events, not uniform sampling.

The dense continuous motion linked to these entities is always retained.

## Spatial relations

The representation supports explicit body-part relations when coordinates alone are insufficient. Examples include:

- in front of / behind;
- above / below;
- left/right relative ordering;
- crossed;
- which forearm is in front at a crossing.

Relations reference body entities and a global time or interval. They are derived observations, not correctness rules.

## Confidence and unknown state

Every inferred/reconstructed layer MUST represent insufficient evidence. Unknown/indeterminate is distinct from false, absent, or incorrect.

Downstream components MUST NOT silently turn an unknown upstream value into a confident interpretation.

## Persistence

Dense arrays SHOULD use an efficient binary representation; metadata and semantic structures MAY use an inspectable structured format such as JSON. Exact serialization is an implementation decision as long as the semantics above remain stable.

Original videos, dense generated artifacts, model weights, caches, and other bulky outputs are not required to live in Git.

## Versioning

Persisted artifacts MUST include a schema/version identifier and enough model/configuration provenance to determine how they were generated. Incompatible representation changes require an explicit version change rather than silent reinterpretation.
