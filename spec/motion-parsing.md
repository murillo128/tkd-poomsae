# Motion parsing specification

## Responsibility

This component converts continuous reconstructed/derived motion into a structured description of the poomsae execution. It owns temporal decomposition and motion semantics, not underlying 3D measurements and not technical correctness.

The parser runs offline after reconstruction and may inspect the complete execution.

## Execution interval

The parser identifies the actual poomsae interval inside the longer synchronized recordings, allowing pre-roll and post-roll outside it.

This is distinct from camera synchronization.

## Hierarchy

The parser produces the semantic hierarchy defined in [motion-representation.md](motion-representation.md):

**PoomsaeExecution → SequenceStep → Actions → Phases / Keyframes**

Every semantic interval remains linked to the dense underlying trajectories.

## SequenceStep

A SequenceStep is a coarse poomsae movement unit grouping actions belonging to the same progression.

It MUST NOT be equated with a literal footstep. It may contain no literal step, several upper-body actions, a kick, a pivot, or a compound sequence.

## Independent tracks and overlapping actions

The parser reasons over physical tracks independently while retaining one shared clock.

An Action may use one or several tracks and may overlap other Actions.

The parser MUST NOT force both arms or both legs to have identical boundaries merely because their actions occur within the same SequenceStep.

## Lower-body semantics

The parser distinguishes these concepts:

### Stance

A lower-body state/configuration, not itself an action.

### Step / placement action

A displacement moving a foot/body between placements or stances.

### Pivot action

A rotation dominated by a supported foot and informed by the ground component.

### Kick action

A leg technique with its own phases and trajectory, not a special case of a step.

### Support context

Support is physical state from the ground component and contextualizes lower-body actions.

A kick that later places the kicking foot forward can therefore be represented as kick → recovery/placement → resulting stance.

## Upper-body and special semantics

Not all arm/hand motion is forced into attack or defense.

An action may carry a known semantic role such as attack, defense, preparation, or special/formal action.

A **SpecialAction** (or equivalent category) is first-class for coordinated, centering, formal, or otherwise meaningful movements that are not naturally attack/defense.

A coordinated two-arm movement may be one action spanning both arm tracks when that is the natural semantic unit.

## Transitions

Motion between meaningful actions is not discarded.

Transitions may contain chamber/preparation, weight transfer, foot relocation, pivots, or other motion. They are retained either as explicit transition actions or structured intervals/phases linking adjacent actions.

## Phases and keyframes

Keyframes represent meaningful motion events rather than regular temporal samples.

Candidate events include:

- motion onset;
- preparation/chamber;
- arm crossing;
- direction change;
- maxima/minima of extension;
- start/end of extension;
- foot lift-off;
- first contact;
- stable placement;
- pivot start/end;
- kick chamber;
- maximum extension or impact candidate;
- retraction.

Different action types may use different event sets. Complex actions may contain more intermediate keyframes than simple ones.

The trajectory between keyframes is always retained.

## Relative body configuration

The parser may attach relations from the shared representation when they define action structure, such as which forearm is in front during a crossing.

It MUST NOT replace relation-sensitive configurations with only approximate absolute joint locations.

## Timing and dynamics

The parser preserves:

- action and phase duration;
- timing between concurrent tracks;
- trajectory shape;
- velocity/acceleration features where useful.

The MVP does not judge whole-poomsae rhythm or presentation. Local timing remains observable technique data and MUST NOT be normalized away.

A normalized phase coordinate may be derived for comparison as long as absolute time is preserved.

## Classification depth

The MVP must distinguish broad classes needed for contextual analysis, including arm action, step/placement, pivot, kick, stance state/transition, special action, and transition.

Exact formal technique naming is desirable but not required for MVP acceptance.

## Automatic and manual boundaries

Automatic segmentation is the target.

During development, boundaries/keyframes may be inspected and corrected manually so parser quality can be evaluated separately from reconstruction quality. Manual edits are stored as edits and must not masquerade as automatic output.

## Acceptance

The parser is accepted when an execution can be navigated as coherent SequenceSteps containing overlapping per-track actions, meaningful phases/keyframes, lower-body distinctions, special actions, transitions, spatial relations, and preserved timing—without any claim that the execution is correct.
