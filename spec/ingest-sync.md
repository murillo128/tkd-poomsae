# Ingest and synchronization specification

## Responsibility

This component owns source-video discovery/validation and mapping each recording onto the global timeline defined in [motion-representation.md](motion-representation.md). It does not calibrate cameras or reconstruct the practitioner.

## Input assumptions

The input is at least two recordings of the same execution.

The component MUST NOT assume:

- recordings start at the same physical time;
- frame zero corresponds across cameras;
- equal frame rates;
- equal resolution or codec;
- a fixed number of cameras.

Cameras are assumed fixed during the analyzed recording.

## Output

The component produces an ingest manifest containing each source and one constant temporal offset mapping source time to global time, together with synchronization confidence and provenance.

A camera may be chosen as timing reference without giving it special geometric authority.

## Automatic alignment

The normal path MUST attempt synchronization automatically.

Available evidence may include:

- shared audio;
- a deliberate visible or audible synchronization event;
- onset of practitioner motion;
- correlation of per-camera pose/motion-energy signals;
- distinctive common motion events across the execution.

The algorithm SHOULD use more than one ambiguous “first movement” cue when richer evidence exists.

An external sync cue may improve robustness but is not mandatory.

## Constant-offset model

For the MVP:

**global_time = source_time + camera_offset**

No clock drift, rate correction, or piecewise temporal warp is required.

Sub-frame offsets are allowed. Downstream consumers may interpolate observations when a requested global time lies between source samples.

## Optional geometric refinement

After pose/calibration data exists, synchronization MAY refine its coarse estimate by seeking better multi-view geometric consistency, provided the final model remains one constant offset per source.

## Execution interval

Synchronization and detection of the actual poomsae start/end are separate concerns. This component aligns cameras; [motion-parsing.md](motion-parsing.md) owns the execution interval.

## Manual override

The automatic estimate, confidence, and any manual offset correction MUST be visible/persisted. Manual adjustment is a debugging/recovery mechanism and must not erase the automatic estimate or masquerade as it.

## Failure behavior

The component MUST fail explicitly if it cannot establish a reliable shared timeline. It MUST NOT silently fall back to equal frame numbers.

A low-confidence camera may be excluded if at least two usable synchronized views remain; exclusions and reasons are persisted.

## Acceptance

Given recordings that begin at different times, the component is accepted when a common physical event can be inspected at the same global time across all retained cameras without pre-trimming the files, and offsets/confidence are persisted.
