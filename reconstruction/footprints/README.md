# Physical footprints and placement geometry

`derive_footprints(final_motion, contacts, calibration, morphology=None,
config=None)` consumes aligned native `Reconstruction` and `Ground` metadata.
Motion must be the final `reconstruction.temporal` product; contact/support is
consumed without reclassification. Inputs must share motion, calibration, scale,
participant and native times. Resolved usable ground is required. Coordinates
already belong to the right-handed world XY ground plane; the calibration
transform is never applied again.

```python
from reconstruction.footprints import publish_footprints, load_footprint_evidence

placements = publish_footprints(
    store, final_motion_handle, contact_handle, calibration_handle, morphology_handle
)
evidence = load_footprint_evidence(placements)
```

Publication creates an independent immutable `Ground` artifact, retaining native
contact/support samples and the original contact-evidence array. It adds canonical
footprints and a versioned `footprint_evidence_json` uint8 array. Its cache identity
includes exact input manifest hashes, configuration and `footprint-v1`. Supplied
motion/calibration revisions must match those used by the contact publisher,
even if their metadata IDs are reused. No models, datasets, vision, or contact
inference are run by this stage.

## Stability and moving contact

A stable placement requires usable heel/forefoot geometry and known contact with
quality, uncertainty and source links. Defaults are diagnostic synthetic settings:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `enter_distance` | 0.015 | Maximum landmark displacement during confirmation |
| `leave_distance` | 0.03 | Maximum displacement retained after confirmation |
| `enter_angle_rad` | 0.06 | Maximum wrapped angle change during confirmation |
| `leave_angle_rad` | 0.12 | Maximum retained angle change |
| `stable_seconds` | 0.12 | Required elapsed native evidence |
| `max_gap_seconds` | 0.15 | Maximum connected sample spacing |
| `uncertainty_multiplier` | 2 | Conservative displacement/angle margin |

Distance thresholds default to `threshold_units="world"`: metres for resolved
metric calibration, explicitly arbitrary world units otherwise. They are never
silently treated as metres with unresolved scale. For scale-invariant thresholds,
select `threshold_units="body_normalized"` and supply morphology. This uses the
named measurement (`left_shank` by default), with matching world unit, source
links, positive value and non-unknown quality. Its relative uncertainty must be
at most 0.1 by default. A conservative value-minus-uncertainty denominator is used.
Missing/unreliable normalization makes stability unavailable; raw axes and
contact/landmark evidence remain accessible.

Confirmation compares shared usable landmarks and the wrapped heel-to-forefoot
angle against a fixed candidate anchor, including uncertainty margins. Comparing
only adjacent samples would allow gradual sliding to appear stationary. Once
confirmed, wider leave thresholds suppress jitter. Relocation or rotation beyond
those thresholds ends the placement. Motion below these tolerances cannot be
resolved as sliding; this is bounded geometric stability, not a friction claim.
Each axis endpoint must introduce native sources not already seen in the connected
contact run. Repeated render queries cannot supply a stability dwell.

A stable event's interval brackets its first and last native candidate samples;
`confirmed_seconds` separately identifies the native sample at which dwell was
satisfied. Its representative pose is the first evidenced pose, with positional
uncertainty covering retained drift. It does not imply constant exact coordinates
throughout that interval. No interval is extended across unknown contact,
no-contact, missing axes, stale evidence or gaps. Isolated samples remain in the
dense trajectory without an invented duration.

Contact with unconfirmed stability or actual movement is grouped separately as
`kind="moving"`, with null canonical position/orientation and no frozen rendering
geometry. Its full changing trajectory remains available. Its reason is
`moving_or_unconfirmed_contact`; it does not claim a sliding classification when
there was insufficient dwell. Unknown/occluded evidence retains null geometry,
reasons and source references, and never creates a confident stationary footprint.

## Rendering, confidence and source navigation

A foot position is the midpoint of heel and forefoot; yaw is their ground-plane
axis direction in radians. Available heel, forefoot and outer-foot points are
retained individually. An axis constrains neither sole width nor a complete
polygon. This implementation emits axes/points only, `polygon=null`, and
`approximated=false`. Any future polygon derived from these landmarks must be
explicitly marked approximated. Degenerate/uncertain axes remain unavailable.
Quality is inferred positional uncertainty and contributor IDs, not an accuracy
probability. Angular uncertainty is retained per native trajectory sample.

Every supplied native time has a left and right trajectory row, including missing
regions. Rows retain motion/contact sample indices, time, anatomical side,
original landmark quality/source IDs, contact region/quality, support and rendering
geometry. Each event has deterministic IDs and indices into those dense rows;
rows link back to their event ID. `reconstruction_id` and `contact_id` identify
the original artifacts. Loaders validate canonical footprint metadata, native
contact/sample alignment, event intervals and bidirectional trajectory links.

## Placement relations

For each stable event, relations are emitted to the latest prior placement on
each side. These are historical geometry comparisons; they do not claim
simultaneous support. `measure_placements` also permits explicit relevant pairs.
The origin is the first placement's center, longitudinal axis is its foot axis,
and positive lateral is its left-hand perpendicular in the XY plane. Stored
axes make this reference frame explicit. Measurements include distance and signed
longitudinal/lateral separation. Relative alignment is displacement bearing minus
first-foot yaw; relative foot angle is second-foot yaw minus first-foot yaw.
Angles wrap to [-pi, pi]; alignment is null for coincident centers.

Lengths are tagged `m` only with resolved metric calibration, otherwise
`arbitrary`. Reliable matching morphology adds separate `body_ratio` values with
its uncertainty and provenance; arbitrary values are never relabeled metric.
Angles remain available regardless of scale. No semantic `SequenceStep`, stance,
pivot classification, technical correctness, or score is produced.

Offline acceptance:

```sh
uv run --frozen pytest tests/test_footprints.py tests/test_ground_contact.py
```

Tests cover stationary jitter, relocation, moving contact, settling, occlusion,
axis-only rendering, uncertainty, repeated sources, gaps, hysteresis, native-rate
variation, coordinate/angle-wrap invariance, arbitrary/body-normalized units,
immutable publication, exact revision rejection and deterministic reprocessing.
