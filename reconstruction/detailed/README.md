# Detailed reconstructed geometry

This module derives geometry from the supplied `MotionSample` without editing it.
`derive_sample(sample, reconstruction_id=..., representation="raw" | "fitted")`
returns a versioned `DetailedSample` at the same global time. Temporal consumers
can call it again on their final landmarks; results are never silently reused
from an earlier fit. Pure helpers expose individual regions, arbitrary named
landmark relations, and forearm crossings.

```python
from reconstruction.detailed import derive_sample

geometry = derive_sample(
    reconstruction.samples[0],
    reconstruction_id=reconstruction.id,
    representation="fitted",
)
```

## Frames and supported quantities

All quaternions are active local-to-parent rotations in canonical `[w,x,y,z]`
order. Frames include world origin and explicit `parent="world"`. Directions
are unit world vectors; a direction alone does not imply an axial twist.

- Body: origin at hip midpoint; x from anatomical left hip to right hip; z is
  shoulder-midpoint minus hip-midpoint, orthogonalized against x; y = z cross x.
  Thus x is anatomical right, y front, z torso-up. Above/below uses this body's
  up axis, not ground vertical. Sides come from landmark labels, never image
  positions. Four sufficiently precise hips/shoulders are required.
- Hand: origin wrist; x pinky MCP to index MCP; y wrist to their MCP midpoint,
  orthogonalized against x; z = x cross y. The normal's sign follows this
  labeled convention; no independent skin-surface/palmar-side claim is made.
  The wrist-to-MCP midpoint axis remains available without a full frame.
  Wrist alignment is the bend between elbow-to-wrist and wrist-to-MCP midpoint.
- Foot: origin heel; x heel to forefoot; y toward medial geometry on the left,
  outer geometry on the right, orthogonalized against x; z = x cross y. Both
  sides use a consistent signed convention. Heel/forefoot alone supports the
  longitudinal axis; missing outer geometry leaves full orientation unknown.
  This does not infer ground contact or sole surface orientation.
- Head: origin ear midpoint; x left ear to right ear; y midpoint to nose,
  orthogonalized against x; z = x cross y. The head-to-torso rotation uses
  `body_rotation.T @ head_rotation`. Neck-to-head direction is separately
  expressed in the torso frame when both named points are supported. No roll,
  twist, or pointing direction is fabricated from an insufficient pair.

Finger configurations retain both interior joint bends (zero = straight):
MCP/PIP/DIP and PIP/DIP/tip for non-thumb digits, CMC/MCP/IP and MCP/IP/tip
for thumb. An extended digit requires both bends plus uncertainty margins to
be below `open_max_flexion`; flexed requires both minus margins above
`fist_min_flexion`. Other cases are indeterminate. `open` requires all five
digits extended; `fist` requires all five flexed. These conservative labels
only describe the measured bends, and do not classify or judge martial hand
shapes. They do not infer missing digits or knuckle flexion outside those bends.

## Relations and uncertainty

`point_relations` compares arbitrary named landmarks along body x/y/z.
`forearm_crossing` intersects elbow/wrist segments projected into the body's
right/up plane, then interpolates **body-front depth at the intersection**.
It distinguishes opposite depths with identical projections. This is a
body-relative crossing descriptor, not camera overlap or physical contact.
Parallel, foreshortened, nearly collinear, and uncertain endpoint intersections
are unknown. Confidently separated projected segments are `not_crossed`.
A supported crossing can have unknown `front_entity`: crossing evidence and
front-order evidence have separate quality fields.

Every quantity retains source IDs and uncertainty. Non-unknown landmarks from
raw, interpolated, or fitted geometry can contribute only with source IDs and
explicit positional uncertainty. Unknown upstream geometry is never filled.
Uncertainties are conservative diagnostic margins, **not calibrated accuracy**:
position errors sum without assuming independence; directions divide by length;
frames additionally penalize short and nearly parallel axes; relative rotations
sum angular uncertainty. Relations include frame uncertainty and require
separation beyond the configurable uncertainty multiplier. Orientation/direction
and bend uncertainty is radians; relation uncertainty is in source world units.
No detector score becomes a correctness probability.

## Immutable publication

`publish_detailed_geometry(store, handle, representation=...)` accepts existing
raw triangulation or articulated-fit publishers, verifies that representation
matches the producer, and preserves all landmarks, existing frames and source
arrays in a new reconstruction artifact. `detailed_geometry_json` is a separate
versioned byte-array payload; `load_detailed_geometry` validates its sample
models. It records source manifest hash, reconstruction ID, raw/fitted role,
world units, algorithm revision, configuration, source evidence and global time.
The artifact cache key includes the exact source revision and configuration.
Future temporal publishers can use the pure helpers without these publisher
role restrictions. This change does not add a pipeline stage or replace
existing articulated orientations.

Run synthetic acceptance with:

```sh
uv run --frozen pytest tests/test_detailed_geometry.py
```

No dataset/model download, accuracy measurement, facial expression, force,
impact, scoring, or contact inference is part of this module.
