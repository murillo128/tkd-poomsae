# Target-based camera calibration

Run `uv run --frozen python -m calibration captures.json --data-root /absolute/data/root`.
The command reads native image files supplied by the operator, detects a shared
ChArUco board, solves camera intrinsics from varied board poses (or uses an
explicit intrinsic profile), solves world-to-camera poses from known board
placements, and stores an immutable `Calibration` artifact under `derived/`.
It does not infer calibration frames or target geometry from Mendeley metadata.

The JSON manifest is versioned. Paths are relative to the manifest or absolute.
Every capture records the SHA-256 of the **encoded file**, its original decoded
size, a crop in original pixels, and clockwise rotation applied after cropping.
Intrinsics are in the final cropped/rotated pixel coordinates. Profile camera,
original size, crop, and rotation must match exactly. `board_to_world` is optional
for a frame; when present it is the row-major 4x4 transform of a stationary,
face-up board on ground `z=0`, in metres. OpenCV's printed board has local +Y
down and local +Z into the paper, so a face-up floor placement has its local +Z
pointing toward world −Z. A camera needs at least one such frame.
Other poses may have no known placement and can still estimate intrinsics.

```json
{
  "version": 1,
  "board": {
    "squares_x": 8, "squares_y": 6,
    "square_length_m": 0.04, "marker_length_m": 0.025,
    "dictionary_id": 0, "legacy_pattern": false
  },
  "captures": [
    {
      "id": "cam-a-floor-001", "source_id": "source-a", "camera_id": "cam-a",
      "image_file": "cam-a/floor-001.png",
      "sha256": "<64 lowercase hex characters>",
      "image_size": [1920, 1080], "crop_xywh": [0, 0, 1920, 1080],
      "rotation_cw": 0,
      "board_to_world": [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
    }
  ],
  "profiles": {}
}
```

Include at least two camera IDs. Without profiles, each camera needs at least
three well-separated board poses, each with eight visible corners; tilted views
and broad image coverage are needed to constrain focal length and distortion.
For a recovery/import route, `profiles` maps camera IDs to objects with
`width_px`, `height_px`, `crop_xywh`, `rotation_cw`, and `intrinsics` (`fx`, `fy`,
`cx`, `cy`, `distortion`). A camera with a profile still needs a known-placement
frame for automatic extrinsics. The artifact labels imported intrinsics per
camera and records per-camera capture IDs, corner count, reprojection error,
pose ambiguity, and source IDs. A failed or ambiguous camera fails the candidate.

`CameraModel` exposes projection, ideal-pixel undistortion and world rays.
World XY is ground, +Z is up; camera +Z is forward; image +Y points down.
Points behind the camera are rejected. Distortion uses OpenCV coefficient order.
This route supplies ground/world alignment evidence to the later ground stage;
it does not perform final ground estimation.

## Natural-scene candidate

Run `uv run --frozen python -m calibration.natural_cli --selection smoke-short
--sync-artifact /absolute/derived/synchronization/ARTIFACT_ID
--output-dir /absolute/output/directory` on already registered native sources.
The synchronization artifact must come from the upstream sync stage, cover the
exact source hashes, and retain the requested cameras. Without it, the command
persists an unavailable result. It samples the intersection of the selected
windows in **global** time (`global = native PTS + effective offset`) and rejects
unverified or misaligned frames. Native PTS and source hashes remain in the
candidate evidence.
Pass `--profiles /absolute/profiles.json` when measured intrinsics are available;
the JSON maps camera IDs to `Intrinsics` objects (`fx`, `fy`, `cx`, `cy`, and
optional `distortion`). The command makes a temporal-median background image
per view, excludes locally varying pixels, matches SIFT features across every
camera pair, and stores one
content-addressed JSON candidate. It never copies or downloads video.

The candidate includes source and sampled-frame identity, pairwise overlap and
degeneracy evidence, relative camera poses, static points, bundle residuals,
coverage, triangulation angles, and conditioning when these can be estimated.
It is **not** an accepted `Calibration`: its first camera and baseline define
only a numerical world/scale gauge. The ground frame and metric scale remain
unresolved. Missing intrinsics are reported as unavailable; resolution alone
does not supply a focal-length prior. All camera intrinsics are held fixed in
bundle adjustment. A later ground and quality stage must evaluate the candidate
before downstream use. `docs/natural-scene-attempt.md` records the initial
registered-data attempt.

## Ground frame and scale resolution

The target route publishes a resolved ground frame and metric scale from the
declared floor placement and measured board square length. Its calibration
artifact records the target capture IDs, source revision, board dimension, units,
and identity source-to-world transform. No ankle/contact inference is used.

For a natural-scene candidate, run `uv run --frozen python -m
calibration.ground_cli --candidate /absolute/natural-scene.json
--sync-artifact /absolute/derived/synchronization/ARTIFACT_ID --data-root
/absolute/data/root` to persist an unresolved calibration with arbitrary-scale
cameras. Add `--evidence /absolute/evidence.json` only when independent evidence
exists. The optional file has `ground` and `size` objects. `ground` supplies an
ID, the candidate's SHA-256 `source_revision`, at least six classified floor
`floor_indices`, `above_indices` establishing the vertical sign, an ordered
`axis_indices` pair fixing +X, ordered `vertical_indices` (lower, upper),
`vertical_reference_id`, `kind` (`scene` or `manual`), and `producer`.
Automatic `scene` evidence must name a `known_upright` reference already in
the candidate's `evidence.vertical_references` with matching point indices,
`source_kind: upright_target`, an identified physical `source_id`, and a
producer distinct from the floor classifier. Both endpoints must have
consistent projected observations in at
least two calibrated cameras. The fitted floor normal must align within 15°
of that independently identified upright direction. An ordinary wall with
points on one side cannot pass this gate. Without such a cue, natural-scene
ground remains unresolved; an operator can instead supply a separately
attributed `manual` recovery revision.
`size` supplies an ID, the same source revision, two `point_indices`, a positive
measured `length`, `unit` (`m` or `cm`), `kind` (`measured` or `manual`), and
`producer`. Manual evidence also requires `author` and `reason`. Indices refer to
the candidate's `static_points` array; a producer must derive floor/sign/axis
classification from independent scene or target evidence. A plane's size alone
never classifies it as floor. Evidence cannot use contacts derived from this
calibration. The SHA-256 is over the candidate's canonical sorted JSON without
whitespace; `calibration.scene_revision()` computes it.

The resolver fits only classified floor points with deterministic RANSAC, checks
inlier fraction, planar coverage, residual, above-floor sign, independent
vertical alignment, and axis direction.
The artifact retains those diagnostics and the single source-to-world transform.
Its camera transforms are expressed in that world frame. A missing or ambiguous
cue leaves ground unresolved; no ground-dependent product may consume it.
Without a measured segment, scale stays arbitrary and metric arrays and ground
measurements remain unavailable. Revisions produce distinct immutable
calibrations while original scene points and observations remain untouched.

## Publication quality gate

`resolve_scene` and `calibration.ground_cli` publish only when the candidate
contains verified synchronization identity, compatible per-view source and
intrinsic evidence, bundle diagnostics, and at least two camera poses with 24
shared static points. The gate recomputes each camera's reprojection residual and
positive-depth fraction from the candidate's retained point observations. It
also checks image coverage, shared-view connectivity, ray intersection angles,
and 3D point conditioning. Every source retained by synchronization must appear
in the candidate; a missing view cannot silently disappear from the project.
A disconnected or inconsistent camera is excluded
with a persisted reason only if at least two independent coherent views remain.
If the remaining pair fails, publication stops. Frontal/lateral names never
provide a pose or orthogonal-camera default.

The defaults below are safety floors. `--thresholds thresholds.json` accepts a
JSON object with these field names to **tighten** limits; the Python API accepts
`QualityThresholds`, and runner calibration settings accept an inline
`thresholds` object. Weaker limits are rejected.

| Field | Default | Meaning |
| --- | ---: | --- |
| `min_shared_points` | 24 | Points observed in two retained views |
| `min_camera_points` | 24 | Correspondences per accepted camera |
| `min_hull_fraction` | 0.03 | Image area spanned by static features |
| `min_grid_cells` | 4 | Occupied cells in the 4×4 image grid |
| `max_camera_p90_px` | 3 | 90th-percentile per-camera reprojection error |
| `max_bundle_p90_px` | 3 | Candidate bundle error when no view is excluded |
| `min_cheirality` | 0.95 | Fraction of observed points in front of a camera |
| `min_angle_deg` | 1 | 10th-percentile triangulation angle |
| `max_condition` | 100 | Ratio of largest to smallest point-cloud singular value |

The persisted `Calibration` records camera, ground, and metric-scale capability
statuses separately. `publication_status` is `complete` only when all three are
resolved; missing ground or measurement evidence yields `partial` with explicit
quality flags. It also retains excluded cameras, candidate/sync/ground/size
evidence identifiers, and up to 24 per-point observed-versus-projected pixel
samples for inspection. Reprojection is an internal consistency diagnostic; it
does not measure MMPose accuracy or real-world 3D accuracy.

The offline runner's `calibration` stage accepts a persisted candidate path and
optional evidence path in `{"calibration":{"candidate":"...",
"evidence":"..."}}`. Its artifact key includes the candidate/evidence file hashes
and synchronization revision. Observation reruns reuse unchanged calibration;
sync or intentional calibration/ground revisions change calibration and
dependent geometry keys without changing registered media or native observations.
