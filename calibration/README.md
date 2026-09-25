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
--output-dir /absolute/output/directory` on already registered native sources.
Pass `--profiles /absolute/profiles.json` when measured intrinsics are available;
the JSON maps camera IDs to `Intrinsics` objects (`fx`, `fy`, `cx`, `cy`, and
optional `distortion`). The command samples native frames at matching source
times, makes a temporal-median background image per view, excludes locally
varying pixels, matches SIFT features across every camera pair, and stores one
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
