# Synthetic multiview fixtures (recipe version 1)

`tests.fixtures.synthetic` is a test harness. It needs only Python 3.11's standard
library and never downloads data or uses a GPU. `make_scene(name)` returns a
repeatable scene. `scene.source_manifest()` and `scene.render_ppm(camera_id,
frame_index)` are the source-facing media recipe. `scene.oracle()` and
`scene.write_oracle(path)` are test-only expected data; keep them out of pipeline
input. The oracle contains camera transforms, intrinsics, distortion, offsets,
global frame times, dense 3D and ideal 2D trajectories, morphology, segment
orientations, calibration features, contacts and semantic events.

World XY is the floor and +Z is vertical. Camera +Z looks into the scene;
image +Y points down. Native PTS is `frame.pts / frame.time_base_den` seconds.
The oracle's global time is source time plus its camera offset. Frames from
different cameras with the same index are generally different physical times.
The generated target is a six-color floor grid with known spacings in the
source manifest. Static non-coplanar features are separate. A real calibration
implementation may use the target definition, but must infer camera placement
from images; the oracle camera/world geometry is reserved for assertions.

| Recipe | Intended evidence or failure | Numerical tolerance in fixture tests |
| --- | --- | --- |
| `clean_two`, `clean_three`, `clean_four` | 2/3/4 calibrated views, target and 3D scene features | ideal projection and transform residual < `1e-10` pixels/world units |
| `mixed_timing` | four rates, resolutions, fractional offsets and native PTS | native PTS/time-base and global mapping < `1e-12` seconds |
| `noisy_missing` | independent 1.5 px Gaussian noise and 12% missing points | compare against oracle with an application-specific robust tolerance; do not expect exact reprojection |
| `outliers` | deterministic 8% point displacement | corrupted points differ by > 30 px before image clipping |
| `bad_camera` | third camera observations shifted by `(35, -24)` px | residual > 30 px on retained sample points |
| `degenerate_baseline` | cameras clustered within 4.5 cm | first-pair ray angle < 0.01 rad at scene center |
| `unknown_scale` | metric scale absent | `metres_per_world_unit is None`; metric outputs unavailable |
| `low_texture` | no static features or calibration grid | feature count = 0; fail calibration observability |
| `occluded` | right-side actor points hidden in camera 0 during kick | missing in camera 0, retained in camera 1 at 1.2 s |

The tolerances describe synthetic arithmetic and injected errors, not real-world
accuracy or MMPose performance. The left forefoot remains fixed during the
support pivot (< `1e-10` world units); the heel travels > `0.05` world units.
Shoulder/hip widths and each upper-arm, forearm, thigh, shin and foot length
match the exported morphology within `1e-10` world units at every dense sample.
Events include pre/post-roll, overlapping independent arm actions, kick,
recovery/placement, pivot, a coordinated two-arm action and left-forearm-front
crossing. Morphology is constant while trajectories and root/head orientation
change with time.

Generate media files outside Git under the shared root when needed:

```python
import json
import os
from pathlib import Path
from tests.fixtures import make_scene

scene = make_scene("mixed_timing")
root = Path(os.environ.get("TKD_DATA_ROOT", Path.home() / ".local/share/tkd-poomsae"))
out = root / "synthetic" / scene.recipe.name / f"v{scene.recipe.version}"
out.mkdir(parents=True, exist_ok=True)
(out / "source-manifest.json").write_text(json.dumps(scene.source_manifest()))
scene.write_oracle(out / "test-only-oracle.json")
for camera in scene.cameras:
    for frame in camera.frames():
        (out / f"{camera.id}-{frame.index:05d}.ppm").write_bytes(
            scene.render_ppm(camera.id, frame.index)
        )
```

The manifest is the deterministic video recipe: each camera has its own frame
index, native PTS/time base, rate and resolution. The PPM frames can be encoded
with any chosen external video tool while preserving those PTS; a constant-FPS
encoder alone may discard the mixed-rate timing evidence. Bulk PPM/video outputs
and generated oracle arrays belong under `TKD_DATA_ROOT`, not in Git.
