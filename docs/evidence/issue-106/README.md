# Controlled reconstruction-to-semantics acceptance

This is a positive **software integration proof**, not a real-video, MMPose,
calibration or parser accuracy benchmark. No Mendeley/MediaPipe data, model
inference, manual labels or expected semantic entities enter this fixture.

`tests/fixtures/controlled_acceptance.py` generates two deterministic 25 Hz
four-second clips and their native observation windows. A stationary torso,
fixed-length arm extensions and a right-leg lift/extension/retraction overlap.
The lower-body chain uses two-bone inverse kinematics; heel, forefoot and outer
foot points provide actual physical contact evidence. Quiet motion before and
after the movements gives the production segmenter observable boundaries.

The fixture supplies metric camera/ground geometry and zero-offset clocks.
Its observations are analytic distorted-pixel projections, **not** detections
from the encoded dots. The triangulator uses an explicitly declared 0.001 px
numerical precision floor for these exact projections, with zero supplied sync
uncertainty and calibration residual. This does not estimate real detector or
camera uncertainty. Default feature, segmentation, arm, lower-body, assembly,
articulated, temporal and ground configurations remain unchanged.

All downstream artifacts come from the production publishers:

1. Native observation persistence and alignment.
2. N-view triangulation, participant morphology, articulated fit and temporal
   reconstruction with detailed geometry and dynamics.
3. Contact, footprints, pivots and ground-view products.
4. Feature extraction, execution/step segmentation, arm and lower-body parsing,
   and semantic assembly.
5. Normal inspection registration, including verified upstream lineage.

The recorded run in [controlled.json](controlled.json) contains 101 reconstructed
and ground samples, one automatic step, three arm actions, one kick, eleven
unknown transition actions and 58 keyframes. Arm and kick intervals overlap;
actions carry step ownership and physical sample links. The unknown transitions
are preserved automatic parser output, not confident action labels. Two runs
against the same root yielded identical reports and artifact manifest hashes.
Dense arrays and video files remain outside Git.

`tests/test_controlled_acceptance.py` checks the positive hierarchy, overlapping
arm/kick intervals, inferred quality, physical producer lineage, and normal
inspection APIs for steps, actions, keyframes, linked physical/native camera
evidence and ground snapshots. A parser configuration change reruns all five
parser publishers while upstream publishers are forbidden. It asserts every
source, observation and physical artifact payload remains byte-identical, and
returning to the default parser configuration reuses the original artifacts.

`web/e2e/controlled.spec.ts` reads the generated entities from the real service,
opens the normal viewer, navigates arm/kick actions by keyboard and a generated
keyframe by mouse, and verifies that timeline, 3D, ground and native camera frame
share the selected time. It also follows the physical sample and camera evidence
links. Chromium saves `controlled-automatic.png` in its test output directory;
CI retains that output via the existing browser-evidence artifact.

## Reproduction

```sh
uv sync --frozen --group dev
uv run --frozen python -m tests.fixtures.controlled_acceptance \
  --root /tmp/tkd-controlled-106
uv run --frozen pytest -q tests/test_controlled_acceptance.py
cd web
npm ci
npm run test:integrated -- --grep controlled
```

The module writes `report.json`, `bundle.json`, source clips and the normal
artifact store under the supplied root. To inspect that persisted run manually:

```sh
TKD_DATA_ROOT=/tmp/tkd-controlled-106/store \
  TKD_ALLOWED_SOURCE_ROOTS='{"controlled":"/tmp/tkd-controlled-106"}' \
  uv run --frozen tkd-poomsae serve --host 127.0.0.1 --port 8000
```

Start the web app with `VITE_API_ROOT=http://127.0.0.1:8000 npm run dev`, then choose
`controlled-acceptance`. The browser suite instead creates an ephemeral copy in
its existing fixture service and owns its servers. Set `TKD_BROWSER_WEB_PORT`
and `TKD_BROWSER_SERVICE_PORT` when the default ports are occupied.

## Validation

The focused Python integration check and real Chromium navigation check passed.
Repository-wide validation results are recorded in the PR; the committed JSON
retains the technical artifact identities, without transient CI bookkeeping.
