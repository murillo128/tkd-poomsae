Ground browser acceptance uses a small offline fixture exported from the shared
synthetic pivot scenario and the authoritative ground-view producer:

```
uv run --frozen python -m tests.fixtures.export_ground_view
cd web
npm ci
npx playwright install chromium
npm run test:browser
```

No dataset or model is downloaded. Tests mock only the HTTP transport and inspect
the real React/SVG app in Chromium. They verify XY projection and foot orientation,
fixed bounds, dynamic/summary agreement, shared selection, layer visibility,
unresolved scale and unavailable ground. Continuous playback also runs with
80 ms snapshot latency (longer than the 50 ms clock tick), verifying visible,
advancing native geometry and exact paused seeks. The fixture retains six native snapshots
and original complete placement/pivot geometry, rather than estimating geometry
in a test or browser. Python transport tests separately cover all native rows,
gaps, pagination, revision conflicts and stale clocks.

Screenshots (`dynamic.png`, `summary.png`, `arbitrary.png`) are bounded to the
ground panel and stay outside Git, under `/tmp/tkd-ground-browser-evidence` by
default. Set `GROUND_EVIDENCE_DIR` to retain them elsewhere; `CHROMIUM_PATH` may
select a locally provisioned browser. Set `GROUND_PORT` when another inspection
session owns the default test port. CI installs Chromium and runs this suite.

Timeline acceptance also starts a disposable Python service on port 5197 using
`tests.fixtures.serve_timeline`. Install the frozen Python dev environment with
`uv sync --frozen --group dev` before running the browser suite. It constructs
small ordinary, two-track SpecialAction and parsing-unavailable projects from
local synthetic motion. HTTP is redirected to that service; semantic reads,
boundary/keyframe apply, undo, reset, optimistic revision conflicts and persistence
all use the real application API and ArtifactStore. No vision, external dataset or
model download occurs. The service hashes every file in both original automatic
artifacts and reports their preservation after browser edit round trips.
Coincident and nonuniform keyframes navigate by stable ID and global time. The
suite also verifies six tracks, independent overlapping bounds, shared selection,
keyboard/pointer controls, zoom/pan, explicit drafts, retained stale/invalid edits,
and raw track/time access without parsing. The timeline screenshot remains under
the configured browser evidence directory outside Git.

A fractional long-extent regression supplies synthetic metadata bounds of
2.2–65 seconds while sending every semantic window read to the unchanged real
service. It checks successful loads, gap-free adjacent windows for all five
collections, selectable canonical entities, and rejection of the old nominal
30-second request (whose actual floating-point duration exceeds 30 seconds).
The client uses 29-second windows to leave rounding headroom under the API cap.

Integrated acceptance (`npm run test:integrated`, also mandatory in
`npm run test:browser`) starts Vite and the actual FastAPI application with a
fresh temporary ArtifactStore via `tests.browser_service`. Its projects have
2/3/4 generated H.264 clips at 25/30/20/24 Hz, nonzero offsets, rotated pixels,
low-confidence and excluded views. `tests.fixtures.integrated_browser` supplies
known 50 Hz synthetic geometry, ground products and overlapping synthetic
semantic actions with one linked event. These are deterministic test inputs,
including handcrafted semantics in the canonical assembly persistence format;
they are **not MMPose output or evidence of reconstruction/parser accuracy**.
Ground products run through the normal contact/footprint/pivot/view publishers.
The integrated suite and its server configuration are type checked before the
browser starts.

`e2e/integrated.spec.ts` verifies positive rendering on the same instant across
source images/2D overlays, WebGL 3D, ground and timeline; action/event/placement
selection and entity-to-source frame provenance; native stepping; persisted
parser editing/reload/undo; real-service error recovery; missing calibration and
scale; keyboard selection/focus, named controls and textual uncertainty at
1024/1280/1600 px; delayed cross-project responses, bounded API windows and
repeated project switching without accumulating blob URLs, timers, canvases or live WebGL contexts.
The camera and provenance suites additionally cover playback video, rapid seek
responses, missing/mismatched observations, excluded views, sync edit/reload,
server rejection and genuine concurrent-revision conflicts. The timeline suite
covers genuine stale parser revisions and invalid edits.

Set `TKD_BROWSER_WEB_PORT` and `TKD_BROWSER_SERVICE_PORT` to use isolated ports
(defaults 5173/18044). Tests always start their own servers and never reuse a
running app. Chromium is the tested browser; Firefox/WebKit are not claimed.
The integrated desktop screenshot and failure-only screenshots/traces are kept
in ignored `web/test-results/integrated`; component output uses
`web/test-results/components` so it cannot erase integrated evidence. CI retains
all browser evidence (including ground output) for seven days. Application CI
runs once on PRs to `main` and `codex/epic-issue-*`, without a duplicate push
trigger or changes to SkillForge dispatch. It generates all fixtures locally and
never downloads a dataset or model weights.
