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
