# Integrated browser acceptance

Issue #49 adds ten integrated cases to the real-service camera/provenance suite.
The complete browser command exercises 38 cases: 23 integrated/camera/provenance
and 15 ground/timeline/3D component cases. No browser test is skipped.

## Inputs and scope

All media and persisted projects are generated locally in temporary storage.
No Mendeley data, model weights, GPU inference or MMPose result is used. The
positive geometry is the synthetic pivot scenario with a synthetic wrist and
known native samples; the two overlapping actions and linked event are
handcrafted canonical semantics, explicitly named `synthetic-*`. They exercise
inspection/persistence, not reconstruction or parser accuracy. Ground products
use the normal contact, footprint, pivot and ground-view publishers. Camera
rates, offsets, exclusions and native identity remain independent of motion time.

The real FastAPI service receives normal reads, parser and synchronization writes.
Test transport interception is limited to deliberate outages or delayed responses;
the separate component suites also use their documented synthetic transports.

## Observed validation (2026-09-26)

Environment: Python 3.11.16, Node 22.20.0, Playwright 1.63.0, headless Chromium
153.0.8010.12, software WebGL via SwiftShader. Firefox/WebKit were not exercised.

- `pytest`: 584 passed; six dataset/model-dependent tests deselected by the normal
  repository marker policy. No model/dataset acceptance claim is made.
- `ruff check src tests contracts storage media sync calibration pose reconstruction`:
  passed.
- `mypy`: passed (150 source files).
- `npm run test`: 24 passed in six files.
- `npm run build` and TypeScript contracts check: passed. The existing Vite bundle
  size advisory remains; there is no throughput benchmark in this acceptance.
- New integrated test/config TypeScript check: passed.
- `npm run test:integrated`: 23 passed.
- `npm run test:components`: 15 passed.

Local browser runs used isolated web/service ports 5249/18449, ground port 5248
and 3D port 5247; the component timeline service owned port 5197. Reproduce with
`uv sync --frozen --group dev`, `cd web`, `npm ci`,
`npx playwright install chromium`, then `npm run test:browser`.

The cases verify shared video/image/2D/3D/ground/timeline time, event/action/ground
selection, entity-to-source evidence, native stepping, playback/pause, rapid
seeks, parser apply/reload/undo, sync writes and conflicts, missing geometry and
scale, excluded/low-confidence cameras, server errors, delayed project responses,
API bounds, resource release, and keyboard/focus/control labels with textual
uncertainty at desktop widths 1024, 1280 and 1600 px.

The representative 1280 px screenshot is retained under ignored
`web/test-results/integrated/*/integrated-desktop.png`. Failure screenshots/traces
are retained only on failure. Ground evidence defaults to `/tmp`; CI places it
under the browser artifact directory. No large media, screenshots or traces are
committed. Application CI runs these mandatory checks on PRs to the default and
epic integration branches with no duplicate push event; SkillForge dispatch is
unchanged.
