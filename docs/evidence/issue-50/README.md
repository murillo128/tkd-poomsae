# Shared-data functional acceptance

This opt-in acceptance uses the already provisioned Mendeley version 1 dataset,
shared model assets and MVP-23 native MMPose windows. It reports functional
capabilities, persistence and reuse; it does not measure prediction accuracy.

`demo-full.json` records offline reuse of the complete initial paired Taegeuk 1
execution. `all-forms-smoke.json` records native 5–7 second windows from all eight
forms. The new catalog entry `all-forms-smoke` keeps original source hashes and
camera IDs and creates no clips. The first form reuses MVP-23 windows; only
missing matching windows invoke MMPose. Missing/ambiguous practitioner and
low-evidence hand states remain in the persisted results.

`functional.json` separates real-data synchronization/calibration/attachment
outcomes from a positive synthetic geometric path. Synthetic inputs are projected
constructed compound motion and supplied metric cameras/ground, not MMPose
predictions. Generated video frames contain those projected dots. Native frame
PTS are read back from the encoded media before observation publication. The
fixture supplies exact zero sync uncertainty and an explicit 0.1 px precision
assumption for its ideal coordinates; these settings are never applied to real
video. Camera calibration estimation from real images is not established by this
supplied-camera fixture. Existing generated-board tests separately exercise the
target calibration estimator.

The synthetic run invokes production alignment, triangulation, articulated fit,
temporal motion, contact, footprint, pivot, ground-view, feature, segmentation,
arm, lower-body and semantic publishers. An empty automatic parser result is a
capability outcome, not a reason to insert action labels. A parser-only semantic assembly
configuration change must leave source, native observation, synchronization,
calibration, raw/final reconstruction and physical ground manifest hashes intact.

All processing harness socket connections are denied and counted. The second
identical processing pass changes cwd to a fresh temporary directory and asserts
zero newly produced immutable artifacts, unchanged existing manifests and identical
outcomes. The observation harness additionally counts model/inference calls and
requires no new calls on the second pass. Dataset integrity verifies all 32 files
against the acquisition receipt; model integrity checks every pinned registry
asset. No acquisition/bootstrap command runs during acceptance.

The interruption receipt records cooperative cancellation at the second safe
check and resume without changing retained complete window hashes. For this
initial execution, the first view was already cached, so cancellation occurred
before the second view. This is not an OS process-kill experiment. Default unit
coverage independently exercises interruption during an unfinished window.

## Reproduce on the provisioned host

Use one absolute `TKD_DATA_ROOT` for both checkouts/cwds. Provision developer
packages once with `uv sync --frozen --group dev` and `cd web && npm ci`; use Node
22+. These developer setup commands are outside the offline processing experiment.
The shared vision runtime/model/data provisioning is described in the existing
vision and dataset documentation; acceptance never silently provisions them.

From the checkout:

```sh
export TKD_DATA_ROOT="${TKD_DATA_ROOT:-$HOME/.local/share/tkd-poomsae}"
export PYTHONPATH="$PWD/src:$PWD"
"$TKD_DATA_ROOT/runtime/vision/bin/python" -m pose.acceptance demo-full \
  --output docs/evidence/issue-50/demo-full.json
"$TKD_DATA_ROOT/runtime/vision/bin/python" -m pose.acceptance all-forms-smoke \
  --output docs/evidence/issue-50/all-forms-smoke.json
.venv/bin/python -m tests.functional_smoke \
  --receipt "$(python3 -c 'import json; print(json.load(open("docs/evidence/issue-50/demo-full.json"))["receipt_path"])')" \
  --receipt "$(python3 -c 'import json; print(json.load(open("docs/evidence/issue-50/all-forms-smoke.json"))["receipt_path"])')" \
  --output docs/evidence/issue-50/functional.json
cd web
npx tsc --project tsconfig.local.json
npx playwright test --config=playwright.local.config.ts
```

The initial observation experiment used `--interrupt` for `all-forms-smoke`.
Omit it once that complete selection receipt exists: the harness deliberately
fails rather than claiming interruption was tested on an immediate receipt hit.

The explicit local browser configuration starts the actual viewer and existing
project service on loopback ports 5350/18550. It does not generate projects at
service startup. Its eight real-video cases verify visible exact frames with
MMPose overlays, raw scores, landmark selection, reopening and unavailable 3D.
A separate synthetic case verifies visible 3D/metric ground, selection and
reopening. Browser requests are restricted to loopback. CI continues to run the
portable generated-fixture suites; it cannot access the host's shared dataset.

Large media/arrays remain in shared storage under
`$TKD_DATA_ROOT/runs/functional-issue-50/` and `$TKD_DATA_ROOT/derived/`.
Inspection bundles there retain complete artifact keys/lineage. Screenshots and
Playwright traces remain in ignored `web/test-results/local-smoke/`; compact
receipts and this report are the committed evidence.

Automatic synchronization can fail on a form. Such a failure is recorded; no
manual cross-camera offsets are invented. For native inspection only, a separate
artifact names the frontal source as the timing reference with an unknown
automatic estimate. Reference zero defines that source's own clock. Other views
remain excluded with unknown offsets and the solver's failure reason. This
supports genuine frontal 2D inspection but is not paired synchronization or a
calibrated result.

The real viewer experiment encountered overlapping wrist/hip hit targets at the
sampled Taegeuk 1 frame. Keyboard focus/Enter selects the retained wrist evidence;
the smoke suite exercises that supported route. Pointer access to every dense
overlapping landmark is not established by this acceptance.

## Observed results (2026-09-26)

- Full initial pair: 1,536 retained native observations, all reused from MVP-23;
  zero detector/wholebody/hand calls on this acceptance run.
- Eight-form smoke: 976 native observations in 32 windows, 972 selected frames,
  with usable body evidence in every view. The first pair's 122 smoke frames were
  reused. Forms 2–8 required 854 detector and 854 wholebody calls, 1,945 hand
  calls and 14 camera model sessions on CPU. The initial bounded run took
  901.748 seconds. These counts establish execution, not prediction accuracy.
- All native receipts reloaded offline. The identical second-cwd observation
  pass made zero additional inference/network calls. Cooperative resume preserved
  all 62 then-existing window manifests.
- Automatic paired synchronization returned estimates for seven forms. Taegeuk 3
  returned `manual timeline needs two attributed views`; its frontal native-clock
  fallback is explicitly separate from automatic synchronization.
- Every real calibration attempt was unavailable: seven reported insufficient
  matched texture and Taegeuk 3 lacked paired synchronization. Real camera poses,
  scale, calibrated ground and full 3D demonstration remain unavailable.
- The separate supplied-camera synthetic run produced 141 triangulated/final
  samples and metric ground snapshots through the production publishers. Semantic
  publication completed with zero steps/actions; it establishes no positive
  semantic-label result. Parser-only assembly configuration changes preserved
  physical/native/source hashes.
- Identical application processing from the second cwd produced zero new immutable
  artifacts, retained identical capability outcomes and verified 315 published
  artifact manifests with zero network-connect attempts. All 32 original dataset
  files (293,982,328 bytes) and all 14 pinned model assets passed integrity checks.

Validation used Python 3.11.16, Node 22.20.0 and headless Chromium with software
WebGL. The default Python suite passed 584 tests (six opt-in tests deselected);
all four registered local/full-data plumbing tests passed. Ruff and mypy (152
source files), 24 web unit tests, the web build, contract TypeScript and local
browser TypeScript checks passed. The existing generated-media browser suites
passed all 38 cases. All nine shared-project viewer cases passed in 80.618 seconds
with no skips or retries; `browser.json` records their results and screenshot
hashes. The 30-second media wait is a test bound, not a latency guarantee.
The normal bundle-size advisory remains. Local model/data
acceptance is provided by these explicit harnesses rather than default CI markers.

CUDA, Firefox/WebKit, full inference on forms 2–8, process-kill recovery, model
accuracy benchmarks and CSV ground-truth comparisons were not exercised. No
external resource blocker remains for this bounded smoke contract; real geometry,
synchronization and empty-parser outcomes are capability gaps retained in evidence.
