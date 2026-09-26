# Local setup and inspection runbook

This guide describes delivered commands and the observed inspection demo.
Requirements and component ownership remain in [spec/](../spec/README.md).
The application observes motion; it makes no scoring, coaching or accuracy claim.
The [final requirement matrix](evidence/issue-52/README.md) evaluates #52 using
separate real-data and controlled positive evidence tracks, as explicitly adopted
in that issue. Real Mendeley camera/world/3D unavailability remains visible;
capture-suitable real validation (#107) and one-command producer wiring (#108)
are follow-ups. Controlled evidence establishes software integration, not accuracy.

## Environment and shared storage

Use Linux x86_64, Python 3.11 and `uv` for the exercised core/CPU vision path,
and Node 22.12+ with npm for the viewer. Select Node 22 explicitly if the host's
default is older. From the checkout, set one persistent absolute root in the host
environment, inherited by every terminal, worktree and executor:

```sh
export TKD_DATA_ROOT="${TKD_DATA_ROOT:-$HOME/.local/share/tkd-poomsae}"
uv sync --frozen --group dev
uv run --frozen tkd-poomsae --help
uv run --frozen tkd-poomsae doctor
cd web
npm ci
cd ..
```

The core lock installs media decoding, storage, analysis APIs and the local
service, without PyTorch/MMPose. `doctor` is diagnostic: inspect its JSON for
asset verification, interpreter, runtime and CUDA availability; exit zero alone
does not mean vision is ready. Persisted inspection needs no vision interpreter.

| Shared namespace | Contents and lifetime |
| --- | --- |
| `datasets/` | Verified original videos/CSVs and acquisition receipt; do not edit originals |
| `models/` | Pinned local configs/checkpoints and device leases; never copied per issue |
| `runtime/vision/` | Separate locked Python 3.11 vision interpreter |
| `derived/` | Immutable artifacts, native observation receipts, opt-in media variants and bounded preview caches |
| `runs/` | Project/source references, mutable run state, inspection indexes, edit history and service jobs |

The unset root defaults to `~/.local/share/tkd-poomsae`. Never put it inside a
checkout, review worktree or `/tmp`. Source references and model/runtime paths
must remain accessible from later worktrees. A project registration refers to
original files; it does not copy or re-encode them.

## Explicit one-time provisioning

The following commands may acquire resources on a new host. Run them once at
the shared root, then verify/reuse them across issues:

```sh
uv run --frozen tkd-poomsae datasets bootstrap mendeley-bjy7vr4xkt-v1
uv run --frozen tkd-poomsae datasets verify
uv run --frozen tkd-poomsae datasets status
uv run --frozen tkd-poomsae models runtime-bootstrap
uv run --frozen tkd-poomsae models bootstrap
uv run --frozen tkd-poomsae doctor
```

[Dataset bootstrap](dataset-bootstrap.md) documents archive pins, locking,
atomic extraction, offline cache hits and explicit `--repair`.
[Vision setup](../vision/README.md) owns runtime/model pins and licences.
The CPU reference is exercised; CUDA is unavailable on this host and needs
separate driver/wheel validation. `TKD_VISION_PYTHON`, if used, must name an
existing compatible absolute interpreter. The core automatically delegates
observation commands to the shared runtime; never install the core's NumPy 2
dependencies into that NumPy 1 vision environment.

The dataset is QingWei Zheng's [Mendeley version 1](https://doi.org/10.17632/bjy7vr4xkt.1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Retain attribution and
licence when sharing demos. It is only for integration, demos and software tests.
Its MediaPipe CSVs are detector output, not ground truth: no training, held-out
validation, accuracy evaluation, ranking or threshold tuning against these CSVs.

## Import and reproduce the demo

From the checkout after provisioning:

```sh
uv run --frozen tkd-poomsae datasets inventory --output /tmp/tkd-inventory.json
uv run --frozen tkd-poomsae datasets register
uv run --frozen tkd-poomsae datasets selections demo-full
uv run --frozen tkd-poomsae observations demo-full --device cpu
```

Inventory measures native PTS and hashes; registration creates the eight stable
Mendeley projects. The selection references the complete first paired execution
without making clips. Observation output reports the shared receipt path and
cache status. Repeating it with unchanged source/model/settings reuses verified
native windows. See [import diagnostics](dataset-import.md).

The small [demo manifest](demo/inspection.json) pins original source hashes,
the native observation receipt and shared inspection bundles. Paths are relative
to `TKD_DATA_ROOT`, not a particular user's home. It contains no media, weights
or dense arrays. It describes the delivered issue-50 acceptance configuration;
it is an inventory, not a CLI `--config` file.

On a host without its shared bundles, reproduce them with the existing offline
acceptance harness. This also exercises the synthetic physical publishers and
a parser-only configuration change, then repeats processing from another cwd.
Run from the checkout; reports go outside Git:

```sh
export PYTHONPATH="$PWD/src:$PWD"
"$TKD_DATA_ROOT/runtime/vision/bin/python" -m pose.acceptance demo-full \
  --output /tmp/tkd-demo-full.json
"$TKD_DATA_ROOT/runtime/vision/bin/python" -m pose.acceptance all-forms-smoke \
  --output /tmp/tkd-all-forms-smoke.json
.venv/bin/python -m tests.functional_smoke \
  --receipt "$(python3 -c 'import json; print(json.load(open("/tmp/tkd-demo-full.json"))["receipt_path"])')" \
  --receipt "$(python3 -c 'import json; print(json.load(open("/tmp/tkd-all-forms-smoke.json"))["receipt_path"])')" \
  --output /tmp/tkd-functional.json
```

The harness denies network connections. Missing local data/models must be
provisioned beforehand. Uncached forms invoke CPU inference (the initial
eight-form smoke took about 15 minutes); cached forms do not. Do not add
`--interrupt` to a completed selection: that test deliberately rejects an
immediate cache hit. Large generated products stay in the shared root.

For inspection-only reuse, verify the demo's pinned references before indexing
the existing bundles. This needs only the core environment and no network or
inference. It intentionally fails if a reference is absent or has changed:

```sh
.venv/bin/python - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
m = json.loads(Path("docs/demo/inspection.json").read_text())
r = Path(os.environ["TKD_DATA_ROOT"])
for demo in (m["real"], m["synthetic"]):
    for name in ("observation_receipt", "inspection_bundle"):
        if name in demo:
            ref = demo[name]
            p = r / ref["relative_path"]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == ref["sha256"], p
    subprocess.run([".venv/bin/tkd-poomsae", "inspection-register", demo["project"],
                    "--artifacts", str(r / demo["inspection_bundle"]["relative_path"])],
                   check=True)
PY
```

Registration verifies artifact/source identities and lineage. If regenerated
bundles differ, inspect producer/config/sync revisions and the new receipts;
do not bypass the hash check or relabel old geometry as current. The manifest's
source hashes can also be compared with the inventory report.

## Start and inspect the local viewer

In one terminal at the checkout:

```sh
uv run --frozen tkd-poomsae serve
```

In another with the same root:

```sh
cd web
npm run dev -- --port 5173 --strictPort
```

Open `http://127.0.0.1:5173`. Select `mendeley-bjy7vr4xkt-v1-taegeuk-1` for
native video/2D overlays, scores and contributing frame evidence. Seek or step
the shared cursor; use keyboard focus/Enter for overlapping landmark targets.
Unavailable real geometry is expected. Select `issue-50-synthetic` separately
for supplied-camera 3D and metric ground; its dot videos and empty semantic
timeline are synthetic, not evidence of real calibration or technique labels.
Reload/reopen to inspect persisted results without processing.

The service defaults to `127.0.0.1:8000`; `/health` reports readiness. It accepts
only loopback. Source API registration needs `TKD_ALLOWED_SOURCE_ROOTS` as a
JSON map of allowed absolute roots; existing dataset inspection does not.
Local HTTP mutations need `X-TKD-Local-Request: 1` and trusted origins.
For custom ports set `VITE_API_ROOT` and configure trusted origins as described
in [the service contract](local-service.md). Do not expose it as a public server.
If only API port 8000 is occupied, use `tkd-poomsae serve --port 18051` and
`VITE_API_ROOT=http://127.0.0.1:18051 npm run dev -- --port 5173 --strictPort`;
the browser origin remains the already trusted 5173 origin. Stop only your own
servers with Ctrl+C when finished.

Stop the manually started service and viewer before automated reproduction.
Wait for processing and inspection registration to finish too: project write
leases temporarily reject inspection reads as busy.
Only one service can own a shared data root, even on different ports; a second
service fails startup with `another local service owns this data root`.
Optional automated reproduction, with Chromium already installed via
`npx playwright install chromium` during developer setup:

```sh
cd web
npx tsc --project tsconfig.local.json
npx playwright test --config=playwright.local.config.ts
```

These nine cases start the actual viewer/service on reserved loopback ports
5350/18550, exercise eight real projects plus the separate synthetic project,
and reopen them. They refuse to reuse another server. Screenshots/traces remain
under ignored `web/test-results/`. [Issue-50 evidence](evidence/issue-50/README.md)
owns the original results; [runbook validation](evidence/issue-51/README.md)
records this reproduction. Portable CI uses generated fixtures instead.

## Analyze, resume and revise

The general runner supports `register`, `analyze PROJECT --through STAGE`,
`status PROJECT`, `resume PROJECT --through STAGE`, `cancel PROJECT` and
`rerun PROJECT STAGE --through STAGE`. `analyze --config settings.json` merges
stage-specific settings; `resume` uses saved settings. Cooperative cancellation
and immutable completed artifacts allow resumption; interrupted unpublished
work is retried. Stage states and reasons are authoritative.

**The default CLI is not a complete automatic pipeline.** Its calibration slot
accepts a persisted scene candidate and its reconstruction slot triangulates;
the other generic slots require explicitly installed Python producers.
`observations`, `sync-solve` and the acceptance harness are separate delivered
entry points. `analyze`/`resume` on default slots may return exit 1 with
`unavailable`, even when inspection products are already indexed. Indexing
products does not install producers. Do not rerun the demo to fix that status.
`status` also returns exit 1 when saved stages include failure/unavailability;
read its JSON diagnostics rather than treating that exit as an absent project.
See [runner APIs and configuration](../pipeline/README.md).

With a configured parser producer, `tkd-poomsae rerun PROJECT parsing
--through parsing` forces only parsing and descendants; changing only
`parsing` settings via `analyze --config` changes its key, leaving upstream
physical artifacts reusable. The demo's executed positive parser-reuse check
is in `tests.functional_smoke`, which invokes production semantic publishers
directly; it is not proof of an installed default CLI parser. Manual semantic
edits use [semantic edit sessions](../src/tkd_poomsae/semantic_edits/README.md),
preserve automatic bytes and are visibly attributed.

Keys bind source/upstream hashes, schema, algorithm, effective settings, model,
calibration and synchronization revisions where consumed. Changed sync offsets
invalidate attachment and timed geometry/semantics, while native observations
remain immutable. Ground/calibration changes invalidate dependent geometry;
parser changes preserve physical motion. Re-register new verified products for
inspection; stale products remain unavailable rather than silently retimed.

## Capture and calibration inputs

Use fixed cameras viewing one practitioner, with at least two independent,
overlapping usable views and shared event coverage. Starts may be asynchronous;
retain untrimmed originals and native timestamps, crop/rotation identity and
timing evidence. Similar filenames or two copies of one view do not establish
independent geometry. Moving cameras are outside the supported fixed-camera
calibration assumption.

Provide separate calibration image captures and a measured ChArUco target via
the versioned capture manifest in [calibration/README.md](../calibration/README.md).
Each image supplies its hash, source/camera identity, original size, crop and
rotation. Supply varied board poses for intrinsics and a known stationary
face-up floor placement per camera, retaining the same camera/lens configuration
as the execution. Measured board dimensions and its board-to-world placement
establish scale/world evidence; the solver estimates camera positions. No manual
camera-position entry is required:

```sh
uv run --frozen python -m calibration captures.json --data-root "$TKD_DATA_ROOT"
```

This publishes a standalone immutable calibration; it is not an `analyze
--config` capture file or automatic wiring into the default scene-only slot.
An operator/developer must bind accepted artifacts through the producer APIs.
The linked calibration documentation also gives the marker-free scene CLI,
measured intrinsic profiles, independent ground/upright/scale evidence and
attributed manual recovery. Never invent scale from body proportions, contact
outputs or frontal/lateral names. No real target captures accompanied this demo;
generated-board tests cover that estimator separately.

## Proven capabilities and limits

The table summarizes [recorded smoke evidence](evidence/issue-50/README.md),
not prediction accuracy or fulfillment of every normative MVP requirement.

| Area | Observed result | Limit / recovery |
| --- | --- | --- |
| Sources and reuse | 32 originals verified; full first pair: 1,536 native observations; eight-form smoke: 976 | Functional testing only; no ground-truth comparison |
| Automatic synchronization | Estimates for seven forms | Taegeuk 3 unavailable; separate frontal reference clock permits 2D only; manual offsets require attribution and stale descendants |
| Marker-free calibration | All real candidates unavailable (insufficient texture or missing paired sync) | No measured intrinsics, accepted real cameras, floor or metric scale; supply reliable independent captures/evidence |
| Scale and ground | Metric synthetic geometry/ground with supplied cameras | Real scale unknown; arbitrary units never imply metres; unresolved ground blocks ground-dependent products |
| Occlusion/detail | Body/hand/foot/head evidence, raw scores and missing/low-evidence states persisted | Occluded or ambiguous points may be unavailable; no guarantee of detailed reconstruction or accuracy |
| Temporal resolution | Dataset native 30 fps PTS retained; exact-frame browsing | Cannot resolve motion between captured frames; sync uncertainty and gaps remain visible; off-grid 3D does not invent a pose |
| 3D and parsing | Controlled production path: 101 samples, metric ground, one automatic step, overlapping arms/kick and 58 keyframes | Supplied cameras/clocks/analytic observations; no real 3D or recognition accuracy claim. Older synthetic fixture has 141 samples and empty semantics |
| Viewer | Nine real/synthetic Chromium cases passed in original smoke | Software WebGL; Firefox/WebKit, CUDA and pointer access to every overlapping point untested |
| Recovery | Offline second cwd: no new artifacts/inference/network; cooperative cancellation/resume | Process-kill recovery and full inference on forms 2–8 untested |

## Controlled positive reconstruction and automatic parsing

This separate fixture reproduces the positive software path integrated by #106.
It supplies metric cameras/ground, zero-offset clocks and analytic projected
2D observations with a declared 0.001 px numerical precision floor. The videos
show those dots; their pixels are not input to MMPose. Production publishers
perform alignment, triangulation, participant morphology/articulated/temporal
reconstruction, ground derivation and all five automatic parser stages with
unchanged parser settings. No semantic entities or manual labels are injected.

The exercised run produced 101 motion/ground samples, one automatic SequenceStep,
three arm actions, one kick, eleven unknown transition actions and 58 keyframes.
The arm/kick intervals overlap. Unknown transitions retain uncertain roles;
these counts are functional outcomes, not technique-recognition accuracy. The
fixture has 20 major body/foot landmarks; detailed hands/head, dynamic root,
SpecialAction, pivots and relations are covered by separate component tests.

From the checkout, with the already configured core environment:

```sh
export TKD_DATA_ROOT="${TKD_DATA_ROOT:-$HOME/.local/share/tkd-poomsae}"
tkd_controlled_root="$TKD_DATA_ROOT/runs/controlled-issue-52"
.venv/bin/python -m tests.fixtures.controlled_acceptance --root "$tkd_controlled_root"
.venv/bin/python -m tests.fixtures.controlled_acceptance --root "$tkd_controlled_root"
.venv/bin/python -m pytest -q tests/test_controlled_acceptance.py
```

The second fixture invocation reuses immutable artifacts. It regenerates the
same deterministic source clips and index, and must retain the report and payload
identities. The Python test independently reruns all five parser publishers with
changed parser settings while forbidding upstream calls and checking original
source/observation/reconstruction/ground bytes. It runs in an isolated temporary
root and needs no dataset, model or network access.

The [demo manifest](demo/inspection.json) pins the controlled sources and bundle
relative to the original shared root. Before inspection, verify these references:

```sh
.venv/bin/python - <<'PY'
import hashlib, json, os
from pathlib import Path
m = json.loads(Path("docs/demo/inspection.json").read_text())["controlled"]
r = Path(os.environ["TKD_DATA_ROOT"])
for ref in [m["inspection_bundle"], *m["sources"]]:
    p = r / ref["relative_path"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == ref["sha256"], p
PY
TKD_DATA_ROOT="$tkd_controlled_root/store" \
  TKD_ALLOWED_SOURCE_ROOTS="{\"controlled\":\"$tkd_controlled_root\"}" \
  .venv/bin/tkd-poomsae serve --host 127.0.0.1 --port 18052
```

The fixture has its own persisted store under the controlled run directory;
this service's root differs from the original Mendeley store. In another terminal
start `VITE_API_ROOT=http://127.0.0.1:18052 npm run dev -- --port 5173 --strictPort`
from `web`. Choose `controlled-acceptance`. Select an automatic arm/kick action
or keyframe and inspect the corresponding native camera frame, 3D/root, ground
and source/physical provenance. Stop these issue-owned servers when finished.

Automated real-browser reproduction creates an isolated controlled project in
the ordinary fixture service and uses the actual viewer:

```sh
cd web
TKD_BROWSER_WEB_PORT=5252 TKD_BROWSER_SERVICE_PORT=18552 npm run test:integrated
```

The `controlled.spec.ts` case navigates generated arm/kick actions and a keyframe,
follows physical/camera evidence links and verifies the same cursor across all
views. The remaining browser cases verify additional semantic classes and manual
revision behavior using their explicitly constructed fixtures. The generic
`analyze` producer slots are unchanged; this supported acceptance route is not
a new one-command CLI implementation.

## Missing resources and safe cleanup

Reads never implicitly download. Missing data: check root, `datasets status`
and `verify`, then bootstrap explicitly. Corrupt publication: inspect first;
dataset `--repair` retains a `v1-replaced-*` copy. Missing model/runtime: inspect
`doctor`, then run the corresponding explicit bootstrap. Missing inspection:
restore/reproduce the verified bundle and register it. Changed originals or
artifact hashes fail closed; preserve evidence and regenerate the affected
revision rather than editing published arrays/manifests. Unavailable calibration
needs better independent evidence, not relaxed quality gates.

Removing an issue/review worktree may remove its `.venv`, `web/node_modules`,
build output and test reports. Never delete shared `datasets`, `models`,
`runtime`, `derived` or `runs` as part of worktree cleanup. No automatic artifact
garbage collector is implemented. Preview caches/opt-in variants are derived,
but delete only identified disposable outputs after stopping users of them;
retain original sources, receipts, referenced immutable artifacts and project
edit/provenance history. A remaining lock file is not proof of a live process.
