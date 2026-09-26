# Runbook reproduction

The [local runbook](../../local-runbook.md) was executed on the persistent Linux
host on 2026-09-26, using the issue's canonical integration snapshot
`78a3bf23596b2612e0bf351b0983c5ef6f9a2f6f`.

Python was 3.11.16; Node 22.20.0 was selected explicitly because the shell default
was Node 21. Core setup used `uv sync --frozen --group dev`; viewer setup used
`npm ci`. The root was the host's existing shared `~/.local/share/tkd-poomsae`,
outside the worktree. The execution sandbox needed write access to the host
package caches and shared root; ordinary operator terminals need those same
filesystem permissions. No per-issue dataset/model download was performed.

## Setup and command behavior

Dataset bootstrap, runtime bootstrap and model bootstrap reused existing verified
publications. Dataset verification covered all 32 files (293,982,328 bytes).
`doctor` verified 14 assets and discovered the shared vision interpreter with
PyTorch 2.1.0+cpu, MMCV 2.1.0, MMEngine 0.10.3, MMDetection 3.3.0 and MMPose
1.3.2. CUDA was unavailable. Inventory, eight-project registration, selection
resolution and `observations demo-full --device cpu` completed successfully.

The CLI lifecycle commands were also executed on a separate
`issue-51-cli-capability` project referring to the same original first pair.
Registration succeeded. `analyze --through observations`, `status`,
`resume --through parsing`, and `rerun parsing --through parsing` returned
exit 1 with unavailable default producers, as documented. They did not replace
the inspected demo or establish a configured CLI parser. The positive physical
and parser-only reuse path uses the existing acceptance harness.

The default service port 8000 was occupied, so this run started its own
`tkd-poomsae serve --port 18051`. `/health` returned `{"status":"ok"}`;
the real demo inspection endpoint retained both pinned source hashes and its
automatic sync product. Vite started on 5173 and was restarted with
`VITE_API_ROOT=http://127.0.0.1:18051`; its HTML was served successfully. The
automated shared-project browser path uses its separate reserved ports.
Its initial startup and diagnostic retry failed because the manually started
service still owned the shared-root lease. The runbook now explicitly requires
stopping that service before browser acceptance; changing ports alone does not
release the lease. The manual issue-owned servers were stopped before rerunning.
A subsequent run overlapped processing/registration and passed seven cases;
forms 7/8 failed during project write contention (including an explicit busy
response). The runbook now requires processing/registration to finish first.

## Offline inspection repeat

Both `pose.acceptance` commands completed with all existing windows reused and
zero inference/network calls on their second-cwd passes: 1,536 observations in
48 windows for `demo-full`, and 976 in 32 windows for `all-forms-smoke`.
`tests.functional_smoke` completed with identical outcomes on its second cwd,
315 verified artifact manifests, zero new artifacts and zero network-connect
attempts. Its parser-only assembly configuration change preserved physical
hashes. The synthetic run retained 141 samples and zero automatic steps/actions.

Both manifest-pinned inspection bundles and the real observation receipt passed
SHA-256 checks before indexing. A separate process changed cwd to `/tmp`, denied
socket connections, verified the same references and registered both projects
using the delivered `Inspection.register` API. Before/after scans of all 1,604
existing immutable artifact manifests were identical: zero new artifacts, zero
network-connect attempts. Mutable inspection indexes were rebuilt; original
media, model weights and immutable artifact bytes were not changed.

Large reports, generated media/arrays, screenshots and traces remain outside Git.
The [demo manifest](../../demo/inspection.json) and compact reproduction receipt
here retain portable references and results. Original inference and capability
evidence remains in [issue-50](../issue-50/README.md).

## Final validation

After stopping manual servers and finishing processing/registration, all nine
shared-project Chromium cases passed without skips or retries. The final
[compact receipt](reproduction.json) records durations, screenshot hashes,
integrity/reuse results and default CLI outcomes. The earlier startup/contention
failures above are retained to explain the required ordering.

The default Python suite passed 584 tests (six opt-in cases deselected).
Ruff, strict mypy (152 files), web/contract/local-browser TypeScript checks,
24 web unit tests and the viewer build passed. Local data/model behavior was
exercised by the explicit harnesses above, not those deselected pytest cases.
Documentation links, demo JSON and Git whitespace checks passed. No source,
dependency, normative specification or CI configuration changed.

## Limits

This is a cache/reuse reproduction, not a second acquisition or new inference
benchmark. No real calibration targets/intrinsic profiles were available, so the
capture-manifest command is documented from its delivered interface and existing
generated-board tests; it was not executed with real captures. Real 3D/scale/
ground remain unavailable. The synthetic case does not establish real camera
estimation, accuracy or positive semantic labels. CUDA, other browsers and
process-kill recovery remain untested. Existing dependency deprecation and
bundle-size advisories were observed without changing dependencies.
