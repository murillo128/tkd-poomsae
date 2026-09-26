# MVP requirement traceability and acceptance decision

**Full-system acceptance: NOT MET.** The integrated implementation has positive
component and inspection tests, but no demonstrated real-input calibrated 3D
execution with automatic overlapping semantics. Issue #52 must return to design
authority; passing regression tests cannot change this decision.

Authority is the nine normative files indexed by [spec/README.md](../../../spec/README.md)
and [issue #52](https://github.com/murillo128/tkd-poomsae/issues/52).
The inspected and executed integration revision is
`110a41287a1f1598b5bf9a7b442431cd8501d09c`. This report adds evidence only;
it changes no specification, algorithm, threshold, model or dataset policy.

## Evidence interpretation

- **S**: deterministic synthetic component tests. Positive geometric/parser
  results establish software invariants on constructed inputs.
- **R**: actual MMPose observations on provisioned Mendeley originals, plus
  processing of those observations. Cache replay verifies identities/reuse;
  original inference evidence remains in [observations](../observations/README.md)
  and [issue-50](../issue-50/README.md). Neither establishes accuracy.
- **B**: real Chromium execution of the actual client/service. Portable tests
  use generated media, supplied geometry and handcrafted semantic entities;
  shared-project tests separately inspect real MMPose observations and synthetic
  geometry. Browser navigation does not prove automatic semantic generation.
- **P**: provisioning, integrity, persistence and offline reuse receipts.

The matrix identifies implementation and regression evidence, **not individual
full-system passes**. “Gap” means positive integrated acceptance is missing even
when the corresponding component tests pass. Unknown regions and unresolved
scale are supported outcomes; absent cameras/ground and all-missing real 3D
cannot stand in for the required positive path.

IDs below combine spec prefix and baseline line number. Every line containing
uppercase MUST/MUST NOT is mapped; required outputs and acceptance paragraphs
without uppercase keywords are included as additional rows. Exact source/spec,
configuration, model and receipt identities accompany [validation.json](validation.json).

## Validation on the integrated revision

Validation ran on 2026-09-26 with Python 3.11.16, Node 22.20.0 and installed
headless Chromium using software WebGL. Setup reused the locked developer
dependencies with `uv sync --frozen --group dev --offline` and `npm ci --offline`.
No dataset, model or runtime bootstrap was needed in this new issue worktree.

| Check | Observed result |
| --- | --- |
| Default `pytest -q` | 584 passed; six opt-in cases deselected and exercised separately below |
| Local/full-data selections, alignment and sync checks | Four passed across two explicit invocations; no skips |
| Existing MMPose model smoke tests | Two passed on the shared CPU runtime, including native-source hand pixels; no skips/downloads |
| Ruff, strict mypy, CLI help | Passed; mypy checked 152 source files |
| Web/contract/local-browser TypeScript, web unit tests and build | Passed; 24 unit tests; existing bundle-size advisory retained |
| Portable real Chromium suites | 23 integrated and 15 component cases passed, no skips/retries |
| Shared-project Chromium suite | Nine real/synthetic inspection/reopen cases passed, no skips/retries; nine screenshot hashes retained |
| Functional publisher replay | Identical to committed issue-50 report; 315 manifests verified, zero new immutable artifacts/network attempts; 32 original files and 14 assets verified |
| Positive functional geometry/semantics | Supplied-camera synthetic path: 141 raw/ground samples, zero automatic steps/actions; every real calibration unavailable |
| SkillForge offline `unittest discover` with `REQUIRE_TMUX_TEST=1` | 126 passed, including real tmux and temporary Git worktrees |

The model checks reused the developer environment's pytest by appending its
site-packages after the vision runtime's own packages, importing the runtime's
NumPy/OpenCV first. This did not install or update the shared vision environment.
`TKD_HAND_TEST_VIDEO` resolved to the original `demo-full` source. These bounded
calls establish runtime execution and pixel mapping, not hand-shape accuracy.

The full pair and eight-form observation acceptance harnesses verified/reused
1,536/976 native observations in 48/32 windows. Both second-cwd passes made zero
inference/network calls. This new worktree reused the same immutable receipts
as the earlier component worktrees; no dataset/model was copied into it.
The functional publisher replay and shared-project browser outcomes are recorded
in [validation.json](validation.json), separately from the synthetic/browser
fixtures above. Original acquisition/inference and repeat bootstrap receipts
remain bound through [issue-50](../issue-50/README.md),
[issue-51](../issue-51/reproduction.json) and the shared receipt identities.

Reproduction commands are the [runbook's acceptance commands](../../local-runbook.md)
and the repository's [development checks](../../contributing.md). For the local
data cases run `pytest -m 'local_data or full_data' tests/test_selections.py
tests/test_alignment.py tests/test_sync_integration.py`; for model smoke use the
shared vision interpreter with the two tests in `tests/test_pose_mmpose_model.py`
and a registered `TKD_HAND_TEST_VIDEO`. The exact runtime/pytest reuse command is:

```sh
PYTHONPATH="$PWD/src:$PWD" "$TKD_DATA_ROOT/runtime/vision/bin/python" - <<'PY'
import os, sys
from pathlib import Path
import numpy, cv2
sys.path.append(str(Path(".venv/lib/python3.11/site-packages").resolve()))
from tkd_poomsae.selections import resolve
os.environ["TKD_HAND_TEST_VIDEO"] = str(resolve("demo-full")[0].source_path)
import pytest
raise SystemExit(pytest.main(["-q", "-m", "model_required",
                             "tests/test_pose_mmpose_model.py"]))
PY
```

The portable browser run used
`TKD_BROWSER_WEB_PORT=5252 TKD_BROWSER_SERVICE_PORT=18552 npm run test:browser`
after the default service port was found occupied. It did not reuse or stop the
other service. The standard shared-project suite uses its reserved 5350/18550
ports after processing finishes. Logs/screenshots/traces remain outside Git;
compact hashes/results are committed here.

The routing CI workflow is path-filtered to workflow/runner files and is not
triggered by this documentation-only delta; its broader native offline suite
was nevertheless run because #52 explicitly requires existing SkillForge checks.
Application PR checks apply to the epic base. Neither local nor remote regression
success is a substitute for the failed full-system acceptance gate.

## General ([spec](../../../spec/general.md))

| Requirement | Implementation and concrete evidence | Integrated limit |
| --- | --- | --- |
| G26: variable cameras, at least two usable synchronized views | `media/reader.py`, `sync/solver.py`, `sync/alignment.py`; S `test_sync_integration.py`, `test_alignment.py`; B `cameras.spec.ts` exercises 2/3/4 views; R seven paired sync estimates | Taegeuk 3 has no paired automatic timeline; estimates are not timing accuracy measurements |
| G28: geometry/time, local duration/velocity/acceleration, trajectories and ordering | `reconstruction/temporal`, `features`; S `test_temporal_motion.py` checks quadratic derivatives, rapid extensions/pivots and morphology; `test_parser_scenarios.py` checks absolute timing | Gap: no real reconstructed kinematics |
| G30, G58: physical evidence separate from semantics, replaceable stages and parser-only reruns | `contracts/models.py`, `storage/store.py`, `pipeline/runner.py`, `reconstruction/semantics`; S persisted parser scenario counts zero upstream calls and preserves bytes; P functional parser configuration rerun preserves physical hashes | Default runner slots are not a wired complete pipeline |
| G32: no scores, deductions, correctness or coaching | Physical/semantic contracts and client inspectors; S contract/parser schemas; B uncertainty/provenance inspection; [runbook](../../local-runbook.md) and demo describe observation only | Optional reference overlay deferred; no correctness inference accepted |
| G76: retain participant proportions | `reconstruction/articulated`, `temporal`; S `test_articulated.py`, `test_temporal_motion.py::test_morphology_lengths_and_vertical_root_survive` | Gap: no real participant fit |
| G84: uncertainty/provenance through reconstruction/derivation | Contracts and all publishers; S `test_triangulation.py`, `test_detailed_geometry.py`, ground/parser unknown cases; B `provenance.spec.ts`, `integrated.spec.ts` | Real unknown geometry is exposed, not recovered |
| Pipeline/core principles/completion: continuous body/hands/feet/head/root, ground, semantics and synchronized inspection from untrimmed multiview input | R complete Taegeuk 1 native pair; S positive components; B shared clock and entity navigation | **Gap: no single positive full-system execution**; supplied-camera synthetic functional path has 141 samples but zero steps/actions |

## Motion representation ([spec](../../../spec/motion-representation.md))

| Requirement | Implementation and concrete evidence | Integrated limit |
| --- | --- | --- |
| MR5: shared semantic ownership | `contracts/models.py`, generated JSON schema and `contracts/types.ts`; S `test_contracts.py`; TypeScript contract check | Shared contracts do not demonstrate all producers working together |
| MR21: global seconds, native camera/frame/PTS, variable rates | `FrameTime`, `ObservationJoin`, media index; S `test_media_reader.py`, `test_alignment.py`, `test_sync_integration.py`; B native stepping across rates | Native identity available on real observations |
| MR39: explicit world/body/segment transforms | Contract transforms, `calibration/ground.py`, detailed/temporal geometry; S `test_calibration_ground.py`, `test_detailed_geometry.py` world-transform/handedness cases | Real world/ground unresolved |
| MR69: global-time query/interpolation | `reconstruction/temporal/core.py`, alignment and inspection APIs; S `test_temporal_motion.py` query/gap/slerp cases, `test_inspection_api.py` | Long gaps and unsupported off-grid geometry remain unavailable |
| MR73, MR75: separate morphology and original pose, derived normalization | Articulated morphology artifact and pose lineage; S `test_articulated.py`, temporal morphology tests, `test_ground_view.py` unresolved scale/normalization | Real morphology unavailable |
| MR132, MR134: unknown differs from false; no confident promotion downstream | `Quality`, tri-state ground and semantic evidence; S unknown triangulation/detail/contact/parser cases; B explicit uncertainty | Required positive capabilities still missing |
| MR144: schema/version/config/model provenance, incompatible revisions | `ArtifactKey`, immutable manifests, native model identities; S `test_contracts.py`, `test_storage.py`, revision/cache tests, `test_semantic_edits.py` stale parser revision; P receipt/model hashes | Versioned receipts establish reproducibility, not accuracy |
| Observation/reconstruction/ground/semantic layers, six tracks, relations, dense trajectory links | Typed artifact contracts and separate publishers; S `test_contracts.py`, `test_semantic_assembly.py`, `test_parser_scenarios.py`; B inspector links | Real semantic and reconstructed layers unavailable |

## Ingest and synchronization ([spec](../../../spec/ingest-sync.md))

| Requirement | Implementation and concrete evidence | Integrated limit |
| --- | --- | --- |
| IS11: no same-start/frame/rate/resolution/codec/fixed-count assumption | `media/reader.py`, cue extraction and offset solver; S generated-media cases in `test_media_reader.py`, `test_sync_integration.py`; B variable-view/rate/offset tests | No pretrim/copy required for full real pair |
| IS29: automatic synchronization attempt | `sync/cues.py`, `sync/solver.py`, `Pipeline.solve_sync`; S `test_sync_cues.py`, `test_sync_solver.py`; R functional attempts for eight forms | Seven estimates, one explicit failure; no accuracy oracle |
| IS63: retain estimate/confidence and attributed manual correction, visibly separate | Sync contract/revision and inspection API; S `test_pipeline.py`, `test_sync_solver.py`, `test_inspection_api.py`; B `provenance.spec.ts` sync edit/conflict/reload | Reference-clock fallback is not a paired sync result |
| IS67: explicit failure; no equal-frame fallback | `TimelineFailure`, exclusions; S ambiguous/disconnected sync cases; R Taegeuk 3 unavailable | Failure retained honestly |
| Constant offset, subframe interpolation, timing-reference without geometric authority, persisted source-bound manifest | Alignment and synchronization artifacts; S `test_alignment.py`, `test_sync_solver.py`; P source/upstream hashes in functional receipt | Separate from parser execution interval and calibration |
| Acceptance: same physical event across retained untrimmed cameras | S imposed shifts and generated event oracle; B global/native mappings; R sync estimates on complete original pair | Real event alignment has no independently measured timing oracle |

## Camera calibration ([spec](../../../spec/camera-calibration.md))

| Requirement | Implementation and concrete evidence | Integrated limit |
| --- | --- | --- |
| CC28: automatic placement; no manual camera position/orientation entry | `calibration/target.py`, `natural.py`; S `test_calibration.py::test_manifest_to_immutable_artifact_from_rendered_frames`, `test_calibration_natural.py::test_nonplanar_variable_camera_scene` | **Gap: every real calibration attempt unavailable**; synthetic functional cameras are supplied, not estimated |
| CC36: explicit plane/quality; no downstream invented floor | `calibration/ground.py`, quality publication gate; S floor/wall/vertical tests, `test_ground_view.py`; B unavailable ground | Real ground/world unresolved |
| CC40, CC42: reliable metric scale or unresolved; no fabricated centimetres | Ground scale evidence and unit contracts; S `test_calibration_ground.py`, `test_calibration_quality.py`, `test_ground_view.py`; B unknown units | Unknown real scale is legitimate; it does not supply missing geometry |
| CC50: reject inconsistent cameras while preserving usable minimum | `calibration/quality.py`; S `test_bad_view_is_excluded_only_with_coherent_retained_pair`, disconnected/omitted/mismatched-sync tests | No accepted real retained set |
| Required intrinsics/extrinsics, world XY floor/Z vertical/axis orientation, diagnostics, persistence and reprojection | Projection/target/natural/ground/quality APIs; S all four calibration suites; P immutable synthetic calibration | Real intrinsics/poses/ground absent; [natural-scene evidence](../../natural-scene-attempt.md) records insufficient texture |

## Pose observation ([spec](../../../spec/pose-observation.md))

| Requirement | Implementation and concrete evidence | Integrated limit |
| --- | --- | --- |
| PO15: raw confidence/visibility with coordinates | Wholebody adapter/mapping and observation contracts; S `test_pose_mmpose.py`; R 133-landmark native records/raw scores; B real observation inspector | Scores are model output, not accuracy |
| PO21: high-resolution hand path localized from wholebody/original pixels | `pose/providers/mmpose/hand.py`, adapter; S `test_pose_hand.py` ROI transforms/crop/quality; R retained hand refinement calls/ROI provenance | Difficult/ambiguous hands remain missing/low evidence |
| PO37: heel/forefoot axis/orientation and useful inner/outer geometry | Mapping and `pose/regions.py`; S `test_pose_regions.py`, `test_detailed_geometry.py`; R foot regions and B overlays | Visibility limits; no individual-toe requirement |
| PO59: occlusion/foreshortening never fabricated as high confidence | Regional quality and identity tracker; S `test_pose_stream.py`, `test_pose_regions.py`, hand rejection tests; R hand states/ambiguous practitioner records | Partial real evidence retained |
| PO63: traceable filtering, preserve genuine rapid motion | Native immutable observation stream and temporal reconstruction; S `test_observation_run.py`, fast extension/pivot temporal tests | Real reconstruction filtering not demonstrated |
| MMPose baseline/body, detailed hands, head/neck, per-view regions, synchronized inspection | P pinned separate runtime/assets; R complete pair and eight-form bounded windows; S mapping/head/region tests; B eight real-project overlay/reopen cases | Native frame inference established historically; cache repeat is not new inference or 3D |

## 4D reconstruction ([spec](../../../spec/4d-reconstruction.md))

| Requirement | Implementation and concrete evidence | Integrated limit |
| --- | --- | --- |
| RC17: all useful N-view observations, per-region subsets/provenance | `reconstruction/triangulation`; S `test_known_geometry_uses_every_view` (variable N), noisy extra views/outliers/distinct subsets | Gap: no calibrated real reconstruction |
| RC39: refined hands/feet survive coarse fit | `articulated`, `detailed`, `temporal`; S `test_articulated.py`, detailed digit/foot and temporal publication tests | Detailed synthetic tests cannot prove real visibility |
| RC47: explicit spatial relations and forearm depth order | `reconstruction/detailed/core.py`; S crossing depth/intersection/uncertainty tests; parser crossing scenarios | Ambiguous ordering stays unknown |
| Temporal coherence, participant shape/pose separation, head/neck, root translation/rotation/height, reprojection/query and uncertain regions | Raw/articulated/detailed/temporal publishers; S `test_temporal_motion.py`, `test_articulated.py`, `test_detailed_geometry.py`; B 3D layers/source evidence; functional 141-sample supplied-camera path | **Gap: real replay of stable full-body/detail/root unavailable** |

## Footwork and ground ([spec](../../../spec/footwork-ground.md))

| Requirement | Implementation and concrete evidence | Integrated limit |
| --- | --- | --- |
| FG47: jitter cannot multiply footprints | `reconstruction/footprints`; S `test_footprints.py` jitter, sliding and gaps | No real calibrated placements |
| Contact/support tri-state and regions; stable L/R footprints/time/orientation/geometry/confidence | `ground`, `footprints`; S `test_ground_contact.py`, `test_footprints.py`; B dynamic/summary ground | Synthetic evidence only for positive ground |
| Step distances/separations/alignment/orientation, metric gating and derived normalization | `footprints`, `ground_view`; S footprint/ground-view tests; B metric/unresolved-scale cases | Real metric and nonmetric calibrated ground unavailable |
| Supported pivot interval/region/rotation/heel/forefoot/center trajectories | `pivots`; S `test_pivots.py`, `test_parser_scenarios.py::test_supported_pivot_uses_detected_ground_event`; B arcs | Must not infer a pivot from unknown support |
| Root planar/vertical paths, deterministic dynamic and summary product, no correctness inference | `ground_view`; S `test_ground_view.py` XY/Z/root quality/cache/parser-independence; B `ground.spec.ts` | Gap: no real accepted ground frame or products |
| Reference compatibility | Shared representation and frame/unit metadata | Optional overlay implementation deferred |

## Motion parsing ([spec](../../../spec/motion-parsing.md))

| Requirement | Implementation and concrete evidence | Integrated limit |
| --- | --- | --- |
| MP27: SequenceStep not literal footstep | `segmentation`, `semantics`; S stationary arms and kick/recovery compound tests | Positive constructed motion only |
| MP35: independent arm/leg boundaries, overlapping multi-track actions | `arms`, `lower_body`, `semantics`; S compound/special and assembly tests; B visible overlapping fixture actions | Browser semantic fixtures are handcrafted |
| MP107: retain relation-sensitive configuration/front forearm | Detailed relations/features/semantic links; S parser crossing/ambiguous crossing and transition tests | No real crossing interpretation established |
| MP118: absolute local timing/durations/dynamics not normalized away | Features and interval/dense-link contracts; S scenario shared invariants and feature derivatives | Real motion unavailable |
| Execution interval within preroll/postroll; hierarchy/dense links; automatic steps/phases/event keyframes | `features`, `segmentation`, `semantics`; S 12 parser scenarios and segmentation pre/post-roll tests | **Gap: synthetic geometric functional path returns zero automatic steps/actions** |
| Broad lower-body kick/placement/recovery/stance/pivot/support distinctions | `lower_body`; S `test_lower_body.py`, compound/pivot/unsupported scenarios | No positive semantics on real input |
| Arm roles and first-class coordinated SpecialAction, retained transitions | `arms`, `semantics`; S stationary/special/crossing and transition cases | Unknown roles preserved; exact technique naming deferred |
| Manual edits attributed/revisioned/reversible and distinct from automatic bytes | `src/tkd_poomsae/semantic_edits`; S apply/reload/undo/concurrent/stale-base tests and persisted parser rerun; B integrated edits/reload/undo | Manual labels cannot repair missing automatic acceptance |
| Parser-only rerun without vision/reconstruction/network | S `test_persisted_full_rerun_edits_and_zero_upstream_or_network_calls`; P functional assembly-config rerun physical hashes unchanged | Harness tests direct publishers; default CLI parser not installed |

## Visualization ([spec](../../../spec/visualization.md))

| Requirement | Implementation and concrete evidence | Integrated limit |
| --- | --- | --- |
| V7: render persisted results; no invented physical/semantic/evaluation outputs | FastAPI inspection and client APIs; S `test_inspection_api.py`; B unavailable products/errors/identity checks | Empty/unavailable output remains visible |
| V32: frame stepping on shared global clock | `playback.ts`, `CameraPanel.tsx`, exact-frame service; S playback/media API tests; B stepping/rate/offset/off-grid cases | Playback metadata absent: overlays suppressed, paused exact stepping remains |
| V102: no known metric values for unresolved scale | `GroundView.tsx`, ground API/unit metadata; B ground and integrated uncertainty cases | Real metric scale unknown |
| V108: no automatic correct/incorrect overlay labels | Descriptive inspectors and contracts | Optional reference overlay deferred |
| Local web/service, heavy ML outside browser, variable cameras/2D overlays/time/confidence/exclusions | `api.py`, `CameraPanel.tsx`, model runtime delegation; B 2/3/4 camera cases and eight real projects | Local native observations work without browser inference |
| Free 3D inspection/layers: joints/hands/feet/floor/root/selected trajectories/frusta/confidence | `ThreePanel.tsx`, `sceneGeometry.ts`, `GeometryInspector.tsx`; B `three.spec.ts` and integrated source selection | Positive geometry synthetic; real frusta/body unavailable |
| Timeline tracks/steps/overlapping actions/phases/keyframes, selection highlights/seek | `Timeline.tsx`, playback state; B `timeline.spec.ts`, integrated action/event selection | Handcrafted semantic navigation does not establish parser success |
| Dynamic/summary top-down feet/root/footprints/axes/distances/contact/pivot/parser overlays and bidirectional placement selection | `GroundView.tsx`; B `ground.spec.ts`, integrated selection/shared time | Synthetic physical products; no real ground path |
| Structured joint/landmark/footprint/action/phase/keyframe confidence and contributing evidence | Inspector APIs/components; S `test_inspection_api.py`; B `provenance.spec.ts`, integrated evidence links | Real native 2D provenance available; missing geometry not invented |
| Acceptance: all views inspect one physical instant with semantic/footprint navigation and uncertainty | B generated integrated/browser tests; shared-project native 2D plus separate synthetic geometry | **Gap: no positive real project containing every required view/product** |

## Selected SHOULDs and optional scope

| Clause | Selection and evidence |
| --- | --- |
| MR51: trace reconstructed landmarks to observations | Selected: endpoint/source/calibration/sync lineage in triangulation and inspector; S provenance tests and B source selection |
| MR138: efficient dense binary arrays; inspectable metadata MAY | Selected: typed NumPy arrays with versioned JSON manifests; S storage/contract roundtrip and immutable hash checks |
| IS39: multiple cues when richer evidence exists | Selected: timestamped audio/visual cue consensus; S cue/solver tests; no claim every real pair has rich cues |
| CC48: static evidence across volume | Selected: target and static-scene estimators with observability gates; moving practitioner alone does not authorize geometry |
| RC31: articulated participant-specific equivalent model | Selected: participant-specific skeleton/morphology; no mandatory SMPL-X mesh |
| V118: layer toggles, sync correction, parser edits, trajectory windows, confidence/provenance | Selected: delivered client controls; B component/integrated/provenance suites; edits remain attributed |
| G32/V106: optional reference overlay | Deferred; no scores/correctness introduced |
| IS55: optional geometric sync refinement | Deferred; constant-offset cue consensus remains the supported synchronization model |
| Optional body mesh, toe detail, detailed face, exact technique names | Deferred; articulated landmarks and broad classes remain mandatory |

## Remaining capability gaps and required return

1. **Real camera/world/ground evidence.** Seven real pairs fail static calibration
   for insufficient matched texture; the eighth lacks paired synchronization.
   No target captures or measured intrinsic profiles accompany the current demo.
   Design authority must identify a supported capture/input and evidence route
   that permits automatic placement and a persisted ground/world. Unresolved
   metric scale can remain explicit; absent geometry cannot.
2. **Positive continuous reconstruction and automatic semantic composition.**
   Use that accepted input to demonstrate morphology-preserving body, detailed
   hands/feet/head/root and ground products with automatic overlapping steps,
   actions, phases/keyframes. Investigate the empty result from the existing
   supplied-camera functional fixture without substituting manual labels or
   tuning against Mendeley CSVs. Positive constructed parser tests currently
   begin at reconstructed-motion inputs and do not close this integration gap.
3. **Executable supported delivery route.** The runbook documents that generic
   ingest/synchronization/observation/ground/parser producers are not installed
   by default. Direct acceptance publishers prove bounded composition, not a
   complete operator CLI path. The repaired design must identify and validate
   the configured entry point, artifact binding and reproducible supported demo.

No spec relaxation, invented calibration, accuracy metric, dataset validation
split, hidden download, manual-only segmentation or 2D fallback is authorized.
The existing [runbook](../../local-runbook.md) honestly reproduces limited
inspection/offline reuse; it does not reproduce the full supported positive MVP.
This report is a draft evidence delivery and design return, not a `review-ready`
handoff or a merge recommendation.
