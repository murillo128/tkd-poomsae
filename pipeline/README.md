# Offline pipeline runner

The CLI records projects and run state below `TKD_DATA_ROOT/runs/projects/` (or
`~/.local/share/tkd-poomsae` by default). Source files stay where registered;
their bytes are hashed for every analysis. Completed artifacts live in the shared
immutable `derived/` store. Ordinary runs never download videos or models.

The canonical operator path is `tkd-poomsae run PROJECT --config settings.json`;
see the [runbook](../docs/local-runbook.md#canonical-offline-project-command) for
resource inputs and reopening. `run PROJECT` resumes saved settings and indexes
inspection automatically. `run PROJECT --config settings.json --rerun parsing`
changes only parser products when only parser settings changed. Existing native
publishers retain their own immutable keys; the runner journal records and
verifies those keys and their lineage before reuse. Missing geometry stays
unavailable while native observations remain usable.

```sh
tkd-poomsae register demo --source left=/data/left.mp4 --source right=/data/right.mp4
tkd-poomsae analyze demo --through observations
tkd-poomsae status demo
tkd-poomsae resume demo --through parsing
tkd-poomsae rerun demo observations --through parsing
tkd-poomsae cancel demo
tkd-poomsae sync-solve demo
tkd-poomsae sync-offset demo right -0.125 --author operator --source visual --reason "event alignment"
```

`--config settings.json` on `analyze` accepts stage-specific effective settings,
for example `{"sync":{"offset":0.25},"parsing":{"threshold":0.6}}`.
Only a changed stage and its descendants get new artifact keys. `rerun` increments
the named stage's generation to force a new immutable artifact and descendants.
`resume` uses the last saved settings and reuses verified complete artifacts.
`status` reports `not-run`, `running`, `complete`, `failed`, `unavailable`, or
`stale`, with progress and diagnostics. A process killed during a stage has no
published result; the next status/resume reports interruption and retries it.

The explicit graph is `ingest → sync → calibration`, and
`ingest → observations`; `sync + observations → attachment` adds global time;
`calibration + attachment → reconstruction → ground → parsing` follows. Thus
`--through observations` can produce native-time 2D evidence without sync or
calibration. When 3D capability is unavailable, completed 2D artifacts remain
readable. The canonical `run` profile installs existing stage publishers through the
registry. Unconfigured legacy projects retain the original explicit slots and
actionable `unavailable` states. The profile is persisted in project registration
so status and analysis use the matching publisher revisions.

The calibration slot can consume a persisted natural-scene candidate via stage
settings `candidate`, optional `evidence`, and optional stricter `thresholds`.
It requires the exact synchronization artifact and hashes candidate/evidence
bytes into its key. A missing candidate leaves calibration unavailable; failed
geometry or mismatched revisions are reported as failed while upstream artifacts
remain available.

Each producer receives its `ArtifactKey`, verified dependency handles, and
effective settings, and returns `StageOutput` with a contract artifact, arrays,
diagnostics, and result version. `Pipeline.report_progress()` lets a running
producer persist fractional progress. Producer revisions and optional model
revisions are part of the key and saved run status. Per-project run locks and
artifact-store locks protect concurrent writers. Cancellation is requested via
`cancel` and checked before publication; producers doing long work should check
for cancellation at their own safe points as they are integrated.

The reconstruction slot publishes immutable raw N-view triangulation from an
accepted `Calibration` and an `Alignment` attachment artifact. Configure an
explicit `participant_id` and optional `triangulation` settings. The standalone
publisher and diagnostic format are documented in
[`reconstruction/triangulation`](../reconstruction/triangulation/README.md).
Body fitting and temporal regularization remain separate derived artifacts.
