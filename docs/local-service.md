# Local project service

`tkd-poomsae serve` listens on `127.0.0.1:8000` by default. It accepts only
loopback bind addresses. The API reads registered projects and their persisted
stage state without loading a vision runtime or fetching assets. Project reads
use IDs; source paths are never returned. Missing local inputs or producers are
reported as unavailable with an offline provisioning instruction.

Registration through the API is disabled until `TKD_ALLOWED_SOURCE_ROOTS` is set
to a JSON object of names and absolute local directories, for example
`{"dataset":"/data/tkd/datasets"}`. A source registration names a root and a
relative file path. The service resolves symlinks and rejects paths outside all
configured roots; links into another configured shared root are valid. The
trusted operator may also register local files with the CLI.

The API endpoints are:

- `GET /api/projects`, `GET /api/projects/{id}` and
  `GET /api/projects/{id}/capabilities` for discovery, status, and producer,
  schema, model, and artifact revisions;
- `POST /api/projects` with `{"id":"demo","sources":{"left":{"root":"dataset","path":"left.mp4"},"right":{"root":"dataset","path":"right.mp4"}}}`;
- `POST /api/projects/{id}/runs` with `{"through":"observations"}` or an
  optional `rerun` stage; `GET /api/runs/{run_id}` for queue, stage progress,
  results, and errors; `POST /api/runs/{run_id}/cancel` for cancellation.
- `GET`/`HEAD /api/projects/{id}/media/{camera}` serves registered source bytes
  with `Accept-Ranges`, ETag/Last-Modified validators, single byte ranges (`206`),
  and `416` for malformed or unsatisfiable ranges. No video is copied or
  re-encoded for this route.
- `GET /api/projects/{id}/media/{camera}/metadata` reports source identity,
  codec, browser playback capability, and first/last native frames. A false
  `browser_playback` directs clients to the exact PNG frame route; it does not
  claim that a browser can seek an unsupported codec precisely.
- `GET /api/projects/{id}/media/{camera}/frames?start=0&limit=100` pages through
  native presentation-order frame indexes (at most 256 entries per request).
  `/frames/nearest?seconds=T` and `/frames/bracket?seconds=T` use **source** time,
  preserving nonzero PTS; synchronization offsets belong to the client/project
  timeline. `/frames/{ordinal}/image` returns the exact oriented PNG frame and
  identifies the delivered ordinal, PTS, and source hash in response headers.

Media routes resolve only registered cameras within configured source roots or
the shared storage `datasets/` namespace. Indexing and exact-frame decoding use
checked open file descriptors, so a path replacement cannot redirect an in-flight
read outside those roots. Source paths remain absent from JSON.
At most four video streams and two frame decodes run at once. The service keeps
at most eight compact indexes in memory and 16 exact PNG previews (64 MiB total)
under shared `derived/media-previews-v1/`, with a matching bounded memory cache.
It never extracts all frames, downloads media, or implicitly re-encodes a
recording. An unsupported container returns `415`;
unavailable or changed source frames return an explicit error.

Every state-changing HTTP request needs `X-TKD-Local-Request: 1`. The service
checks the Host and Origin headers and answers CORS preflight only for trusted
origins, `GET`/`HEAD`/`POST`, and the local request, content type, range, and
validator headers.
Actual responses echo only a trusted Origin; there is no wildcard CORS. The
default accepted browser origins are the local Vite and API ports;
`create_app(..., trusted_origins=...)` can set them for an embedded client.
Cross-site requests without a trusted Origin remain rejected. This is a local
operator service, not multiuser auth.

The service keeps at most 16 pending jobs and two workers by default. One job
per project can be pending or active. Workers hold a shared local analysis
lease while invoking the offline pipeline, so analysis jobs serialize even if
the worker count is raised. Stage producers retain responsibility for any
physical GPU lease they require. Job records live under
`TKD_DATA_ROOT/runs/service-jobs/`; jobs interrupted by a service restart are
marked `interrupted` and can be submitted again. The pipeline's immutable
artifacts and per-project state provide resumability. Normal API requests do
not download datasets or model weights.

Processed artifact inspection uses a persisted per-project SQLite index. The
trusted producer/operator binds existing `ArtifactKey`s with
`Inspection(pipeline).register(project, products, observations=window_keys,
lineage=upstream_keys)` or:

```sh
tkd-poomsae inspection-register PROJECT --artifacts artifact-keys.json
```

The bundle contains `products` (required `sync`, optional `reconstruction`,
`ground`, `semantics`), an optional `observations` array of native window
keys to index, and an optional `lineage` array of upstream keys to verify. Each
key uses the same JSON fields as the immutable store and the
semantic-edit CLI: `layer`, `inputs`, `schema_version`, `algorithm_revision`,
`config_digest`, and any optional model/calibration/sync revisions. Register the
final reconstruction and ground product, not an earlier raw intermediate.
Registration verifies source hashes, layer identities and physical/semantic
lineage before assigning a clock, including for previously unindexed motion.
Direct motion keys must bind the current source hashes and synchronization
manifest. Published derived motion instead requires upstream reconstruction,
alignment and native observation keys in `lineage`, following their immutable
input references to the current synchronization and sources. Unverifiable or
mismatched lineage is rejected. The upstream key inventory is bounded to 8 MiB
and reconstruction ancestry to eight edges. Registration indexes each native
observation window separately and preserves the first clock binding of every
timed artifact. It cannot relabel an old product as current after a sync
revision. Registration reads persisted data only and
never invokes a producer. No HTTP endpoint accepts artifact keys or file paths.
Missing products remain explicitly unavailable until registered; this does not
make an uninstalled producer available.

The inspection endpoints below share the existing Host, Origin, registered-media
and local-mutation policy:

- `GET /api/projects/{id}/inspection` reports source/artifact revisions, edit
  revisions, availability, array descriptors and request limits.
- `GET .../inspection/{product}/window?collection=samples&start=T&end=U` returns
  a closed global-time window. Products are `observations`, `reconstruction`,
  `ground`, `semantics`. Observations use collection `observations`; reconstruction
  uses `samples` or `joints`; ground uses `samples`, `footprints`, `pivots`,
  `measurements`; semantics uses `steps`, `stances`, `actions`, `phases`,
  `keyframes`. Intervals overlap the requested window. Rows retain confidence,
  native identities, null missing geometry and landmark masks. Geometry units
  are `m` only for metric reconstruction; unresolved scale is `arbitrary`.
  Root/landmark trajectories are native sample windows, without invented samples.
- `GET .../inspection/entities?id=ENTITY_ID` resolves a selected joint,
  footprint, action, phase or keyframe and follows retained native observation
  IDs to camera/frame/PTS evidence. Sample IDs are `{artifact_id}/samples/{index}`;
  joint IDs append `/{landmark_name}`. Persisted semantic/footprint IDs stay
  unchanged. Native observation entities return the effective global `frame`
  and retain their original `native_frame`, matching window and evidence times.
  Their `artifact_revision` identifies the owning native window's manifest.
  Indexes created before native ownership was recorded require re-registration
  for native entity lookup and return an explicit conflict until then.
  If a source has no usable synchronization offset, native frame/PTS evidence
  stays available, but its mapped `frame.global_seconds` and `offset_seconds`
  are null with a `global_time_reason` also reported in `source_evidence_reason`.
  Window, entity/evidence and time lookup share the same offset policy: current
  or persisted manual overrides take precedence; otherwise excluded cameras
  and missing automatic estimates are unavailable. A retained timing reference
  with no automatic estimate has its genuine zero offset.
  Missing or truncated contributing evidence has an explicit reason. Partial
  resolution retains valid evidence and reports bounded
  `source_evidence_unavailable_ids` and `source_evidence_unavailable_count` for
  the IDs examined; `evidence_truncated` reports unexamined excess IDs/rows.
  Reprojection diagnostics describe internal consistency, not accuracy.
- `GET .../inspection/time/{camera}?seconds=T` maps global time using persisted
  offsets and current manual revisions, then uses the registered native media
  index. It returns bracket/nearest ordinal, source hash, PTS, source/global
  times and nearest-frame gap; outside-coverage lookup is labeled explicitly.
  It never implies that a bracket has reconstructed/interpolated geometry.
- `GET .../inspection/{product}/arrays/{array_id}?start=T&end=U` returns an NPZ
  with `values`, optional `missing_mask`, and UTF-8 JSON `metadata` stored as
  `uint8`. Use `allow_pickle=False` when opening it. Only registered arrays whose
  leading axis is `native_time`, `time` or `sample` and whose length matches the
  indexed samples are exposed. JSON evidence blobs and arbitrary storage files
  are not array transports.
- `POST .../inspection/sync-offset` takes `camera`, `offset_seconds`, integer
  `expected_revision` and `author`/`source`/`reason`. Expected revisions are
  compared under the pipeline's project lock. Original automatic estimates stay
  visible; revised offsets immediately affect native observation/time lookup.
  Reconstruction, ground and semantics become stale and return no current rows.
- `POST .../inspection/parser-edits` takes `automatic_revision` (the registered
  automatic manifest hash), integer `expected_revision`, `command`
  (`apply`, `undo`, `reset`), typed `operations` and attribution. It delegates to
  `SemanticEditor` and updates only the effective semantic index. Automatic bytes
  and physical products stay intact. Each automatic artifact has a separate
  project-scoped edit session. CLI edits outside this service make its semantic
  index explicitly stale until re-registration, rather than silently serving
  an earlier effective revision.

Windows are at most 30 seconds, 256 rows (`limit`, default 100), and 2 MiB per
response. `next_cursor` is a keyset cursor for another page of the same window;
clients must bind it to the response `revision`. Arrays must fit one page.
JSON, time and array responses have ETags; window/entity/array requests may supply
`expected_revision` to reject superseded requests. Clients should abort old
window fetches when the cursor changes; disconnects cancel window row assembly.
Project leases exclude concurrent publication/editing during reads. Readers use
indexed rows and mmap slices, not complete dense recording JSON. Changed
registered artifact files fail closed until registration verifies them again.
The compact product inventory and edit inputs are each capped at 8 MiB; oversized
semantic artifacts must be partitioned by their owning producer before service
editing. These limits cover service reads/edits; trusted registration may read
one complete persisted physical artifact while constructing its disk index.
