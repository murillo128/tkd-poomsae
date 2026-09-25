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
