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

Every state-changing HTTP request needs `X-TKD-Local-Request: 1`. The service
checks the Host and Origin headers, rejects cross-site requests, and does not
enable wildcard CORS. The default accepted browser origins are the local Vite
and API ports; `create_app(..., trusted_origins=...)` can set them for an
embedded client. This is a local operator service, not multiuser auth.

The service keeps at most 16 pending jobs and two workers by default. One job
per project can be pending or active. Workers hold a shared local analysis
lease while invoking the offline pipeline, so analysis jobs serialize even if
the worker count is raised. Stage producers retain responsibility for any
physical GPU lease they require. Job records live under
`TKD_DATA_ROOT/runs/service-jobs/`; jobs interrupted by a service restart are
marked `interrupted` and can be submitted again. The pipeline's immutable
artifacts and per-project state provide resumability. Normal API requests do
not download datasets or model weights.
