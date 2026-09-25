# Shared artifact storage

`StorageRoot.from_env()` uses the absolute `TKD_DATA_ROOT`, or
`$HOME/.local/share/tkd-poomsae` when unset. The same root can be mounted by
multiple worktrees. Its `datasets/`, `models/`, `derived/`, and `runs/` directories
have separate purposes. Storage writes only under `derived/`. Dataset and model
readers call `StorageRoot.existing(namespace, relative_path, sha256=...)`; missing
assets raise `MissingResource` with a provisioning hint. Readers never download.

An `ArtifactKey` names a single versioned layer. `inputs` maps names to SHA-256
hashes of actual source bytes (`hash_file`) or upstream artifact identities.
`config_digest` should be `hash_config` of **only the effective settings used by
that producer**, and should match `metadata.provenance.config_digest`. Include
schema, algorithm and model revisions; include calibration and synchronization
revisions when the producer consumes them. A parser setting therefore changes
the semantics key without changing an observation or reconstruction key. Do not
include runtime durations, wall-clock times, process IDs, or checkout paths.

Use `ArtifactStore.get(key)` to verify an existing artifact. It returns validated
contract metadata and read-only memory-mapped `.npy` arrays. A slice such as
`handle.read_array("points", slice(100, 200))` uses the mapped file. A missing
artifact raises `MissingResource`; altered or incomplete content raises
`CorruptArtifact` and is never reused.

Use `ArtifactStore.get_or_create(key, producer, timeout=..., cancelled=...)` to
produce once per key. The producer returns one validated contract model and a
mapping of NumPy arrays whose IDs, dtypes and shapes match its dense descriptors.
Masks are separate boolean arrays with matching axes and shape. The producer
runs under an operating-system file lock; another process waits for its complete
result. A bounded wait raises `WriteTimeout`, or `Cancelled` if the supplied
callback requests cancellation. Operating-system locks release when a writer
dies; surviving lock files are harmless and are not interpreted as active work.
Publication uses a temporary directory beside the destination and an atomic
rename after data, manifest and directories are flushed. Only the publishing
process removes its own temporary directory on failure. Published artifacts are
never overwritten or silently repaired. No storage garbage collection runs.

Metadata uses the `contracts` 1.0.0 JSON schema and has no executable payload.
Arrays use `.npy` with pickle disabled. A completion manifest binds the key,
metadata and every array to a SHA-256 digest. `get` checks all hashes, descriptor
types and shapes before a cache hit. Cross-artifact semantic references should
be validated with `contracts.validate_bundle` when assembling a complete bundle.
