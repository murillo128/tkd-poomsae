# Pinned Mendeley dataset on shared storage

The version-1 registration is in
`datasets/mendeley-bjy7vr4xkt-v1/registration.json`. It pins the publisher's
`Data.zip` file ID, length, SHA-256, and all 32 ZIP member paths and sizes. The
dataset is CC BY 4.0 and is limited to functional integration, demos, and
software tests. Its MediaPipe CSVs are detector output, not ground truth.

Set `TKD_DATA_ROOT` to one absolute, persistent path shared by every issue
worktree and optional local executor. With the variable unset, the root is
`$HOME/.local/share/tkd-poomsae`. The extraction path is always
`$TKD_DATA_ROOT/datasets/mendeley/bjy7vr4xkt/v1/`; the directory keeps the
publisher's `Data/` hierarchy. Do not set the root to a checkout or temporary
CI directory. Configure the host or runner environment once, so each process
inherits the same root; no launcher path change is needed.

On the persistent host, run:

```sh
tkd-poomsae datasets bootstrap mendeley-bjy7vr4xkt-v1
tkd-poomsae datasets verify
tkd-poomsae datasets status
```

Bootstrap locks the version directory across processes. It downloads into a
separate staging directory, resumes a partial archive only when a strong ETag
still matches and the server honors its byte range, and checks the publisher
SHA-256 before extracting. It rejects unexpected names, traversal, symlinks,
duplicate entries, and unregistered expansion sizes. It writes an
`acquisition.json` receipt with SHA-256 and size for every extracted file, then
publishes the completed directory atomically. `verify` hashes every file against
that local receipt without network access. A verified second bootstrap also
uses no network. The archive and partial transfer metadata remain outside the
published directory and are removed after success.

If the published directory fails verification, bootstrap stops. Run the same
bootstrap command with `--repair` only after deciding to replace that local
copy. Repair moves the old directory to a sibling named `v1-replaced-*` before
acquisition so its contents remain available for inspection. Remove retained
copies manually when they are no longer needed. An interrupted download has no
published completion receipt and can be resumed by rerunning bootstrap.

`datasets status` is the original quick size/presence check. Use `datasets
verify` when cryptographic confirmation is required. Offline consumers only
resolve local registered files; they never bootstrap or repair implicitly.

## Host acquisition evidence (2026-09-25)

One explicit bootstrap on the persistent execution host downloaded the pinned
286,266,253-byte archive. Its measured SHA-256 was
`8447119933de8032296c1d60dc0ba5f5586fe9cd2d8560c267bd1d53a29c7940`,
matching registration. The completed dataset has 32 files and 293,982,328
extracted bytes. The local receipt, which holds every file's size and SHA-256,
has SHA-256
`3eeea9e4b101ea15822b14781919a64d48b206cebf0c3549f63c27c4a2e964ea`.

From a temporary directory and from the issue worktree, a registered CSV
resolved at the same persistent root. With socket connections disabled in both
process contexts, `verify` returned the same counts and receipt digest, and a
second `bootstrap` returned `cache_hit: true` with `downloaded_bytes: 0`.
Neither the media nor the host's personal path is tracked.
