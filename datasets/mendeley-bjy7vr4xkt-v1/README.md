# Pinned Mendeley registration

`registration.json` is the version-1 source record for DOI
[`10.17632/bjy7vr4xkt.1`](https://doi.org/10.17632/bjy7vr4xkt.1).
The creator is QingWei Zheng. The publisher licenses the dataset under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); exported demos must
retain this attribution, source link, licence, and usage policy.

The publisher's [version-1 snapshot](https://data.mendeley.com/public-api/datasets/bjy7vr4xkt/snapshot/1)
describes eight Taegeuk forms with frontal and lateral 1920×1080/30 fps video
and paired 12-joint MediaPipe CSV observations. These are publisher claims,
not measurements of local files. The CSV is detector output, not evaluation
ground truth or detailed hand, foot, or head observation. This dataset is only
for functional integration, demos, and software tests. Do not use it for
training, held-out validation, MMPose accuracy measurement, model ranking, or
threshold optimization against those CSVs.

The publisher's [version-1 file listing](https://data.mendeley.com/public-api/datasets/bjy7vr4xkt/files?folder_id=root&version=1&%24start=0&%24limit=1000)
contains one source file, `Data.zip`, with stable file/content IDs, reported
size, and publisher SHA-256. The 32 payload paths, uncompressed/compressed
sizes, and CRC-32 values in `registration.json` were read from that file's ZIP
central directory using an HTTP byte-range request. CRC-32 is ZIP directory
metadata, not an independent payload hash. Runtime SHA-256 values and a
complete acquisition receipt belong to issue #7.

[`inventory.schema.json`](inventory.schema.json) defines the inventory fields.
`archive` describes the publisher file; each `members` entry describes an
original relative path inside it. Paths are unique and must stay inside the
versioned extraction root. The provider also rejects duplicate/traversal paths.

The inventory contains 16 MP4 and 16 CSV files. It lists no separate camera
calibration, 3D annotation, or audio files. It cannot establish whether an MP4
contains an audio stream or whether usable calibration information can be
derived from frames. Intrinsics, extrinsics, metric scale, manual 3D labels,
and exact synchronization remain unestablished. The publisher's ±1-frame
synchronization description does not replace measuring actual video timing.

The offline provider expects extraction under
`${TKD_DATA_ROOT}/datasets/mendeley/bjy7vr4xkt/v1/`, preserving paths such as
`Data/video/Taegeuk_1st/Frontal/1st_Frontal.mp4`. When `TKD_DATA_ROOT` is
unset, the shared root is `$HOME/.local/share/tkd-poomsae`, outside the
worktree. Run `tkd-poomsae datasets status` from any directory to inspect local
presence and sizes. It never downloads or modifies data. The status and file
resolver report `tkd-poomsae datasets bootstrap mendeley-bjy7vr4xkt-v1`
when provisioning is needed; issue #7 owns that bootstrap command. `available`
means all registered extracted files exist with their published ZIP sizes; the
bootstrap receipt is responsible for cryptographic payload verification.
