# Mendeley execution import

`tkd-poomsae datasets inventory --output report.json` reads every registered
local file, hashes source bytes, decodes video for native presentation timestamps,
and parses each paired CSV. It does not download, copy, edit, or re-encode source
files. `tkd-poomsae datasets register` writes eight project manifests under the
shared `runs/projects/` root and content-addressed source references under
`runs/dataset-sources/`. Both commands require the pinned version-1 dataset to
be available locally. The inventory JSON is the per-file diagnostic report;
large media and CSV payloads stay outside Git.

Projects use stable IDs `mendeley-bjy7vr4xkt-v1-taegeuk-1` through `-8` and
camera IDs `mendeley-frontal` and `mendeley-lateral`. These are source labels,
not calibrated camera geometry. Project manifests reference shared source
objects by SHA-256 and retain the original absolute source paths, CSV paths,
publisher attribution, licence, and usage policy. `open_registered_project(id)`
checks those paths against the source objects and current video hashes before
returning a project. Registration is repeatable for the same unchanged inputs;
changed project/source identities are reported as conflicts.

The optional `mediapipe-csv` observation provider maps the twelve named CSV
joints to canonical left/right shoulder, elbow, wrist, hip, knee, and ankle.
Rows retain their original `frame_index`, `time_ms`, normalized coordinates,
and detector score. Missing triplets become unknown landmarks. Coordinates are
converted from normalized stored-image coordinates to oriented pixels, without
clipping off-frame detector values. Only the video's native PTS supplies a
mapped observation's source time; CSV indices are checked against presentation
ordinals and CSV times are compared with PTS. Missing/duplicate indices,
out-of-range coordinates, and time discrepancies are reported. No rows are
silently shifted, trimmed, or resampled. CSV output is detector evidence for
functional import and overlay only. It is not ground truth for MMPose accuracy,
model selection, or thresholds, and cannot supply unsupported hand, foot, head,
or neck detail. MMPose remains the required pose provider.

## Local capability report (2026-09-25)

Full offline decoding measured all 16 MP4s and parsed all 16 paired CSVs from
the verified local version-1 acquisition. All videos are H.264, stored and
oriented at 1920×1080, with no audio stream. All decoded frame PTS use time base
1/15360, start at PTS 0, and have median 512-tick intervals (30 fps). The observed
properties agree with the publisher's codec, dimensions, and frame-rate claims.
Native video PTS and CSV `time_ms` agree within 1 ms at every present CSV index.
The table reports measured video frame counts and CSV row counts; source hashes,
per-view durations, first/last PTS, and individual discrepancies are available
from `datasets inventory`.

| Form | Frontal frames / CSV rows | Lateral frames / CSV rows | CSV index gaps | Off-frame joint coordinates |
| --- | ---: | ---: | ---: | ---: |
| Taegeuk 1 | 768 / 768 | 768 / 768 | 0 | 6 |
| Taegeuk 2 | 884 / 884 | 885 / 884 | 1 lateral | 38 |
| Taegeuk 3 | 1064 / 1064 | 1064 / 1064 | 0 | 136 |
| Taegeuk 4 | 1034 / 1034 | 1038 / 1038 | 0 | 176 |
| Taegeuk 5 | 1053 / 1053 | 1053 / 1013 | 40 lateral | 317 |
| Taegeuk 6 | 1194 / 1194 | 1196 / 1196 | 0 | 75 |
| Taegeuk 7 | 1203 / 1203 | 1206 / 1206 | 0 | 169 |
| Taegeuk 8 | 1464 / 1464 | 1466 / 1466 | 0 | 22 |

Taegeuk 1 has two readable paired views and is the initial demo. No registered
file supplies camera intrinsics, extrinsics, calibration targets, metric scale,
manual 3D labels, or detailed hand/foot/head observations. Frontal/lateral
labels alone do not establish geometric overlap or exact synchronization.
Those capabilities remain unknown. This inventory is a functional input check,
not an evaluation of pose accuracy.

Shared virtual test windows and opt-in derived variants are documented in
[Shared test media selections](test-media-selections.md).
