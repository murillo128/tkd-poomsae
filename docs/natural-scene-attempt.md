# Initial natural-scene attempt on registered Mendeley views

Issue #16 used the shared local version-1 Taegeuk 1 frontal/lateral recordings.
The first audit found that sampling equal raw source PTS did not establish
synchronization; its four feature matches are exploratory only and are **not**
calibration evidence.

The repaired route required a source-bound synchronization artifact. An offline
attempt used the repository's `ingest`, `extract_cues`, `sources_from_manifest`,
and `publish_offsets` APIs on the registered sources, with cue/cache/artifact
outputs under `/tmp/issue16-sync-*`. No video was downloaded or copied. The
resulting artifact ID was
`a6aa99eaa5694ec3b7e5ab31be58a40e62a8c85b376a31fa557d4c6e41d6283c`.
It reported effective offsets of 0 seconds for frontal and −20.167 seconds for
lateral. The dataset has no exact cross-camera timing oracle, so this is a
solver output, not a measured accuracy claim.

Two calibration attempts used that artifact:

```sh
uv run --frozen python -m calibration.natural_cli --selection smoke-short \
  --sync-artifact /tmp/issue16-sync-store/derived/synchronization/a6aa99eaa5694ec3b7e5ab31be58a40e62a8c85b376a31fa557d4c6e41d6283c \
  --output-dir /tmp/tkd-issue-16-candidates
uv run --frozen python -m calibration.natural_cli --selection demo-full \
  --sync-artifact /tmp/issue16-sync-store/derived/synchronization/a6aa99eaa5694ec3b7e5ab31be58a40e62a8c85b376a31fa557d4c6e41d6283c \
  --output-dir /tmp/tkd-issue-16-candidates
```

The `smoke-short` 5–7 second source windows had **no common global interval**
under those offsets and returned `unavailable`. The `demo-full` windows had an
overlap; native frames were sampled at matched global times and their PTS were
retained. Median backgrounds yielded 253 frontal and 530 lateral SIFT features.
Only **two** reciprocal cross-view matches remained, below the 24 required to
estimate epipolar geometry, so this attempt also returned `unavailable` with
`insufficient matched texture`. No camera poses, static 3D points, metric scale,
or ground frame were accepted. Measured intrinsics were unavailable as well.

The local diagnostic JSON files retain source hashes, sync identity, native
frame PTS, and the observed outcomes without image pixels or copied media.

Issue #18's publication gate was applied to those retained candidate records.
The synchronized `smoke-short` record remains `unavailable` because its selected
windows have no common global interval. The `demo-full` records remain
`unavailable` because reciprocal matches were below the 24-point minimum; none
contains accepted cameras or a bundle suitable for publication. The resulting
capabilities are: camera geometry unavailable, ground/world frame unresolved,
metric scale unresolved, and downstream world projection unavailable. The sync
artifact and candidate diagnostic JSON remain usable for inspection. No
calibration or reconstruction was fabricated from the frontal/lateral filenames.
