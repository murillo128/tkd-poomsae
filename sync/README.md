# Synchronization cues

`extract_cues(recording, config=CueConfig(), cache_dir=path)` accepts one
`media.Recording` and returns audio and visual `CueStream`s in **original source
PTS seconds**. It does not estimate offsets or choose a matching event. Feed
candidate events and complete sample series to the later alignment layer.

Each sample carries a source-time interval, raw amplitude and normalized
amplitude. Audio uses native frame PTS and short RMS windows; visual evidence
uses differences between small grayscale frames sampled from the original
video. Both streams retain local event candidates, confidence, missing spans
and diagnostics. Confidence describes cue quality, not the probability that
an event is a unique correspondence across cameras. Repeated events are
retained and flagged. The motion stream works without pose or calibration.

The result carries source SHA-256, effective-config digest and algorithm
revision. A supplied `cache_dir` stores a JSON cue result under the source and
config hashes. Identical inputs reuse it without decoding. Source files must
still match the ingest metadata. Cache data is derived and should live outside
Git, such as under `data/`.

Audio and video settings in `CueConfig` bound analysis resolution and rate.
Silence, absent audio, low motion, low contrast, clipping, missing spans,
continuous noise and probable camera shake are reported rather than promoted
to confident anchors. A camera-shake diagnostic signals a possible violation
of the fixed-camera assumption; it does not silently repair the recording.

## Constant-offset alignment

`sources_from_manifest(manifest, cues_by_source_id)` binds cues to the exact
ingested source hashes and native frame intervals. `solve_offsets(sources)`
correlates audio and motion over candidate fractional shifts, checks useful
overlap windows and competing peaks, then solves a connected set of pairwise
offsets. The selected timing reference has offset zero. Pair estimates retain
scores, peak separation, window scores, cue kinds, overlap and rejection reasons.
An ambiguous or weak pair is not treated as reliable. Cameras with inconsistent
or insufficient evidence are excluded only when two usable overlapping views
remain; otherwise `TimelineFailure` explains why no timeline was published.

`global_time = source_time + effective_seconds`. Candidate resolution is the
slower sample interval of each cue pair; tests accept errors within two 50 ms
sample intervals. There is one constant offset per source. `source_interval`,
`global_interval`, `common_interval`, and explicit exclusion reasons are stored
in the synchronization artifact. An excluded source has no automatic offset;
its quality is `unknown`.

`publish_offsets(store, key, sources)` writes the result through the immutable
artifact store. For a registered project, `tkd-poomsae sync-solve PROJECT`
indexes the source recordings, extracts cached cues, and publishes that artifact
under the runner's synchronization key. It fails explicitly if a reliable shared
timeline cannot be established. Original media and native-time cues remain
untouched.

`tkd-poomsae sync-offset PROJECT SOURCE_ID OFFSET_SECONDS --author NAME
--source METHOD --reason TEXT` records an absolute manual revision in runner
state. A later `sync-solve` publishes its effective offset while retaining the
automatic estimate and its quality separately. The CLI source ID here is the
registered camera name. Revision history survives reload and invalidates sync,
attachment, reconstruction, ground and parsing. Ingest, calibration, and
native-time observations remain reusable.

When automatic pairs are ambiguous, an attributed manual offset may establish
a two-view timeline relative to one zero-offset reference. The artifact keeps
the rejected pair diagnostics and marks automatic quality `unknown`; the
reference's zero is a timing convention, not an automatic estimate. A nonzero
manual revision of the selected reference is rejected, including through the
runner once that reference is known.

## Regression evidence

`tests/test_sync_integration.py` encodes independent synthetic recordings and
checks decode, cue extraction, offset solving, and frame lookup at known global
event times. It covers two to four views, visual-only and audio-assisted cues,
unequal and variable frame timing, nonzero PTS, different durations, repeated
events, a bad camera, disjoint manually imposed intervals, and manual revisions.
Offset acceptance is one 50 ms visual sample plus one source frame and a 50 ms
encoding/window margin. This bound applies to the synthetic event oracle, not to
the Mendeley pair.

Pytest writes `.pytest_cache/sync-regression-results.json` with test status,
source SHA-256 identities for executed cases, and a cue-config digest. Pass
`--sync-results=PATH` to save it elsewhere. Marker-filtered cases have explicit
`skipped` status. `pytest -m local_data tests/test_sync_integration.py` uses the
registered `smoke-short` selection and a shared imposed-shift variant as an
offline functional smoke test; it does not certify synchronization or pose
accuracy.
