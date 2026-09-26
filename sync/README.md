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

## Global-time observation queries

`sync.alignment.ObservationJoin(observations, synchronization, config)` snapshots
native observation contracts and the persisted `Synchronization` offsets. Use
`ObservationJoin.from_windows(handles, synchronization, config)` to load verified
native window artifacts. The join materializes the supplied windows; callers can select bounded windows
for large executions. Both paths are offline readers; neither calls inference,
decodes media, provisions a runtime, or downloads models/data. PTS and source
seconds must agree, source IDs must match synchronization, and each source must
have unique native timestamps and one camera identity.

```python
from sync.alignment import JoinConfig, ObservationJoin

join = ObservationJoin.from_windows(native_windows, sync_handle.metadata,
                                    JoinConfig(max_bracket_factor=2))
query = join.query(5.125)
coverage = join.coverage()
```

A query uses `source_seconds = global_seconds - effective_seconds` independently
for each retained source. It exposes `exact`, `bracket`, or `unknown` time lookup,
original endpoint observations with camera/frame/PTS identities, weights, per-point
quality and explicit missing masks. `bracket` describes the source lookup; its
geometry can still be unknown. Exact usable points remain `observed`; linear
estimates are `interpolated`, with no invented raw detector scores or visibility.
Endpoint observations retain original scores, model provenance, ROIs, subject
selection, wholebody and refined channels, and original regional geometry.
The effective offset on the query maps those native endpoints to the current
clock without rewriting their original frame metadata.

Interpolation needs two observed, usable endpoints on the same selected track.
Unknown track identity, ambiguous/missing practitioners, rejected regional quality,
missing/occluded points and provider-channel gaps stay unknown. Canonical masks
also gate source-channel estimates, so raw wholebody fingers cannot replace an
unavailable refined hand. Each source keeps its complete landmark vocabulary,
including outside coverage where its geometry is explicitly null. Available
sources are reported separately for every canonical landmark; a usable wrist
does not imply a usable hand or foot.

By default bracket spacing must not exceed twice the local median frame interval.
`JoinConfig.local_radius` selects up to five neighboring intervals on each side;
the tested gap is excluded from this median to avoid inflating its own limit.
With only one interval, that interval defines cadence. An optional
`max_bracket_seconds` supplies an additional absolute cap. Exact matching allows
only numerical tolerance (default 1 ns); no extrapolation is performed. Query
provenance includes the local median and effective bracket limit. Foot/head axes
are interpolated only with compatible providers, complete supported endpoints and
usable canonical landmarks. Their angle is derived from the interpolated axis;
a collapsed axis remains degenerate. `regional_quality` distinguishes observed,
interpolated and unknown regional estimates. Original regional provider outputs remain
available in endpoints even when derived geometry is unavailable.

`coverage()` reports closed global-time spans per source and per landmark,
including isolated exact samples and gaps caused by rejected brackets. It also
honors the retained synchronization source interval. Excluded sources and sources
without native observations have explicit reasons and no usable coverage.

`query_many(global_times)` accepts an explicit sampling grid. Requesting 60 Hz
from 30 fps sources produces labeled estimates between native observations and
adds no higher-frequency observed evidence. `publish_alignment(store, windows,
sync_handle, global_times, config)` caches only this derived product in the
immutable `alignment` layer. Its identity binds the native window manifests,
synchronization manifest, algorithm revision, configuration and requested times.
The version-1 byte payload retains settings, queries, masks and coverage;
`load_alignment(handle)` returns typed queries and coverage. Changed offsets,
reference clock, grid or interpolation settings produce new alignment artifacts
while leaving native inference windows unchanged.

Run `pytest tests/test_alignment.py` for synthetic mixed-rate, fractional-offset,
nonzero-PTS, gap/mask, coverage, reference-clock and persistence regressions.
`pytest -m local_data tests/test_alignment.py` reads existing `smoke-short`
observation receipts with imposed fractional offsets, with networking disabled.
It never infers missing observations or compares accuracy with dataset CSVs.
