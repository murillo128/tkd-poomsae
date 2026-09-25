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
