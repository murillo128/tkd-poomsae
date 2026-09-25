# Local media ingest

`ingest([(camera_id, path), ...])` reads existing local files and returns a
versioned `IngestManifest`. It requires two distinct recordings. Paths are source
references: ingest hashes and decodes them for indexing, but never edits, copies
or downloads them. The manifest contains a shared `contracts.Source` for each
camera, native stream and frame time bases, presentation-order PTS, keyframe
markers, codec, stored and oriented dimensions, source duration, audio presence,
SHA-256, and both stored/oriented pixel transforms. Source time is absolute
`PTS * time_base`, including nonzero starts.

`MediaReader(recording, max_decode_frames=N)` reuses that recording's metadata
index for `nearest`, `bracket`, `frame`, and `decode_window`. Decode windows hold
only requested RGB frames and decoder buffers; no full-video pixel cache or frame
extraction is created. A window that cannot be reached from a keyframe within
`N` decoded frames fails explicitly. `FrameRef.ordinal` identifies a frame within
the indexed presentation sequence. It is not a claimed source frame number;
`Recording.frame_time()` leaves the shared contract's native `frame_index` unset
when the container supplies none. Missing PTS and decode failures are errors,
and source files changed since indexing must be ingested again.

Rotation degrees follow FFmpeg display-matrix convention (counterclockwise).
Returned pixels are oriented RGB. Apply `stored_to_oriented` to overlay points
from stored pixels, or `oriented_to_stored` for the reverse. These matrices use
integer pixel-center coordinates, including translation by width/height minus
one. Synchronization offsets and confidence belong to the later sync layer;
the reader only accepts an explicit offset when creating a `FrameTime`.
