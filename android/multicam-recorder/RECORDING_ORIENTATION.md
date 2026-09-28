# v1.9: saved rotation and immediate START

The camera sensor still supplies native 1920x1080 high-speed buffers directly to the recorder. No GPU rotation, extra capture Surface, decoder or encoder is inserted. Preview geometry, automatic exposure, autofocus, v1.8 light telemetry, bitrate and pre-roll remain unchanged.

## Saved orientation

Physical device orientation comes from OrientationEventListener, not the locked Activity/display orientation. Quarter-turn selection has a 10-degree hysteresis beyond the 45-degree boundary. For the rear camera the clockwise output hint is `(sensorOrientation + physicalClockwise) % 360`. If physical orientation is unknown (for example a phone lying flat), the last known physical axis is explicitly labelled uncertain; if no physical reading exists, a named display fallback is used. These choices are recorded in the sidecar and uncertainty is exposed as a save warning.

ARM captures the initial orientation for MediaRecorder.setOrientationHint, which only sets container metadata. START snapshots orientation again, because positioning the phone during pre-roll must not leave the entire saved performance tagged as portrait. After STOP finalizes the private file and before Gallery publication, Mp4Orientation updates/verifies the **single video track's 36-byte tkhd display matrix**. It does not decode, re-encode, remux, change sample tables or touch H.264 samples/PTS/DTS. The parser validates parent bounds, standard/extended box lengths, tkhd v0/v1 and a pure rotation matrix before writing; multiple video tracks, unknown transforms, fragmented/truncated inputs are rejected. Failed verification is reported, not silently presented as a successful orientation fix.

The final file uses START orientation, or ARM orientation when no START occurred. A file has one playback matrix: keep the phone fixed after START. If orientation changed during pre-roll, those setup frames may not be upright under the final matrix. This does not implement per-frame dynamic rotation. JSON records `orientation_at_arm`, `orientation_at_start`, `output_rotation_cw`, verification/warning state and rotation basis.

## Immediate start

After ARM/READY, START sends the current common requested instant with **zero intentional delay** and a zero-millisecond unsynchronized fallback. Controller-only mode also has no countdown. Actual device callback times remain separate from requested times, so network/dispatch latency is not misrepresented as hardware synchronization. ARM continues to write pre-roll; START remains a logical marker, not a camera restart. Older controllers that send future start targets still work through the existing scheduling path.

## Analysis recommendation

Keep originals in native encoded orientation plus correct MP4 metadata. During analysis, establish one orientation normalization point in the decoder/input adapter; rotate decoded frames once when needed and keep any calibration/keypoints in the same coordinate system. Do not both auto-rotate in the decoder and manually apply the metadata again. This change does not rewrite the analysis pipeline.

## Validation

JVM tests cover physical/display sign, locked-display disagreement, hysteresis, all four rotations, v0/v1 and extended headers, malformed/multiple-track rejection and byte preservation outside the matrix. Android instrumentation checks immediate START, physical orientation independent of a portrait UI lock, and MediaMetadataRetriever recognition of all four output rotations while MediaExtractor returns identical sample payload hashes/timestamps.

`rotation-fixture.mp4` is a 2 KB synthetic testsrc clip generated locally, not user footage:
`ffmpeg -f lavfi -i testsrc=size=64x48:rate=30 -frames:v 3 -c:v libx264 -pix_fmt yuv420p -bf 0 -an -movflags +faststart rotation-fixture.mp4`

The capture invariant guard allows only five marked storage-orientation hooks, in addition to passive light observation, then reproduces the original capture/control source hashes. No preview or network implementation change is included.

Primary API references:
- https://developer.android.com/reference/android/media/MediaRecorder#setOrientationHint(int)
- https://developer.android.com/reference/android/view/OrientationEventListener
- https://developer.android.com/reference/android/hardware/camera2/CaptureRequest#JPEG_ORIENTATION (sensor/device sign convention; not a JPEG capture change)

Emulator validation does not replace physical S21 / multi-phone acceptance. The private user video and its frames are not committed.
