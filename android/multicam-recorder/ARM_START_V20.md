# APK 20: ARM prepares, START keeps a recording

This experimental build preserves the continuous 1080p120 Camera2/MediaRecorder path that the user observed near 119 fps. ARM still needs internal encoded working data in this implementation: it writes only to an app-private `files/capture-temporary/warmup-*.mp4`. It does **not** create a final recording or Gallery item. This is not a claim of zero disk writes during ARM.

STOP before START closes the pipeline, deletes its private temporary and reports discarded, not saved. No JSON or MP4 is published. ARM without START is limited to three minutes, then cancelled transparently. A generation token invalidates pending camera/session callbacks, metering timers and delayed START tasks, so cancellation cannot resurrect capture. Process startup removes abandoned never-started working files only in the owned directory. A fsynced START marker protects started recordings for manual recovery if finalization fails; unrelated files are never removed.

START remains immediate, with no intentional countdown. Actual START time is kept separately from the common requested time. After STOP, `RecordingTrim` remuxes from the sync frame at or before the estimated START offset. This intentionally keeps the dependencies needed to decode the first movement instead of jumping forward to the next keyframe. The resulting file may contain a reference-frame lead; the Camera/Session dialogs and `start_in_final_video_us` report that lead. Its clock basis is elapsed time since MediaRecorder.start(), not exact sensor-frame alignment.

No pixel decoding/re-encoding is used to save the final clip. Samples remain in decode order, including B-picture PTS earlier than the preceding keyframe. Relative PTS must match within 25 microseconds (native container timebase rounding); compressed sample payloads/flags/order are hashed and checked before committing the output. The existing physical-orientation finalization then verifies the final matrix before Gallery publication. The working file is deleted after verified final output/sidecar creation. Failure publishes no untrimmed Gallery video and reports the private recovery path.

`pre_roll_ns` now means the reference-frame lead in the FINAL file, not the complete internal warm-up. New fields record internal warm-up, discarded duration, trim mode, clock basis, first retained source timestamp, payload hash and maximum timestamp rounding. Light statistics still describe all observed high-speed metadata, including warm-up; they are not per-encoded-frame labels.

The capture invariant gate verifies the original 24 Camera2/encoder configuration invocations and burst-before-recorder order while allowing this authorized storage/lifecycle rewrite. PreviewGeometry, CameraPreview, light-monitor classes and orientation helpers remain byte-identical. Protocol v3 gains an optional `discarded` flag on stopped notifications; install APK 20 on all phones for consistent labels.

Validation adds private-file cleanup/recovery and session-generation tests. Android tests construct a synthetic two-second 120fps-timestamp stream by repeating the existing three-picture closed-GOP rotation fixture, then verify retained sync boundaries, payload/PTS preservation, output rotation, decoding, failure cleanup and the real engine's no-START discard branch. A separate reordered-PTS test uses synthetic P pictures with leading timestamps; it is not a claim of testing a Samsung hardware encoder. Physical S21 recording, actual GOP cadence and multi-phone behavior still require device confirmation.

Primary API references:
- https://developer.android.com/reference/android/media/MediaExtractor#seekTo(long,int)
- https://developer.android.com/reference/android/media/MediaMuxer#writeSampleData(int,java.nio.ByteBuffer,android.media.MediaCodec.BufferInfo)

Build identity: versionCode 20, versionName 2.0-test, package dev.murillo.tkd.multicam.v20, label TKD MultiCam 20.
