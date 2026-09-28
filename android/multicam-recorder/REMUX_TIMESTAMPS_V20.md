# Reordered presentation timestamps in APK 20

The first Android test execution exposed a native MediaMuxer limitation: MPEG4Writer normalizes timestamps against the first sample and rejects a later picture whose PTS precedes that first sample, even if all supplied timestamps are nonnegative. Merely subtracting the minimum source PTS does not fix it. The user S21 stream exhibits leading-PTS pictures after sync frames, so this case must not be skipped or hidden by weakening the test.

The private staging mux now writes unchanged H.264 samples/flags in their original decoder order using monotonic staging times. After closing it, `CompositionTimes` reads its bounded single-video metadata and replaces the composition offsets with a signed CTTS table derived from the original per-sample PTS, normalized against the minimum retained PTS. Durations are updated, staging edit lists removed, and the replacement moov is appended while the old moov becomes free space. Media payload and chunk offsets do not move. Original DTS values are not promised: MediaExtractor does not expose them here; the output retains decode order and reconstructs a valid decode timeline plus original relative presentation timing.

Before a final file can be committed or published, the existing verification reopens it through Android MediaExtractor and checks every compressed sample hash/order/flag and all relative PTS (25 us maximum native timebase rounding). The same previously failing regression test remains intact. This is still lossless with respect to compressed pictures and their presentation timing: no decoder, encoder or pixel processing is used in the save path.

A local Java/FFmpeg cross-check on a 240-sample synthetic B-frame MP4, with a first sync PTS later than following pictures, preserved every compressed payload and reproduced the requested PTS exactly at the fixture timescale. This is synthetic validation, not a physical S21 run. Android instrumentation also tests actual native remux, extraction and decoding.

Reference: AOSP MPEG4Writer.cpp, Track::threadEntry (timestampUs -= previousPausedDurationUs; subsequent nonnegative timestamp guard):
https://android.googlesource.com/platform/frameworks/av/+/master/media/libstagefright/MPEG4Writer.cpp
