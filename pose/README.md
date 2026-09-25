# Wholebody observation adapter

`pose.providers.mmpose.MMPoseAdapter` accepts a `Recording` and an iterable of
its display-oriented `DecodedFrame` values. Call `infer` once per bounded window
(at most `max_frames`, default 32); consume its iterator before starting another
window. The iterator yields one `PoseFrame` per input frame, including frames
without detections. `FrameTime` retains native PTS, time base, source and camera
IDs without requiring synchronization. Candidate indices are local to each frame
and are not person tracks.

Each candidate keeps its detector box and score, all 133 COCO WholeBody named
points with raw scores and model visibility when available, plus original-image
hand refinements and the boxes used for those crops. The optional
`canonical_landmarks(candidate)` projection maps the 63 directly supported
points to the shared `Landmark2D` contract. It does not invent pelvis, spine,
neck, or head points, and it does not turn detector scores into quality scores.
Face points and hand wrist duplicates remain available in the raw candidate.
Artifact IDs and final observation assembly belong to the integration stage.

The adapter uses verified local registry paths. It never provisions assets and
has no whole-frame production fallback. OpenMMLab's topdown API receives the
original display-oriented pixels and returns coordinates in that same image;
it undoes its own resize and crop. `PixelTransform` provides an explicit inverse
for callers that preprocess frames before using a different backend. Do not
apply that inverse again to coordinates returned by this adapter.

The optional vision interpreter in `vision/README.md` has the pinned models but
does not include the core PyAV reader. A caller must supply already decoded
frames in that interpreter or install compatible media dependencies there.
No network is needed for inference after model bootstrap. The explicit
`model_required` test runs detector, wholebody and hand models on a bounded
synthetic image with network access disabled.
