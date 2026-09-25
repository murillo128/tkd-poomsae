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
hand refinements and the boxes used for those crops. Each `hand_observations`
entry retains the independent 21 coarse and 21 refined points, per-point raw
scores/visibility and observed/inferred/unknown state, anatomical side,
ambiguity flag, model hash, ROI bounds, source overlap, padding, and reversible
ROI-to-source affine transform. A tiny or poorly supported hand stays unknown;
an edge-truncated, mostly occluded, or individually unsupported finger cannot
become observed from model confidence alone. Overlapping left/right ROIs flag
ambiguous handedness without
swapping names. `refined_hands` is a compatibility view of observed points only.
The optional `canonical_landmarks(candidate)` projection maps the 63 supported
points to the shared `Landmark2D` contract. It does not invent pelvis, spine,
neck, or head points, and it does not turn detector scores into quality scores.
Face points and hand wrist duplicates remain available in the raw candidate.
The replaceable `RegionalProvider` projects wholebody heels, medial/lateral toe
points, eyes, nose and shoulders into independent left-foot, right-foot and head
states. Available foot axes run from heel to toe midpoint in original-image
pixels. Collinear or foreshortened geometry has no orientation. The head axis
is only a projected eye-midpoint-to-nose direction; it is not 3D head direction.
`regional_geometry` can be copied into an `Observation` with canonical landmarks;
the frame carries source identity and model identity while the observation
provenance records the chosen model/configuration. No foot-only crop is sent to
the wholebody model.
Artifact IDs and final observation assembly belong to the integration stage.

`pose.stream.PractitionerTracker` assembles one camera/source stream in native
frame order. Call `observe` for each `PoseFrame` with an artifact ID and producer
provenance; keep one tracker per camera. Its initial automatic selection requires
a clearly larger practitioner box. Later selections use box and torso continuity,
not candidate index or detector score. Ambiguous matches and gaps retain the
prior identity anchor without choosing a spectator. Pass an explicit
`operator_candidate_index` to recover and record that selection. The tracker
does not join cameras in global time or interpolate a missing frame.

The resulting `Observation` stores subject candidate boxes, raw detector scores,
match costs and selection reasons. `wholebody_landmarks` preserves coarse named
coordinates, scores and visibility; `refined_landmarks` preserves independent
hand evidence, including unknown points and their original scores. `landmarks`
is the selected derived view: unsupported coordinates become null with unknown
quality, including low raw visibility. Detailed hand points require usable
refinement; coarse wholebody fingers remain source evidence when a crop is too
small or refinement is unavailable. Rapid valid movement remains unsmoothed. `region_quality`
contains independent body, hand, foot and head usability masks and reason codes.
These fields are per view; downstream reconstruction can combine useful regions
from different cameras. No optional temporal filter is applied.

The adapter uses verified local registry paths. It never provisions assets and
has no whole-frame production fallback. OpenMMLab's topdown API receives the
original display-oriented pixels and returns coordinates in that same image;
it undoes its own resize and crop. `PixelTransform` provides an explicit inverse
for callers that preprocess frames before using a different backend. Do not
apply that inverse again to coordinates returned by this adapter.

`HandROIConfig` bounds native crop size, margins, source visibility, score gates,
rotation, and optional mirroring. The adapter copies each eligible ROI from the
display-oriented original BGR frame at 1:1 pixel scale, pads out-of-frame area,
and feeds the crop to the pinned hand model. Its raw crop/model output is cached
under `${TKD_DATA_ROOT}/derived/hand-refinement-cache/` by source frame,
crop content, model config/checkpoint/runtime, side, ROI transform, and settings.
The cache does not alter wholebody observations or download assets.

The optional vision interpreter in `vision/README.md` has the pinned models but
does not include the core PyAV reader. A caller must supply already decoded
frames in that interpreter or install compatible media dependencies there.
No network is needed for inference after model bootstrap. The explicit
`model_required` test runs detector, wholebody and hand models on a bounded
synthetic image with network access disabled.
With `TKD_HAND_TEST_VIDEO` pointing to a registered local video, a second
`model_required` test decodes one native frame and runs the hand model on an
unresized crop without network access. It checks model execution and topology,
not hand accuracy. Integrated video examples belong to the shared observation
smoke run.
