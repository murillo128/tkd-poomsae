# Initial natural-scene attempt on registered Mendeley views

Issue #16 tested the `smoke-short` Taegeuk 1 frontal/lateral selection from the
shared local version-1 registration. The command was:

```sh
uv run --frozen python -m calibration.natural_cli --selection smoke-short \
  --output-dir /tmp/tkd-issue-16-candidates
```

The reader verified the registered source hashes and decoded five native frames
per view at source seconds 5.2, 5.6, 6.0, 6.4 and 6.8. No video was downloaded
or copied. The median backgrounds produced 303 frontal and 1,306 lateral SIFT
features, with stable-pixel fractions 0.882 and 0.956. Reciprocal ratio
matching found **four** cross-view matches, below the 24 needed to fit and
validate epipolar geometry. The route returned `unavailable` with
`insufficient matched texture`; it produced no camera poses or 3D points.
Measured intrinsics were not supplied, which independently prevents a
Euclidean camera solution. The attempt does not establish a successful 3D
reconstruction or a metric/ground frame. Its local diagnostic JSON is under
`/tmp/tkd-issue-16-candidates/`; it contains source hashes and summary
evidence, not image pixels.
