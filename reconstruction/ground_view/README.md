# Ground-view product

`reconstruction.ground_view` assembles backend-owned dynamic and whole-execution
summary geometry from final temporal reconstruction, calibrated ground and the
physical pivot artifact (which retains contacts and placements). It does not run
contact, footprint, pivot or semantic detectors.

```python
from reconstruction.ground_view import publish_ground_view, load_ground_view

handle = publish_ground_view(store, temporal_motion, calibration, physical_pivots)
summary = load_ground_view(handle)
snapshot = summary.query(global_seconds=0.4)
# Both objects have model_dump(mode="json") for a local service/viewer.
```

`summary.root_trajectory` retains every native global time in seconds, XY position
and vertical Z, native motion index, stable selection ID, quality/uncertainty,
source IDs and reasons. Root geometry is used when available; a reconstructed
pelvis can supply a labelled fallback when root is absent. Missing geometry stays
null. `root_path_indices` splits paths at missing roots and gaps longer than
`GroundViewConfig.max_gap_seconds` (default 0.15 s); a viewer must draw each run
separately. Vertical movement is retained even when XY is unchanged.

Projection takes XY and Z from **already ground-aligned world coordinates**, a
right-handed frame with Z up, under `spec/motion-representation.md`. It does not
apply `GroundFrame.source_to_world` a second time, recenter the participant, rotate
into body space, or substitute footprint centres for root samples. `world_unit`
is `m` only for resolved metric scale, otherwise `arbitrary`.

`summary.physical` preserves the upstream pivot series, including original foot
landmarks, dense foot axes/points, available footprint shapes, stable and moving
placements, contact/support, pivot angular and translation trajectories, source
indices, quality reasons and paired placement measurements. It does not fabricate
sole polygons from foot axes. Metric, arbitrary-world and anatomy-normalized
`body_ratio` measurements retain separate unit labels; normalization evidence is
kept in `physical.placements.normalization`. Original participant geometry is
never overwritten. A future reference may use this same representation; alignment
and comparison are not performed here.

`contact_events` gives each native contact/support sample a stable ID, global time
and exact foot-trajectory links. These are sample events rather than invented
continuous contact intervals. Placement/pivot event IDs and trajectory references
are preserved unchanged. `scene_bounds` encloses available projected XY geometry;
its Z limits describe the root height range. Empty geometry has no bounds.

Queries use the global clock. At a native timestamp, the snapshot agrees exactly
with summary rows and event links. Between adjacent native samples within the gap
limit, it returns the preceding **display snapshot**, explicitly labelled
`native_snapshot`, with separate requested/sampled times and bracket. This is not
interpolated contact/support or reconstructed motion. No extrapolation occurs.
Long time gaps and out-of-execution queries return `unavailable` with a reason;
individual missing roots/feet remain missing in otherwise available snapshots.
Query results are detached copies and can be selected/modified by a consumer
without changing the persisted summary.

For unresolved or unusable ground, `derive_ground_view(motion, calibration)`
returns an explicit unavailable product with retained timestamps, null geometry,
no physical events and no scene bounds. No image-coordinate fallback exists.
Immutable publication requires usable ground and the matching physical artifact,
as do the upstream contact/placement publishers; unavailable results can be
served directly from pure derivation rather than published as a valid `Ground`.

Publication uses the existing versioned `Ground` envelope. It retains all upstream
arrays byte for byte, and adds `ground_view_json` (versioned inspection payload),
`ground_view_root` (float64 columns `[global_seconds, X, Y, Z]`) and an equal-shaped
boolean `ground_view_root_missing` mask (`true` means missing; time is always
valid). Binary missing coordinates are zero-filled only for storage; consumers
must honor the mask. The JSON contract exposes nulls and states. Mixed-column
units are explicit in the payload: seconds in column 0, `world_unit` in 1–3.

Cache identity includes exact motion, calibration and physical-pivot manifest
hashes, algorithm revision and effective configuration. Same-ID revised inputs
are rejected when they disagree with upstream footprint lineage. Semantic/parser
artifacts are neither inputs nor cache dependencies. Republishing after parser
changes reuses the same physical ground-view artifact without derivation.
`load_ground_view` checks the inspection payload, binary root arrays, physical
payload, canonical contact/placement/pivot metadata and provenance for agreement.

Offline acceptance uses shared synthetic helpers and local immutable artifacts:

```sh
uv run --frozen pytest tests/test_ground_view.py
```

These tests establish projection, units, gaps, query/summary agreement,
repeatability, lineage and parser-independent caching. They make no real-data
reconstruction or contact-accuracy claims and download no data or models.
