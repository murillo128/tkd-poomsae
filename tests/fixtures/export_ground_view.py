"""Regenerate the browser ground fixture from offline physical producers.

Run: uv run --frozen python -m tests.fixtures.export_ground_view
"""

import json
from pathlib import Path

from tests.test_ground_view import view
from tests.test_pivots import sequence

source, contacts = sequence(sign=-1)
for i, sample in enumerate(source.samples):
    sample.root_xyz_world = (
        -0.1 + sample.global_seconds * 0.3,
        0.3 + sample.global_seconds * 0.3,
        1,
    )
series = view(source, contacts)
assert series.physical is not None
assert series.scene_bounds is not None
runs = {i: n for n, run in enumerate(series.root_path_indices) for i in run}
frames = []
for i in (0, 10, 20, 30, 40, 50):
    frame = series.query(series.root_trajectory[i].global_seconds).model_dump(
        mode="json"
    )
    frame |= {"id": series.root_trajectory[i].id, "path_run": runs.get(i)}
    frames.append(frame)
meta = {
    "world_unit": series.world_unit,
    "participant_id": series.participant_id,
    "scene_bounds": series.scene_bounds.model_dump(mode="json"),
    "start_seconds": 0,
    "end_seconds": 1,
    "max_gap_seconds": 0.15,
}
summary = {
    "ground_frames": frames,
    "placements": [
        e.model_dump(mode="json") | {"id": e.footprint.id + "/placement"}
        for e in series.physical.placements.events
    ],
    "rotations": [
        e.model_dump(mode="json") | {"id": e.pivot.id + "/rotation"}
        for e in series.physical.events
    ],
    "placement_relations": [
        e.model_dump(mode="json") | {"id": f"relation-{i}"}
        for i, e in enumerate(series.physical.placements.relations)
    ],
}
Path("web/tests/fixtures/ground.json").write_text(
    json.dumps({"meta": meta, "summary": summary}, separators=(",", ":")) + "\n"
)
