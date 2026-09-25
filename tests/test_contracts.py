"""Synthetic interchange examples: no video, detector, or network needed."""

import copy
import json
from typing import Any, cast

import jsonschema
import pytest
from pydantic import ValidationError

from contracts.models import FrameTime, validate_artifact, validate_bundle
from contracts.schema import SCHEMA_PATH, schema_text

PROV = {
    "producer": "synthetic-test",
    "model": "fixture",
    "model_version": "1",
    "config_digest": "0" * 64,
}
GOOD = {"state": "observed", "score": 0.7, "source_ids": []}
UNKNOWN: dict[str, Any] = {"state": "unknown", "score": None, "source_ids": []}
IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def base(kind: str, identifier: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "id": identifier,
        "schema_version": "1.0.0",
        "provenance": PROV,
    }


def bundle() -> list[dict[str, Any]]:
    project = base("project", "project") | {
        "source_ids": ["source-left", "source-right"],
        "participant_ids": ["person"],
    }
    sources = [
        base("source", f"source-{side}")
        | {
            "camera_id": side,
            "width_px": 1920,
            "height_px": 1080,
            "time_base_num": 1,
            "time_base_den": den,
        }
        for side, den in (("left", 30), ("right", 60))
    ]
    sync = base("synchronization", "sync") | {
        "offsets": [
            {"source_id": "source-left", "automatic_seconds": 0.5, "quality": GOOD},
            {
                "source_id": "source-right",
                "automatic_seconds": -0.25,
                "manual_correction_seconds": 0.25,
                "quality": GOOD,
            },
        ]
    }
    calibration = base("calibration", "cal") | {
        "scale": "arbitrary",
        "world_unit": "arbitrary",
        "camera_status": "resolved",
        "ground_z": 0,
        "ground_status": "resolved",
        "source_revision": "fixture",
        "ground_frame": {
            "source_to_world": IDENTITY,
            "plane_normal_source": [0, 0, 1],
            "plane_offset_source": 0,
            "inlier_count": 3, "sample_count": 3, "coverage": 1,
            "rms_residual": 0, "normal_uncertainty_rad": 0,
            "axis_uncertainty_rad": 0,
            "evidence_kind": "scene", "evidence_ids": ["fixture-floor"],
            "evidence_producer": "fixture",
            "source_revision": "fixture",
        },
        "quality": GOOD,
        "cameras": [
            {
                "camera_id": side,
                "source_id": f"source-{side}",
                "intrinsics": {"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
                "world_to_camera": IDENTITY,
                "quality": GOOD,
            }
            for side in ("left", "right")
        ],
    }
    observation = base("observation", "obs-left") | {
        "frame": {
            "source_id": "source-left",
            "camera_id": "left",
            "frame_index": 30,
            "pts": 30,
            "time_base_num": 1,
            "time_base_den": 30,
            "source_seconds": 1.0,
            "offset_seconds": 0.5,
            "global_seconds": 1.5,
        },
        "landmarks": [
            {
                "name": "left_index_tip",
                "xy_px": [120, 200],
                "raw_score": {"value": 8, "range_min": 0, "range_max": 10},
                "quality": GOOD,
            },
            {"name": "right_heel", "xy_px": None, "quality": UNKNOWN},
        ],
        "regions": [{"part": "left_hand", "xywh_px": [100, 180, 50, 60]}],
    }
    reconstruction = base("reconstruction", "motion") | {
        "calibration_id": "cal",
        "participant_id": "person",
        "scale": "arbitrary",
        "samples": [
            {
                "global_seconds": 1.5,
                "root_xyz_world": [0, 0, 1],
                "root_orientation": {"wxyz": [1, 0, 0, 0]},
                "landmarks": [
                    {
                        "name": "left_index_tip",
                        "xyz_world": [0.5, 0, 1.2],
                        "quality": GOOD | {"source_ids": ["obs-left"]},
                    }
                ],
                "segments": [
                    {
                        "segment": "left_forearm",
                        "parent": "root",
                        "orientation": {"wxyz": [1, 0, 0, 0]},
                        "quality": GOOD,
                    }
                ],
                "quality": GOOD | {"source_ids": ["obs-left"]},
            }
        ],
        "arrays": [
            {
                "id": "positions",
                "dtype": "float64",
                "shape": [1, 3],
                "axes": ["time", "xyz"],
                "unit": "arbitrary",
                "missing_mask_id": "mask",
            },
            {"id": "mask", "dtype": "bool", "shape": [1, 3], "axes": ["time", "xyz"]},
        ],
    }
    morphology = base("morphology", "morph") | {
        "participant_id": "person",
        "measurements": [
            {
                "name": "shoulder_width",
                "value": 0.8,
                "unit": "arbitrary",
                "quality": GOOD,
            },
            {
                "name": "hip_width_ratio",
                "value": 0.3,
                "unit": "body_ratio",
                "quality": GOOD,
            },
        ],
    }
    ground = base("ground", "ground") | {
        "reconstruction_id": "motion",
        "scale": "arbitrary",
        "samples": [
            {
                "global_seconds": 1.5,
                "left": {"state": "contact", "region": "forefoot", "quality": GOOD},
                "right": {"state": "unknown", "quality": UNKNOWN},
                "support": "unknown",
            }
        ],
        "footprints": [
            {
                "id": "print-1",
                "foot": "left",
                "interval": {"start": 1.2, "end": 1.8},
                "xy_ground": [0, 0],
                "yaw_rad": 0,
                "quality": GOOD,
            }
        ],
        "pivots": [
            {
                "id": "pivot-1",
                "foot": "left",
                "interval": {"start": 1.4, "end": 1.7},
                "region": "forefoot",
                "rotation_rad": 0.2,
                "quality": GOOD,
            }
        ],
        "measurements": [
            {"name": "step_length", "value": 0.5, "unit": "arbitrary", "quality": GOOD}
        ],
    }
    semantics = base("semantics", "sem") | {
        "reconstruction_id": "motion",
        "ground_id": "ground",
        "execution": {"start": 1, "end": 3},
        "steps": [
            {
                "id": "step",
                "interval": {"start": 1, "end": 3},
                "action_ids": ["left-block", "right-block"],
            }
        ],
        "stances": [
            {
                "id": "stance-1",
                "interval": {"start": 1, "end": 2.2},
                "label": "front_stance",
                "quality": GOOD,
            }
        ],
        "actions": [
            {
                "id": "left-block",
                "step_id": "step",
                "tracks": ["left_arm"],
                "category": "arm",
                "role": "defense",
                "interval": {"start": 1, "end": 2},
            },
            {
                "id": "right-block",
                "step_id": "step",
                "tracks": ["right_arm"],
                "category": "arm",
                "role": "defense",
                "interval": {"start": 1.4, "end": 2.3},
            },
        ],
        "phases": [
            {
                "id": "extend",
                "action_id": "left-block",
                "name": "extension",
                "interval": {"start": 1.2, "end": 1.8},
            }
        ],
        "keyframes": [
            {
                "id": "crossing",
                "action_id": "left-block",
                "phase_id": "extend",
                "global_seconds": 1.5,
                "event": "forearms_cross",
            }
        ],
        "relations": [
            {
                "id": "cross",
                "subject": "left_forearm",
                "object": "right_forearm",
                "relation": "crossed",
                "front_entity": "left_forearm",
                "interval": {"start": 1.4, "end": 1.6},
                "quality": GOOD,
            }
        ],
    }
    edits = base("manual_edits", "edits") | {
        "automatic_semantics_id": "sem",
        "edits": [
            {
                "id": "edit-1",
                "target_id": "crossing",
                "field_path": "/global_seconds",
                "replacement": 1.55,
                "author": "analyst",
            }
        ],
    }
    artifacts = [
        project,
        *sources,
        sync,
        calibration,
        observation,
        reconstruction,
        morphology,
        ground,
        semantics,
        edits,
    ]
    # JSON artifacts have value semantics; shared Python fixture dicts must not
    # make unrelated quality fields alias one another in mutation tests.
    return cast(list[dict[str, Any]], json.loads(json.dumps(artifacts)))


def test_all_layers_json_round_trip_and_schema() -> None:
    data = bundle()
    models = validate_bundle(data)
    encoded = json.dumps([m.model_dump(mode="json") for m in models], allow_nan=False)
    assert [
        m.model_dump(mode="json") for m in validate_bundle(json.loads(encoded))
    ] == json.loads(encoded)
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    for item in json.loads(encoded):
        jsonschema.validate(item, schema)
    assert SCHEMA_PATH.read_text() == schema_text()


def test_time_and_scale_are_explicit() -> None:
    data = bundle()
    assert data[5]["frame"]["global_seconds"] == 1.5
    assert data[6]["scale"] == "arbitrary"
    assert data[8]["samples"][0]["support"] == "unknown"
    assert (
        data[9]["actions"][0]["interval"]["end"]
        > data[9]["actions"][1]["interval"]["start"]
    )
    assert data[9]["relations"][0]["front_entity"] == "left_forearm"


def test_different_native_rates_can_share_one_global_instant() -> None:
    left = FrameTime.model_validate(bundle()[5]["frame"])
    right = FrameTime(
        source_id="source-right",
        camera_id="right",
        frame_index=90,
        pts=90,
        time_base_num=1,
        time_base_den=60,
        source_seconds=1.5,
        offset_seconds=0.0,
        global_seconds=1.5,
    )
    assert left.frame_index != right.frame_index
    assert left.global_seconds == right.global_seconds


def test_unknown_contact_differs_from_known_no_contact() -> None:
    data = bundle()
    sample = data[8]["samples"][0]
    assert sample["right"]["state"] == "unknown"
    sample["right"] = {"state": "no_contact", "quality": GOOD}
    sample["support"] = "left"
    assert validate_bundle(data)[8].kind == "ground"


def test_ground_frame_and_arbitrary_scale_gate_outputs() -> None:
    data = bundle()
    calibration = data[4]
    calibration["ground_status"] = "unresolved"
    calibration["ground_z"] = None
    calibration["ground_frame"] = None
    with pytest.raises(ValueError, match="resolved calibration ground"):
        validate_bundle(data)
    calibration["ground_status"] = "resolved"
    calibration["ground_z"] = 0
    calibration["ground_frame"] = bundle()[4]["ground_frame"]
    data[6]["arrays"][0]["unit"] = "m"
    with pytest.raises(ValueError, match="metric arrays"):
        validate_bundle(data)


@pytest.mark.parametrize(
    ("index", "path", "valid_source"),
    [
        (3, ["offsets", 0, "quality"], "source-left"),
        (4, ["quality"], "source-left"),
        (4, ["cameras", 0, "quality"], "source-left"),
        (5, ["landmarks", 0, "quality"], "source-left"),
        (6, ["samples", 0, "quality"], "obs-left"),
        (6, ["samples", 0, "landmarks", 0, "quality"], "obs-left"),
        (6, ["samples", 0, "segments", 0, "quality"], "obs-left"),
        (7, ["measurements", 0, "quality"], "motion"),
        (8, ["samples", 0, "left", "quality"], "motion"),
        (8, ["samples", 0, "right", "quality"], "motion"),
        (8, ["footprints", 0, "quality"], "motion"),
        (8, ["pivots", 0, "quality"], "motion"),
        (8, ["measurements", 0, "quality"], "motion"),
        (9, ["stances", 0, "quality"], "ground"),
        (9, ["relations", 0, "quality"], "motion"),
    ],
)
def test_all_quality_sources_resolve_to_the_evidence_layer(
    index: int, path: list[str | int], valid_source: str
) -> None:
    data = bundle()
    quality: Any = data[index]
    for key in path:
        quality = quality[key]
    quality["source_ids"] = [valid_source]
    validate_bundle(data)
    quality["source_ids"] = ["nonexistent-observation"]
    with pytest.raises(ValueError, match="quality source"):
        validate_bundle(data)
    quality["source_ids"] = ["project"]
    with pytest.raises(ValueError, match="quality source"):
        validate_bundle(data)


@pytest.mark.parametrize(
    ("index", "path", "bad"),
    [
        (0, ["schema_version"], "2.0.0"),
        (5, ["frame", "global_seconds"], 30.0),
        (5, ["frame", "time_base_den"], 60),
        (5, ["landmarks", 1, "xy_px"], [0, 0]),
        (5, ["landmarks", 0, "raw_score", "value"], 11),
        (6, ["arrays", 0, "shape"], [1, 2]),
        (6, ["arrays", 0, "axes"], ["time"]),
        (6, ["samples", 0, "landmarks", 0, "quality", "source_ids"], ["missing"]),
        (7, ["participant_id"], "missing"),
        (8, ["measurements", 0, "unit"], "m"),
        (8, ["samples", 0, "support"], "left"),
        (9, ["actions", 0, "step_id"], "missing"),
        (9, ["relations", 0, "subject"], "imaginary_segment"),
        (9, ["keyframes", 0, "global_seconds"], 5.0),
        (10, ["edits", 0, "target_id"], "missing"),
    ],
)
def test_reject_corrupt_artifacts(
    index: int, path: list[str | int], bad: object
) -> None:
    data = copy.deepcopy(bundle())
    target: Any = data[index]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad
    with pytest.raises((ValueError, ValidationError)):
        validate_bundle(data)


def test_json_never_emits_nan() -> None:
    item = bundle()[5]
    item["landmarks"][0]["xy_px"][0] = float("nan")
    with pytest.raises(ValidationError):
        validate_artifact(item)
