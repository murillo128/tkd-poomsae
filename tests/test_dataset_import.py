"""Synthetic CSV contracts and source-PTS precedence for optional observations."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from media import ingest
from storage import StorageRoot, hash_file
from tests.test_media_reader import video
from tkd_poomsae import dataset_import
from tkd_poomsae.dataset import MendeleyDatasetProvider
from tkd_poomsae.dataset_import import (
    CSV_FIELDS,
    JOINTS,
    CSVSchemaError,
    _recording_facts,
    inventory,
    open_registered_project,
    parse_csv,
    register_all,
)


def row(index: int = 0, time_ms: float = 0) -> dict[str, str]:
    result = {"frame_index": str(index), "time_ms": str(time_ms)}
    for joint in JOINTS:
        result.update(
            {f"{joint}_x": "0.25", f"{joint}_y": "0.5", f"{joint}_score": "0.8"}
        )
    return result


def write_csv(
    path: Path, rows: list[dict[str, str]], *, fields: tuple[str, ...] = CSV_FIELDS
) -> Path:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_missing_joint_and_canonical_body_mapping(tmp_path: Path) -> None:
    values = row()
    for suffix in ("x", "y", "score"):
        values[f"L_wrist_{suffix}"] = "NaN"
    parsed = parse_csv(write_csv(tmp_path / "pose.csv", [values]))
    assert parsed.missing_landmarks == 1
    left = video(tmp_path / "left.mkv", [200, 240])
    right = video(tmp_path / "right.mkv", [400, 440])
    recording = ingest([("front", left), ("side", right)])[0]
    observation = parsed.observation(parsed.rows[0], recording)
    assert observation.provenance.producer == "mediapipe-csv"
    assert observation.provenance.model == "MediaPipe"
    assert observation.frame.pts == recording.frames[0].pts
    assert observation.frame.frame_index is None
    assert observation.frame.source_seconds == pytest.approx(0.2)
    assert len(observation.landmarks) == 12
    assert {point.name for point in observation.landmarks} == {
        name.replace("L_", "left_").replace("R_", "right_") for name in JOINTS
    }
    wrist = next(point for point in observation.landmarks if point.name == "left_wrist")
    assert wrist.xy_px is None and wrist.quality.state == "unknown"
    assert all(
        "hand" not in point.name and "head" not in point.name
        for point in observation.landmarks
    )


def test_duplicate_missing_indices_and_native_pts_mismatch(tmp_path: Path) -> None:
    parsed = parse_csv(
        write_csv(
            tmp_path / "pose.csv", [row(0, 0), row(2, 80), row(2, 80), row(4, 160)]
        )
    )
    assert parsed.duplicate_indices == (2,)
    assert parsed.missing_indices == (1, 3)
    first = video(tmp_path / "a.mkv", [200, 240, 280, 320])
    second = video(tmp_path / "b.mkv", [0, 40, 80, 120])
    recording = ingest([("a", first), ("b", second)])[0]
    facts = _recording_facts(recording, parsed)
    issues = facts["discrepancies"]
    assert issues["duplicate_csv_indices"] == (2,)
    assert issues["missing_csv_indices"] == (1, 3)
    assert issues["csv_indices_without_video_frame"] == [4]
    assert issues["csv_time_vs_video_pts"]["count"] == 3
    assert facts["first_source_seconds"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="no video frame"):
        parsed.observation(parsed.rows[-1], recording)


def test_off_frame_coordinate_is_reported_without_clipping(tmp_path: Path) -> None:
    values = row()
    values["L_wrist_x"] = "1.01"
    parsed = parse_csv(write_csv(tmp_path / "offscreen.csv", [values]))
    assert parsed.out_of_bounds == ((0, "L_wrist"),)
    first = video(tmp_path / "a.mkv", [0, 40])
    second = video(tmp_path / "b.mkv", [100, 140])
    recording = ingest([("a", first), ("b", second)])[0]
    observation = parsed.observation(parsed.rows[0], recording)
    wrist = next(point for point in observation.landmarks if point.name == "left_wrist")
    assert wrist.xy_px is not None and wrist.xy_px[0] > recording.source.width_px - 1
    assert (
        _recording_facts(recording, parsed)["discrepancies"][
            "csv_coordinates_outside_frame"
        ]["count"]
        == 1
    )


@pytest.mark.parametrize(
    "change",
    [
        {"L_hip_x": "inf"},
        {"R_knee_score": "-0.01"},
        {"L_wrist_y": "NaN"},
        {"frame_index": "-1"},
        {"time_ms": "not-a-number"},
    ],
)
def test_invalid_values_fail_with_line(change: dict[str, str], tmp_path: Path) -> None:
    values = row()
    values.update(change)
    with pytest.raises(CSVSchemaError, match=":2:"):
        parse_csv(write_csv(tmp_path / "bad.csv", [values]))


def test_malformed_columns_and_row_length(tmp_path: Path) -> None:
    with pytest.raises(CSVSchemaError, match="columns"):
        parse_csv(write_csv(tmp_path / "missing.csv", [row()], fields=CSV_FIELDS[:-1]))
    path = write_csv(tmp_path / "short.csv", [row()])
    with path.open("a") as stream:
        stream.write("1,33\n")
    with pytest.raises(CSVSchemaError, match="malformed column count"):
        parse_csv(path)


def test_two_projects_reuse_source_objects_and_reopen_from_any_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = StorageRoot(tmp_path)
    provider = MendeleyDatasetProvider(root)
    paths = [tmp_path / "front.mp4", tmp_path / "side.mp4"]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index + 1]))
    views = [
        {
            "camera_id": f"camera-{index}",
            "path": str(path),
            "sha256": hash_file(path),
            "source_id": f"source:{hash_file(path)}",
            "size_bytes": path.stat().st_size,
        }
        for index, path in enumerate(paths)
    ]
    report = {
        "dataset": {"source": "test"},
        "timing_authority": "video PTS",
        "csv_usage": "optional",
        "projects": [
            {"execution_id": f"mendeley-test-{number}", "views": views}
            for number in (1, 2)
        ],
    }
    monkeypatch.setattr(dataset_import, "inventory", lambda _provider: report)
    register_all(provider)
    register_all(provider)
    original = [
        open_registered_project(f"mendeley-test-{number}", root=root)
        for number in (1, 2)
    ]
    monkeypatch.chdir(tmp_path / "..")
    reopened = [
        open_registered_project(f"mendeley-test-{number}", root=root)
        for number in (1, 2)
    ]
    assert original == reopened
    assert original[0]["source_objects"] == original[1]["source_objects"]
    assert len(list((root.namespace("runs") / "dataset-sources").glob("*.json"))) == 2
    paths[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="source reference changed"):
        open_registered_project("mendeley-test-1", root=root)


def test_demo_falls_back_to_first_readable_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = MendeleyDatasetProvider(StorageRoot(tmp_path))
    provider.path.mkdir(parents=True)
    bad = provider.path / "bad.mp4"
    bad.write_bytes(b"invalid video")
    front = video(provider.path / "front.mkv", [0, 40])
    side = video(provider.path / "side.mkv", [100, 140])
    annotation = write_csv(provider.path / "pose.csv", [row(0, 0), row(1, 40)])
    pairs = {
        "1st": {"Frontal": (bad, annotation), "Lateral": (side, annotation)},
        "2nd": {"Frontal": (front, annotation), "Lateral": (side, annotation)},
    }
    monkeypatch.setattr(provider, "status", lambda: SimpleNamespace(state="available"))
    monkeypatch.setattr(
        provider, "resolve_file", lambda relative: provider.path / relative
    )
    monkeypatch.setattr(dataset_import, "_pair_paths", lambda _provider: pairs)
    report = inventory(provider)
    assert report["demo_execution_id"] == "mendeley-bjy7vr4xkt-v1-taegeuk-2"
    assert len(report["unreadable_forms"]) == 1
    assert report["unreadable_forms"][0]["form"] == "1st"
