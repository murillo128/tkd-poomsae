"""Decoded media through native cues, alignment, and global-time frame lookup."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from contracts.models import Synchronization
from media import MediaReader, Recording, ingest
from storage import StorageRoot, hash_config, hash_file
from sync import (
    CueConfig,
    SyncSource,
    TimelineFailure,
    extract_cues,
    solve_offsets,
    sources_from_manifest,
)
from tkd_poomsae.media_variants import VariantRecipe, materialize
from tkd_poomsae.selections import Window

from .test_sync_cues import recording

EVENTS = (0.75, 1.4, 2.35, 3.15, 4.55, 5.3, 6.45)
PERIODIC = tuple(0.7 * index for index in range(1, 12))
CONFIG = CueConfig(video_hz=20)


@pytest.fixture(autouse=True)
def _report_config(request: pytest.FixtureRequest) -> None:
    request.node.user_properties.append(
        ("sync_config_digest", hash_config(asdict(CONFIG)))
    )
    request.node.user_properties.append(("fixture_revision", "decoded-sync-v1"))


def _record_inputs(
    request: pytest.FixtureRequest,
    recordings: dict[str, Recording],
    scenario: str,
) -> None:
    request.node.user_properties.append(
        (
            "input_identity",
            {
                "scenario": scenario,
                "source_sha256": {
                    name: item.sha256 for name, item in recordings.items()
                },
            },
        )
    )


def _sources(
    tmp_path: Path,
    cameras: tuple[tuple[str, float, int], ...],
    *,
    events: tuple[float, ...] = EVENTS,
    audio: bool = True,
    vfr_camera: str | None = None,
    bad_camera: str | None = None,
) -> tuple[dict[str, Recording], dict[str, SyncSource]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        name: recording(
            tmp_path / f"{name}.mkv",
            video_rate=rate,
            audio_rate=8000 if audio else None,
            pre_roll=start,
            events=events,
            duration=7.2 + index * 0.2,
            visual_column=8 + 8 * index,
            vfr=name == vfr_camera,
            static=name == bad_camera,
            silent=name == bad_camera,
        )
        for index, (name, start, rate) in enumerate(cameras)
    }
    manifest = ingest(list(paths.items()))
    by_camera = {item.camera_id: item for item in manifest}
    cues = {
        item.source_id: extract_cues(item, config=CONFIG, cache_dir=tmp_path / "cues")
        for item in manifest
    }
    return by_camera, sources_from_manifest(manifest, cues)


@pytest.mark.parametrize(
    "cameras,audio,vfr_camera",
    [
        ((("a", 0.4, 20), ("b", 0.73, 25)), False, None),
        ((("a", 0.4, 20), ("b", 0.73, 25), ("c", 1.08, 30)), True, None),
        (
            (("a", 0.4, 20), ("b", 0.73, 25), ("c", 1.08, 30), ("d", 0.16, 24)),
            True,
            "d",
        ),
    ],
)
def test_decoded_multicamera_alignment_and_lookup(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    cameras: tuple[tuple[str, float, int], ...],
    audio: bool,
    vfr_camera: str | None,
) -> None:
    recordings, sources = _sources(
        tmp_path, cameras, audio=audio, vfr_camera=vfr_camera
    )
    _record_inputs(
        request, recordings, f"{len(cameras)}-camera-audio-{audio}-vfr-{vfr_camera}"
    )
    by_id = {item.source_id: item for item in recordings.values()}
    reference = recordings["a"].source_id
    result = solve_offsets(sources, reference=reference)
    rows = {row.source_id: row for row in result.offsets}
    assert len(rows) == len(cameras)
    assert all(row.retained and row.exclusion_reason is None for row in rows.values())
    assert result.common_interval is not None
    assert result.common_interval.end - result.common_interval.start > 5
    for name, start, rate in cameras:
        item = recordings[name]
        row = rows[item.source_id]
        expected = cameras[0][1] - start
        # One 50 ms motion bin plus one video frame, with a 50 ms margin for
        # encoding and audio-window placement. This is a synthetic oracle.
        tolerance = CONFIG.video_hz**-1 + rate**-1 + 0.05
        assert row.effective_seconds == pytest.approx(expected, abs=tolerance)
        assert row.automatic_seconds == pytest.approx(expected, abs=tolerance)
        assert row.source_interval is not None
        assert row.global_interval is not None
        assert row.global_interval.start == pytest.approx(
            row.source_interval.start + row.effective_seconds
        )
        reader = MediaReader(item)
        for event in (EVENTS[1], EVENTS[3], EVENTS[5]):
            global_seconds = cameras[0][1] + event + 0.075
            ref = reader.nearest(global_seconds - row.effective_seconds)
            frame = reader.frame(ref)
            stamped = item.frame_time(frame.ref, row.effective_seconds)
            assert stamped.global_seconds == pytest.approx(
                global_seconds, abs=tolerance + rate**-1 / 2
            )
            assert stamped.source_seconds == pytest.approx(frame.ref.seconds)
            assert frame.rgb.shape[:2] == (48, 64)
            nearby = (
                reader.frame(
                    reader.nearest(global_seconds + delta - row.effective_seconds)
                )
                for delta in (-0.05, 0, 0.05)
            )
            assert any(
                np.count_nonzero(candidate.rgb[:, :, 0] > 150) > 450
                for candidate in nearby
            )
    if vfr_camera:
        times = [ref.seconds for ref in recordings[vfr_camera].frames]
        assert len({round(b - a, 3) for a, b in zip(times, times[1:])}) > 1
    assert all(by_id[row.source_id].sha256 for row in result.offsets)


def test_reference_change_preserves_pairwise_correspondence(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    cameras = (("a", 0.4, 20), ("b", 0.73, 25), ("c", 1.08, 30))
    recordings, sources = _sources(tmp_path, cameras, audio=False)
    _record_inputs(request, recordings, "reference-change-visual")
    first = solve_offsets(sources, reference=recordings["a"].source_id)
    second = solve_offsets(sources, reference=recordings["c"].source_id)
    a = {row.source_id: row.effective_seconds for row in first.offsets}
    c = {row.source_id: row.effective_seconds for row in second.offsets}
    for left in a:
        for right in a:
            assert a[left] - a[right] == pytest.approx(c[left] - c[right])
    assert c[recordings["c"].source_id] == pytest.approx(0)


def test_ambiguous_and_empty_evidence_fail_closed(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    cameras = (("a", 0.4, 20), ("b", 0.73, 25))
    repeated_recordings, repeated = _sources(
        tmp_path / "repeated", cameras, events=PERIODIC, audio=False
    )
    _record_inputs(request, repeated_recordings, "repeated-visual")
    with pytest.raises(TimelineFailure):
        solve_offsets(repeated)
    empty_recordings, empty = _sources(
        tmp_path / "empty", cameras, events=(), audio=False, bad_camera="b"
    )
    _record_inputs(request, empty_recordings, "silent-static")
    with pytest.raises(TimelineFailure):
        solve_offsets(empty)


def test_no_common_overlap_and_bad_camera_exclusion(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    cameras: tuple[tuple[str, float, int], ...] = (("a", 0.4, 20), ("b", 0.73, 25))
    disjoint_recordings, disjoint = _sources(tmp_path / "disjoint", cameras)
    _record_inputs(request, disjoint_recordings, "manual-disjoint-intervals")
    second_id = next(
        key for key, source in disjoint.items() if source.start_seconds > 0.5
    )
    with pytest.raises(TimelineFailure, match="overlapping views"):
        solve_offsets(
            disjoint,
            reference=disjoint_recordings["a"].source_id,
            overrides={
                second_id: {
                    "offset_seconds": 30.0,
                    "author": "operator",
                    "source": "external clock",
                    "reason": "recording was disjoint",
                }
            },
        )
    cameras = (("a", 0.4, 20), ("b", 0.73, 25), ("bad", 1.08, 30))
    recordings, sources = _sources(tmp_path / "bad", cameras, bad_camera="bad")
    _record_inputs(request, recordings, "one-bad-camera")
    result = solve_offsets(sources)
    retained = {row.source_id for row in result.offsets if row.retained}
    assert retained == {recordings[name].source_id for name in ("a", "b")}
    bad = next(
        row for row in result.offsets if row.source_id == recordings["bad"].source_id
    )
    assert bad.exclusion_reason and bad.automatic_seconds is None
    assert bad.global_interval is None
    assert any(d.startswith("excluded:") for d in result.diagnostics)


def test_manual_roundtrip_changes_lookup_without_changing_native_data(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    recordings, sources = _sources(
        tmp_path, (("a", 0.4, 20), ("b", 0.73, 25)), audio=True
    )
    _record_inputs(request, recordings, "manual-correction")
    reference = recordings["a"].source_id
    automatic = solve_offsets(sources, reference=reference)
    source_id = recordings["b"].source_id
    before = next(row for row in automatic.offsets if row.source_id == source_id)
    native_hash = hash_file(recordings["b"].path)
    cue_cache = sorted((tmp_path / "cues").rglob("*.json"))
    cue_hashes = {str(path): hash_file(path) for path in cue_cache}
    override = {
        "offset_seconds": before.effective_seconds + 0.24,
        "author": "operator",
        "source": "frame inspection",
        "reason": "correct correspondence",
    }
    changed = solve_offsets(
        sources, reference=reference, overrides={source_id: override}
    )
    restored = Synchronization.model_validate_json(changed.model_dump_json())
    after = next(row for row in restored.offsets if row.source_id == source_id)
    assert after.automatic_seconds == before.automatic_seconds
    assert after.effective_seconds == pytest.approx(override["offset_seconds"])
    assert after.manual_correction_seconds == pytest.approx(0.24)
    assert (after.manual_author, after.manual_source, after.manual_reason) == (
        "operator",
        "frame inspection",
        "correct correspondence",
    )
    reader = MediaReader(recordings["b"])
    target = 0.4 + EVENTS[3]
    old_ref = reader.nearest(target - before.effective_seconds)
    new_ref = reader.nearest(target - after.effective_seconds)
    assert old_ref.ordinal != new_ref.ordinal
    assert reader.frame(old_ref).ref == old_ref
    assert reader.frame(new_ref).ref == new_ref
    assert hash_file(recordings["b"].path) == native_hash
    assert {str(path): hash_file(path) for path in cue_cache} == cue_hashes


@pytest.mark.local_data
def test_registered_selection_imposed_shift_smoke(
    local_data_windows: tuple[Window, ...],
    request: pytest.FixtureRequest,
) -> None:
    """Functional import/variant plumbing; original paired views are not an oracle."""
    assert len(local_data_windows) == 2
    window = local_data_windows[0]
    assert window.source_path.is_file()
    original_hash = hash_file(window.source_path)
    request.node.user_properties.append(
        (
            "input_identity",
            {"selection": "smoke-short", "source_sha256": original_hash},
        )
    )
    root = StorageRoot.from_env()
    variant = materialize(window, VariantRecipe(start_offset_seconds=0.25), root=root)
    assert variant.manifest["identity"]["source_sha256"] == original_hash
    assert variant.manifest["identity"]["recipe"]["start_offset_seconds"] == 0.25
    assert variant.manifest["time_mappings"]
    indexed = ingest([("original", window.source_path), ("imposed", variant.path)])
    mapping = variant.manifest["time_mappings"][0]
    assert indexed[1].frames[0].seconds == pytest.approx(
        mapping["variant_seconds"], abs=0.001
    )
    assert mapping["variant_seconds"] == pytest.approx(0.25, abs=0.001)
    assert hash_file(window.source_path) == original_hash
