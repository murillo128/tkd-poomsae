"""Shared virtual windows and derived media use synthetic sources by default."""

from __future__ import annotations

from pathlib import Path

import pytest

from media import DecodeError, index_recording
from storage import CorruptArtifact, StorageRoot, hash_file
from tests.test_media_reader import video
from tkd_poomsae import selections
from tkd_poomsae.media_variants import VariantRecipe, materialize
from tkd_poomsae.selections import Window


def test_two_worktrees_resolve_same_virtual_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = video(tmp_path / "source.mkv", [i * 40 for i in range(12)])
    digest = hash_file(source)
    monkeypatch.setattr(
        selections,
        "catalog",
        lambda: {
            "short": {
                "executions": [
                    {
                        "execution_id": "test",
                        "views": [
                            {
                                "camera_id": "front",
                                "source_sha256": digest,
                                "start_seconds": 0.08,
                                "end_seconds": 0.32,
                            }
                        ],
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(
        selections,
        "open_registered_project",
        lambda *_args, **_kwargs: {
            "views": [
                {
                    "camera_id": "front",
                    "sha256": digest,
                    "path": str(source),
                    "first_source_seconds": 0,
                    "last_source_seconds": 0.44,
                }
            ]
        },
    )
    first = tmp_path / "worktree-a"
    second = tmp_path / "worktree-b"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    a = selections.resolve("short", root=StorageRoot(tmp_path))
    monkeypatch.chdir(second)
    b = selections.resolve("short", root=StorageRoot(tmp_path))
    assert a == b
    assert a[0].to_source_seconds(a[0].to_window_seconds(0.24)) == pytest.approx(0.24)


def test_explicit_missing_local_selection_has_bootstrap_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError()

    monkeypatch.setattr(selections, "open_registered_project", missing)
    with pytest.raises(ValueError, match="datasets bootstrap"):
        selections.resolve("smoke-short", root=StorageRoot(tmp_path))


def test_variant_is_verified_and_reused_with_roundtrip_lineage(tmp_path: Path) -> None:
    source = video(tmp_path / "source.mkv", [i * 40 for i in range(25)])
    before = hash_file(source)
    window = Window("test", "front", before, source, 0.2, 0.6)
    recipe = VariantRecipe(
        fps=15,
        width=16,
        height=12,
        start_offset_seconds=0.12,
        pre_roll_seconds=0.08,
        post_roll_seconds=0.08,
        drop_source_ordinals=(10,),
    )
    root = StorageRoot(tmp_path / "shared")
    first = materialize(window, recipe, root=root)
    second = materialize(window, recipe, root=root)
    assert not first.cache_hit and second.cache_hit
    assert first.path == second.path
    assert hash_file(source) == before
    assert first.manifest == second.manifest
    assert first.manifest["geometry_independent_camera"] is False
    assert first.manifest["output_audio_present"] is False
    recording = index_recording("variant", first.path)
    refs = recording.frames
    assert (recording.source.width_px, recording.source.height_px) == (16, 12)
    mappings = first.manifest["time_mappings"]
    assert len(refs) == len(mappings)
    assert all(
        abs(ref.seconds - item["variant_seconds"]) < 0.001
        for ref, item in zip(refs, mappings)
    )
    assert 10 not in {item["source_ordinal"] for item in mappings}
    assert all(
        abs(
            window.to_source_seconds(window.to_window_seconds(item["source_seconds"]))
            - item["source_seconds"]
        )
        < 1e-9
        for item in mappings
    )
    assert mappings[0]["variant_seconds"] >= 0.119
    first.path.write_bytes(b"changed")
    with pytest.raises(CorruptArtifact, match="changed media variant"):
        materialize(window, recipe, root=root)


def test_variant_rejects_upsampling(tmp_path: Path) -> None:
    source = video(tmp_path / "source.mkv", [i * 100 for i in range(8)])
    window = Window("test", "front", hash_file(source), source, 0, 0.7)
    with pytest.raises(ValueError, match="upsampling"):
        materialize(
            window, VariantRecipe(fps=30), root=StorageRoot(tmp_path / "shared")
        )


def test_corrupt_tail_variant_retains_exact_recipe_and_source(tmp_path: Path) -> None:
    source = video(tmp_path / "source.mkv", [i * 40 for i in range(25)])
    original = hash_file(source)
    window = Window("test", "front", original, source, 0, 0.9)
    recipe = VariantRecipe(corrupt_tail_bytes=200)
    artifact = materialize(window, recipe, root=StorageRoot(tmp_path / "shared"))
    assert artifact.manifest["identity"]["recipe"]["corrupt_tail_bytes"] == 200
    assert hash_file(source) == original
    with pytest.raises(DecodeError):
        index_recording("damaged", artifact.path)


@pytest.mark.local_data
def test_registered_local_smoke_is_readable(
    local_data_windows: tuple[Window, ...],
) -> None:
    assert len(local_data_windows) == 2
    for window in local_data_windows:
        recording = index_recording(window.camera_id, window.source_path)
        assert recording.sha256 == window.source_sha256
        assert any(
            window.start_seconds <= ref.seconds <= window.end_seconds
            for ref in recording.frames
        )


@pytest.mark.full_data
def test_every_registered_form_resolves_offline() -> None:
    windows = selections.resolve("all-forms")
    assert len(windows) == 16
    assert len({window.execution_id for window in windows}) == 8
