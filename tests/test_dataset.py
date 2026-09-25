"""Offline checks of the pinned registration and shared-root resolver."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from storage import MissingResource, StorageRoot
from tkd_poomsae.dataset import (
    MendeleyDatasetProvider,
    UnsupportedVersion,
    load_registration,
)


def _populate(provider: MendeleyDatasetProvider) -> None:
    for member in provider.members:
        path = provider.path / member.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.truncate(member.size_bytes)


def test_published_inventory_and_testing_only_policy() -> None:
    record = load_registration()
    schema = json.loads(
        files("datasets")
        .joinpath("mendeley-bjy7vr4xkt-v1/inventory.schema.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(record["inventory"])
    provider = MendeleyDatasetProvider()
    assert record["source"]["version"] == 1
    assert record["source"]["doi"] == "10.17632/bjy7vr4xkt.1"
    assert record["inventory"]["archive"]["file_id"] == (
        "5a5bd049-cee1-4219-823b-b91305fdd63e"
    )
    assert record["inventory"]["archive"]["sha256"] == (
        "8447119933de8032296c1d60dc0ba5f5586fe9cd2d8560c267bd1d53a29c7940"
    )
    assert len(provider.members) == 32
    assert sum(item.kind == "video" for item in provider.members) == 16
    assert sum(item.kind == "detector_csv" for item in provider.members) == 16
    assert record["usage_policy"]["csv_is_ground_truth"] is False
    assert "MMPose accuracy measurement" in record["usage_policy"]["prohibited"]
    assert "training" in record["usage_policy"]["prohibited"]
    exported = provider.demo_metadata()
    assert exported["source"]["creator"] == "QingWei Zheng"
    assert exported["source"]["licence"]["id"] == "CC-BY-4.0"
    assert exported["usage_policy"]["csv_is_ground_truth"] is False


def test_missing_and_incomplete_dataset(tmp_path: Path) -> None:
    provider = MendeleyDatasetProvider(StorageRoot(tmp_path))
    missing = provider.status()
    assert missing.state == "missing"
    assert missing.expected_files == 32
    assert missing.present_files == 0
    assert missing.bootstrap_command == provider.bootstrap_command
    with pytest.raises(MissingResource, match="datasets bootstrap"):
        provider.resolve_file(provider.members[0].path)

    first = provider.members[0]
    path = provider.path / first.path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"incomplete")
    incomplete = provider.status()
    assert incomplete.state == "incomplete"
    assert first.path in incomplete.wrong_size_paths
    with pytest.raises(MissingResource, match="Incomplete dataset file"):
        provider.resolve_file(first.path)


def test_valid_dataset_and_cwd_independence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = MendeleyDatasetProvider(StorageRoot(tmp_path / "shared"))
    _populate(provider)
    monkeypatch.chdir(tmp_path)
    other_cwd_provider = MendeleyDatasetProvider(StorageRoot(tmp_path / "shared"))
    assert other_cwd_provider.status().state == "available"
    assert other_cwd_provider.status().bootstrap_command is None
    first = other_cwd_provider.members[0]
    assert other_cwd_provider.resolve_file(first.path) == provider.path / first.path
    with pytest.raises(ValueError, match="unregistered"):
        other_cwd_provider.resolve_file("Data/other.mp4")


def test_wrong_version_and_incomplete_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(UnsupportedVersion, match="only version 1"):
        MendeleyDatasetProvider(version=2)
    monkeypatch.setattr("tkd_poomsae.dataset._REGISTRATION", "missing.json")
    with pytest.raises(ValueError, match="registration is incomplete"):
        load_registration()
