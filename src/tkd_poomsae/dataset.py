"""Pinned, offline Mendeley dataset registration and read-only file provider."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from storage import CorruptArtifact, MissingResource, StorageRoot

DATASET_ID = "bjy7vr4xkt"
VERSION = 1
_REGISTRATION = "mendeley-bjy7vr4xkt-v1/registration.json"


class InvalidRegistration(ValueError):
    """The committed inventory cannot safely identify the pinned dataset."""


class UnsupportedVersion(ValueError):
    """Only the registered publisher version can be resolved."""


@dataclass(frozen=True)
class DatasetFile:
    path: str
    size_bytes: int
    kind: Literal["video", "detector_csv"]
    crc32: str


@dataclass(frozen=True)
class DatasetStatus:
    state: Literal["available", "missing", "incomplete"]
    root: Path
    expected_files: int
    present_files: int
    missing_paths: tuple[str, ...]
    wrong_size_paths: tuple[str, ...]
    bootstrap_command: str | None


def _safe_path(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not value.startswith("/")
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in value.split("/"))
        and PurePosixPath(value).as_posix() == value
    )


def load_registration() -> dict[str, Any]:
    """Load and validate the committed registration, independent of cwd."""
    path = files("datasets").joinpath(_REGISTRATION)
    try:
        record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        source = record["source"]
        inventory = record["inventory"]
        archive = inventory["archive"]
        resolution = record["resolution"]
        policy = record["usage_policy"]
        members = inventory["members"]
        if (
            record["schema_version"] != 1
            or source["dataset_id"] != DATASET_ID
            or source["version"] != VERSION
            or source["doi"] != "10.17632/bjy7vr4xkt.1"
            or source["licence"]["id"] != "CC-BY-4.0"
            or archive["filename"] != "Data.zip"
            or not archive["file_id"]
            or re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]) is None
            or archive["size_bytes"] <= 0
            or resolution["namespace"] != "datasets"
            or resolution["relative_root"] != "mendeley/bjy7vr4xkt/v1"
            or not resolution["bootstrap_command"]
            or policy["csv_is_ground_truth"] is not False
            or inventory["archive_member_count"] != len(members)
            or len(members) != 32
        ):
            raise InvalidRegistration("pinned dataset registration is incomplete")
        seen: set[str] = set()
        counts = {"video": 0, "detector_csv": 0}
        for member in members:
            name = member["path"]
            kind = member["kind"]
            if (
                not _safe_path(name)
                or name in seen
                or kind not in counts
                or not isinstance(member["size_bytes"], int)
                or member["size_bytes"] <= 0
                or not isinstance(member["crc32"], str)
                or re.fullmatch(r"[0-9a-f]{8}", member["crc32"]) is None
            ):
                raise InvalidRegistration("invalid or duplicate archive member")
            if (kind == "video") != name.endswith(".mp4") or (
                kind == "detector_csv"
            ) != name.endswith(".csv"):
                raise InvalidRegistration("archive member kind does not match path")
            seen.add(name)
            counts[kind] += 1
        if counts != {"video": 16, "detector_csv": 16}:
            raise InvalidRegistration("expected 16 videos and 16 detector CSV files")
        ordinals = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th")
        expected_paths = {
            f"Data/{directory}/Taegeuk_{ordinal}/{view}/{ordinal}_{view}.{suffix}"
            for ordinal in ordinals
            for view in ("Frontal", "Lateral")
            for directory, suffix in (("video", "mp4"), ("annotations", "csv"))
        }
        if seen != expected_paths:
            raise InvalidRegistration("publisher archive member paths are incomplete")
        return record
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise InvalidRegistration("pinned dataset registration is incomplete") from exc


class MendeleyDatasetProvider:
    """Resolve only registered version-1 files already on shared local storage."""

    def __init__(
        self, root: StorageRoot | None = None, *, version: int = VERSION
    ) -> None:
        if version != VERSION:
            raise UnsupportedVersion(
                f"Mendeley {DATASET_ID} version {version} is not registered; "
                "only version 1 is supported"
            )
        self.root = root or StorageRoot.from_env()
        self.registration = load_registration()
        self.relative_root: str = self.registration["resolution"]["relative_root"]
        self.bootstrap_command: str = self.registration["resolution"][
            "bootstrap_command"
        ]
        self.members = tuple(
            DatasetFile(
                path=item["path"],
                size_bytes=item["size_bytes"],
                kind=item["kind"],
                crc32=item["crc32"],
            )
            for item in self.registration["inventory"]["members"]
        )
        self._by_path = {member.path: member for member in self.members}

    @property
    def path(self) -> Path:
        return self.root.namespace("datasets") / self.relative_root

    def status(self) -> DatasetStatus:
        missing: list[str] = []
        wrong_size: list[str] = []
        present = 0
        for member in self.members:
            try:
                path = self.root.existing(
                    "datasets", f"{self.relative_root}/{member.path}"
                )
            except MissingResource:
                missing.append(member.path)
                continue
            except CorruptArtifact:
                wrong_size.append(member.path)
                continue
            if path.stat().st_size != member.size_bytes:
                wrong_size.append(member.path)
            else:
                present += 1
        state: Literal["available", "missing", "incomplete"]
        if not missing and not wrong_size:
            state = "available"
        elif present == 0 and not wrong_size:
            state = "missing"
        else:
            state = "incomplete"
        return DatasetStatus(
            state=state,
            root=self.path,
            expected_files=len(self.members),
            present_files=present,
            missing_paths=tuple(missing),
            wrong_size_paths=tuple(wrong_size),
            bootstrap_command=None if state == "available" else self.bootstrap_command,
        )

    def resolve_file(self, relative_path: str) -> Path:
        """Return a registered extracted file after a local size check; never fetch."""
        if relative_path not in self._by_path:
            raise ValueError(f"unregistered dataset path: {relative_path!r}")
        try:
            path = self.root.existing(
                "datasets", f"{self.relative_root}/{relative_path}"
            )
        except MissingResource as exc:
            raise MissingResource(f"{exc}; run: {self.bootstrap_command}") from exc
        if path.stat().st_size != self._by_path[relative_path].size_bytes:
            raise MissingResource(
                f"Incomplete dataset file: {path}; run: {self.bootstrap_command}"
            )
        return path

    def demo_metadata(self) -> dict[str, Any]:
        """Source and usage details that exported demos must retain."""
        return {
            "source": self.registration["source"].copy(),
            "usage_policy": self.registration["usage_policy"].copy(),
        }
