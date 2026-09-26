"""Pinned, virtual test windows over registered local Mendeley sources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from storage import MissingResource, StorageRoot
from tkd_poomsae.dataset_import import open_registered_project


@dataclass(frozen=True)
class Window:
    execution_id: str
    camera_id: str
    source_sha256: str
    source_path: Path
    start_seconds: float
    end_seconds: float

    def to_window_seconds(self, source_seconds: float) -> float:
        return source_seconds - self.start_seconds

    def to_source_seconds(self, window_seconds: float) -> float:
        return window_seconds + self.start_seconds


def catalog() -> dict[str, Any]:
    path = files("datasets").joinpath("mendeley-bjy7vr4xkt-v1/selections.json")
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("unsupported selection catalog")
    return cast(dict[str, Any], record["selections"])


def resolve(name: str, *, root: StorageRoot | None = None) -> tuple[Window, ...]:
    """Resolve verified local project sources; never acquire or copy media."""
    selections = catalog()
    if name not in selections:
        raise ValueError(f"unknown selection {name!r}; choose: {', '.join(selections)}")
    root = root or StorageRoot.from_env()
    windows: list[Window] = []
    for execution in selections[name]["executions"]:
        execution_id = execution["execution_id"]
        try:
            project = open_registered_project(execution_id, root=root)
        except (FileNotFoundError, MissingResource) as exc:
            raise ValueError(
                "registered local data missing; run: "
                "tkd-poomsae datasets bootstrap mendeley-bjy7vr4xkt-v1 "
                "&& tkd-poomsae datasets register"
            ) from exc
        by_camera = {view["camera_id"]: view for view in project["views"]}
        for selected in execution["views"]:
            camera_id = selected["camera_id"]
            view = by_camera.get(camera_id)
            if view is None or view["sha256"] != selected["source_sha256"]:
                raise ValueError(
                    f"selection {name} source identity changed: {camera_id}"
                )
            start = float(selected["start_seconds"])
            end = float(selected["end_seconds"])
            if not (
                view["first_source_seconds"]
                <= start
                < end
                <= view["last_source_seconds"]
            ):
                raise ValueError(f"selection {name} outside readable PTS: {camera_id}")
            windows.append(
                Window(
                    execution_id,
                    camera_id,
                    view["sha256"],
                    Path(view["path"]),
                    start,
                    end,
                )
            )
    return tuple(windows)
