"""Offline Mendeley execution inventory and optional MediaPipe CSV observations."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, cast

from contracts.models import (
    FrameTime,
    Landmark,
    Landmark2D,
    Observation,
    Provenance,
    Quality,
    RawScore,
)
from media import IngestError, Recording, ingest
from pipeline import Pipeline
from storage import ArtifactStore, MissingResource, StorageRoot, hash_config, hash_file
from tkd_poomsae.dataset import MendeleyDatasetProvider

JOINTS = (
    "L_shoulder",
    "R_shoulder",
    "L_elbow",
    "R_elbow",
    "L_wrist",
    "R_wrist",
    "L_hip",
    "R_hip",
    "L_knee",
    "R_knee",
    "L_ankle",
    "R_ankle",
)
JOINT_MAP = {
    name: f"{'left' if name[0] == 'L' else 'right'}_{name[2:]}" for name in JOINTS
}
CSV_FIELDS = ("frame_index", "time_ms") + tuple(
    f"{joint}_{field}" for joint in JOINTS for field in ("x", "y", "score")
)
FORMS = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th")
VIEWS = ("Frontal", "Lateral")
PROVIDER = "mediapipe-csv"


class CSVSchemaError(ValueError):
    """A CSV cannot be interpreted as this optional observation provider."""


@dataclass(frozen=True)
class CSVRow:
    source_row: int
    frame_index: int
    time_seconds: float
    # None denotes a missing detector landmark, never an inferred point.
    joints: dict[str, tuple[float, float, float] | None]


@dataclass(frozen=True)
class CSVInventory:
    path: Path
    sha256: str
    rows: tuple[CSVRow, ...]
    duplicate_indices: tuple[int, ...]
    missing_indices: tuple[int, ...]
    missing_landmarks: int
    out_of_bounds: tuple[tuple[int, str], ...]

    def observation(self, row: CSVRow, recording: Recording) -> Observation:
        """Map one CSV row to canonical body landmarks using matched native PTS.

        The CSV frame index is checked against the video's presentation ordinal;
        the video's PTS supplies time. This mapping is valid only after callers
        inspect timing diagnostics. The CSV time is retained in the inventory.
        """
        if row.frame_index < 0 or row.frame_index >= len(recording.frames):
            raise ValueError("CSV index has no video frame")
        frame = recording.frames[row.frame_index]
        source_time = recording.frame_time(frame)
        transform = recording.stored_to_oriented
        landmarks: list[Landmark2D] = []
        for raw_name, point in row.joints.items():
            if point is None:
                landmarks.append(
                    Landmark2D(
                        name=cast(Landmark, JOINT_MAP[raw_name]),
                        xy_px=None,
                        quality=Quality(
                            state="unknown", source_ids=[recording.source_id]
                        ),
                    )
                )
                continue
            x, y, score = point
            stored_x = x * (recording.stored_width_px - 1)
            stored_y = y * (recording.stored_height_px - 1)
            xy = (
                transform[0][0] * stored_x
                + transform[0][1] * stored_y
                + transform[0][2],
                transform[1][0] * stored_x
                + transform[1][1] * stored_y
                + transform[1][2],
            )
            landmarks.append(
                Landmark2D(
                    name=cast(Landmark, JOINT_MAP[raw_name]),
                    xy_px=xy,
                    raw_score=RawScore(value=score, range_min=0, range_max=1),
                    quality=Quality(
                        state="observed", score=score, source_ids=[recording.source_id]
                    ),
                )
            )
        return Observation(
            id=f"{recording.source_id}:{self.sha256}:row-{row.source_row}",
            schema_version="1.0.0",
            kind="observation",
            provenance=Provenance(
                producer=PROVIDER,
                model="MediaPipe",
                model_version=None,
                config_digest=hash_config(
                    {"provider": PROVIDER, "csv_sha256": self.sha256, "schema": 1}
                ),
            ),
            frame=FrameTime(
                source_id=source_time.source_id,
                camera_id=source_time.camera_id,
                # CSV index is a detector claim, not a native MP4 index.
                frame_index=None,
                pts=source_time.pts,
                time_base_num=source_time.time_base_num,
                time_base_den=source_time.time_base_den,
                source_seconds=source_time.source_seconds,
                offset_seconds=0,
                global_seconds=source_time.source_seconds,
            ),
            landmarks=landmarks,
        )


def parse_csv(path: Path) -> CSVInventory:
    """Parse exact named schema, preserving source indices, times and missing points."""
    rows: list[CSVRow] = []
    missing = 0
    out_of_bounds: list[tuple[int, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or tuple(reader.fieldnames) != CSV_FIELDS:
            raise CSVSchemaError(f"{path}: unexpected CSV columns")
        for line, values in enumerate(reader, start=2):
            if None in values or any(value is None for value in values.values()):
                raise CSVSchemaError(f"{path}:{line}: malformed column count")
            try:
                index = int(values["frame_index"])
                time_ms = float(values["time_ms"])
                if index < 0 or not math.isfinite(time_ms) or time_ms < 0:
                    raise ValueError("invalid frame index or time")
                joints: dict[str, tuple[float, float, float] | None] = {}
                for name in JOINTS:
                    fields = [
                        values[f"{name}_{suffix}"].strip()
                        for suffix in ("x", "y", "score")
                    ]
                    if all(
                        value.lower() in ("", "nan", "na", "null") for value in fields
                    ):
                        joints[name] = None
                        missing += 1
                    elif any(
                        value.lower() in ("", "nan", "na", "null") for value in fields
                    ):
                        raise ValueError(f"partial missing joint {name}")
                    else:
                        point = tuple(float(value) for value in fields)
                        if any(not math.isfinite(value) for value in point):
                            raise ValueError(f"non-finite joint {name}")
                        if not 0 <= point[2] <= 1:
                            raise ValueError(f"out-of-range score {name}")
                        if not 0 <= point[0] <= 1 or not 0 <= point[1] <= 1:
                            out_of_bounds.append((index, name))
                        joints[name] = point  # type: ignore[assignment]
                rows.append(CSVRow(line, index, time_ms / 1000, joints))
            except (TypeError, ValueError) as exc:
                raise CSVSchemaError(f"{path}:{line}: {exc}") from exc
    if not rows:
        raise CSVSchemaError(f"{path}: no observations")
    counts = Counter(row.frame_index for row in rows)
    indices = set(counts)
    return CSVInventory(
        path=path,
        sha256=hash_file(path),
        rows=tuple(rows),
        duplicate_indices=tuple(
            sorted(index for index, count in counts.items() if count > 1)
        ),
        missing_indices=tuple(sorted(set(range(max(indices) + 1)) - indices)),
        missing_landmarks=missing,
        out_of_bounds=tuple(out_of_bounds),
    )


def _pair_paths(
    provider: MendeleyDatasetProvider,
) -> dict[str, dict[str, tuple[Path, Path]]]:
    """Pair by actual relative structure and basename; reject unknown layouts."""
    groups: dict[str, dict[str, dict[str, Path]]] = {}
    for member in provider.members:
        parts = member.path.split("/")
        if (
            len(parts) != 5
            or parts[0] != "Data"
            or parts[2] not in {f"Taegeuk_{form}" for form in FORMS}
            or parts[3] not in VIEWS
        ):
            raise ValueError(f"unrecognized dataset path: {member.path}")
        form = parts[2].removeprefix("Taegeuk_")
        view = parts[3]
        if parts[4] != f"{form}_{view}.{('mp4' if member.kind == 'video' else 'csv')}":
            raise ValueError(f"unpaired dataset basename: {member.path}")
        slot = groups.setdefault(form, {}).setdefault(view, {})
        if member.kind in slot:
            raise ValueError(f"duplicate {member.kind} for {form}/{view}")
        slot[member.kind] = provider.path / member.path
    if set(groups) != set(FORMS):
        raise ValueError("dataset form inventory is incomplete")
    result: dict[str, dict[str, tuple[Path, Path]]] = {}
    for form in FORMS:
        if set(groups[form]) != set(VIEWS):
            raise ValueError(f"incomplete views for Taegeuk {form}")
        result[form] = {}
        for view in VIEWS:
            pair = groups[form][view]
            if set(pair) != {"video", "detector_csv"}:
                raise ValueError(f"incomplete video/CSV pair for Taegeuk {form}/{view}")
            result[form][view] = pair["video"], pair["detector_csv"]
    return result


def _recording_facts(recording: Recording, csv_data: CSVInventory) -> dict[str, Any]:
    frames = recording.frames
    deltas = [right.seconds - left.seconds for left, right in zip(frames, frames[1:])]
    fps = 1 / median(deltas) if deltas else None
    mismatches = []
    unmapped = []
    for row in csv_data.rows:
        if row.frame_index >= len(frames):
            unmapped.append(row.frame_index)
        else:
            delta = row.time_seconds - frames[row.frame_index].seconds
            if abs(delta) > 0.001:
                mismatches.append(
                    {"frame_index": row.frame_index, "delta_ms": round(delta * 1000, 4)}
                )
    issues: dict[str, Any] = {}
    if csv_data.duplicate_indices:
        issues["duplicate_csv_indices"] = csv_data.duplicate_indices
    if csv_data.missing_indices:
        issues["missing_csv_indices"] = csv_data.missing_indices
    if csv_data.out_of_bounds:
        issues["csv_coordinates_outside_frame"] = {
            "count": len(csv_data.out_of_bounds),
            "examples": csv_data.out_of_bounds[:5],
        }
    if unmapped:
        issues["csv_indices_without_video_frame"] = sorted(set(unmapped))
    if mismatches:
        issues["csv_time_vs_video_pts"] = {
            "count": len(mismatches),
            "examples": mismatches[:5],
            "max_abs_delta_ms": max(abs(item["delta_ms"]) for item in mismatches),
        }
    if len(csv_data.rows) != len(frames):
        issues["csv_row_count_vs_video_frame_count"] = [len(csv_data.rows), len(frames)]
    if (recording.stored_width_px, recording.stored_height_px) != (1920, 1080):
        issues["publisher_dimensions"] = [1920, 1080]
    if fps is not None and abs(fps - 30) > 0.01:
        issues["publisher_fps"] = 30
    if recording.codec not in {"h264", "avc1"}:
        issues["publisher_codec"] = "H.264"
    return {
        "source_id": recording.source_id,
        "path": str(recording.path),
        "sha256": recording.sha256,
        "size_bytes": recording.size_bytes,
        "codec": recording.codec,
        "stored_dimensions_px": [recording.stored_width_px, recording.stored_height_px],
        "oriented_dimensions_px": [
            recording.source.width_px,
            recording.source.height_px,
        ],
        "rotation_degrees": recording.rotation_degrees,
        "frame_count": len(frames),
        "median_fps_from_pts": fps,
        "pts_time_base": [frames[0].time_base_num, frames[0].time_base_den],
        "first_pts": frames[0].pts,
        "last_pts": frames[-1].pts,
        "first_source_seconds": frames[0].seconds,
        "last_source_seconds": frames[-1].seconds,
        "duration_seconds": recording.duration_seconds,
        "audio_present": recording.audio_present,
        "csv": {
            "path": str(csv_data.path),
            "sha256": csv_data.sha256,
            "provider": PROVIDER,
            "row_count": len(csv_data.rows),
            "first_index": csv_data.rows[0].frame_index,
            "last_index": csv_data.rows[-1].frame_index,
            "first_time_seconds": csv_data.rows[0].time_seconds,
            "last_time_seconds": csv_data.rows[-1].time_seconds,
            "missing_landmarks": csv_data.missing_landmarks,
        },
        "discrepancies": issues,
    }


def inventory(provider: MendeleyDatasetProvider | None = None) -> dict[str, Any]:
    """Read every registered local file and return measured, functional diagnostics."""
    provider = provider or MendeleyDatasetProvider()
    status = provider.status()
    if status.state == "missing":
        raise ValueError(f"dataset {status.state}; run: {provider.bootstrap_command}")
    paired = _pair_paths(provider)
    projects: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for form, views in paired.items():
        try:
            for view in VIEWS:
                for path in views[view]:
                    provider.resolve_file(path.relative_to(provider.path).as_posix())
            manifest = ingest(
                [(f"mendeley-{view.lower()}", views[view][0]) for view in VIEWS]
            )
            items = []
            for view, recording in zip(VIEWS, manifest.recordings):
                csv_data = parse_csv(views[view][1])
                items.append(
                    {
                        "camera_id": recording.camera_id,
                        "view_label": view,
                        **_recording_facts(recording, csv_data),
                    }
                )
        except (IngestError, CSVSchemaError, MissingResource, OSError) as exc:
            unreadable.append({"form": form, "reason": str(exc)})
            continue
        projects.append(
            {
                "execution_id": (
                    f"mendeley-bjy7vr4xkt-v1-taegeuk-{FORMS.index(form) + 1}"
                ),
                "form": f"Taegeuk {FORMS.index(form) + 1}",
                "views": items,
            }
        )
    demo = projects[0]["execution_id"] if projects else None
    demo_reason = (
        "Taegeuk 1 pair exists and both views decoded"
        if demo is not None and projects[0]["form"] == "Taegeuk 1"
        else "first readable registered form in ordinal order; Taegeuk 1 unavailable"
        if demo is not None
        else "no readable paired form"
    )
    return {
        "schema_version": 1,
        "dataset": provider.demo_metadata(),
        "publisher_claims": provider.registration["published_description"],
        "timing_authority": (
            "native video presentation timestamps; CSV index/time are optional "
            "detector metadata"
        ),
        "csv_usage": (
            "optional mediapipe-csv observations for import/overlay only; "
            "never model accuracy ground truth"
        ),
        "demo_execution_id": demo,
        "demo_reason": demo_reason,
        "unreadable_forms": unreadable,
        "capabilities": {
            "calibration_evidence": (
                "no registered calibration file or known camera geometry"
            ),
            "metric_scale": "unknown",
            "view_overlap": (
                "unmeasured; frontal/lateral labels do not establish overlap"
            ),
            "available_regions": [
                "shoulders",
                "elbows",
                "wrists",
                "hips",
                "knees",
                "ankles",
            ],
            "missing_regions": ["hands", "detailed feet", "head", "neck"],
            "intrinsics": "unknown",
            "extrinsics": "unknown",
            "synchronization": "unknown",
        },
        "projects": projects,
    }


def register_all(provider: MendeleyDatasetProvider | None = None) -> dict[str, Any]:
    """Register eight projects and shared source objects without copying media."""
    provider = provider or MendeleyDatasetProvider()
    report = inventory(provider)
    pipeline = Pipeline(ArtifactStore(provider.root))
    sources_root = provider.root.namespace("runs") / "dataset-sources"
    sources_root.mkdir(parents=True, exist_ok=True)
    for project in report["projects"]:
        project_id = project["execution_id"]
        source_map = {view["camera_id"]: view["path"] for view in project["views"]}
        project_dir = pipeline._directory(project_id)
        if project_dir.exists():
            existing = json.loads((project_dir / "project.json").read_text())
            if existing.get("sources") != source_map:
                raise ValueError(f"existing project {project_id} has different sources")
        else:
            pipeline.register(project_id, source_map)
        for view in project["views"]:
            source_path = sources_root / f"{view['sha256']}.json"
            source_record = {
                "schema_version": 1,
                "id": view["source_id"],
                "path": view["path"],
                "sha256": view["sha256"],
                "size_bytes": view["size_bytes"],
            }
            if source_path.exists():
                if json.loads(source_path.read_text()) != source_record:
                    raise ValueError(f"source object conflict: {source_path}")
            else:
                source_path.write_text(json.dumps(source_record, sort_keys=True) + "\n")
        import_path = project_dir / "dataset-import.json"
        payload = {
            "schema_version": 1,
            "execution_id": project_id,
            "source_objects": [view["sha256"] for view in project["views"]],
            "views": project["views"],
            "dataset": report["dataset"],
            "timing_authority": report["timing_authority"],
            "csv_usage": report["csv_usage"],
        }
        serialized = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        if import_path.exists() and import_path.read_text() != serialized:
            raise ValueError(f"existing project {project_id} has different inventory")
        if not import_path.exists():
            import_path.write_text(serialized)
    return report


def open_registered_project(
    execution_id: str, *, root: StorageRoot | None = None
) -> dict[str, Any]:
    """Read a project and verify that its shared source references still match."""
    root = root or StorageRoot.from_env()
    project_dir = Pipeline(ArtifactStore(root))._directory(execution_id)
    manifest = json.loads((project_dir / "dataset-import.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("invalid project manifest")
    paths = json.loads((project_dir / "project.json").read_text())["sources"]
    if manifest["execution_id"] != execution_id:
        raise ValueError("project manifest identity mismatch")
    for view in manifest["views"]:
        digest = view["sha256"]
        source = json.loads(
            (root.namespace("runs") / "dataset-sources" / f"{digest}.json").read_text()
        )
        if (
            source["sha256"] != digest
            or source["path"] != view["path"]
            or source["id"] != view["source_id"]
            or paths[view["camera_id"]] != view["path"]
            or hash_file(Path(source["path"])) != digest
        ):
            raise ValueError(f"source reference changed: {digest}")
    return cast(dict[str, Any], manifest)
