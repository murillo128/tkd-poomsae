"""Offline, resumable native-time observation windows for local selections."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import BaseModel

from contracts.models import DenseArray, Observation, Provenance
from media import MediaReader, index_recording
from pose.providers.mmpose.adapter import MMPoseAdapter
from pose.providers.mmpose.hand import HandROIConfig
from pose.stream import PractitionerTracker, TrackingConfig
from storage import (
    ArtifactHandle,
    ArtifactKey,
    ArtifactStore,
    MissingResource,
    hash_config,
    hash_file,
)
from tkd_poomsae.selections import Window, resolve
from tkd_poomsae.vision.assets import registry

REVISION = "native-observation-windows-v1"
ARRAY_ID = "window_records_json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class ObservationSettings:
    device: str = "cpu"
    max_frames: int = 32
    max_people: int = 4
    detector_threshold: float = 0.3
    hand_roi: HandROIConfig = HandROIConfig()
    tracking: TrackingConfig = TrackingConfig()

    def __post_init__(self) -> None:
        if self.max_frames < 1 or self.max_frames > 128:
            raise ValueError("max_frames must be between 1 and 128")


@dataclass(frozen=True)
class ObservationRecord:
    observation: Observation
    model_identity: dict[str, Any]
    inference_settings: dict[str, Any]
    hand_observations: dict[str, Any]
    hand_roi_to_source: dict[str, Any]
    source_orientation: dict[str, Any]
    tracker_state: dict[str, Any]


def _model_revision() -> str:
    # The pinned model bytes and runtime lock are identities, never download hints.
    lock = Path(__file__).resolve().parents[1] / "vision/requirements.lock"
    return hash_config({"registry": registry(), "runtime_lock_sha256": hash_file(lock)})


def _key(
    window: Window, refs: tuple[Any, ...], settings: ObservationSettings
) -> ArtifactKey:
    return ArtifactKey(
        layer="observation",
        inputs={"source": window.source_sha256},
        schema_version="1.0.0",
        algorithm_revision=REVISION,
        model_revision=_model_revision(),
        config_digest=hash_config(
            {
                "camera_id": window.camera_id,
                "first_ordinal": refs[0].ordinal,
                "last_ordinal": refs[-1].ordinal,
                "first_pts": refs[0].pts,
                "last_pts": refs[-1].pts,
                "preprocessing": "display-oriented-rgb-v1",
                "tracking_revision": "practitioner-tracker-v1",
                "regional_revision": "wholebody-regions-v1",
                "settings": asdict(settings),
            }
        ),
    )


def _record_bytes(records: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
    encoded = json.dumps(
        {"version": 1, "records": records},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8).copy()


def load_window_records(handle: ArtifactHandle) -> list[ObservationRecord]:
    """Reload typed observations and retained model/ROI source evidence."""
    if not isinstance(handle.metadata, Observation):
        raise ValueError("window artifact is not an observation")
    data = handle.read_array(ARRAY_ID)
    payload = json.loads(data.tobytes().decode("utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError("unsupported observation window payload")
    records = [
        ObservationRecord(
            observation=Observation.model_validate(record["observation"]),
            model_identity=dict(record["model_identity"]),
            inference_settings=dict(record["inference_settings"]),
            hand_observations=dict(record["hand_observations"]),
            hand_roi_to_source=dict(record["hand_roi_to_source"]),
            source_orientation=dict(record["source_orientation"]),
            tracker_state=dict(record["tracker_state"]),
        )
        for record in payload["records"]
    ]
    observations = [record.observation for record in records]
    if not observations or observations[0].frame != handle.metadata.frame:
        raise ValueError("observation window metadata disagrees with payload")
    if any(
        later.frame.source_seconds <= earlier.frame.source_seconds
        or later.frame.source_id != earlier.frame.source_id
        for earlier, later in zip(observations, observations[1:])
    ):
        raise ValueError("observation window is not in native-time order")
    return records


def load_window(handle: ArtifactHandle) -> list[Observation]:
    """Verify the published payload and return frame contracts."""
    return [record.observation for record in load_window_records(handle)]


def _receipt_path(store: ArtifactStore, identity: dict[str, Any]) -> Path:
    return (
        store.root.namespace("derived")
        / "observation-runs"
        / f"{hash_config(identity)}.json"
    )


def _publish_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".observation-run-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_receipt(
    path: Path, store: ArtifactStore
) -> tuple[dict[str, Any], list[Observation]]:
    """Reload all typed frames offline when the caller wants them in memory."""
    receipt = verify_receipt(path, store)
    observations = [
        observation
        for entry in receipt["windows"]
        for observation in load_window(store.get(ArtifactKey(**entry["key"])))
    ]
    return receipt, observations


def verify_receipt(path: Path, store: ArtifactStore) -> dict[str, Any]:
    """Verify a full result with memory bounded by one window."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, dict)
        or receipt.get("version") != 1
        or not isinstance(receipt.get("windows"), list)
    ):
        raise ValueError("unsupported observation run receipt")
    frame_count = 0
    selected_count = 0
    usable_by_view: dict[str, int] = {}
    for entry in receipt["windows"]:
        key = ArtifactKey(**entry["key"])
        if key.digest != entry["digest"]:
            raise ValueError("observation window key changed")
        frames = load_window(store.get(key))
        if len(frames) != entry["frames"]:
            raise ValueError("observation window frame count changed")
        frame_count += len(frames)
        selected_count += sum(
            frame.subject_selection is not None
            and frame.subject_selection.state == "selected"
            for frame in frames
        )
        view = str(entry["view"])
        usable_by_view[view] = usable_by_view.get(view, 0) + sum(
            any(
                region.part == "body" and region.usable
                for region in frame.region_quality
            )
            for frame in frames
        )
    if (
        frame_count != receipt["frame_count"]
        or selected_count != receipt["selected_frame_count"]
        or usable_by_view != receipt["usable_by_view"]
        or not usable_by_view
        or any(count == 0 for count in usable_by_view.values())
    ):
        raise ValueError("observation run count or usable evidence changed")
    return cast(dict[str, Any], receipt)


def run_selection(
    selection: str,
    *,
    store: ArtifactStore | None = None,
    settings: ObservationSettings | None = None,
    cancelled: Callable[[], bool] | None = None,
    adapter_factory: Callable[..., Any] = MMPoseAdapter,
) -> dict[str, Any]:
    """Infer missing complete windows only; final receipt appears after all views."""
    store = store or ArtifactStore()
    settings = settings or ObservationSettings()
    windows = resolve(selection, root=store.root)
    identity = {
        "selection": selection,
        "sources": [
            [
                w.execution_id,
                w.camera_id,
                w.source_sha256,
                w.start_seconds,
                w.end_seconds,
            ]
            for w in windows
        ],
        "model_revision": _model_revision(),
        "algorithm_revision": REVISION,
        "settings": asdict(settings),
    }
    path = _receipt_path(store, identity)
    if path.is_file():
        receipt = verify_receipt(path, store)
        if receipt["identity"] != identity:
            raise ValueError("observation receipt identity changed")
        return receipt | {"cached": True, "receipt_path": str(path)}
    entries: list[dict[str, Any]] = []
    frame_count = 0
    selected_count = 0
    usable_by_view: dict[str, int] = {}
    for window in windows:
        if cancelled is not None and cancelled():
            raise RuntimeError("observation run cancelled")
        recording = index_recording(window.camera_id, window.source_path)
        if recording.sha256 != window.source_sha256:
            raise ValueError("registered source changed during observation indexing")
        refs = tuple(
            ref
            for ref in recording.frames
            if window.start_seconds <= ref.seconds <= window.end_seconds
        )
        if not refs:
            raise ValueError(f"selection contains no native frames: {window.camera_id}")
        reader = MediaReader(recording, max_decode_frames=settings.max_frames)
        view = f"{window.execution_id}:{window.camera_id}"
        usable_by_view[view] = 0
        adapter: MMPoseAdapter | None = None
        tracker = PractitionerTracker(settings.tracking)
        with ExitStack() as sessions:
            for start in range(0, len(refs), settings.max_frames):
                chunk = refs[start : start + settings.max_frames]
                key = _key(window, chunk, settings)
                try:
                    handle = store.get(key)
                    records = load_window_records(handle)
                except MissingResource:
                    if adapter is None:
                        adapter = adapter_factory(
                            device=settings.device,
                            max_frames=settings.max_frames,
                            max_people=settings.max_people,
                            detector_threshold=settings.detector_threshold,
                            hand_roi=settings.hand_roi,
                            hand_cache_root=store.root.namespace("derived")
                            / "hand-refinement-cache",
                            cancelled=cancelled,
                        )
                        sessions.enter_context(adapter.session())

                    def produce() -> tuple[
                        Observation, dict[str, np.ndarray[Any, Any]]
                    ]:
                        assert adapter is not None
                        records: list[dict[str, Any]] = []
                        decoded = reader.decode_window(chunk[0].ordinal, len(chunk))
                        for pose in adapter.infer(recording, decoded):
                            observation = tracker.observe(
                                pose,
                                artifact_id=f"observation:{key.digest}:{pose.ordinal}",
                                provenance=Provenance(
                                    producer="pose.observation_run",
                                    model="MMPose RTMPose wholebody+hand",
                                    model_version=key.model_revision,
                                    config_digest=key.config_digest,
                                ),
                            )
                            candidate = (
                                pose.candidates[
                                    observation.subject_selection.candidate_index
                                ]
                                if observation.subject_selection is not None
                                and observation.subject_selection.candidate_index
                                is not None
                                else None
                            )
                            records.append(
                                {
                                    "observation": observation.model_dump(mode="json"),
                                    "model_identity": pose.model_identity,
                                    "inference_settings": pose.inference_settings,
                                    "hand_observations": {
                                        side: _jsonable(asdict(hand))
                                        for side, hand in (
                                            candidate.hand_observations.items()
                                            if candidate is not None
                                            else ()
                                        )
                                    },
                                    "hand_roi_to_source": {
                                        side: hand.roi.roi_to_source_affine
                                        for side, hand in (
                                            candidate.hand_observations.items()
                                            if candidate is not None
                                            else ()
                                        )
                                        if hand.roi is not None
                                    },
                                    "source_orientation": {
                                        "stored_to_oriented": getattr(
                                            recording, "stored_to_oriented", None
                                        ),
                                        "oriented_to_stored": getattr(
                                            recording, "oriented_to_stored", None
                                        ),
                                    },
                                    "tracker_state": tracker.snapshot(),
                                }
                            )
                        if len(records) != len(chunk):
                            raise ValueError(
                                "inference did not return every native frame"
                            )
                        array = _record_bytes(records)
                        first = Observation.model_validate(records[0]["observation"])
                        metadata = first.model_copy(
                            update={
                                "id": f"observation-window:{key.digest}",
                                "arrays": [
                                    DenseArray(
                                        id=ARRAY_ID,
                                        dtype="uint8",
                                        shape=[len(array)],
                                        axes=["json_byte"],
                                    )
                                ],
                            }
                        )
                        return metadata, {ARRAY_ID: array}

                    # The atomic store makes a killed writer retryable.
                    handle = store.get_or_create(key, produce, cancelled=cancelled)
                    records = load_window_records(handle)
                observations = [record.observation for record in records]
                if len(observations) != len(chunk) or any(
                    obs.frame.pts != ref.pts
                    or obs.frame.source_id != recording.source_id
                    for obs, ref in zip(observations, chunk)
                ):
                    raise ValueError(
                        "cached observation window does not match native frames"
                    )
                tracker.restore(records[-1].tracker_state)
                frame_count += len(observations)
                selected_count += sum(
                    obs.subject_selection is not None
                    and obs.subject_selection.state == "selected"
                    for obs in observations
                )
                usable_by_view[view] += sum(
                    any(
                        region.part == "body" and region.usable
                        for region in obs.region_quality
                    )
                    for obs in observations
                )
                entries.append(
                    {
                        "view": view,
                        "digest": key.digest,
                        "frames": len(observations),
                        "key": {
                            "layer": key.layer,
                            "inputs": dict(key.inputs),
                            "schema_version": key.schema_version,
                            "algorithm_revision": key.algorithm_revision,
                            "model_revision": key.model_revision,
                            "config_digest": key.config_digest,
                        },
                    }
                )
    if selected_count == 0 or any(count == 0 for count in usable_by_view.values()):
        raise ValueError("one or more views have no usable practitioner observations")
    receipt = {
        "version": 1,
        "identity": identity,
        "windows": entries,
        "frame_count": frame_count,
        "selected_frame_count": selected_count,
        "usable_by_view": usable_by_view,
    }
    _publish_receipt(path, receipt)
    verify_receipt(path, store)
    return receipt | {"cached": False, "receipt_path": str(path)}
