"""Window publication, resume, and offline reuse without model dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from contracts.models import FrameTime, RawScore
from media import FrameRef
from pose.observation_run import (
    ObservationSettings,
    _key,
    load_receipt,
    load_window_records,
    run_selection,
)
from pose.providers.mmpose.adapter import NamedPoint, PersonCandidate, PoseFrame
from pose.providers.mmpose.hand import HandObservation, missing_points
from pose.providers.mmpose.mapping import NAMES
from storage import ArtifactKey, ArtifactStore, StorageRoot, hash_file
from tkd_poomsae.selections import Window


def test_resume_and_cross_directory_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"registered source")
    digest = hash_file(source)
    refs = tuple(FrameRef(i, i, 1, 10, i == 0) for i in range(4))
    window = Window("execution", "camera", digest, source, 0, 0.3)

    def frame_time(ref: FrameRef) -> FrameTime:
        return FrameTime(
            source_id=f"source:{digest}",
            camera_id="camera",
            pts=ref.pts,
            time_base_num=1,
            time_base_den=10,
            source_seconds=ref.seconds,
            offset_seconds=0,
            global_seconds=ref.seconds,
        )

    recording = SimpleNamespace(
        sha256=digest,
        frames=refs,
        source_id=f"source:{digest}",
        frame_time=frame_time,
    )

    class Reader:
        def __init__(self, _recording: Any, *, max_decode_frames: int) -> None:
            assert max_decode_frames >= 1

        def decode_window(self, start: int, count: int) -> Iterator[Any]:
            for ref in refs[start : start + count]:
                yield SimpleNamespace(ref=ref)

    import pose.observation_run as module

    monkeypatch.setattr(module, "resolve", lambda _name, root: (window,))
    monkeypatch.setattr(module, "index_recording", lambda *_args: recording)
    monkeypatch.setattr(module, "MediaReader", Reader)
    calls: list[int] = []
    fail_at: list[int] = [2]
    missing_mode: list[bool] = []
    score = RawScore(value=0.9, range_min=0, range_max=1)
    points = tuple(NamedPoint(name, (30, 30), score) for name in NAMES)
    candidate = PersonCandidate(
        index=0,
        bbox_xyxy_px=(10, 10, 90, 90),
        detector_score=score,
        landmarks=points,
        refined_hands={},
        refined_hand_boxes={},
        hand_observations={
            "left": HandObservation(
                "left",
                points[91:112],
                missing_points("left", "too_small"),
                None,
                "skipped",
                False,
            )
        },
    )

    class Adapter:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        @contextmanager
        def session(self) -> Iterator[Adapter]:
            yield self

        def infer(self, _recording: Any, decoded: Iterator[Any]) -> Iterator[PoseFrame]:
            for item in decoded:
                calls.append(item.ref.ordinal)
                if item.ref.ordinal in fail_at:
                    raise RuntimeError("interrupted during next window")
                yield PoseFrame(
                    frame=frame_time(item.ref),
                    ordinal=item.ref.ordinal,
                    image_size=(100, 100),
                    candidates=() if missing_mode else (candidate,),
                    model_identity={"wholebody": {"checkpoint_sha256": "test"}},
                    inference_settings={"device": "fake"},
                )

    store = ArtifactStore(StorageRoot(tmp_path / "shared"))
    settings = ObservationSettings(max_frames=2)
    later_origin = Window("other-execution", "camera", digest, source, 0.2, 0.3)
    assert _key(window, refs[2:], settings).digest != _key(
        later_origin, refs[2:], settings
    ).digest
    renamed = Window("other-execution", "camera", digest, source, 0, 0.3)
    assert _key(window, refs[2:], settings).digest == _key(
        renamed, refs[2:], settings
    ).digest
    with pytest.raises(RuntimeError, match="interrupted"):
        run_selection(
            "smoke-short", store=store, settings=settings, adapter_factory=Adapter
        )
    assert calls == [0, 1, 2]
    fail_at.clear()
    result = run_selection(
        "smoke-short", store=store, settings=settings, adapter_factory=Adapter
    )
    assert result["frame_count"] == 4
    assert result["selected_frame_count"] == 4
    assert calls == [0, 1, 2, 2, 3]
    receipt, observations = load_receipt(Path(result["receipt_path"]), store)
    assert receipt["frame_count"] == len(observations) == 4
    assert [item.frame.pts for item in observations] == [0, 1, 2, 3]
    assert observations[2].subject_selection is not None
    assert observations[2].subject_selection.method == "temporal"
    first_key = ArtifactKey(**receipt["windows"][0]["key"])
    source_record = load_window_records(store.get(first_key))[0]
    assert (
        source_record.hand_observations["left"]["coarse"][0]["raw_score"]["value"]
        == 0.9
    )
    monkeypatch.chdir(tmp_path / "shared")
    cached = run_selection(
        "smoke-short", store=store, settings=settings, adapter_factory=Adapter
    )
    assert cached["cached"] is True
    assert calls == [0, 1, 2, 2, 3]
    changed = run_selection(
        "smoke-short",
        store=store,
        settings=ObservationSettings(max_frames=2, detector_threshold=0.4),
        adapter_factory=Adapter,
    )
    assert changed["cached"] is False
    assert calls[-4:] == [0, 1, 2, 3]
    missing_mode.append(True)
    with pytest.raises(ValueError, match="no usable practitioner observations"):
        run_selection(
            "smoke-short",
            store=store,
            settings=ObservationSettings(max_frames=2, detector_threshold=0.5),
            adapter_factory=Adapter,
        )
