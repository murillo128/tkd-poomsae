"""Registered media stays local while HTTP ranges and native frames stay exact."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg  # type: ignore[import-untyped]
import numpy as np
from fastapi.testclient import TestClient

from media import index_recording
from pipeline import Pipeline
from storage import ArtifactStore, StorageRoot
from tests.test_media_reader import video
from tkd_poomsae.api import create_app


def client_for(tmp_path: Path, left: Path, right: Path) -> TestClient:
    pipe = Pipeline(ArtifactStore(StorageRoot(tmp_path / "store")))
    pipe.register("demo", {"left": left, "right": right})
    return TestClient(
        create_app(pipe, allowed_roots={"local": tmp_path}),
        base_url="http://localhost",
    )


def test_registered_video_ranges_validators_and_source_bytes(tmp_path: Path) -> None:
    left = video(tmp_path / "left.mkv", [400, 440, 500, 580])
    right = video(tmp_path / "right.mkv", [900, 940, 980])
    original = left.read_bytes()
    url = "/api/projects/demo/media/left"
    with client_for(tmp_path, left, right) as client:
        full = client.get(url)
        assert full.status_code == 200
        assert full.content == original
        assert full.headers["content-type"] == "video/x-matroska"
        assert full.headers["content-length"] == str(len(original))
        assert full.headers["accept-ranges"] == "bytes"
        assert client.get(url + "/metadata").json()["first_frame"]["pts"] == 400
        preflight = client.options(
            url,
            headers={
                "origin": "http://localhost:5173",
                "access-control-request-method": "HEAD",
                "access-control-request-headers": "range,if-range",
            },
        )
        assert preflight.status_code == 204
        assert "HEAD" in preflight.headers["access-control-allow-methods"]
        cross_origin = client.get(url, headers={"origin": "http://localhost:5173"})
        assert "Content-Range" in cross_origin.headers["access-control-expose-headers"]
        head = client.head(url)
        assert head.status_code == 200 and not head.content
        assert head.headers["content-length"] == str(len(original))
        first = client.get(url, headers={"range": "bytes=2-9"})
        assert first.status_code == 206 and first.content == original[2:10]
        assert first.headers["content-range"] == f"bytes 2-9/{len(original)}"
        assert first.headers["content-length"] == "8"
        suffix = client.get(url, headers={"range": "bytes=-7"})
        assert suffix.status_code == 206 and suffix.content == original[-7:]
        assert (
            client.head(url, headers={"range": "bytes=0-1"}).headers["content-range"]
            == f"bytes 0-1/{len(original)}"
        )
        for bad in (
            "bytes=99-10",
            "bytes=-0",
            "bytes=1-2,4-5",
            "bytes=abc",
            "bytes=999999-",
            "bytes=" + "9" * 5000 + "-",
        ):
            invalid = client.get(url, headers={"range": bad})
            assert invalid.status_code == 416
            assert invalid.headers["content-range"] == f"bytes */{len(original)}"
        assert (
            client.get(url, headers={"if-none-match": full.headers["etag"]}).status_code
            == 304
        )
        assert (
            client.get(
                url, headers={"if-modified-since": full.headers["last-modified"]}
            ).status_code
            == 304
        )
        assert (
            client.get(
                url, headers={"if-none-match": "W/" + full.headers["etag"]}
            ).status_code
            == 304
        )
        no_range = client.get(
            url, headers={"range": "bytes=0-1", "if-range": '"different"'}
        )
        assert no_range.status_code == 200 and no_range.content == original
    assert left.read_bytes() == original
    assert hashlib.sha256(original).hexdigest() in full.headers["etag"]


def test_native_frame_index_bracket_and_oriented_exact_image(tmp_path: Path) -> None:
    left = video(tmp_path / "left.mkv", [400, 440, 500, 580])
    plain = video(tmp_path / "plain.mkv", [900, 940, 980])
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(plain),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    with client_for(tmp_path, left, rotated) as client:
        base = "/api/projects/demo/media/right/frames"
        page = client.get(base, params={"start": 1, "limit": 1}).json()
        assert page["total"] == 3 and len(page["frames"]) == 1
        frame = page["frames"][0]
        reference = index_recording("right", rotated)
        metadata = client.get("/api/projects/demo/media/right/metadata").json()
        assert metadata["browser_playback"] is False
        assert metadata["source_sha256"] == reference.sha256
        assert frame["ordinal"] == 1
        assert frame["pts"] == reference.frames[1].pts
        assert frame["source_seconds"] == reference.frames[1].seconds
        assert frame["rotation_degrees"] == 90
        assert frame["stored_to_oriented"] == [
            list(row) for row in reference.stored_to_oriented
        ]
        assert frame["source_sha256"] == reference.sha256
        assert client.get(base, params={"limit": 257}).status_code == 422
        middle = (reference.frames[1].seconds + reference.frames[2].seconds) / 2
        bracket = client.get(base + "/bracket", params={"seconds": middle}).json()
        assert bracket["before"]["ordinal"] == 1
        assert bracket["after"]["ordinal"] == 2
        nearest = client.get(
            base + "/nearest", params={"seconds": reference.frames[1].seconds}
        ).json()
        assert nearest["frame"]["ordinal"] == 1
        image = client.get(base + "/1/image")
        decoded = cv2.imdecode(
            np.frombuffer(image.content, dtype="uint8"),
            cv2.IMREAD_COLOR,
        )
        assert image.status_code == 200
        assert image.headers["x-source-pts"] == str(reference.frames[1].pts)
        assert decoded is not None and decoded.shape[:2] == (32, 24)
        cache = tmp_path / "store" / "derived" / "media-previews-v1"
        assert (cache / f"{reference.sha256}-1.png").read_bytes() == image.content
        assert client.get(base + "/99/image").status_code == 404


def test_only_registered_sources_in_permitted_roots(tmp_path: Path) -> None:
    left = video(tmp_path / "left.mkv", [0, 40, 80])
    right = video(tmp_path / "right.mkv", [0, 40, 80])
    pipe = Pipeline(ArtifactStore(StorageRoot(tmp_path / "store")))
    pipe.register("demo", {"left": left, "right": right})
    outside = tmp_path.parent / "outside-media.mkv"
    outside.write_bytes(left.read_bytes())
    (tmp_path / "escape.mkv").symlink_to(outside)
    with TestClient(
        create_app(pipe, allowed_roots={"local": tmp_path}),
        base_url="http://localhost",
    ) as client:
        base = "/api/projects/demo/media/"
        assert client.get(base + "unknown").status_code == 404
        assert client.get("/api/projects/unknown/media/left").status_code == 404
        project_file, _ = pipe._files("demo")
        original = project_file.read_text()
        try:
            project_file.write_text(
                original.replace(str(left), str(tmp_path / "escape.mkv"))
            )
            assert client.get(base + "left").status_code == 403
        finally:
            project_file.write_text(original)


def test_shared_dataset_root_is_permitted_without_api_registration_root(
    tmp_path: Path,
) -> None:
    root = StorageRoot(tmp_path / "store")
    dataset = root.namespace("datasets")
    dataset.mkdir(parents=True)
    left = video(dataset / "left.mkv", [0, 40, 80])
    right = video(dataset / "right.mkv", [0, 40, 80])
    pipe = Pipeline(ArtifactStore(root))
    pipe.register("demo", {"left": left, "right": right})
    with TestClient(
        create_app(pipe, allowed_roots={}), base_url="http://localhost"
    ) as client:
        assert client.get("/api/projects/demo/media/left").content == left.read_bytes()
        assert (
            client.get("/api/projects/demo/media/left/metadata").json()["source_sha256"]
            == hashlib.sha256(left.read_bytes()).hexdigest()
        )
