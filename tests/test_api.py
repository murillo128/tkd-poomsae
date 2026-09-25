"""Local service tests use only synthetic pipeline producers and source bytes."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from threading import Event
from typing import Any

from fastapi.testclient import TestClient

from pipeline import Pipeline, Stage, StageOutput
from pipeline.runner import DEPENDENCIES, STAGE_LAYERS
from storage import ArtifactKey, ArtifactStore, StorageRoot
from tests.test_pipeline import setup
from tkd_poomsae.api import create_app

WRITE = {"x-tkd-local-request": "1"}


def wait_for(client: TestClient, job_id: str, status: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        row: dict[str, Any] = client.get(f"/api/runs/{job_id}").json()
        if row["status"] == status:
            return row
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {status}: {row}")


def test_registration_discovery_capabilities_and_offline_job(tmp_path: Path) -> None:
    pipe = setup(tmp_path, Counter())
    app = create_app(pipe, allowed_roots={"shared": tmp_path})
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/api/projects").json() == {"projects": ["demo"]}
        data = {
            "id": "other",
            "sources": {
                side: {"root": "shared", "path": f"{side}.video"}
                for side in ("left", "right")
            },
        }
        assert client.post("/api/projects", json=data, headers=WRITE).status_code == 201
        detail = client.get("/api/projects/other").json()
        assert detail["sources"] == ["left", "right"]
        assert str(tmp_path) not in json.dumps(detail)
        capability = client.get("/api/projects/other/capabilities").json()
        assert capability["stages"]["ingest"]["schema_version"] == "1.0.0"
        assert capability["stages"]["ingest"]["available"]
        response = client.post(
            "/api/projects/other/runs", json={"through": "observations"}, headers=WRITE
        )
        assert response.status_code == 202
        row = wait_for(client, response.json()["id"], "complete")
        assert row["stages"]["observations"]["key"]
        assert row["stages"]["observations"]["software_revision"] == "1"
        assert (
            client.get("/api/projects/other").json()["state"]["stages"]["observations"][
                "status"
            ]
            == "complete"
        )


def test_registration_rejects_escape_and_invalid_ids(tmp_path: Path) -> None:
    pipe = setup(tmp_path, Counter())
    outside = tmp_path.parent / "outside-source.video"
    outside.write_bytes(b"outside")
    (tmp_path / "escape.video").symlink_to(outside)
    app = create_app(pipe, allowed_roots={"shared": tmp_path})
    with TestClient(app, base_url="http://localhost") as client:
        base: dict[str, Any] = {
            "id": "another",
            "sources": {
                "left": {"root": "shared", "path": "left.video"},
                "right": {"root": "shared", "path": "right.video"},
            },
        }
        for bad in (
            "../outside-source.video",
            "escape.video",
            "/etc/passwd",
            "https://host/x",
        ):
            base["sources"]["left"]["path"] = bad
            response = client.post("/api/projects", json=base, headers=WRITE)
            assert response.status_code == 400
        base["sources"]["left"]["path"] = "left.video"
        base["id"] = "../bad"
        assert client.post("/api/projects", json=base, headers=WRITE).status_code == 400
        assert client.get("/api/projects/../bad").status_code != 200
        assert client.get("/api/runs/../../etc/passwd").status_code != 200


def test_registration_accepts_symlink_into_another_allowed_root(tmp_path: Path) -> None:
    pipe = setup(tmp_path, Counter())
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "camera.video").write_bytes(b"shared")
    (tmp_path / "shared-link.video").symlink_to(shared / "camera.video")
    app = create_app(pipe, allowed_roots={"input": tmp_path, "dataset": shared})
    with TestClient(app, base_url="http://localhost") as client:
        response = client.post(
            "/api/projects",
            headers=WRITE,
            json={
                "id": "linked",
                "sources": {
                    "left": {"root": "input", "path": "shared-link.video"},
                    "right": {"root": "input", "path": "right.video"},
                },
            },
        )
        assert response.status_code == 201
        assert client.get("/api/projects/linked").status_code == 200


def test_project_id_cannot_follow_directory_symlink(tmp_path: Path) -> None:
    pipe = setup(tmp_path, Counter())
    projects = pipe.store.root.namespace("runs") / "projects"
    (projects / "alias").symlink_to(projects / "demo", target_is_directory=True)
    with TestClient(
        create_app(pipe, allowed_roots={"input": tmp_path}),
        base_url="http://localhost",
    ) as client:
        assert client.get("/api/projects/alias").status_code == 404
        assert client.get("/api/projects").json() == {"projects": ["demo"]}


def test_offline_inspection_reports_unavailable_and_stale(tmp_path: Path) -> None:
    default = Pipeline(ArtifactStore(StorageRoot(tmp_path / "default")))
    for side in ("left", "right"):
        (tmp_path / f"{side}.video").write_bytes(side.encode())
    default.register(
        "demo", {side: tmp_path / f"{side}.video" for side in ("left", "right")}
    )
    with TestClient(
        create_app(default, allowed_roots={"input": tmp_path}),
        base_url="http://localhost",
    ) as client:
        capability = client.get("/api/projects/demo/capabilities").json()
        assert not capability["stages"]["ingest"]["available"]
        assert "provision it offline" in capability["stages"]["ingest"]["reason"]
        response = client.post(
            "/api/projects/demo/runs", json={"through": "ingest"}, headers=WRITE
        )
        assert (
            wait_for(client, response.json()["id"], "unavailable")["stages"]["ingest"][
                "status"
            ]
            == "unavailable"
        )

    pipe = setup(tmp_path, Counter())
    pipe.analyze("demo", "ingest")
    (tmp_path / "left.video").write_bytes(b"changed")
    with TestClient(
        create_app(pipe, allowed_roots={"input": tmp_path}),
        base_url="http://localhost",
    ) as client:
        stage = client.get("/api/projects/demo/capabilities").json()["stages"]["ingest"]
        assert stage["status"]["status"] == "stale"
        assert stage["reason"] == "inputs or revision changed"


def test_origin_host_and_csrf_policy(tmp_path: Path) -> None:
    pipe = setup(tmp_path, Counter())
    app = create_app(pipe, allowed_roots={"shared": tmp_path})
    with TestClient(app, base_url="http://localhost") as client:
        url = "/api/projects/demo/runs"
        assert client.post(url, json={"through": "ingest"}).status_code == 403
        assert (
            client.post(
                url,
                json={"through": "ingest"},
                headers={**WRITE, "origin": "https://evil.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                url,
                json={"through": "ingest"},
                headers={**WRITE, "sec-fetch-site": "cross-site"},
            ).status_code
            == 403
        )
        response = client.get("/api/projects", headers={"host": "evil.example"})
        assert response.status_code == 400
        response = client.get("/api/projects", headers={"host": "localhost.evil:8000"})
        assert response.status_code == 400
        response = client.get(
            "/api/projects", headers={"origin": "https://evil.example"}
        )
        assert response.status_code == 403


def test_conflict_cancellation_and_restart_recovery(tmp_path: Path) -> None:
    pipe = setup(tmp_path, Counter())
    entered, release = Event(), Event()
    original = pipe.stages["ingest"]

    def slow(key: ArtifactKey, inputs: Any, settings: Any) -> StageOutput:
        entered.set()
        assert release.wait(5)
        return original.producer(key, inputs, settings)

    pipe.stages["ingest"] = Stage(
        "ingest", slow, STAGE_LAYERS["ingest"], DEPENDENCIES["ingest"]
    )
    app = create_app(pipe, allowed_roots={"shared": tmp_path}, queue_size=1)
    with TestClient(app, base_url="http://localhost") as client:
        response = client.post(
            "/api/projects/demo/runs", json={"through": "ingest"}, headers=WRITE
        )
        job_id = response.json()["id"]
        try:
            assert entered.wait(5)
            assert (
                wait_for(client, job_id, "running")["stages"]["ingest"]["status"]
                == "running"
            )
            assert (
                client.post(
                    "/api/projects/demo/runs", json={"through": "ingest"}, headers=WRITE
                ).status_code
                == 409
            )
            assert client.post(f"/api/runs/{job_id}/cancel", headers=WRITE).json()[
                "cancel_requested"
            ]
        finally:
            release.set()
        assert wait_for(client, job_id, "cancelled")["error"]
    job_path = pipe.store.root.namespace("runs") / "service-jobs" / f"{job_id}.json"
    record = json.loads(job_path.read_text(encoding="utf-8"))
    record["status"] = "running"
    job_path.write_text(json.dumps(record), encoding="utf-8")
    with TestClient(
        create_app(pipe, allowed_roots={"shared": tmp_path}),
        base_url="http://localhost",
    ) as client:
        recovered = client.get(f"/api/runs/{job_id}").json()
        assert recovered["status"] == "interrupted"
        assert "submit a new run" in recovered["error"]


def test_pending_queue_is_bounded_and_can_cancel_before_execution(
    tmp_path: Path,
) -> None:
    pipe = setup(tmp_path, Counter())
    sources = {side: tmp_path / f"{side}.video" for side in ("left", "right")}
    pipe.register("second", sources)
    pipe.register("third", sources)
    entered, release = Event(), Event()
    original = pipe.stages["ingest"]

    def slow(key: ArtifactKey, inputs: Any, settings: Any) -> StageOutput:
        entered.set()
        assert release.wait(5)
        return original.producer(key, inputs, settings)

    pipe.stages["ingest"] = Stage(
        "ingest", slow, STAGE_LAYERS["ingest"], DEPENDENCIES["ingest"]
    )
    app = create_app(pipe, workers=1, queue_size=1)
    with TestClient(app, base_url="http://localhost") as client:
        first = client.post(
            "/api/projects/demo/runs", json={"through": "ingest"}, headers=WRITE
        ).json()["id"]
        try:
            assert entered.wait(5)
            second = client.post(
                "/api/projects/second/runs",
                json={"through": "ingest"},
                headers=WRITE,
            ).json()["id"]
            assert (
                client.post(
                    "/api/projects/third/runs",
                    json={"through": "ingest"},
                    headers=WRITE,
                ).status_code
                == 429
            )
            cancelled = client.post(f"/api/runs/{second}/cancel", headers=WRITE)
            assert cancelled.json()["status"] == "cancelled"
            replacement = client.post(
                "/api/projects/third/runs",
                json={"through": "ingest"},
                headers=WRITE,
            )
            assert replacement.status_code == 202
            replacement_id = replacement.json()["id"]
            assert (
                client.get(f"/api/runs/{replacement_id}").json()["status"] == "queued"
            )
            assert (
                client.post(f"/api/runs/{replacement_id}/cancel", headers=WRITE).json()[
                    "status"
                ]
                == "cancelled"
            )
        finally:
            release.set()
        assert wait_for(client, first, "complete")["status"] == "complete"
        assert client.get(f"/api/runs/{second}").json()["status"] == "cancelled"
