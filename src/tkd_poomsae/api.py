"""Loopback inspection and bounded offline-job API."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline import Pipeline
from pipeline.runner import STAGE_ORDER
from storage import MissingResource
from tkd_poomsae.jobs import BusyProject, JobQueue, QueueFull, required_stages


class SourceRegistration(BaseModel):
    root: str
    path: str


class ProjectRegistration(BaseModel):
    id: str
    sources: dict[str, SourceRegistration]


class RunRequest(BaseModel):
    through: str = "parsing"
    rerun: str | None = None


def _source_path(roots: dict[str, Path], source: SourceRegistration) -> Path:
    root = roots.get(source.root)
    relative = Path(source.path)
    if (
        root is None
        or not source.path
        or relative.is_absolute()
        or "\\" in source.path
        or any(part in {"", ".", ".."} for part in source.path.split("/"))
    ):
        raise ValueError("invalid registered source path or root")
    resolved = (root / relative).resolve()
    if not any(resolved.is_relative_to(allowed) for allowed in roots.values()) or (
        not resolved.is_file()
    ):
        raise ValueError("source is missing or outside its allowed root")
    return resolved


def create_app(
    pipeline: Pipeline | None = None,
    *,
    allowed_roots: dict[str, Path] | None = None,
    trusted_origins: set[str] | None = None,
    workers: int = 2,
    queue_size: int = 16,
) -> FastAPI:
    pipe = pipeline or Pipeline()
    if allowed_roots is None:
        configured = json.loads(os.environ.get("TKD_ALLOWED_SOURCE_ROOTS", "{}"))
        if not isinstance(configured, dict):
            raise ValueError("TKD_ALLOWED_SOURCE_ROOTS must be a JSON object")
        allowed_roots = {name: Path(value) for name, value in configured.items()}
    if any(not path.is_absolute() for path in allowed_roots.values()):
        raise ValueError("allowed source roots must be absolute")
    roots = {name: path.resolve() for name, path in allowed_roots.items()}
    origins = (
        trusted_origins
        if trusted_origins is not None
        else {
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        }
    )
    jobs = JobQueue(pipe, workers=workers, capacity=queue_size)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        jobs.start()
        try:
            yield
        finally:
            jobs.stop()

    service = FastAPI(title="TKD Poomsae", version="0.1.0", lifespan=lifespan)

    @service.middleware("http")
    async def local_request_policy(request: Request, call_next: Any) -> Any:
        hosts = request.headers.getlist("host")
        try:
            if len(hosts) != 1:
                raise ValueError("one Host header required")
            host = urlsplit(f"//{hosts[0].lower()}")
            _ = host.port  # Invalid ports raise ValueError.
            trusted_host = (
                host.hostname in {"localhost", "127.0.0.1", "::1"}
                and not host.path
                and not host.query
                and not host.fragment
                and not host.username
                and not host.password
            )
        except ValueError:
            trusted_host = False
        if not trusted_host:
            return JSONResponse({"detail": "untrusted host"}, status_code=400)
        origin = request.headers.get("origin")
        if origin is not None and origin not in origins:
            return JSONResponse({"detail": "untrusted origin"}, status_code=403)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if request.headers.get("x-tkd-local-request") != "1" or (
                request.headers.get("sec-fetch-site") == "cross-site"
            ):
                return JSONResponse(
                    {"detail": "local request header required"}, status_code=403
                )
        return await call_next(request)

    @service.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    def project_status(project: str) -> dict[str, Any]:
        try:
            return pipe.status(project)
        except MissingResource as exc:
            raise HTTPException(
                409, "registered source unavailable; provision it locally"
            ) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @service.get("/api/projects")
    def list_projects() -> dict[str, list[str]]:
        directory = pipe.store.root.namespace("runs") / "projects"
        return {
            "projects": sorted(
                path.name
                for path in directory.iterdir()
                if path.is_dir()
                and not path.is_symlink()
                and (path / "project.json").is_file()
                and not (path / "project.json").is_symlink()
            )
            if directory.is_dir()
            else []
        }

    @service.post("/api/projects", status_code=201)
    def register_project(data: ProjectRegistration) -> dict[str, str]:
        if not roots:
            raise HTTPException(403, "configure TKD_ALLOWED_SOURCE_ROOTS first")
        try:
            pipe.register(
                data.id,
                {
                    name: _source_path(roots, source)
                    for name, source in data.sources.items()
                },
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"id": data.id}

    @service.get("/api/projects/{project}")
    def get_project(project: str) -> dict[str, Any]:
        state = project_status(project)
        project_path, _ = pipe._files(project)
        data = json.loads(project_path.read_text(encoding="utf-8"))
        return {"id": project, "sources": sorted(data["sources"]), "state": state}

    @service.get("/api/projects/{project}/capabilities")
    def capabilities(project: str) -> dict[str, Any]:
        state = project_status(project)
        return {
            "project": project,
            "stages": {
                name: {
                    "dependencies": list(stage.dependencies),
                    "available": stage.capability_reason is None,
                    "reason": stage.capability_reason
                    or (
                        "; ".join(state["stages"][name].get("diagnostics", []))
                        if state["stages"][name]["status"] in {"stale", "unavailable"}
                        else None
                    ),
                    "schema_version": stage.schema_version,
                    "software_revision": stage.revision,
                    "model_revision": stage.model_revision,
                    "status": state["stages"][name],
                }
                for name, stage in pipe.stages.items()
            },
        }

    @service.post("/api/projects/{project}/runs", status_code=202)
    def create_run(project: str, data: RunRequest) -> dict[str, str]:
        project_status(project)
        if data.through not in STAGE_ORDER or (
            data.rerun is not None and data.rerun not in STAGE_ORDER
        ):
            raise HTTPException(422, "unknown pipeline stage")
        if data.rerun is not None and data.rerun not in required_stages(data.through):
            raise HTTPException(422, "rerun stage must be required by through stage")
        try:
            return {"id": jobs.submit(project, data.through, data.rerun)}
        except BusyProject as exc:
            raise HTTPException(409, str(exc)) from exc
        except QueueFull as exc:
            raise HTTPException(429, str(exc)) from exc

    @service.get("/api/runs/{job_id}")
    def get_run(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404, "unknown run") from exc

    @service.post("/api/runs/{job_id}/cancel")
    def cancel_run(job_id: str) -> dict[str, Any]:
        try:
            return jobs.cancel(job_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404, "unknown run") from exc

    return service


app = create_app()
