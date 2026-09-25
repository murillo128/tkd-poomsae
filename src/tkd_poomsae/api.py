"""Loopback inspection and bounded offline-job API."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from media import IngestError, MediaReader
from pipeline import Pipeline
from pipeline.runner import STAGE_ORDER
from storage import MissingResource
from tkd_poomsae.jobs import BusyProject, JobQueue, QueueFull, required_stages
from tkd_poomsae.media_access import (
    MediaAccess,
    MediaAccessError,
    RegisteredMedia,
    byte_range,
    frame_info,
    last_modified,
)


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
    media = MediaAccess(pipe, roots)

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

        def with_cors(response: Response) -> Response:
            if origin is not None:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Expose-Headers"] = (
                    "ETag, Last-Modified, Accept-Ranges, Content-Range, "
                    "Content-Length, X-Source-Ordinal, X-Source-PTS, X-Source-SHA256"
                )
                vary = response.headers.get("Vary", "")
                if "origin" not in {part.strip().lower() for part in vary.split(",")}:
                    response.headers["Vary"] = f"{vary}, Origin" if vary else "Origin"
            return response

        if request.method == "OPTIONS" and request.headers.get(
            "access-control-request-method"
        ):
            requested_method = request.headers["access-control-request-method"]
            requested_headers = {
                name.strip().lower()
                for name in request.headers.get(
                    "access-control-request-headers", ""
                ).split(",")
                if name.strip()
            }
            if (
                origin is None
                or not request.url.path.startswith("/api/")
                or requested_method not in {"GET", "HEAD", "POST"}
                or requested_headers
                - {
                    "x-tkd-local-request",
                    "content-type",
                    "range",
                    "if-none-match",
                    "if-modified-since",
                    "if-range",
                }
            ):
                return JSONResponse(
                    {"detail": "unsupported preflight"}, status_code=403
                )
            response = Response(status_code=204)
            response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, POST"
            response.headers["Access-Control-Allow-Headers"] = (
                "X-TKD-Local-Request, Content-Type, Range, If-None-Match, "
                "If-Modified-Since, If-Range"
            )
            response.headers["Vary"] = (
                "Origin, Access-Control-Request-Method, Access-Control-Request-Headers"
            )
            return with_cors(response)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if request.headers.get("x-tkd-local-request") != "1" or (
                request.headers.get("sec-fetch-site") == "cross-site" and origin is None
            ):
                return with_cors(
                    JSONResponse(
                        {"detail": "local request header required"}, status_code=403
                    )
                )
        return with_cors(await call_next(request))

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

    def registered_media(project: str, camera: str) -> RegisteredMedia:
        try:
            return media.resolve(project, camera)
        except MediaAccessError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        except IngestError as exc:
            raise HTTPException(422, f"source cannot be indexed: {exc}") from exc

    @service.api_route(
        "/api/projects/{project}/media/{camera}", methods=["GET", "HEAD"]
    )
    def source_video(project: str, camera: str, request: Request) -> Response:
        item = registered_media(project, camera)
        recording = item.recording
        etag = f'"{recording.sha256}"'
        modified = last_modified(recording.modified_ns)
        headers = {
            "Accept-Ranges": "bytes",
            "ETag": etag,
            "Last-Modified": modified,
            "Cache-Control": "private, no-cache",
            "X-Content-Type-Options": "nosniff",
        }
        match = request.headers.get("if-none-match")
        since = request.headers.get("if-modified-since")
        matches_etag = match is not None and any(
            tag.strip().removeprefix("W/") in {etag, "*"} for tag in match.split(",")
        )
        matches_date = False
        if match is None and since is not None:
            try:
                parsed = parsedate_to_datetime(since)
                if parsed.tzinfo is not None:
                    matches_date = parsed >= datetime.fromtimestamp(
                        recording.modified_ns / 1e9, UTC
                    ).replace(microsecond=0)
            except (TypeError, ValueError, OverflowError):
                pass
        if matches_etag or matches_date:
            return Response(status_code=304, headers=headers)
        size = recording.size_bytes
        requested = request.headers.get("range")
        if_range = request.headers.get("if-range")
        if requested and if_range and if_range not in {etag, modified}:
            requested = None
        selected = byte_range(requested, size) if requested else None
        if requested and selected is None:
            return Response(
                status_code=416,
                headers={
                    **headers,
                    "Content-Range": f"bytes */{size}",
                    "Content-Length": "0",
                },
            )
        start, end = selected if selected is not None else (0, size - 1)
        headers["Content-Length"] = str(end - start + 1)
        if selected is not None:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        status = 206 if selected is not None else 200
        if request.method == "HEAD":
            return Response(
                status_code=status, media_type=item.content_type, headers=headers
            )
        if not media.streams.acquire(blocking=False):
            raise HTTPException(503, "media stream capacity reached")
        try:
            stream = media.open_stream(item)
        except MediaAccessError as exc:
            media.streams.release()
            raise HTTPException(exc.status, str(exc)) from exc

        def chunks() -> Iterator[bytes]:
            try:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    remaining -= len(block)
                    yield block
            finally:
                stream.close()
                media.streams.release()

        return StreamingResponse(
            chunks(), status_code=status, media_type=item.content_type, headers=headers
        )

    @service.get("/api/projects/{project}/media/{camera}/metadata")
    def media_metadata(project: str, camera: str) -> dict[str, Any]:
        item = registered_media(project, camera)
        recording = item.recording
        return {
            "source_id": recording.source_id,
            "source_sha256": recording.sha256,
            "camera_id": camera,
            "content_type": item.content_type,
            "codec": recording.codec,
            "browser_playback": item.browser_playback,
            "browser_playback_reason": (
                None
                if item.browser_playback
                else "source codec/container requires exact PNG frame fallback"
            ),
            "frame_count": len(recording.frames),
            "first_frame": frame_info(recording, 0),
            "last_frame": frame_info(recording, len(recording.frames) - 1),
        }

    @service.get("/api/projects/{project}/media/{camera}/frames")
    def source_frames(
        project: str, camera: str, start: int = 0, limit: int = 100
    ) -> dict[str, Any]:
        recording = registered_media(project, camera).recording
        if start < 0 or limit < 1 or limit > 256:
            raise HTTPException(
                422, "frame page must have nonnegative start and limit 1..256"
            )
        end = min(start + limit, len(recording.frames))
        return {
            "source_id": recording.source_id,
            "source_sha256": recording.sha256,
            "total": len(recording.frames),
            "start": start,
            "frames": [frame_info(recording, ordinal) for ordinal in range(start, end)],
        }

    @service.get("/api/projects/{project}/media/{camera}/frames/nearest")
    def nearest_frame(project: str, camera: str, seconds: float) -> dict[str, Any]:
        if not -1e12 < seconds < 1e12:
            raise HTTPException(422, "invalid source time")
        recording = registered_media(project, camera).recording
        ref = MediaReader(recording).nearest(seconds)
        return {
            "requested_source_seconds": seconds,
            "frame": frame_info(recording, ref.ordinal),
        }

    @service.get("/api/projects/{project}/media/{camera}/frames/bracket")
    def bracket_frames(project: str, camera: str, seconds: float) -> dict[str, Any]:
        if not -1e12 < seconds < 1e12:
            raise HTTPException(422, "invalid source time")
        recording = registered_media(project, camera).recording
        before, after = MediaReader(recording).bracket(seconds)
        return {
            "requested_source_seconds": seconds,
            "before": frame_info(recording, before.ordinal) if before else None,
            "after": frame_info(recording, after.ordinal) if after else None,
        }

    @service.get("/api/projects/{project}/media/{camera}/frames/{ordinal}/image")
    def exact_frame(project: str, camera: str, ordinal: int) -> Response:
        item = registered_media(project, camera)
        recording = item.recording
        if ordinal < 0 or ordinal >= len(recording.frames):
            raise HTTPException(404, "unknown source frame")
        try:
            content = media.preview(item, ordinal)
        except MediaAccessError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        except IngestError as exc:
            raise HTTPException(409, f"source frame unavailable: {exc}") from exc
        info = frame_info(recording, ordinal)
        return Response(
            content,
            media_type="image/png",
            headers={
                "X-Source-Ordinal": str(ordinal),
                "X-Source-PTS": str(info["pts"]),
                "X-Source-SHA256": recording.sha256,
                "Cache-Control": "private, no-cache",
            },
        )

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
