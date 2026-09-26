"""HTTP transport for registered artifact inspection and application edits."""

from __future__ import annotations

import asyncio
import io
import math
from collections.abc import Callable
from threading import Event
from typing import Any, Literal, TypeVar

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from contracts.models import SemanticEditOperation, StrictModel, Synchronization
from media import IngestError, MediaReader
from pipeline.runner import RevisionConflict
from storage import ArtifactKey
from storage.store import StorageError
from tkd_poomsae.inspection import MAX_BYTES, Inspection, InspectionError, encoded
from tkd_poomsae.media_access import MediaAccess, MediaAccessError, frame_info
from tkd_poomsae.semantic_edits import (
    EditError,
    IncompatibleAutomaticBase,
    StaleRevision,
)

T = TypeVar("T")


class SyncEdit(StrictModel):
    expected_revision: int = Field(ge=0, strict=True)
    camera: str
    offset_seconds: float
    author: str = Field(min_length=1)
    source: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ParserEdit(StrictModel):
    expected_revision: int = Field(ge=0, strict=True)
    automatic_revision: str
    command: Literal["apply", "undo", "reset"] = "apply"
    operations: list[SemanticEditOperation] = Field(default_factory=list, max_length=64)
    author: str = Field(min_length=1)
    source: str = Field(min_length=1)
    reason: str = Field(min_length=1)


def routes(service: FastAPI, inspection: Inspection, media: MediaAccess) -> None:
    service.state.inspection = inspection

    @service.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> Response:
        if "/inspection" not in request.url.path:
            return await request_validation_exception_handler(request, exc)
        # Rejected NaN/Infinity input must not itself leak nonfinite JSON.
        return JSONResponse(
            {
                "detail": [
                    {"loc": e["loc"], "msg": e["msg"], "type": e["type"]}
                    for e in exc.errors()
                ]
            },
            status_code=422,
        )

    def guarded(action: Callable[[], T]) -> T:
        try:
            return action()
        except (InspectionError, MediaAccessError) as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        except (RevisionConflict, StaleRevision, IncompatibleAutomaticBase) as exc:
            raise HTTPException(409, str(exc)) from exc
        except EditError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, StorageError, IngestError) as exc:
            raise HTTPException(
                409, "registered resource unavailable or changed"
            ) from exc
        except ValueError as exc:
            raise HTTPException(422, "invalid project or edit") from exc

    def cached(
        request: Request, value: dict[str, Any], expected: str | None
    ) -> Response:
        if expected is not None and value["revision"] != expected:
            raise HTTPException(
                409, "inspection revision changed; discard stale request"
            )
        content = encoded(value)
        if len(content) > MAX_BYTES:
            raise HTTPException(413, "inspection response exceeds 2 MiB")
        import hashlib

        etag = '"' + hashlib.sha256(content).hexdigest() + '"'
        headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
        if any(
            t.strip().removeprefix("W/") in {etag, "*"}
            for t in request.headers.get("if-none-match", "").split(",")
        ):
            return Response(status_code=304, headers=headers)
        return Response(content, media_type="application/json", headers=headers)

    @service.get("/api/projects/{project}/inspection")
    def capabilities(project: str, request: Request) -> Response:
        def read() -> Response:
            with inspection.lease(project):
                headers, clock = inspection.headers(project)
                value = {
                    "revision": inspection.revision(headers, clock),
                    "source_revisions": clock["sources"],
                    "sync_revision": clock["sync_revision"],
                    "limits": {
                        "window_seconds": 30,
                        "rows": 256,
                        "response_bytes": 2 * 1024 * 1024,
                    },
                    "products": {
                        name: {
                            "id": h["id"],
                            "artifact_revision": h["manifest_revision"],
                            "effective_edit_revision": h.get("edit_revision", 0),
                            "available": inspection.current(
                                h, clock, name.startswith("observations:")
                            ),
                            "reason": None
                            if inspection.current(
                                h, clock, name.startswith("observations:")
                            )
                            else "source/sync revision changed",
                            "arrays": h["arrays"],
                            "provenance": h["provenance"],
                        }
                        for name, h in headers.items()
                        if not name.startswith("observations:")
                    },
                }
                return cached(request, value, None)

        return guarded(read)

    @service.get("/api/projects/{project}/inspection/{product}/window")
    async def window(
        project: str,
        product: str,
        request: Request,
        start: float,
        end: float,
        collection: str = "samples",
        limit: int = 100,
        cursor: int = 0,
        expected_revision: str | None = None,
    ) -> Response:
        cancelled = Event()

        async def watch() -> None:
            while not cancelled.is_set():
                if await request.is_disconnected():
                    cancelled.set()
                    return
                await asyncio.sleep(0.05)

        watcher = asyncio.create_task(watch())

        def read() -> Response:
            with inspection.lease(project):
                return cached(
                    request,
                    inspection.window(
                        project,
                        product,
                        collection,
                        start,
                        end,
                        limit,
                        cursor,
                        cancelled,
                    ),
                    expected_revision,
                )

        try:
            return await run_in_threadpool(guarded, read)
        finally:
            cancelled.set()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    @service.get("/api/projects/{project}/inspection/entities")
    def entity(
        project: str, id: str, request: Request, expected_revision: str | None = None
    ) -> Response:
        def read() -> Response:
            with inspection.lease(project):
                return cached(
                    request, inspection.entity(project, id), expected_revision
                )

        return guarded(read)

    @service.get("/api/projects/{project}/inspection/{product}/arrays/{array_id}")
    def array(
        project: str,
        product: str,
        array_id: str,
        start: float,
        end: float,
        request: Request,
        expected_revision: str | None = None,
    ) -> Response:
        def read() -> Response:
            with inspection.lease(project):
                metadata, values, mask = inspection.array(
                    project, product, array_id, start, end
                )
                if (
                    expected_revision is not None
                    and metadata["revision"] != expected_revision
                ):
                    raise InspectionError(409, "inspection revision changed")
                buffer = io.BytesIO()
                arrays = {
                    "values": values,
                    "metadata": np.frombuffer(
                        encoded(metadata | {"rows": []}), dtype=np.uint8
                    ),
                }
                if mask is not None:
                    arrays["missing_mask"] = mask
                np.savez(buffer, allow_pickle=False, **arrays)
                import hashlib

                content = buffer.getvalue()
                if len(content) > MAX_BYTES:
                    raise InspectionError(413, "array response exceeds 2 MiB")
                etag = '"' + hashlib.sha256(content).hexdigest() + '"'
                if request.headers.get("if-none-match") in {etag, "W/" + etag, "*"}:
                    return Response(status_code=304, headers={"ETag": etag})
                return Response(
                    content,
                    media_type="application/x-npz",
                    headers={
                        "Cache-Control": "private, no-cache",
                        "ETag": etag,
                    },
                )

        return guarded(read)

    @service.get("/api/projects/{project}/inspection/time/{camera}")
    def time_mapping(
        project: str, camera: str, seconds: float, request: Request
    ) -> Response:
        def read() -> Response:
            if not math.isfinite(seconds):
                raise InspectionError(422, "finite global seconds required")
            with inspection.lease(project):
                headers, clock = inspection.headers(project)
                sync = headers.get("sync")
                if not sync:
                    raise InspectionError(409, "registered synchronization unavailable")
                item = media.resolve(project, camera)
                offset = next(
                    (
                        o
                        for o in sync["offsets"]
                        if o["source_id"] == item.recording.source_id
                    ),
                    None,
                )
                if offset is None:
                    raise InspectionError(409, "source has no synchronization offset")
                state = inspection.pipe.status(project)
                edit = (
                    state["config"]
                    .get("sync", {})
                    .get("manual_offsets", {})
                    .get(camera)
                )
                effective = (
                    edit["offset_seconds"] if edit else offset.get("manual_seconds")
                )
                if effective is None:
                    if not offset["retained"]:
                        raise InspectionError(
                            409, offset["exclusion_reason"] or "camera excluded"
                        )
                    effective = (offset["automatic_seconds"] or 0) + (
                        offset.get("manual_correction_seconds") or 0
                    )
                source_seconds = seconds - effective
                reader = MediaReader(item.recording)
                before, after = reader.bracket(source_seconds)
                nearest = reader.nearest(source_seconds)

                def frame(ordinal: int) -> dict[str, Any]:
                    value = frame_info(item.recording, ordinal)
                    return value | {
                        "global_seconds": value["source_seconds"] + effective
                    }

                value = {
                    "revision": inspection.revision(headers, clock),
                    "sync_revision": clock["sync_revision"],
                    "camera": camera,
                    "source_id": item.recording.source_id,
                    "global_seconds": seconds,
                    "source_seconds": source_seconds,
                    "effective_offset_seconds": effective,
                    "offset": offset,
                    "manual_revision": edit,
                    "before": frame(before.ordinal) if before else None,
                    "after": frame(after.ordinal) if after else None,
                    "nearest": frame(nearest.ordinal),
                    "nearest_gap_seconds": abs(nearest.seconds - source_seconds),
                    "interpolation": "exact"
                    if abs(nearest.seconds - source_seconds) < 1e-9
                    else "bracket"
                    if before and after
                    else "outside_coverage",
                    "geometry_interpolated": False,
                }
                return cached(request, value, None)

        return guarded(read)

    @service.post("/api/projects/{project}/inspection/sync-offset")
    def sync_edit(project: str, data: SyncEdit) -> dict[str, Any]:
        def edit() -> dict[str, Any]:
            # Application function compares under its owning run lock.
            headers, _ = inspection.headers(project)
            sync = headers.get("sync")
            synchronization: Synchronization | None = None
            if sync:
                inspection.checked_path(sync)
                candidate = inspection.pipe.store.get(
                    ArtifactKey(**sync["key"])
                ).metadata
                if not isinstance(candidate, Synchronization):
                    raise InspectionError(409, "synchronization unavailable")
                synchronization = candidate
            revision = inspection.pipe.revise_sync_offset(
                project,
                data.camera,
                data.offset_seconds,
                author=data.author,
                source=data.source,
                reason=data.reason,
                expected_revision=data.expected_revision,
                synchronization=synchronization,
            )
            return {"revision": data.expected_revision + 1, "edit": revision}

        return guarded(edit)

    @service.post("/api/projects/{project}/inspection/parser-edits")
    def parser_edit(project: str, data: ParserEdit) -> dict[str, Any]:
        def edit() -> dict[str, Any]:
            with inspection.lease(project, write=True):
                automatic = inspection.automatic(project)
                from storage import hash_file

                if (
                    hash_file(automatic.path / "manifest.json")
                    != data.automatic_revision
                ):
                    raise InspectionError(409, "automatic parser revision changed")
                editor = inspection.editor(project)
                attribution = {
                    "source": data.source,
                    "author": data.author,
                    "reason": data.reason,
                }
                if data.command == "apply":
                    view = editor.apply(
                        automatic,
                        data.expected_revision,
                        data.operations,
                        **attribution,
                    )
                else:
                    if data.operations:
                        raise InspectionError(
                            422, "undo/reset cannot include operations"
                        )
                    view = getattr(editor, data.command)(
                        automatic, data.expected_revision, **attribution
                    )
                inspection.update_semantics(project, view)
                return view.model_dump(mode="json", exclude={"semantics"})

        return guarded(edit)
