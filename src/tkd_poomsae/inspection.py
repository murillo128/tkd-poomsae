"""Registered persisted products indexed for bounded, offline inspection.

Registration is a trusted application operation, never an HTTP file/key input.
SQLite holds individual native samples so readers never load dense metadata.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

from contracts.models import (
    Alignment,
    Calibration,
    Ground,
    Observation,
    Reconstruction,
    Semantics,
    Synchronization,
    SyncOffset,
)
from pipeline import Pipeline
from pipeline.runner import _read_json
from pose.observation_run import load_window
from storage import ArtifactHandle, ArtifactKey, hash_config, hash_file
from tkd_poomsae.semantic_edits import SemanticEditor
from tkd_poomsae.semantic_edits.session import EffectiveSemanticView

MAX_WINDOW = 30.0
MAX_ROWS = 256
MAX_BYTES = 2 * 1024 * 1024
MAX_EDIT_BYTES = 8 * 1024 * 1024
COLLECTIONS = {
    "observations",
    "samples",
    "footprints",
    "pivots",
    "measurements",
    "steps",
    "stances",
    "actions",
    "phases",
    "keyframes",
    "joints",
    "ground_frames",
    "placements",
    "rotations",
    "placement_relations",
    "contact_events",
}


class InspectionError(ValueError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status


def encoded(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


def _signature(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


class Inspection:
    def __init__(
        self,
        pipe: Pipeline,
        *,
        source_hashes: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        self.pipe = pipe
        self.source_hashes = source_hashes

    @contextmanager
    def connection(self, project: str) -> Iterator[sqlite3.Connection]:
        directory = self.pipe._directory(project)
        if not (directory / "project.json").is_file():
            raise InspectionError(404, "unknown project")
        path = directory / "inspection.sqlite3"
        if path.is_symlink():
            raise InspectionError(409, "invalid inspection index")
        db = sqlite3.connect(path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS products (
                        name TEXT PRIMARY KEY, header TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS entities (
                        product TEXT, collection TEXT, id TEXT, start REAL,
                        end REAL, ordinal INTEGER, source_id TEXT, body TEXT,
                        PRIMARY KEY(product, collection, id));
                    CREATE TABLE IF NOT EXISTS bindings (
                        product TEXT, manifest TEXT, clock TEXT,
                        PRIMARY KEY(product, manifest));
                    CREATE TABLE IF NOT EXISTS observation_owners (
                        id TEXT PRIMARY KEY, product TEXT NOT NULL);
                    CREATE INDEX IF NOT EXISTS native_window ON entities
                        (product, collection, source_id, start);
                    CREATE INDEX IF NOT EXISTS native_ordinal ON entities
                        (product, collection, ordinal);
                    CREATE INDEX IF NOT EXISTS time_window ON entities
                        (product, collection, start);
                    CREATE INDEX IF NOT EXISTS entity_identity ON entities(id);
                """)
                yield db
        finally:
            db.close()

    def clock(self, project: str) -> dict[str, Any]:
        project_path, state_path = self.pipe._files(project)
        data, state = _read_json(project_path), _read_json(state_path)
        sources = (
            self.source_hashes(project)
            if self.source_hashes
            else {name: hash_file(Path(path)) for name, path in data["sources"].items()}
        )
        return {
            "sources": sources,
            "sync_revision": len(state.get("sync_revisions", [])),
            "sync_config": hash_config(state["config"].get("sync", {})),
            "sync_generation": state["generation"].get("sync", 0),
        }

    @staticmethod
    def _row(
        db: sqlite3.Connection,
        product: str,
        collection: str,
        identifier: str,
        ordinal: int,
        body: dict[str, Any],
        *,
        time: float | None = None,
    ) -> None:
        if "landmarks" in body:
            body = body | {
                "missing_mask": {
                    p["name"]: p.get("xyz_world", p.get("xy_px")) is None
                    for p in body["landmarks"]
                }
            }
        interval = body.get("interval", {})
        start = (
            time
            if time is not None
            else body.get("global_seconds", interval.get("start"))
        )
        end = time if time is not None else interval.get("end", start)
        payload = encoded(body | {"id": identifier})
        if len(payload) > MAX_BYTES - 4096:
            raise InspectionError(413, "individual entity exceeds 2 MiB")
        db.execute(
            "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                product,
                collection,
                identifier,
                start,
                end,
                ordinal,
                body.get("frame", {}).get("source_id"),
                payload.decode(),
            ),
        )

    @contextmanager
    def lease(self, project: str, *, write: bool = False) -> Iterator[None]:
        directory = self.pipe._directory(project)
        if not (directory / "project.json").is_file():
            raise InspectionError(404, "unknown project")
        with (directory / "run.lock").open("a+b") as lock:
            try:
                fcntl.flock(
                    lock, (fcntl.LOCK_EX if write else fcntl.LOCK_SH) | fcntl.LOCK_NB
                )
            except BlockingIOError as exc:
                raise InspectionError(409, "project is busy; retry inspection") from exc
            yield

    def register(
        self,
        project: str,
        products: Mapping[str, ArtifactKey],
        *,
        observations: Iterable[ArtifactKey] = (),
        lineage: Iterable[ArtifactKey] = (),
    ) -> None:
        with self.lease(project, write=True):
            self._register(
                project, products, observations=observations, lineage=lineage
            )
            if "semantics" in products:
                automatic = self.pipe.store.get(products["semantics"])
                editor = SemanticEditor(
                    self.pipe.store, project + ":" + automatic.path.name
                )
                self.update_semantics(project, editor.view(automatic))

    def _register(
        self,
        project: str,
        products: Mapping[str, ArtifactKey],
        *,
        observations: Iterable[ArtifactKey] = (),
        lineage: Iterable[ArtifactKey] = (),
    ) -> None:
        """Atomically bind verified artifacts to this project's current clock.

        Call after publication, using final reconstruction/ground/parser keys.
        Native windows are indexed one at a time; inference is never invoked.
        """
        allowed = {
            "sync": "synchronization",
            "calibration": "calibration",
            "reconstruction": "reconstruction",
            "ground": "ground",
            "semantics": "semantics",
        }
        if set(products) - allowed.keys() or "sync" not in products:
            raise ValueError("registered synchronization and known products required")
        handles = {name: self.pipe.store.get(key) for name, key in products.items()}
        if any(key.layer != allowed[name] for name, key in products.items()):
            raise ValueError("product layer mismatch")
        sync = handles["sync"].metadata
        assert isinstance(sync, Synchronization)
        clock = self.clock(project)
        source_ids = {f"source:{digest}" for digest in clock["sources"].values()}
        if {o.source_id for o in sync.offsets} != source_ids:
            raise ValueError("sync sources disagree with registered project")
        state = _read_json(self.pipe._files(project)[1])
        for camera, edit in (
            state["config"].get("sync", {}).get("manual_offsets", {}).items()
        ):
            offset = next(
                o
                for o in sync.offsets
                if o.source_id == f"source:{clock['sources'][camera]}"
            )
            if offset.effective_seconds != edit["offset_seconds"]:
                raise ValueError("sync artifact does not include current manual offset")
        motion = handles.get("reconstruction")
        if motion:
            self._verify_motion_lineage(
                products["reconstruction"],
                products["sync"],
                handles["sync"],
                clock["sources"],
                lineage,
            )
        calibration = handles.get("calibration")
        if calibration:
            value = calibration.metadata
            if not isinstance(value, Calibration):
                raise ValueError("calibration product required")
            if any(
                clock["sources"].get(camera.camera_id) is None
                or camera.source_id != "source:" + clock["sources"][camera.camera_id]
                for camera in value.cameras
            ):
                raise ValueError("calibration sources disagree with registered project")
            if motion and (
                not isinstance(motion.metadata, Reconstruction)
                or motion.metadata.calibration_id != value.id
                or motion.metadata.scale != value.scale
            ):
                raise ValueError("calibration must bind exact reconstruction frame")
            if motion:
                pinned = products["reconstruction"].calibration_revision
                if pinned and pinned != hash_file(calibration.path / "manifest.json"):
                    raise ValueError(
                        "calibration revision disagrees with reconstruction"
                    )
        ground = handles.get("ground")
        semantic = handles.get("semantics")
        if ground and (
            not motion
            or not isinstance(ground.metadata, Ground)
            or ground.metadata.reconstruction_id != motion.metadata.id
        ):
            raise ValueError("ground must bind exact reconstruction")
        if semantic and (
            not ground
            or not motion
            or not isinstance(semantic.metadata, Semantics)
            or semantic.metadata.ground_id != ground.metadata.id
            or semantic.metadata.reconstruction_id != motion.metadata.id
        ):
            raise ValueError("semantics must bind exact physical products")
        if semantic and motion:
            from reconstruction.semantics import load_semantic_evidence

            if not isinstance(motion.metadata, Reconstruction):
                raise ValueError("final native reconstruction required")
            semantic_times = load_semantic_evidence(semantic)["motion_times"]
            if semantic_times != [s.global_seconds for s in motion.metadata.samples]:
                raise ValueError(
                    "semantic indices disagree with final native motion clock"
                )
        with self.connection(project) as db:
            db.execute("DELETE FROM entities")
            db.execute("DELETE FROM products")
            db.execute("DELETE FROM observation_owners")
            for name, handle in handles.items():
                self._index(db, name, handle, products[name], clock)
            for key in observations:
                handle = self.pipe.store.get(key)
                if not isinstance(handle.metadata, Observation):
                    raise ValueError("native observation artifact required")
                self._index(db, "observations:" + key.digest, handle, key, clock)
                frames = (
                    load_window(handle)
                    if "window_records_json" in handle.files
                    else [handle.metadata]
                )
                for i, observation in enumerate(frames):
                    if observation.frame.source_id not in source_ids:
                        raise ValueError("observation source is not registered")
                    offset = next(
                        o
                        for o in sync.offsets
                        if o.source_id == observation.frame.source_id
                    )
                    self._row(
                        db,
                        "observations",
                        "observations",
                        observation.id,
                        i,
                        observation.model_dump(mode="json"),
                        time=observation.frame.source_seconds,
                    )
                    db.execute(
                        "INSERT INTO observation_owners VALUES (?, ?)",
                        (observation.id, "observations:" + key.digest),
                    )
            if (
                db.execute("SELECT sum(length(header)) FROM products").fetchone()[0]
                > MAX_EDIT_BYTES
            ):
                raise InspectionError(413, "registered product inventory exceeds 8 MiB")
            if self.clock(project) != clock:
                raise InspectionError(409, "project clock changed during registration")

    def _verify_motion_lineage(
        self,
        motion_key: ArtifactKey,
        sync_key: ArtifactKey,
        synchronization: ArtifactHandle,
        sources: dict[str, str],
        lineage: Iterable[ArtifactKey],
    ) -> None:
        """Prove the immutable motion inputs before stamping any inspection clock.

        Derived publishers bind raw/fit inputs by manifest hash; their upstream
        keys are supplied by the trusted operator, then verified through storage.
        A direct producer may bind source hashes and the synchronization manifest
        in its key. An index's previous bindings are never a substitute for proof.
        """
        sync_revision = hash_file(synchronization.path / "manifest.json")
        catalog: dict[str, ArtifactKey] = {}
        size = 0
        for key in lineage:
            size += len(encoded(key.__dict__ | {"inputs": dict(key.inputs)}))
            if size > MAX_EDIT_BYTES:
                raise InspectionError(413, "upstream key inventory exceeds 8 MiB")
            manifest = self.pipe.store._path(key) / "manifest.json"
            # Full content verification occurs when the referenced key is opened.
            if manifest.is_symlink() or manifest.parent.is_symlink():
                raise InspectionError(409, "invalid upstream artifact manifest")
            catalog[key.digest] = key
            catalog[hash_file(manifest)] = key
        verified: set[str] = set()
        sync = synchronization.metadata
        assert isinstance(sync, Synchronization)

        def resolve(revision: str, layer: str) -> tuple[ArtifactKey, ArtifactHandle]:
            key = catalog.get(revision)
            if key is None or key.layer != layer:
                raise InspectionError(409, "motion upstream lineage unavailable")
            return key, self.pipe.store.get(key)

        def alignment(revision: str) -> None:
            key, handle = resolve(revision, "alignment")
            metadata = handle.metadata
            if not isinstance(metadata, Alignment) or (
                metadata.synchronization_id != sync.id
                or not (
                    key.sync_revision == sync_revision
                    or key.inputs.get("sync") == sync_key.digest
                )
            ):
                raise InspectionError(409, "motion uses a different synchronization")
            if not metadata.observation_digests:
                raise InspectionError(409, "motion source lineage unavailable")
            for digest in metadata.observation_digests:
                window_key, window = resolve(digest, "observation")
                if (
                    not isinstance(window.metadata, Observation)
                    or window.metadata.frame.source_id
                    not in {"source:" + source_hash for source_hash in sources.values()}
                    or window.metadata.frame.source_id.removeprefix("source:")
                    not in window_key.inputs.values()
                ):
                    raise InspectionError(
                        409, "motion observation source lineage changed"
                    )

        def motion(key: ArtifactKey, depth: int) -> None:
            if key.digest in verified:
                return
            if depth > 8:
                raise InspectionError(
                    409, "motion upstream lineage exceeds depth bound"
                )
            handle = self.pipe.store.get(key)
            if not isinstance(handle.metadata, Reconstruction):
                raise InspectionError(409, "reconstruction lineage required")
            if key.sync_revision == sync_revision and set(sources.values()) <= set(
                key.inputs.values()
            ):
                verified.add(key.digest)
                return
            producer = handle.metadata.provenance.producer
            if producer == "reconstruction.triangulation":
                revision = key.inputs.get("attachment")
                if revision is None:
                    raise InspectionError(409, "motion alignment lineage unavailable")
                alignment(revision)
            else:
                parents = {
                    "reconstruction.articulated": ("raw_reconstruction",),
                    "reconstruction.detailed": ("source_reconstruction",),
                    "reconstruction.temporal": ("raw", "fit"),
                }.get(producer)
                if parents is None:
                    raise InspectionError(
                        409, "motion source/sync lineage unverifiable"
                    )
                for parent in parents:
                    revision = key.inputs.get(parent)
                    if revision is None:
                        raise InspectionError(
                            409, "motion reconstruction lineage unavailable"
                        )
                    parent_key, _ = resolve(revision, "reconstruction")
                    motion(parent_key, depth + 1)
            verified.add(key.digest)

        motion(motion_key, 0)

    def _index(
        self,
        db: sqlite3.Connection,
        name: str,
        handle: ArtifactHandle,
        key: ArtifactKey,
        clock: dict[str, Any],
    ) -> None:
        if (
            name == "semantics"
            and sum(
                p.stat().st_size
                for p in [
                    handle.path / "metadata.json",
                    *(handle.path / f for f in handle.files.values()),
                ]
            )
            > MAX_EDIT_BYTES
        ):
            raise InspectionError(413, "semantic input exceeds 8 MiB")
        body = handle.metadata.model_dump(mode="json")
        arrays = body.get("arrays", [])
        header = {
            k: v for k, v in body.items() if k not in COLLECTIONS and k != "arrays"
        }
        # Large relations/semantic evidence belong to registered structured entities.
        header.pop("relations", None)
        if name.startswith("observations:"):
            header = {
                k: body[k] for k in ("id", "kind", "schema_version", "provenance")
            }
        header.update(
            {
                "key": key.__dict__ | {"inputs": dict(key.inputs)},
                "clock": clock,
                "manifest_revision": hash_file(handle.path / "manifest.json"),
                "sample_count": len(body.get("samples", [])),
                "arrays": arrays,
                "files": dict(handle.files),
                "signatures": {
                    str(p.relative_to(handle.path)): _signature(p)
                    for p in [
                        handle.path / "manifest.json",
                        handle.path / "metadata.json",
                        *(handle.path / f for f in handle.files.values()),
                    ]
                },
            }
        )
        if name in {"calibration", "reconstruction", "ground", "semantics"}:
            binding = encoded(clock).decode()
            previous = db.execute(
                "SELECT clock FROM bindings WHERE product=? AND manifest=?",
                (name, header["manifest_revision"]),
            ).fetchone()
            if previous and previous[0] != binding:
                raise InspectionError(
                    409, "cannot relabel an old timed artifact under a new clock"
                )
            db.execute(
                "INSERT OR IGNORE INTO bindings VALUES (?, ?, ?)",
                (name, header["manifest_revision"], binding),
            )
        if name == "ground" and "ground_view_json" in handle.files:
            from reconstruction.ground_view import load_ground_view

            view = load_ground_view(handle)
            if not view.root_trajectory:
                raise ValueError("nonempty ground-view product required")
            header["ground_view"] = {
                "world_unit": view.world_unit,
                "participant_id": view.participant_id,
                "scene_bounds": view.scene_bounds.model_dump(mode="json")
                if view.scene_bounds
                else None,
                "start_seconds": view.root_trajectory[0].global_seconds,
                "end_seconds": view.root_trajectory[-1].global_seconds,
                "max_gap_seconds": view.config.max_gap_seconds,
            }
            runs = {i: n for n, run in enumerate(view.root_path_indices) for i in run}
            assert view.physical is not None
            pivot_ids: dict[int, list[str]] = {}
            for event in view.physical.events:
                for index in event.rotation_indices:
                    pivot_ids.setdefault(index // 2, []).append(event.pivot.id)
            for i, root in enumerate(view.root_trajectory):
                feet = view.physical.placements.trajectory[2 * i : 2 * i + 2]
                contact = view.contact_events[i]
                snapshot = {
                    "requested_seconds": root.global_seconds,
                    "sampled_seconds": root.global_seconds,
                    "status": "native",
                    "bracket_seconds": None,
                    "root": root.model_dump(mode="json"),
                    "feet": [foot.model_dump(mode="json") for foot in feet],
                    "contact_event": contact.model_dump(mode="json"),
                    "placement_ids": sorted({f.event_id for f in feet if f.event_id}),
                    "pivot_ids": pivot_ids.get(i, []),
                    "reasons": [],
                }
                self._row(
                    db,
                    name,
                    "contact_events",
                    contact.id,
                    i,
                    contact.model_dump(mode="json"),
                    time=root.global_seconds,
                )
                self._row(
                    db,
                    name,
                    "ground_frames",
                    root.id,
                    i,
                    snapshot | {"path_run": runs.get(i)},
                    time=root.global_seconds,
                )
            for collection, rows in (
                ("placements", view.physical.placements.events),
                ("rotations", view.physical.events),
                ("placement_relations", view.physical.placements.relations),
            ):
                for i, entity in enumerate(rows):
                    row = entity.model_dump(mode="json")
                    if collection == "placements":
                        row |= {"interval": row["footprint"]["interval"]}
                        identifier = row["footprint"]["id"] + "/placement"
                    elif collection == "rotations":
                        row |= {"interval": row["pivot"]["interval"]}
                        identifier = row["pivot"]["id"] + "/rotation"
                    else:
                        identifier = f"{body['id']}/placement_relations/{i}"
                    self._row(db, name, collection, identifier, i, row)
        db.execute(
            "INSERT INTO products VALUES (?, ?)", (name, encoded(header).decode())
        )
        for collection in COLLECTIONS - {"joints", "observations"}:
            for i, row in enumerate(body.get(collection, [])):
                identifier = row.get("id", f"{body['id']}/{collection}/{i}")
                self._row(db, name, collection, identifier, i, row)
                if name == "reconstruction" and collection == "samples":
                    for point in row["landmarks"]:
                        self._row(
                            db,
                            name,
                            "joints",
                            f"{identifier}/{point['name']}",
                            i,
                            point
                            | {
                                "global_seconds": row["global_seconds"],
                                "motion_sample_index": i,
                            },
                        )

    def headers(self, project: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if not (self.pipe._directory(project) / "inspection.sqlite3").exists():
            return {}, self.clock(project)
        with self.connection(project) as db:
            headers = {
                r["name"]: json.loads(r["header"])
                for r in db.execute("SELECT * FROM products")
            }
        semantic = headers.get("semantics")
        edit_db = self.pipe.store.root.namespace("runs") / "semantic-edits.sqlite3"
        if semantic and edit_db.exists():
            session = project + ":" + ArtifactKey(**semantic["key"]).digest
            with sqlite3.connect(f"file:{edit_db}?mode=ro", uri=True) as db:
                row = db.execute(
                    "SELECT revision FROM semantic_edit_sessions WHERE session=?",
                    (session,),
                ).fetchone()
            if row and row[0] != semantic.get("edit_revision", 0):
                raise InspectionError(
                    409, "semantic edit index is stale; register products again"
                )
        return headers, self.clock(project)

    def current(
        self, header: dict[str, Any], clock: dict[str, Any], native: bool = False
    ) -> bool:
        return header["clock"]["sources"] == clock["sources"] and (
            native or header["clock"] == clock
        )

    def checked_path(self, header: dict[str, Any]) -> Path:
        key = ArtifactKey(**header["key"])
        path = self.pipe.store.root.namespace("derived") / key.layer / key.digest
        for relative, signature in header["signatures"].items():
            item = path / relative
            if (
                self.pipe.store.root.namespace("derived").is_symlink()
                or path.is_symlink()
                or path.parent.is_symlink()
                or item.is_symlink()
                or item.parent.is_symlink()
                or _signature(item) != signature
            ):
                raise InspectionError(
                    409, "registered artifact changed; register it again"
                )
        return path

    def revision(self, headers: dict[str, Any], clock: dict[str, Any]) -> str:
        return hash_config(
            {
                "clock": clock,
                "products": {
                    name: [h["manifest_revision"], h.get("edit_revision", 0)]
                    for name, h in headers.items()
                },
            }
        )

    def window(
        self,
        project: str,
        product: str,
        collection: str,
        start: float,
        end: float,
        limit: int,
        cursor: int = 0,
        cancelled: Event | None = None,
    ) -> dict[str, Any]:
        if cancelled and cancelled.is_set():
            raise InspectionError(499, "inspection request cancelled")
        allowed = {
            "observations": {"observations"},
            "reconstruction": {"samples", "joints"},
            "ground": {
                "samples",
                "footprints",
                "pivots",
                "measurements",
                "ground_frames",
                "placements",
                "rotations",
                "placement_relations",
                "contact_events",
            },
            "semantics": {"steps", "stances", "actions", "phases", "keyframes"},
        }
        if (
            collection not in allowed.get(product, set())
            or product not in {"observations", "reconstruction", "ground", "semantics"}
            or not all(math.isfinite(t) for t in (start, end))
            or end < start
            or end - start > MAX_WINDOW
            or not 1 <= limit <= MAX_ROWS
            or not 0 <= cursor <= 2**63 - 1
        ):
            raise InspectionError(
                422, "invalid collection/window/page; maximum 30s and 256 rows"
            )
        headers, clock = self.headers(project)
        native = product == "observations"
        relevant = (
            [h for name, h in headers.items() if name.startswith("observations:")]
            if native
            else [headers[product]]
            if product in headers
            else []
        )
        available = bool(relevant) and all(
            self.current(h, clock, native) for h in relevant
        )
        result: dict[str, Any] = {
            "product": product,
            "collection": collection,
            "revision": self.revision(headers, clock),
            "sync_revision": clock["sync_revision"],
            "source_revisions": clock["sources"],
            "artifact_revisions": [h["manifest_revision"] for h in relevant],
            "available": available,
            "reason": None
            if available
            else "artifact unavailable or stale for current source/sync revision",
            "unit": "px"
            if native
            else (
                "m"
                if relevant
                and headers.get("reconstruction", relevant[0]).get("scale") == "metric"
                else "arbitrary"
            ),
            "effective_edit_revision": headers.get("semantics", {}).get(
                "edit_revision", 0
            ),
            "ground_view": headers.get(product, {}).get("ground_view"),
            "origin": headers.get(product, {}).get("origin", "automatic"),
            "rows": [],
            "next_cursor": None,
        }
        if not available:
            return result
        for header in relevant:
            self.checked_path(header)
        with self.connection(project) as db:
            # A keyset cursor bounds query work independently of recording length.
            if native:
                sync = headers.get("sync")
                if not sync:
                    return result | {
                        "available": False,
                        "reason": "synchronization unavailable",
                    }
                edits = (
                    _read_json(self.pipe._files(project)[1])["config"]
                    .get("sync", {})
                    .get("manual_offsets", {})
                )
                # Current offsets change query bounds on the native source-time index.
                # Select one bounded page across all sources before JSON decoding.
                offsets = sync["offsets"]
                predicates: list[str] = []
                parameters: list[Any] = [product, collection, cursor]
                effective_offsets: dict[str, float] = {}
                for offset in offsets:
                    source_id = offset["source_id"]
                    camera = next(
                        k
                        for k, v in clock["sources"].items()
                        if source_id == f"source:{v}"
                    )
                    try:
                        effective = self.effective_offset(offset, edits.get(camera))
                    except InspectionError:
                        continue
                    effective_offsets[source_id] = effective
                    predicates.append("(source_id=? AND start>=? AND start<=?)")
                    parameters.extend([source_id, start - effective, end - effective])
                if not predicates:
                    return result | {"reason": "no retained source has a usable offset"}
                rows = db.execute(
                    "SELECT rowid AS cursor, body FROM entities "
                    "WHERE product=? AND collection=? AND rowid>? AND ("
                    + " OR ".join(predicates)
                    + ") ORDER BY rowid LIMIT ?",
                    (*parameters, limit + 1),
                )
            else:
                rows = db.execute(
                    "SELECT rowid AS cursor, body FROM entities WHERE product=? "
                    "AND collection=? "
                    "AND (start IS NULL OR (start<=? AND end>=?)) AND rowid>? "
                    "ORDER BY rowid LIMIT ?",
                    (product, collection, end, start, cursor, limit + 1),
                )
            size = len(encoded(result))
            for row in rows:
                if cancelled and cancelled.is_set():
                    raise InspectionError(499, "inspection request cancelled")
                value = json.loads(row["body"])
                if native:
                    effective = effective_offsets[value["frame"]["source_id"]]
                    value["frame"]["offset_seconds"] = effective
                    value["frame"]["global_seconds"] = (
                        value["frame"]["source_seconds"] + effective
                    )
                length = len(encoded(value))
                if length > MAX_BYTES - 4096:
                    raise InspectionError(
                        413, "individual entity exceeds response bound"
                    )
                if len(result["rows"]) == limit or size + length > MAX_BYTES - 4096:
                    break
                result["rows"].append(value)
                result["next_cursor"] = row["cursor"]
                size += length + 1
            else:
                result["next_cursor"] = None
        if self.clock(project) != clock:
            raise InspectionError(
                409, "clock changed during query; discard stale response"
            )
        return result

    def ground_snapshot(self, project: str, seconds: float) -> dict[str, Any]:
        """Read one indexed native snapshot; preserve the producer's gap policy."""
        if not math.isfinite(seconds):
            raise InspectionError(422, "finite global seconds required")
        result = self.window(project, "ground", "ground_frames", seconds, seconds, 1)
        metadata = result["ground_view"]
        answer: dict[str, Any] = {
            "revision": result["revision"],
            "ground_view": metadata,
            "available": False,
            "reason": result["reason"],
            "snapshot": None,
        }
        if not result["available"] or metadata is None:
            return answer | {
                "reason": result["reason"] or "ground-view product unavailable"
            }
        if not metadata["start_seconds"] <= seconds <= metadata["end_seconds"]:
            return answer | {"reason": "outside_execution"}
        with self.connection(project) as db:
            before = db.execute(
                "SELECT body, start FROM entities WHERE product='ground' "
                "AND collection='ground_frames' AND start<=? "
                "ORDER BY start DESC LIMIT 1",
                (seconds,),
            ).fetchone()
            after = db.execute(
                "SELECT start FROM entities WHERE product='ground' "
                "AND collection='ground_frames' AND start>? ORDER BY start LIMIT 1",
                (seconds,),
            ).fetchone()
        if before is None:
            return answer | {"reason": "outside_execution"}
        snapshot = json.loads(before["body"])
        snapshot["requested_seconds"] = seconds
        if before["start"] != seconds:
            if (
                after is None
                or after["start"] - before["start"] > metadata["max_gap_seconds"]
            ):
                return answer | {"reason": "native_time_gap"}
            snapshot |= {
                "status": "native_snapshot",
                "bracket_seconds": [before["start"], after["start"]],
                "reasons": ["display_snapshot_not_interpolated_physics"],
            }
        return answer | {"available": True, "reason": None, "snapshot": snapshot}

    def automatic(self, project: str) -> ArtifactHandle:
        headers, clock = self.headers(project)
        header = headers.get("semantics")
        if not header or not self.current(header, clock):
            raise InspectionError(409, "current automatic semantics unavailable")
        path = self.checked_path(header)
        if (
            sum(
                p.stat().st_size
                for p in [
                    path / "metadata.json",
                    *(path / f for f in header["files"].values()),
                ]
            )
            > MAX_EDIT_BYTES
        ):
            raise InspectionError(413, "semantic edit input exceeds 8 MiB")
        return self.pipe.store.get(ArtifactKey(**header["key"]))

    def editor(self, project: str) -> SemanticEditor:

        # A changed automatic artifact starts a separate session, retaining history.
        automatic = self.automatic(project)
        return SemanticEditor(self.pipe.store, project + ":" + automatic.path.name)

    def time_bounds(self, project: str, product: str) -> dict[str, float] | None:
        """Expose indexed global bounds without opening a dense artifact."""
        with self.connection(project) as db:
            start, end = db.execute(
                "SELECT MIN(start), MAX(end) FROM entities WHERE product=?",
                (product,),
            ).fetchone()
        return None if start is None else {"start": start, "end": end}

    def update_semantics(self, project: str, view: EffectiveSemanticView) -> None:
        with self.connection(project) as db:
            db.execute("DELETE FROM entities WHERE product='semantics'")
            header = json.loads(
                db.execute(
                    "SELECT header FROM products WHERE name='semantics'"
                ).fetchone()[0]
            )
            header["edit_revision"] = view.revision
            header["manual_edits_id"] = view.manual_edits_id
            header["origin"] = view.origin
            db.execute(
                "UPDATE products SET header=? WHERE name='semantics'",
                (encoded(header).decode(),),
            )
            for collection in ("steps", "stances", "actions", "phases", "keyframes"):
                for i, value in enumerate(getattr(view.semantics, collection)):
                    self._row(
                        db,
                        "semantics",
                        collection,
                        value.id,
                        i,
                        value.model_dump(mode="json"),
                    )

    @staticmethod
    def effective_offset(
        offset: dict[str, Any] | None, edit: dict[str, Any] | None = None
    ) -> float:
        if offset is None:
            raise InspectionError(409, "source has no synchronization offset")
        manual = (edit or {}).get("offset_seconds", offset.get("manual_seconds"))
        if manual is not None:
            return float(manual)
        if not offset["retained"]:
            raise InspectionError(409, offset["exclusion_reason"] or "camera excluded")
        try:
            return SyncOffset.model_validate(offset).effective_seconds
        except ValueError:
            raise InspectionError(
                409,
                offset["exclusion_reason"]
                or "source has no usable synchronization offset",
            ) from None

    def effective_frame(
        self,
        project: str,
        frame: dict[str, Any],
        headers: dict[str, Any],
        clock: dict[str, Any],
    ) -> dict[str, Any]:
        sync = headers.get("sync", {})
        offset = next(
            (
                o
                for o in sync.get("offsets", [])
                if o["source_id"] == frame["source_id"]
            ),
            None,
        )
        camera = next(
            k
            for k, v in clock["sources"].items()
            if frame["source_id"] == "source:" + v
        )
        edits = (
            _read_json(self.pipe._files(project)[1])["config"]
            .get("sync", {})
            .get("manual_offsets", {})
        )
        try:
            effective = self.effective_offset(offset, edits.get(camera))
        except InspectionError as error:
            return frame | {
                "offset_seconds": None,
                "global_seconds": None,
                "global_time_reason": str(error),
            }
        return frame | {
            "offset_seconds": effective,
            "global_seconds": frame["source_seconds"] + effective,
        }

    def entity(self, project: str, identifier: str) -> dict[str, Any]:
        if not identifier or len(identifier) > 512:
            raise InspectionError(422, "invalid entity ID")
        headers, clock = self.headers(project)
        with self.connection(project) as db:
            row = db.execute(
                "SELECT * FROM entities WHERE id=? LIMIT 1", (identifier,)
            ).fetchone()
            landmark_name = None
            if row is None and "/" in identifier:
                parent, landmark_name = identifier.rsplit("/", 1)
                row = db.execute(
                    "SELECT * FROM entities WHERE id=? AND product='observations'",
                    (parent,),
                ).fetchone()
            if row is None:
                raise InspectionError(404, "unknown entity ID")
            product = row["product"]
            if product == "observations":
                owner = db.execute(
                    "SELECT product FROM observation_owners WHERE id=?", (row["id"],)
                ).fetchone()
                if owner is None or owner[0] not in headers:
                    raise InspectionError(
                        409, "native entity owner unavailable; re-register observations"
                    )
                relevant = [headers[owner[0]]]
            else:
                relevant = [headers[product]]
            if not all(
                self.current(h, clock, product == "observations") for h in relevant
            ):
                raise InspectionError(409, "selected entity belongs to stale artifact")
            for header in relevant:
                self.checked_path(header)
            value = json.loads(row["body"])
            if landmark_name is not None:
                point = next(
                    (p for p in value["landmarks"] if p["name"] == landmark_name),
                    None,
                )
                if point is None:
                    raise InspectionError(404, "unknown observation landmark")
                value = point | {
                    "id": identifier,
                    "frame": value["frame"],
                    "provenance": value["provenance"],
                }
            indices = set(value.get("motion_sample_indices", []))
            for link in value.get("motion_links", []):
                indices.update(link["motion_sample_indices"])
            if "motion_sample_index" in value:
                indices.add(value["motion_sample_index"])
            # For ground entities the physical interval selects contributing samples.
            if not indices and product == "ground":
                indices.update(
                    r[0]
                    for r in db.execute(
                        "SELECT ordinal FROM entities WHERE "
                        "product='reconstruction' AND collection='samples' AND "
                        "start>=? AND start<=? LIMIT ?",
                        (row["start"], row["end"], MAX_ROWS + 1),
                    )
                )
            evidence: list[dict[str, Any]] = []
            physical_evidence: list[dict[str, Any]] = []
            source_ids = set(value.get("quality", {}).get("source_ids", []))
            for index in sorted(indices)[:MAX_ROWS]:
                sample = db.execute(
                    "SELECT body FROM entities WHERE product='reconstruction' "
                    "AND collection='samples' AND ordinal=?",
                    (index,),
                ).fetchone()
                if sample:
                    motion = json.loads(sample[0])
                    physical_evidence.append(
                        {
                            "id": motion["id"],
                            "global_seconds": motion["global_seconds"],
                            "quality": motion["quality"],
                        }
                    )
                    if row["collection"] != "joints":
                        for point in motion["landmarks"]:
                            source_ids.update(point["quality"]["source_ids"])
            unresolved: list[str] = []
            for source_id in sorted(source_ids)[:MAX_ROWS]:
                source = db.execute(
                    "SELECT body FROM entities WHERE product='observations' AND id=?",
                    (source_id,),
                ).fetchone()
                if source is None:
                    unresolved.append(source_id)
                if source:
                    observation = json.loads(source[0])
                    evidence.append(
                        {
                            "observation_id": source_id,
                            "frame": self.effective_frame(
                                project, observation["frame"], headers, clock
                            ),
                            "native_frame": observation["frame"],
                            "quality": observation.get("quality"),
                            "landmarks": observation["landmarks"],
                            "provenance": observation["provenance"],
                        }
                    )
            if product == "observations":
                evidence.append(
                    {
                        "observation_id": row["id"],
                        "frame": self.effective_frame(
                            project, value["frame"], headers, clock
                        ),
                        "native_frame": value["frame"],
                    }
                )
                value = value | {
                    "native_frame": value["frame"],
                    "frame": evidence[-1]["frame"],
                }
        truncated = len(indices) > MAX_ROWS or len(source_ids) > MAX_ROWS
        reasons: list[str] = []
        if unresolved or not evidence:
            reasons.append(
                f"{len(unresolved)} contributing native observation IDs unavailable"
                if unresolved
                else "contributing native observation IDs unavailable"
            )
        if truncated:
            reasons.append("contributing source evidence truncated")
        reasons.extend(
            sorted(
                {
                    item["frame"]["global_time_reason"]
                    for item in evidence
                    if item["frame"].get("global_time_reason")
                }
            )
        )
        result = {
            "entity": value,
            "product": product,
            "source_evidence": evidence,
            "physical_evidence": physical_evidence,
            "source_evidence_reason": "; ".join(reasons) if reasons else None,
            "source_evidence_unavailable_count": len(unresolved),
            "source_evidence_unavailable_ids": unresolved,
            "evidence_truncated": truncated,
            "revision": self.revision(headers, clock),
            "artifact_revision": relevant[0]["manifest_revision"],
            "provenance": relevant[0]["provenance"],
            "generating_revisions": {
                name: relevant[0]["key"].get(name)
                for name in (
                    "algorithm_revision",
                    "model_revision",
                    "schema_version",
                    "config_digest",
                    "calibration_revision",
                    "sync_revision",
                )
            },
            "origin": relevant[0].get("origin", "automatic"),
            "unit": "px"
            if product == "observations"
            else (
                "m"
                if relevant[0].get(
                    "scale", headers.get("reconstruction", {}).get("scale")
                )
                == "metric"
                or relevant[0].get("ground_view", {}).get("world_unit") == "m"
                else "arbitrary"
            ),
            "effective_edit_revision": headers.get("semantics", {}).get(
                "edit_revision", 0
            ),
            "diagnostic_meaning": "reprojection is internal consistency, not accuracy",
        }
        if len(encoded(result)) > MAX_BYTES:
            raise InspectionError(413, "entity evidence exceeds response bound")
        if self.clock(project) != clock:
            raise InspectionError(409, "clock changed during entity query")
        return result

    def array(
        self,
        project: str,
        product: str,
        array_id: str,
        start: float,
        end: float,
    ) -> tuple[dict[str, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
        window = self.window(project, product, "samples", start, end, MAX_ROWS)
        if not window["available"]:
            raise InspectionError(409, window["reason"])
        headers, _ = self.headers(project)
        header = headers[product]
        descriptor = next((a for a in header["arrays"] if a["id"] == array_id), None)
        if (
            not descriptor
            or not descriptor["axes"]
            or descriptor["axes"][0] not in {"native_time", "time", "sample"}
        ):
            raise InspectionError(
                422, "only registered native sample arrays are queryable"
            )
        if window["next_cursor"] is not None:
            raise InspectionError(413, "array window exceeds 256 native samples")
        path = self.checked_path(header)
        with self.connection(project) as db:
            bounds = db.execute(
                "SELECT min(ordinal), max(ordinal), count(*) FROM entities WHERE "
                "product=? AND collection='samples' AND start>=? AND start<=?",
                (product, start, end),
            ).fetchone()
        first, last, count = bounds
        if descriptor["shape"][0] != header["sample_count"]:
            raise InspectionError(
                422, "array does not align with indexed native samples"
            )
        selection = slice(first, last + 1) if count else slice(0, 0)
        data = self._array_slice(header, path, array_id, selection)
        mask_id = descriptor.get("missing_mask_id")
        mask = self._array_slice(header, path, mask_id, selection) if mask_id else None
        if data.nbytes + (mask.nbytes if mask is not None else 0) > MAX_BYTES:
            raise InspectionError(413, "array window exceeds 2 MiB")
        if data.dtype.kind == "f" and not np.all(np.isfinite(data)):
            raise InspectionError(409, "nonfinite persisted array")
        return (
            window
            | {"array": descriptor, "sample_start": first, "sample_count": count},
            data,
            mask,
        )

    @staticmethod
    def _array_slice(
        header: dict[str, Any],
        directory: Path,
        array_id: str,
        selection: slice,
    ) -> np.ndarray[Any, Any]:
        relative = header["files"][array_id]
        path = directory / relative
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            stat = os.fstat(fd)
            signature = [
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ]
            if signature != header["signatures"][relative] or os.readlink(
                f"/proc/self/fd/{fd}"
            ) != str(path):
                raise InspectionError(409, "registered array changed while opening")
            # The mmap owns its descriptor after construction; pin the authorized
            # inode during np.load so path replacement cannot redirect the read.
            result: np.ndarray[Any, Any] = np.load(
                f"/proc/self/fd/{fd}",
                mmap_mode="r",
                allow_pickle=False,
            )[selection]
            return result
        finally:
            os.close(fd)
