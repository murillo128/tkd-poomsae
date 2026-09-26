"""Durable optimistic sessions over immutable revision artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from typing import Literal

from pydantic import Field

from contracts.models import (
    ManualEdits,
    Provenance,
    SemanticEditOperation,
    Semantics,
    StrictModel,
)
from reconstruction.semantics import load_semantic_evidence, load_semantics
from reconstruction.semantics.core import REVISION as AUTOMATIC_REVISION
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

from .core import EditError, IncompatibleAutomaticBase, StaleRevision, apply_operations

REVISION = "semantic-edits-v1"


class EffectiveSemanticView(StrictModel):
    """Explicit annotation envelope; never validation ground truth by default."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    session_id: str
    revision: int = Field(ge=0)
    automatic_semantics_id: str
    automatic_manifest_sha256: str
    manual_edits_id: str | None
    origin: Literal["automatic", "manual"]
    validation_ground_truth: Literal[False] = False
    semantics: Semantics


class SemanticEditor:
    def __init__(self, store: ArtifactStore, session_id: str) -> None:
        if not session_id or session_id.strip() != session_id:
            raise EditError("nonempty session ID required")
        self.store = store
        self.session_id = session_id

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        root = self.store.root.namespace("runs")
        root.mkdir(parents=True, exist_ok=True)
        with (
            closing(sqlite3.connect(root / "semantic-edits.sqlite3", timeout=30)) as db,
            db,
        ):
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS semantic_edit_sessions "
                "(session TEXT PRIMARY KEY, automatic_id TEXT NOT NULL, "
                "digest TEXT NOT NULL, parser TEXT NOT NULL, revision INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS semantic_edit_revisions "
                "(session TEXT NOT NULL, revision INTEGER NOT NULL, "
                "artifact_key TEXT NOT NULL, PRIMARY KEY(session, revision))"
            )
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db

    def _base(
        self, automatic: ArtifactHandle
    ) -> tuple[Semantics, str, str, list[float]]:
        try:
            semantics = load_semantics(automatic)
        except ValueError as exc:
            raise IncompatibleAutomaticBase(str(exc)) from exc
        payload = load_semantic_evidence(automatic)
        key = ArtifactKey(
            layer="semantics",
            inputs=payload["input_revisions"],
            schema_version=semantics.schema_version,
            algorithm_revision=AUTOMATIC_REVISION,
            config_digest=semantics.provenance.config_digest,
        )
        verified = self.store.get(key)
        if verified.path != automatic.path or verified.metadata != semantics:
            raise IncompatibleAutomaticBase(
                "automatic handle differs from stored artifact"
            )
        if semantics.manual_edits_id is not None:
            raise IncompatibleAutomaticBase("automatic parser output required")
        digest = hash_file(automatic.path / "manifest.json")
        parser = semantics.provenance.model
        if not parser:
            raise IncompatibleAutomaticBase("automatic parser revision required")
        times = list(load_semantic_evidence(automatic)["motion_times"])
        return semantics, digest, parser, times

    def _head(
        self, db: sqlite3.Connection, automatic: Semantics, digest: str, parser: str
    ) -> int:
        row = db.execute(
            "SELECT automatic_id, digest, parser, revision "
            "FROM semantic_edit_sessions WHERE session=?",
            (self.session_id,),
        ).fetchone()
        if row is None:
            return 0
        if tuple(row[:3]) != (automatic.id, digest, parser):
            raise IncompatibleAutomaticBase(
                "session is pinned to a different automatic artifact/parser revision; "
                "create a separate session for a rerun, retaining existing edits"
            )
        return int(row[3])

    def _load(self, db: sqlite3.Connection, revision: int) -> ManualEdits:
        row = db.execute(
            "SELECT artifact_key FROM semantic_edit_revisions "
            "WHERE session=? AND revision=?",
            (self.session_id, revision),
        ).fetchone()
        if row is None:
            raise EditError("missing edit revision")
        handle = self.store.get(ArtifactKey(**json.loads(row[0])))
        metadata = handle.metadata
        if not isinstance(metadata, ManualEdits) or (
            metadata.session_id,
            metadata.revision,
            metadata.provenance.producer,
        ) != (self.session_id, revision, "semantic_edits"):
            raise EditError("revision identity mismatch")
        return metadata

    def _materialize(
        self,
        db: sqlite3.Connection,
        automatic: Semantics,
        times: list[float],
        state: int,
    ) -> Semantics:
        chain = []
        while state:
            item = self._load(db, state)
            if item.command != "apply" or item.parent_state_revision is None:
                raise EditError("invalid active state history")
            chain.append(item)
            state = item.parent_state_revision
        view = automatic
        for item in reversed(chain):
            view = apply_operations(view, times, item.operations)
        return view

    def _view(
        self,
        db: sqlite3.Connection,
        automatic: Semantics,
        digest: str,
        times: list[float],
        head: int,
    ) -> EffectiveSemanticView:
        edits = self._load(db, head) if head else None
        state = edits.state_revision if edits else 0
        assert state is not None
        view = self._materialize(db, automatic, times, state)
        if state:
            assert edits is not None
            view.id = f"effective:{edits.id}"
            view.automatic_semantics_id = automatic.id
            view.manual_edits_id = edits.id
            view.provenance = edits.provenance.model_copy(deep=True)
            view = Semantics.model_validate(view.model_dump())
        return EffectiveSemanticView(
            session_id=self.session_id,
            revision=head,
            automatic_semantics_id=automatic.id,
            automatic_manifest_sha256=digest,
            manual_edits_id=edits.id if edits else None,
            origin="manual" if state else "automatic",
            semantics=view,
        )

    def view(self, automatic: ArtifactHandle) -> EffectiveSemanticView:
        semantics, digest, parser, times = self._base(automatic)
        with self._connection() as db:
            head = self._head(db, semantics, digest, parser)
            return self._view(db, semantics, digest, times, head)

    def revision(self, number: int) -> ManualEdits:
        """Reload an immutable audit record, including superseded/reset revisions."""
        with self._connection() as db:
            return self._load(db, number)

    def apply(
        self,
        automatic: ArtifactHandle,
        expected_revision: int,
        operations: list[SemanticEditOperation],
        *,
        source: str,
        author: str,
        reason: str,
    ) -> EffectiveSemanticView:
        return self._commit(
            automatic,
            expected_revision,
            "apply",
            operations,
            source=source,
            author=author,
            reason=reason,
        )

    def undo(
        self,
        automatic: ArtifactHandle,
        expected_revision: int,
        *,
        source: str,
        author: str,
        reason: str,
    ) -> EffectiveSemanticView:
        return self._commit(
            automatic,
            expected_revision,
            "undo",
            [],
            source=source,
            author=author,
            reason=reason,
        )

    def reset(
        self,
        automatic: ArtifactHandle,
        expected_revision: int,
        *,
        source: str,
        author: str,
        reason: str,
    ) -> EffectiveSemanticView:
        return self._commit(
            automatic,
            expected_revision,
            "reset",
            [],
            source=source,
            author=author,
            reason=reason,
        )

    def _commit(
        self,
        automatic: ArtifactHandle,
        expected: int,
        command: Literal["apply", "undo", "reset"],
        operations: list[SemanticEditOperation],
        *,
        source: str,
        author: str,
        reason: str,
    ) -> EffectiveSemanticView:
        semantics, digest, parser, times = self._base(automatic)
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise StaleRevision("nonnegative integer expected revision required")
        with self._connection(write=True) as db:
            head = self._head(db, semantics, digest, parser)
            if head != expected:
                raise StaleRevision(
                    f"stale edit revision: expected {expected}, current {head}"
                )
            current = self._load(db, head) if head else None
            parent_state = current.state_revision if current else 0
            assert parent_state is not None
            if command == "undo":
                if current is None:
                    raise EditError("no edit to undo")
                # Undo reset restores its previous state. Repeated undo of applies
                # walks active history, rather than toggling the last control command.
                if current.command == "reset":
                    state = current.parent_state_revision
                elif parent_state:
                    state = self._load(db, parent_state).parent_state_revision
                else:
                    raise EditError("no active edit to undo")
            else:
                state = head + 1 if command == "apply" else 0
            body = {
                "kind": "manual_edits",
                "schema_version": "1.0.0",
                "automatic_semantics_id": semantics.id,
                "automatic_manifest_sha256": digest,
                "automatic_parser_revision": parser,
                "session_id": self.session_id,
                "revision": head + 1,
                "base_revision": head,
                "state_revision": state,
                "parent_state_revision": parent_state,
                "command": command,
                "operations": [o.model_dump(mode="json") for o in operations],
                "source": source,
                "author": author,
                "reason": reason,
            }
            config_digest = hash_config(body)
            key = ArtifactKey(
                layer="manual_edits",
                inputs={"automatic": digest},
                schema_version="1.0.0",
                algorithm_revision=REVISION,
                config_digest=config_digest,
            )
            edits = ManualEdits.model_validate(
                body
                | {
                    "id": f"manual_edits:{key.digest}",
                    "provenance": Provenance(
                        producer="semantic_edits",
                        model=REVISION,
                        config_digest=config_digest,
                    ),
                }
            )
            if command == "apply":
                active = self._materialize(db, semantics, times, parent_state)
                apply_operations(active, times, edits.operations)
            self.store.get_or_create(key, lambda: (edits, {}))
            serialized_key = json.dumps(
                {
                    "layer": key.layer,
                    "inputs": dict(key.inputs),
                    "schema_version": key.schema_version,
                    "algorithm_revision": key.algorithm_revision,
                    "config_digest": key.config_digest,
                },
                sort_keys=True,
            )
            db.execute(
                "INSERT INTO semantic_edit_revisions VALUES (?, ?, ?)",
                (self.session_id, head + 1, serialized_key),
            )
            db.execute(
                "INSERT INTO semantic_edit_sessions VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(session) DO UPDATE SET revision=excluded.revision",
                (self.session_id, semantics.id, digest, parser, head + 1),
            )
            return self._view(db, semantics, digest, times, head + 1)
