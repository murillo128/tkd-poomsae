"""CLI adapter for annotation sessions; imports no vision runtime."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from contracts.models import SemanticEditOperation
from storage import ArtifactKey, ArtifactStore

from .session import SemanticEditor


def configure(subcommands: Any) -> None:
    parser = subcommands.add_parser(
        "semantic-edits", help="Edit immutable parser output offline"
    )
    commands = parser.add_subparsers(dest="edit_command", required=True)
    for name in ("view", "apply", "undo", "reset"):
        command = commands.add_parser(name)
        command.add_argument("session", help="Stable annotation session ID")
        command.add_argument(
            "--automatic-key",
            type=Path,
            required=True,
            help="JSON ArtifactKey for the existing automatic semantics",
        )
        if name != "view":
            command.add_argument("--expected-revision", type=int, required=True)
            command.add_argument("--source", required=True)
            command.add_argument("--author", required=True)
            command.add_argument("--reason", required=True)
        if name == "apply":
            command.add_argument(
                "--operations",
                type=Path,
                required=True,
                help="JSON array of typed semantic edit operations",
            )


def run(args: argparse.Namespace) -> int:
    try:
        store = ArtifactStore()
        automatic = store.get(ArtifactKey(**json.loads(args.automatic_key.read_text())))
        editor = SemanticEditor(store, args.session)
        if args.edit_command == "view":
            result = editor.view(automatic)
        else:
            provenance = dict(
                source=args.source, author=args.author, reason=args.reason
            )
            if args.edit_command == "apply":
                operations = TypeAdapter(list[SemanticEditOperation]).validate_json(
                    args.operations.read_text()
                )
                result = editor.apply(
                    automatic, args.expected_revision, operations, **provenance
                )
            elif args.edit_command == "undo":
                result = editor.undo(automatic, args.expected_revision, **provenance)
            else:
                result = editor.reset(automatic, args.expected_revision, **provenance)
        print(result.model_dump_json(indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"tkd-poomsae: {exc}", file=sys.stderr)
        return 1
