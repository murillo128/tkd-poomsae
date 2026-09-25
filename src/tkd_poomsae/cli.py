"""Command line entry point."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import uvicorn

from pipeline import Pipeline
from tkd_poomsae.dataset import MendeleyDatasetProvider, UnsupportedVersion


def main() -> int:
    """Run the local observation service or show CLI help."""
    parser = argparse.ArgumentParser(
        prog="tkd-poomsae",
        description="Local TKD Poomsae observation tools",
    )
    subcommands = parser.add_subparsers(dest="command")
    serve = subcommands.add_parser("serve", help="Start the local API service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    datasets = subcommands.add_parser("datasets", help="Inspect registered datasets")
    dataset_commands = datasets.add_subparsers(dest="dataset_command", required=True)
    dataset_status = dataset_commands.add_parser(
        "status", help="Check the pinned local dataset"
    )
    dataset_status.add_argument("--version", type=int, default=1)
    register = subcommands.add_parser("register", help="Register a local project")
    register.add_argument("project")
    register.add_argument("--source", action="append", required=True, metavar="ID=PATH")
    analyze = subcommands.add_parser("analyze", help="Analyze through a stage")
    analyze.add_argument("project")
    analyze.add_argument("--through", default="parsing")
    analyze.add_argument("--config", type=Path, help="JSON stage settings file")
    resume = subcommands.add_parser("resume", help="Resume analysis")
    resume.add_argument("project")
    resume.add_argument("--through", default="parsing")
    status = subcommands.add_parser("status", help="Show persisted stage status")
    status.add_argument("project")
    rerun = subcommands.add_parser("rerun", help="Force one stage and its descendants")
    rerun.add_argument("project")
    rerun.add_argument("stage")
    rerun.add_argument("--through", default="parsing")
    cancel = subcommands.add_parser("cancel", help="Request cancellation")
    cancel.add_argument("project")
    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run("tkd_poomsae.api:app", host=args.host, port=args.port)
        return 0
    if args.command == "datasets" and args.dataset_command == "status":
        try:
            dataset_result = MendeleyDatasetProvider(version=args.version).status()
        except UnsupportedVersion as exc:
            parser.error(str(exc))
        payload = asdict(dataset_result)
        payload["root"] = str(dataset_result.root)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command is None:
        parser.print_help()
        return 0
    pipeline = Pipeline()
    try:
        if args.command == "register":
            sources = dict(item.split("=", 1) for item in args.source)
            if len(sources) != len(args.source):
                raise ValueError("source identifiers must be unique")
            pipeline.register(args.project, sources)
            result = pipeline.status(args.project)
        elif args.command == "status":
            result = pipeline.status(args.project)
        elif args.command == "cancel":
            pipeline.cancel(args.project)
            result = {"cancel_requested": True}
        elif args.command == "rerun":
            result = pipeline.analyze(args.project, args.through, rerun=args.stage)
        else:
            config = (
                json.loads(args.config.read_text(encoding="utf-8"))
                if (args.command == "analyze" and args.config is not None)
                else None
            )
            result = pipeline.analyze(args.project, args.through, config=config)
        print(json.dumps(result, sort_keys=True))
        return (
            1
            if any(
                stage["status"] in {"failed", "unavailable"}
                for stage in result.get("stages", {}).values()
            )
            else 0
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"tkd-poomsae: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
