"""Command line entry point."""

import argparse
import json
from dataclasses import asdict

import uvicorn

from tkd_poomsae.dataset import MendeleyDatasetProvider, UnsupportedVersion


def main() -> None:
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
    status = dataset_commands.add_parser(
        "status", help="Check the pinned local dataset"
    )
    status.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run("tkd_poomsae.api:app", host=args.host, port=args.port)
    elif args.command == "datasets" and args.dataset_command == "status":
        try:
            result = MendeleyDatasetProvider(version=args.version).status()
        except UnsupportedVersion as exc:
            parser.error(str(exc))
        payload = asdict(result)
        payload["root"] = str(result.root)
        print(json.dumps(payload, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
