"""Command line entry point."""

import argparse

import uvicorn


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
    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run("tkd_poomsae.api:app", host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
