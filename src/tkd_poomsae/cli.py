"""Command line entry point."""

import argparse
import json
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from pipeline import Pipeline
from storage.store import StorageError
from tkd_poomsae.dataset import MendeleyDatasetProvider, UnsupportedVersion
from tkd_poomsae.dataset_bootstrap import AcquisitionError, bootstrap, verify
from tkd_poomsae.dataset_import import inventory, register_all
from tkd_poomsae.media_variants import VariantRecipe, materialize
from tkd_poomsae.selections import catalog, resolve


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
    dataset_verify = dataset_commands.add_parser(
        "verify", help="Hash every file against the local acquisition receipt"
    )
    dataset_verify.add_argument("--version", type=int, default=1)
    dataset_bootstrap = dataset_commands.add_parser(
        "bootstrap", help="Acquire a pinned dataset onto shared storage"
    )
    dataset_bootstrap.add_argument("dataset", choices=["mendeley-bjy7vr4xkt-v1"])
    dataset_bootstrap.add_argument(
        "--repair", action="store_true", help="Replace a corrupt local publication"
    )
    dataset_inventory = dataset_commands.add_parser(
        "inventory", help="Measure all local Mendeley executions without writing"
    )
    dataset_inventory.add_argument(
        "--output", type=Path, help="Write a compact JSON report"
    )
    dataset_commands.add_parser(
        "register", help="Register all local Mendeley projects and shared sources"
    )
    dataset_selections = dataset_commands.add_parser(
        "selections", help="List or resolve pinned virtual media windows"
    )
    dataset_selections.add_argument("name", nargs="?")
    dataset_variant = dataset_commands.add_parser(
        "variant", help="Generate or reuse a shared derived media variant"
    )
    dataset_variant.add_argument("selection")
    dataset_variant.add_argument("execution")
    dataset_variant.add_argument("camera")
    dataset_variant.add_argument("--recipe", type=Path, required=True)
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
    observations = subcommands.add_parser(
        "observations", help="Infer or reuse a local native-time selection"
    )
    observations.add_argument(
        "selection", choices=["smoke-short", "demo-full", "all-forms"]
    )
    observations.add_argument("--device", default="cpu")
    observations.add_argument("--max-frames", type=int, default=32)
    rerun.add_argument("--through", default="parsing")
    cancel = subcommands.add_parser("cancel", help="Request cancellation")
    cancel.add_argument("project")
    offset = subcommands.add_parser("sync-offset", help="Revise a camera offset")
    offset.add_argument("project")
    offset.add_argument("source_id")
    offset.add_argument("offset_seconds", type=float)
    offset.add_argument("--author", required=True)
    offset.add_argument("--source", required=True)
    offset.add_argument("--reason", required=True)
    solve = subcommands.add_parser("sync-solve", help="Publish camera alignment")
    solve.add_argument("project")
    subcommands.add_parser("doctor", help="Report model assets and device capability")
    models = subcommands.add_parser(
        "models", help="Provision or exercise pinned models"
    )
    model_commands = models.add_subparsers(dest="model_command", required=True)
    model_commands.add_parser(
        "bootstrap", help="Download and verify pinned model assets"
    )
    model_commands.add_parser(
        "runtime-bootstrap", help="Provision the shared locked vision interpreter"
    )
    for name in ("smoke", "infer"):
        command = model_commands.add_parser(name, help=f"Run local {name} inference")
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--device", default="cpu")
    from tkd_poomsae.semantic_edits.cli import configure as configure_edits

    configure_edits(subcommands)
    args = parser.parse_args()
    if args.command == "semantic-edits":
        from tkd_poomsae.semantic_edits.cli import run as run_edits

        try:
            return run_edits(args)
        except StorageError as exc:
            print(f"tkd-poomsae: {exc}", file=sys.stderr)
            return 1
    if args.command in {"observations", "doctor"} or (
        args.command == "models" and args.model_command in {"smoke", "infer"}
    ):
        from tkd_poomsae.vision.runtime import delegate, runtime_python

        try:
            if args.command != "doctor" or runtime_python().is_file():
                delegated = delegate(sys.argv[1:])
                if delegated is not None:
                    return delegated
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"tkd-poomsae: {exc}", file=sys.stderr)
            return 1
    if args.command == "observations":
        try:
            result = Pipeline().observe_selection(
                args.selection, device=args.device, max_frames=args.max_frames
            )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            print(f"tkd-poomsae: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "serve":
        if args.host not in {"localhost", "127.0.0.1", "::1"}:
            parser.error("the local API must bind to a loopback address")
        import uvicorn

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
    if args.command == "datasets" and args.dataset_command in {"inventory", "register"}:
        try:
            result = (
                inventory() if args.dataset_command == "inventory" else register_all()
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"tkd-poomsae: {exc}", file=sys.stderr)
            return 1
        output = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.dataset_command == "inventory" and args.output is not None:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        return 0
    if args.command == "datasets" and args.dataset_command in {"selections", "variant"}:
        try:
            if args.dataset_command == "selections":
                result = (
                    {"selections": catalog()}
                    if args.name is None
                    else {
                        "name": args.name,
                        "windows": [
                            {**asdict(window), "source_path": str(window.source_path)}
                            for window in resolve(args.name)
                        ],
                    }
                )
            else:
                windows = resolve(args.selection)
                matches = [
                    window
                    for window in windows
                    if window.execution_id == args.execution
                    and window.camera_id == args.camera
                ]
                if len(matches) != 1:
                    raise ValueError("selection has no unique execution/camera window")
                recipe = VariantRecipe(**json.loads(args.recipe.read_text()))
                variant = materialize(matches[0], recipe)
                result = {
                    "path": str(variant.path),
                    "cache_hit": variant.cache_hit,
                    "manifest": variant.manifest,
                }
        except (OSError, ValueError, RuntimeError, TypeError, StorageError) as exc:
            print(f"tkd-poomsae: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "datasets" and args.dataset_command in {"bootstrap", "verify"}:
        try:
            if args.dataset_command == "verify":
                if args.version != 1:
                    raise UnsupportedVersion("only Mendeley version 1 is registered")
                result = verify()
            else:
                result = bootstrap(repair=args.repair)
        except (AcquisitionError, UnsupportedVersion, OSError) as exc:
            print(f"tkd-poomsae: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "doctor":
        from tkd_poomsae.vision.assets import (
            ModelAssetError,
            models_root,
            verified_paths,
        )

        report: dict[str, object] = {
            "models_root": str(models_root()),
            "interpreter": sys.executable,
        }
        try:
            report["assets"] = f"verified ({len(verified_paths())})"
        except ModelAssetError as error:
            report["assets"] = str(error)
        try:
            import mmcv  # type: ignore[import-not-found]
            import mmdet  # type: ignore[import-not-found]
            import mmengine  # type: ignore[import-not-found]
            import mmpose  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]

            report["runtime"] = {
                "torch": torch.__version__,
                "mmcv": mmcv.__version__,
                "mmengine": mmengine.__version__,
                "mmdet": mmdet.__version__,
                "mmpose": mmpose.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cpu_threads": torch.get_num_threads(),
            }
        except ImportError as error:
            report["runtime"] = f"Optional vision environment unavailable: {error}"
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "models":
        from tkd_poomsae.vision.assets import ModelAssetError
        from tkd_poomsae.vision.assets import bootstrap as bootstrap_models

        try:
            if args.model_command == "runtime-bootstrap":
                from tkd_poomsae.vision.runtime import bootstrap_runtime

                print(json.dumps(bootstrap_runtime(), sort_keys=True))
            elif args.model_command == "bootstrap":
                installed = bootstrap_models()
                print(json.dumps({"installed": installed, "cache_hit": not installed}))
            else:
                from tkd_poomsae.vision.inference import infer_image

                interrupted = False

                def cancel_inference(_signal: int, _frame: object) -> None:
                    nonlocal interrupted
                    interrupted = True

                signal.signal(signal.SIGINT, cancel_inference)
                signal.signal(signal.SIGTERM, cancel_inference)
                print(
                    json.dumps(
                        infer_image(
                            args.input,
                            device=args.device,
                            smoke=args.model_command == "smoke",
                            cancelled=lambda: interrupted,
                        )
                    )
                )
            return 0
        except (
            ModelAssetError,
            RuntimeError,
            ValueError,
            OSError,
            subprocess.CalledProcessError,
        ) as error:
            print(f"tkd-poomsae: {error}", file=sys.stderr)
            return 1
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
        elif args.command == "sync-offset":
            result = pipeline.revise_sync_offset(
                args.project,
                args.source_id,
                args.offset_seconds,
                author=args.author,
                source=args.source,
                reason=args.reason,
            )
        elif args.command == "sync-solve":
            handle = pipeline.solve_sync(args.project)
            result = handle.metadata.model_dump(mode="json")
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
