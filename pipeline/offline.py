"""Canonical wiring of existing offline publishers into the resumable runner."""

from __future__ import annotations

import fcntl
from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

from contracts.models import Calibration, Reconstruction, Synchronization
from media import ingest
from pipeline.runner import (
    DEPENDENCIES,
    STAGE_LAYERS,
    STAGE_ORDER,
    CapabilityUnavailable,
    Pipeline,
    PublishedOutput,
    RunCancelled,
    Stage,
    StageOutput,
    _read_json,
    _scene_calibration,
    _write_json,
    key_data,
)
from storage import ArtifactHandle, ArtifactKey, ArtifactStore, hash_config, hash_file

REVISION = "offline-publishers-v1"


def stage_identities() -> dict[str, tuple[str, str | None]]:
    """Invalidate journals when any constituent publisher/model revision changes."""
    from pose.observation_run import REVISION as observation_revision
    from pose.observation_run import _model_revision

    modules = {
        "ingest": (),
        "sync": (),
        "calibration": (),
        "observations": (),
        "attachment": ("sync.alignment",),
        "reconstruction": (
            "reconstruction.triangulation.artifact",
            "reconstruction.articulated.core",
            "reconstruction.temporal.core",
        ),
        "ground": tuple(
            "reconstruction." + name + ".core"
            for name in ("ground", "footprints", "pivots", "ground_view")
        ),
        "parsing": tuple(
            "reconstruction." + name + ".core"
            for name in ("features", "segmentation", "arms", "lower_body", "semantics")
        ),
    }
    return {
        name: (
            hash_config(
                {
                    "wiring": REVISION,
                    "stage": name,
                    "publishers": [
                        getattr(import_module(m), "REVISION") for m in paths
                    ],
                    "native": observation_revision if name == "observations" else None,
                }
            ),
            _model_revision() if name == "observations" else None,
        )
        for name, paths in modules.items()
    }


class PublisherStore(ArtifactStore):
    """Retain keys returned by existing publishers for verified journal reuse."""

    def __init__(self, store: ArtifactStore) -> None:
        super().__init__(store.root)
        self.keys: dict[str, ArtifactKey] = {}

    def get(self, key: ArtifactKey) -> ArtifactHandle:
        handle = super().get(key)
        self.keys[str(handle.path)] = key
        return handle

    def key(self, handle: ArtifactHandle) -> ArtifactKey:
        return self.keys[str(handle.path)]


def configured_pipeline(project: str, store: ArtifactStore | None = None) -> Pipeline:
    """Wire provisioned resources; never bootstrap or synthesize capabilities."""
    store = store or ArtifactStore()
    publisher = PublisherStore(store)
    pipe = Pipeline(publisher)
    pipe.profile = REVISION
    sources = _read_json(pipe._files(project)[0])["sources"]

    def cancelled() -> bool:
        if pipe._cancelled(project):
            raise RunCancelled("analysis cancelled; resume to continue")
        return False

    def result(handle: ArtifactHandle) -> PublishedOutput:
        return PublishedOutput(publisher.key(handle), tuple(publisher.keys.values()))

    def ingest_stage(
        key: ArtifactKey,
        _inputs: Mapping[str, ArtifactHandle],
        settings: Mapping[str, Any],
    ) -> StageOutput:
        if settings:
            raise ValueError("ingest takes no settings; register sources first")
        manifest = ingest(list(sources.items()))
        if any(key.inputs[r.camera_id] != r.sha256 for r in manifest):
            raise ValueError("source changed during ingestion")
        source = manifest[0].source.model_copy(deep=True)
        source.provenance.config_digest = key.config_digest
        return StageOutput(source)

    def sync_stage(
        key: ArtifactKey,
        _inputs: Mapping[str, ArtifactHandle],
        settings: Mapping[str, Any],
    ) -> StageOutput:
        from sync import extract_cues, sources_from_manifest
        from sync.solver import solve_offsets

        if set(settings) - {"reference", "manual_offsets"}:
            raise ValueError("sync accepts reference and attributed manual_offsets")
        manifest = ingest(list(sources.items()))
        cues = {}
        for recording in manifest:
            cancelled()
            cues[recording.source_id] = extract_cues(
                recording, cache_dir=store.root.namespace("runs") / "cues"
            )
        by_camera = {r.camera_id: r.source_id for r in manifest}
        sync = solve_offsets(
            sources_from_manifest(manifest, cues),
            reference=by_camera[settings["reference"]]
            if "reference" in settings
            else None,
            overrides={
                by_camera[c]: r for c, r in settings.get("manual_offsets", {}).items()
            },
            artifact_id=key.digest,
            config_digest=key.config_digest,
        )
        return StageOutput(sync, diagnostics=tuple(sync.diagnostics))

    def calibration_stage(
        key: ArtifactKey,
        inputs: Mapping[str, ArtifactHandle],
        settings: Mapping[str, Any],
    ) -> StageOutput:
        if "artifact" not in settings:
            if set(settings) - {"candidate", "evidence", "thresholds"}:
                raise ValueError("unsupported calibration settings")
            if "candidate" not in settings:
                raise CapabilityUnavailable(
                    "Supply calibration.artifact (export with `python -m calibration "
                    "--key-output`) or candidate/evidence; see docs/local-runbook.md"
                )
            output = _scene_calibration(key, inputs, settings)
            cal = output.artifact.model_copy(deep=True)
        else:
            if set(settings) != {"artifact"}:
                raise ValueError(
                    "calibration artifact cannot be combined with scene settings"
                )
            path = Path(settings["artifact"])
            if not path.is_file():
                raise CapabilityUnavailable(
                    f"Missing calibration key file: {path}; see calibration/README.md"
                )
            handle = publisher.get(ArtifactKey(**_read_json(path)))
            cal = handle.metadata.model_copy(deep=True)
        if not isinstance(cal, Calibration):
            raise ValueError("calibration artifact required")
        hashes = {n: hash_file(Path(p)) for n, p in sources.items()}
        if not {c.camera_id for c in cal.cameras} <= set(hashes) or any(
            c.source_id != "source:" + hashes[c.camera_id] for c in cal.cameras
        ):
            raise ValueError(
                "calibration cameras must bind the registered source bytes"
            )
        sync = inputs["sync"].metadata
        assert isinstance(sync, Synchronization)
        links = [
            link.removeprefix("synchronization:")
            for link in cal.evidence_links
            if link.startswith("synchronization:")
        ]
        if links and links != [sync.id]:
            raise ValueError("calibration uses a different synchronization revision")
        if cal.quality.state == "unknown" or len(cal.cameras) < 2:
            raise CapabilityUnavailable(
                "calibration geometry is unavailable; "
                "supply accepted independent captures/evidence"
            )
        cal.provenance.config_digest = key.config_digest
        return StageOutput(cal, diagnostics=tuple(cal.quality_flags))

    def observation_stage(
        _key: ArtifactKey,
        _inputs: Mapping[str, ArtifactHandle],
        settings: Mapping[str, Any],
    ) -> PublishedOutput:
        from pose.observation_run import ObservationSettings, run_windows
        from tkd_poomsae.selections import Window
        from tkd_poomsae.vision.assets import ModelAssetError

        if set(settings) - {"device", "max_frames", "start_seconds", "end_seconds"}:
            raise ValueError(
                "observations accepts device, max_frames, start_seconds, end_seconds"
            )
        windows = []
        for recording in ingest(list(sources.items())):
            start = float(settings.get("start_seconds", recording.frames[0].seconds))
            end = float(settings.get("end_seconds", recording.frames[-1].seconds))
            if (
                not recording.frames[0].seconds
                <= start
                < end
                <= recording.frames[-1].seconds
            ):
                raise ValueError("observation window is outside registered source PTS")
            windows.append(
                Window(
                    project,
                    recording.camera_id,
                    recording.sha256,
                    recording.path,
                    start,
                    end,
                )
            )

        # Initialize models only for an uncached native window. Complete windows
        # and parser-only reruns remain usable in the core interpreter.
        def adapter_factory(**kwargs: Any) -> Any:
            from pose.providers.mmpose.adapter import MMPoseAdapter

            try:
                return MMPoseAdapter(**kwargs)
            except (ImportError, RuntimeError) as exc:
                raise CapabilityUnavailable(
                    f"Vision unavailable: {exc}; run `tkd-poomsae models bootstrap` "
                    "and `tkd-poomsae models runtime-bootstrap` explicitly; run again"
                ) from exc

        try:
            receipt = run_windows(
                project,
                tuple(windows),
                store=publisher,
                settings=ObservationSettings(
                    device=settings.get("device", "cpu"),
                    max_frames=settings.get("max_frames", 32),
                ),
                cancelled=cancelled,
                adapter_factory=adapter_factory,
            )
        except (ModelAssetError, ImportError) as exc:
            raise CapabilityUnavailable(
                f"Vision unavailable: {exc}; explicitly run "
                "`tkd-poomsae models bootstrap` and "
                "`tkd-poomsae models runtime-bootstrap`, then run again"
            ) from exc
        keys = [ArtifactKey(**entry["key"]) for entry in receipt["windows"]]
        return PublishedOutput(keys[0], tuple(keys))

    def observation_keys() -> list[ArtifactKey]:
        state = _read_json(pipe._files(project)[1])
        return [ArtifactKey(**k) for k in state["stages"]["observations"]["lineage"]]

    def attachment_stage(
        _key: ArtifactKey,
        inputs: Mapping[str, ArtifactHandle],
        settings: Mapping[str, Any],
    ) -> PublishedOutput:
        from pose.observation_run import load_window
        from sync.alignment import JoinConfig, publish_alignment

        if set(settings) - {"join"}:
            raise ValueError(
                "attachment accepts join settings; grid uses reference native PTS"
            )
        sync = inputs["sync"].metadata
        assert isinstance(sync, Synchronization)
        reference = next(
            o for o in sync.offsets if o.source_id == sync.reference_source_id
        )
        windows = [publisher.get(k) for k in observation_keys()]
        times = sorted(
            {
                o.frame.source_seconds + reference.effective_seconds
                for w in windows
                for o in load_window(w)
                if o.frame.source_id == reference.source_id
            }
        )
        return result(
            publish_alignment(
                publisher,
                windows,
                inputs["sync"],
                times,
                JoinConfig.model_validate(settings.get("join", {})),
            )
        )

    def reconstruction_stage(
        _key: ArtifactKey,
        inputs: Mapping[str, ArtifactHandle],
        settings: Mapping[str, Any],
    ) -> PublishedOutput:
        from reconstruction.articulated import FitConfig
        from reconstruction.temporal import TemporalConfig, publish_temporal_motion
        from reconstruction.triangulation import (
            TriangulationConfig,
            publish_triangulation,
        )

        if set(settings) - {"participant_id", "triangulation", "fit", "temporal"}:
            raise ValueError("unsupported reconstruction settings")
        participant = settings.get("participant_id")
        if not isinstance(participant, str) or not participant:
            raise ValueError("reconstruction requires explicit participant_id")
        raw = publish_triangulation(
            publisher,
            inputs["calibration"],
            inputs["attachment"],
            participant,
            TriangulationConfig.model_validate(settings.get("triangulation", {})),
        )
        if not isinstance(raw.metadata, Reconstruction) or not any(
            point.xyz_world is not None and point.quality.state != "unknown"
            for sample in raw.metadata.samples
            for point in sample.landmarks
        ):
            raise CapabilityUnavailable(
                "no supported triangulated motion; inspect calibration, "
                "synchronization and observations"
            )
        cancelled()
        return result(
            publish_temporal_motion(
                publisher,
                raw,
                TemporalConfig.model_validate(settings.get("temporal", {})),
                FitConfig.model_validate(settings.get("fit", {})),
            ).motion
        )

    def stage_handle(name: str) -> ArtifactHandle:
        record = _read_json(pipe._files(project)[1])["stages"][name]
        if "artifact_key" in record:
            return publisher.get(ArtifactKey(**record["artifact_key"]))
        state = _read_json(pipe._files(project)[1])
        key = pipe._expected_keys(_read_json(pipe._files(project)[0]), state)[name]
        return publisher.get(key)

    def ground_stage(
        _key: ArtifactKey,
        inputs: Mapping[str, ArtifactHandle],
        settings: Mapping[str, Any],
    ) -> PublishedOutput:
        from reconstruction.footprints import FootprintConfig, publish_footprints
        from reconstruction.ground import ContactConfig, publish_contacts
        from reconstruction.ground_view import GroundViewConfig, publish_ground_view
        from reconstruction.pivots import PivotConfig, publish_pivots

        if set(settings) - {"contact", "footprints", "pivots", "view"}:
            raise ValueError("unsupported ground settings")
        cal = stage_handle("calibration")
        assert isinstance(cal.metadata, Calibration)
        if cal.metadata.ground_status != "resolved":
            raise CapabilityUnavailable(
                "ground is unresolved; supply independent ground/scale evidence"
            )
        contact_config = ContactConfig.model_validate(settings.get("contact", {}))
        if contact_config.units == "metric" and cal.metadata.scale_status != "resolved":
            raise CapabilityUnavailable(
                "metric ground requires independently resolved scale"
            )
        motion = inputs["reconstruction"]
        reconstruction_record = _read_json(pipe._files(project)[1])["stages"][
            "reconstruction"
        ]
        morphology_key = next(
            ArtifactKey(**k)
            for k in reconstruction_record["lineage"]
            if k["layer"] == "morphology"
        )
        morphology = publisher.get(morphology_key)
        contacts = publish_contacts(
            publisher,
            motion,
            cal,
            morphology=morphology,
            config=contact_config,
        )
        feet = publish_footprints(
            publisher,
            motion,
            contacts,
            cal,
            morphology=morphology,
            config=FootprintConfig.model_validate(settings.get("footprints", {})),
        )
        pivots = publish_pivots(
            publisher, feet, PivotConfig.model_validate(settings.get("pivots", {}))
        )
        return result(
            publish_ground_view(
                publisher,
                motion,
                cal,
                pivots,
                GroundViewConfig.model_validate(settings.get("view", {})),
            )
        )

    def parsing_stage(
        _key: ArtifactKey,
        inputs: Mapping[str, ArtifactHandle],
        settings: Mapping[str, Any],
    ) -> PublishedOutput:
        from reconstruction.arms import ArmConfig, publish_arm_actions
        from reconstruction.features import FeatureConfig, publish_features
        from reconstruction.lower_body import LowerBodyConfig, publish_lower_body
        from reconstruction.segmentation import SegmentationConfig, publish_segmentation
        from reconstruction.semantics import AssemblyConfig, publish_semantics

        if set(settings) - {
            "features",
            "segmentation",
            "arms",
            "lower_body",
            "semantics",
        }:
            raise ValueError("unsupported parsing settings")
        motion, ground = stage_handle("reconstruction"), inputs["ground"]
        features = publish_features(
            publisher,
            motion,
            ground,
            FeatureConfig.model_validate(settings.get("features", {})),
        )
        segments = publish_segmentation(
            publisher,
            features,
            SegmentationConfig.model_validate(settings.get("segmentation", {})),
        )
        arms = publish_arm_actions(
            publisher,
            features,
            segments,
            motion,
            ArmConfig.model_validate(settings.get("arms", {})),
        )
        legs = publish_lower_body(
            publisher,
            features,
            ground,
            segments,
            LowerBodyConfig.model_validate(settings.get("lower_body", {})),
        )
        return result(
            publish_semantics(
                publisher,
                features,
                segments,
                arms,
                legs,
                AssemblyConfig.model_validate(settings.get("semantics", {})),
            )
        )

    producers: dict[str, Callable[..., StageOutput | PublishedOutput]] = dict(
        ingest=ingest_stage,
        sync=sync_stage,
        calibration=calibration_stage,
        observations=observation_stage,
        attachment=attachment_stage,
        reconstruction=reconstruction_stage,
        ground=ground_stage,
        parsing=parsing_stage,
    )
    identities = stage_identities()
    pipe.stages = {
        name: Stage(
            name,
            producers[name],
            STAGE_LAYERS[name],
            DEPENDENCIES[name],
            revision=identities[name][0],
            model_revision=identities[name][1],
            publishes=name not in {"ingest", "sync", "calibration"},
        )
        for name in STAGE_ORDER
    }
    return pipe


def run_project(
    project: str,
    *,
    config: Mapping[str, Mapping[str, Any]] | None = None,
    through: str = "parsing",
    rerun: str | None = None,
) -> dict[str, Any]:
    """Analyze and persist the verified inspection inventory in one invocation."""
    from tkd_poomsae.inspection import Inspection

    pipe = configured_pipeline(project)
    project_path = pipe._files(project)[0]
    with (project_path.parent / "run.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = _read_json(project_path)
        data["pipeline_profile"] = REVISION
        _write_json(project_path, data)
    state = pipe.analyze(project, through, config=config, rerun=rerun)
    keys = pipe._expected_keys(_read_json(pipe._files(project)[0]), state)
    products = {}
    for stage, product in (
        ("sync", "sync"),
        ("calibration", "calibration"),
        ("reconstruction", "reconstruction"),
        ("ground", "ground"),
        ("parsing", "semantics"),
    ):
        record = state["stages"][stage]
        if record["status"] == "complete" and record["key"] == keys[stage].digest:
            products[product] = (
                ArtifactKey(**record["artifact_key"])
                if "artifact_key" in record
                else keys[stage]
            )
    obs = state["stages"]["observations"]
    observations = (
        [ArtifactKey(**k) for k in obs.get("lineage", [])]
        if obs["status"] == "complete" and obs["key"] == keys["observations"].digest
        else []
    )
    lineage = {
        k.digest: k
        for record in state["stages"].values()
        if record["status"] == "complete"
        for k in (ArtifactKey(**item) for item in record.get("lineage", []))
    }
    if "sync" in products:
        Inspection(pipe).register(
            project, products, observations=observations, lineage=lineage.values()
        )
        bundle_path = pipe._directory(project) / "inspection-bundle.json"
        _write_json(
            bundle_path,
            {
                "products": {name: key_data(key) for name, key in products.items()},
                "observations": [key_data(key) for key in observations],
                "lineage": [key_data(key) for key in lineage.values()],
            },
        )
        state["inspection_bundle"] = str(bundle_path)
    state["inspection_registered"] = "sync" in products
    return state
