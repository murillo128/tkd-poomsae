"""Positive reconstruction-to-parser acceptance; no real-video accuracy claims."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from contracts.models import Ground, Reconstruction, Semantics
from reconstruction.features import FeatureConfig
from storage import hash_file
from tests.fixtures.controlled_acceptance import PROJECT, build, parse, report
from tkd_poomsae.api import create_app


@pytest.mark.synthetic
def test_controlled_reconstruction_semantics_inspection_and_parser_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe, store, handles = build(tmp_path)
    semantics = handles["semantics"].metadata
    motion = handles["reconstruction"].metadata
    ground = handles["ground"].metadata
    assert isinstance(semantics, Semantics)
    assert isinstance(motion, Reconstruction) and isinstance(ground, Ground)
    assert semantics.execution and semantics.steps and semantics.actions
    assert semantics.provenance.producer == "reconstruction.semantics"
    assert semantics.reconstruction_id == motion.id and semantics.ground_id == ground.id
    arms = [a for a in semantics.actions if a.category == "arm"]
    kick = next(a for a in semantics.actions if a.category == "kick")
    assert any(
        a.interval.start < kick.interval.end and kick.interval.start < a.interval.end
        for a in arms
    )
    assert arms and semantics.phases and semantics.keyframes
    assert all(a.quality.state == "inferred" for a in arms)
    assert all(step.action_ids for step in semantics.steps)
    assert all(
        a.step_id in {s.id for s in semantics.steps} and a.motion_links
        for a in semantics.actions
    )
    assert len(motion.samples) == len(ground.samples) == 101
    assert all(s.root_xyz_world is not None for s in motion.samples)
    assert handles["raw"].metadata.provenance.producer == "reconstruction.triangulation"
    assert (
        handles["fitted"].metadata.provenance.producer == "reconstruction.articulated"
    )
    assert motion.provenance.producer == "reconstruction.temporal"
    assert any(s.right.state == "no_contact" for s in ground.samples)

    api = f"/api/projects/{PROJECT}/inspection"
    app = create_app(pipe, allowed_roots={"fixture": tmp_path})
    with TestClient(app, base_url="http://localhost") as client:
        meta = client.get(api).json()
        assert all(p["available"] for p in meta["products"].values())
        assert meta["products"]["semantics"]["origin"] == "automatic"
        for collection in ("steps", "actions", "keyframes"):
            response = client.get(
                api + "/semantics/window",
                params=dict(collection=collection, start=0, end=4, limit=256),
            )
            assert response.status_code == 200, response.text
            assert response.json()["rows"]
        action = arms[0]
        response = client.get(api + "/entities", params={"id": action.id})
        assert response.status_code == 200, response.text
        detail = response.json()
        assert detail["source_evidence"] and detail["physical_evidence"]
        time = action.interval.start
        for product in ("reconstruction", "ground"):
            response = client.get(
                api + f"/{product}/window",
                params={"start": time, "end": time, "limit": 256},
            )
            assert response.status_code == 200, response.text
            assert response.json()["rows"][0]["global_seconds"] == time
        response = client.get(api + "/ground/snapshot", params={"seconds": time})
        assert response.status_code == 200, response.text
        assert response.json()["snapshot"]["root"] is not None

    # Snapshot all physical payload bytes (not only metadata IDs). Parser-only
    # re-execution must never call an upstream producer or modify source media.
    physical = [
        h
        for name, h in handles.items()
        if name not in {"features", "segmentation", "arms", "legs", "semantics"}
    ]
    paths = [p for h in physical for p in h.path.rglob("*") if p.is_file()]
    paths += list(tmp_path.glob("*.mp4"))
    paths += list(
        store.root.namespace("derived").joinpath("observation").rglob("*.json")
    )
    paths += list(
        store.root.namespace("derived").joinpath("observation").rglob("*.npy")
    )
    before = {p: hash_file(p) for p in paths}

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("parser-only rerun invoked an upstream producer")

    for module, function in (
        ("sync.alignment", "publish_alignment"),
        ("reconstruction.triangulation.artifact", "publish_triangulation"),
        ("reconstruction.articulated.artifact", "publish_articulated_fit"),
        ("reconstruction.temporal.artifact", "publish_temporal_motion"),
        ("reconstruction.ground.artifact", "publish_contacts"),
        ("reconstruction.footprints.artifact", "publish_footprints"),
        ("reconstruction.pivots.artifact", "publish_pivots"),
        ("reconstruction.ground_view.artifact", "publish_ground_view"),
    ):
        monkeypatch.setattr(importlib.import_module(module), function, forbidden)
    created = store.created
    changed = parse(
        store,
        handles["reconstruction"],
        handles["ground"],
        FeatureConfig(min_excursion=0.03),
    )
    assert store.created == created + 5
    assert changed["semantics"].path != handles["semantics"].path
    updated = changed["semantics"].metadata
    assert isinstance(updated, Semantics) and updated.steps
    assert any(a.category == "arm" for a in updated.actions)
    created = store.created
    assert (
        parse(store, handles["reconstruction"], handles["ground"])["semantics"].path
        == handles["semantics"].path
    )
    assert store.created == created
    assert {p: hash_file(p) for p in paths} == before
    evidence = report(handles)
    assert evidence["steps"] > 0 and not evidence["manual_labels"]
