"""Native ground-view transport and disk indexing on synthetic physical artifacts."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from reconstruction.footprints import publish_footprints
from reconstruction.ground import publish_contacts
from reconstruction.ground_view import publish_ground_view
from reconstruction.pivots import publish_pivots
from storage import ArtifactKey, hash_file
from tests.test_ground_contact import calibration, persist
from tests.test_inspection_api import BASE, client
from tests.test_inspection_api import inspection as base_inspection
from tests.test_pivots import sequence
from tkd_poomsae.inspection import Inspection


@pytest.fixture
def registered_inspection(tmp_path: Path) -> tuple[Inspection, dict[str, ArtifactKey]]:
    factory = cast(
        Callable[[Path], tuple[Inspection, dict[str, ArtifactKey]]],
        getattr(base_inspection, "__wrapped__"),
    )
    return factory(tmp_path)


def indexed_view(index: Inspection) -> None:
    store = index.pipe.store
    source, _ = sequence(sign=-1)
    motion, cal = persist(store, source), persist(store, calibration())
    contacts = publish_contacts(store, motion, cal)
    placements = publish_footprints(store, motion, contacts, cal)
    pivots = publish_pivots(store, placements)
    product = publish_ground_view(store, motion, cal, pivots)
    key = ArtifactKey(
        layer="ground",
        inputs={
            "motion": hash_file(motion.path / "manifest.json"),
            "calibration": hash_file(cal.path / "manifest.json"),
            "pivots": hash_file(pivots.path / "manifest.json"),
        },
        schema_version="1.0.0",
        algorithm_revision="ground-view-v2",
        config_digest=product.metadata.provenance.config_digest,
    )
    # Exercise trusted registration's index boundary with a real persisted product;
    # the shared fixture independently establishes source/clock registration policy.
    with index.connection("demo") as db:
        db.execute("DELETE FROM products WHERE name='ground'")
        db.execute("DELETE FROM entities WHERE product='ground'")
        index._index(db, "ground", product, key, index.clock("demo"))


def test_indexed_ground_snapshot_pages_revision_and_units(
    registered_inspection: tuple[Inspection, dict[str, ArtifactKey]],
) -> None:
    index, _ = registered_inspection
    indexed_view(index)
    with client(index) as http:
        response = http.get(BASE + "/ground/snapshot?seconds=0.4")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["available"]
        assert data["ground_view"]["world_unit"] == "m"
        assert data["snapshot"]["sampled_seconds"] == 0.4
        assert data["snapshot"]["pivot_ids"]
        page = http.get(
            BASE + "/ground/window",
            params={
                "collection": "ground_frames",
                "start": 0,
                "end": 1,
                "limit": 3,
                "expected_revision": data["revision"],
            },
        ).json()
        assert len(page["rows"]) == 3 and page["next_cursor"] is not None
        native = http.get(
            BASE + "/ground/window",
            params={
                "collection": "ground_frames",
                "start": 0.4,
                "end": 0.4,
            },
        ).json()["rows"][0]
        assert native == data["snapshot"]
        preceding = http.get(BASE + "/ground/snapshot?seconds=0.405").json()
        assert preceding["snapshot"]["status"] == "native_snapshot"
        assert preceding["snapshot"]["sampled_seconds"] == 0.4
        assert preceding["snapshot"]["feet"] == native["feet"]
        assert preceding["snapshot"]["bracket_seconds"] == [0.4, 0.42]
        assert (
            http.get(BASE + "/ground/snapshot?seconds=2").json()["reason"]
            == "outside_execution"
        )
        assert http.get(BASE + "/ground/snapshot?seconds=NaN").status_code == 422
        assert (
            http.get(
                BASE + "/ground/snapshot?seconds=.4&expected_revision=old"
            ).status_code
            == 409
        )
        for collection in ("placements", "rotations", "placement_relations"):
            rows = http.get(
                BASE + "/ground/window",
                params={
                    "collection": collection,
                    "start": 0,
                    "end": 1,
                },
            ).json()["rows"]
            assert rows
        # Snapshot is a disk-index read, not a repeated dense artifact load.
        index.pipe.store.get = lambda *args, **kwargs: pytest.fail("dense read")  # type: ignore[method-assign]
        assert http.get(BASE + "/ground/snapshot?seconds=.4").json()["available"]


def test_ground_without_view_and_stale_clock_are_unavailable(
    registered_inspection: tuple[Inspection, dict[str, ArtifactKey]],
) -> None:
    index, _ = registered_inspection
    with client(index) as http:
        assert (
            http.get(BASE + "/ground/snapshot?seconds=0").json()["reason"]
            == "ground-view product unavailable"
        )
    indexed_view(index)
    with index.connection("demo") as db:
        # Exercise the query policy across a large native gap.
        db.execute(
            "DELETE FROM entities WHERE product='ground' "
            "AND collection='ground_frames' AND start>.4 AND start<.7"
        )
    with client(index) as http:
        assert (
            http.get(BASE + "/ground/snapshot?seconds=.5").json()["reason"]
            == "native_time_gap"
        )
    index.pipe.revise_sync_offset(
        "demo",
        "left",
        3,
        author="test",
        source="test",
        reason="test",
        expected_revision=0,
    )
    with client(index) as http:
        data = http.get(BASE + "/ground/snapshot?seconds=.4").json()
        assert not data["available"] and data["snapshot"] is None
