"""Keep core tests independent of external services and model downloads."""

import json
import socket
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from storage import hash_config
from tkd_poomsae.selections import resolve

SUITES = ("synthetic", "local_data", "full_data", "model_required")
EXECUTED: dict[str, set[str]] = {name: set() for name in SUITES}
SYNC_RESULTS: dict[str, dict[str, Any]] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--sync-results",
        default=".pytest_cache/sync-regression-results.json",
        help="Path for compact synchronization regression results",
    )


def _sync_item(item: pytest.Item, status: str) -> None:
    if item.path.name != "test_sync_integration.py":
        return
    SYNC_RESULTS.setdefault(
        item.nodeid,
        {
            "test": item.nodeid,
            "input_case": item.nodeid.split("::")[-1],
            "sync_config_digest": hash_config(asdict(getattr(item, "module").CONFIG)),
            "suite": "local_data"
            if item.get_closest_marker("local_data")
            else "synthetic",
            "status": status,
        },
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        _sync_item(item, "collected")
        if not any(item.get_closest_marker(name) for name in SUITES):
            item.add_marker(pytest.mark.synthetic)


def pytest_deselected(items: list[pytest.Item]) -> None:
    for item in items:
        _sync_item(item, "skipped")
        if item.nodeid in SYNC_RESULTS:
            SYNC_RESULTS[item.nodeid]["status"] = "skipped"
            SYNC_RESULTS[item.nodeid]["reason"] = "marker filter"


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call":
        for name in SUITES:
            if name in report.keywords:
                EXECUTED[name].add(report.nodeid)
    if report.nodeid in SYNC_RESULTS and (
        report.when == "call" or report.failed or report.skipped
    ):
        row = SYNC_RESULTS[report.nodeid]
        row["status"] = (
            "skipped" if report.skipped else "failed" if report.failed else "passed"
        )
        row["duration_seconds"] = round(report.duration, 3)
        row.update(dict(report.user_properties))
        if report.skipped:
            row["reason"] = (
                str(report.longrepr[-1])
                if isinstance(report.longrepr, tuple)
                else "skipped"
            )


def pytest_sessionfinish(session: pytest.Session) -> None:
    if not SYNC_RESULTS:
        return
    path = Path(session.config.getoption("--sync-results"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "results": list(SYNC_RESULTS.values())},
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def pytest_terminal_summary(terminalreporter: Any) -> None:
    terminalreporter.write_sep("-", "executed test suites (test calls only)")
    for name in SUITES:
        terminalreporter.write_line(f"{name}: {len(EXECUTED[name])} ran")


@pytest.fixture
def local_data_windows() -> tuple[Any, ...]:
    try:
        return resolve("smoke-short")
    except (ValueError, OSError) as exc:
        pytest.fail(f"local data unavailable: {exc}")


@pytest.fixture(autouse=True)
def disable_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail if an application test attempts an outbound connection."""

    def deny_connect(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Core tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", deny_connect)
    monkeypatch.setattr(socket, "create_connection", deny_connect)
    yield
