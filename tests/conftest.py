"""Keep core tests independent of external services and model downloads."""

import socket
from collections.abc import Iterator
from typing import Any

import pytest

from tkd_poomsae.selections import resolve

SUITES = ("synthetic", "local_data", "full_data", "model_required")
EXECUTED: dict[str, set[str]] = {name: set() for name in SUITES}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if not any(item.get_closest_marker(name) for name in SUITES):
            item.add_marker(pytest.mark.synthetic)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call":
        for name in SUITES:
            if name in report.keywords:
                EXECUTED[name].add(report.nodeid)


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
