"""Keep core tests independent of external services and model downloads."""

import socket
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def disable_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail if an application test attempts an outbound connection."""

    def deny_connect(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Core tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", deny_connect)
    monkeypatch.setattr(socket, "create_connection", deny_connect)
    yield
