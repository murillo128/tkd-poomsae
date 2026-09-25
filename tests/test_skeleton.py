"""Offline smoke checks for the first application boundary."""

import subprocess
import sys

from fastapi.testclient import TestClient

from tkd_poomsae.api import app


def test_local_api_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from tkd_poomsae.cli import main; main()", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "serve" in result.stdout
