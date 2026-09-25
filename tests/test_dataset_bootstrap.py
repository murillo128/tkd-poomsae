"""Exercise the acquisition protocol against a local publisher stand-in."""

from __future__ import annotations

import hashlib
import io
import json
import socket
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from storage import StorageRoot
from tkd_poomsae import dataset_bootstrap as acquisition

_SOCKET_CONNECT = socket.socket.connect
_CREATE_CONNECTION = socket.create_connection


def _fixture(
    extra: str | None = None, *, symlink: bool = False
) -> tuple[bytes, dict[str, Any]]:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Data/video/test.mp4", b"video-bytes" * 40)
        archive.writestr("Data/annotations/test.csv", b"csv-bytes" * 20)
        if extra:
            if symlink:
                link = zipfile.ZipInfo(extra)
                link.create_system = 3
                link.external_attr = 0o120777 << 16
                archive.writestr(link, b"../../escape")
            else:
                archive.writestr(extra, b"unexpected")
    content = output.getvalue()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = [
            {
                "path": item.filename,
                "size_bytes": item.file_size,
                "compressed_size_bytes": item.compress_size,
                "crc32": f"{item.CRC:08x}",
            }
            for item in archive.infolist()
            if item.filename != extra
        ]
    record = {
        "source": {"dataset_id": "bjy7vr4xkt", "version": 1},
        "inventory": {
            "archive": {
                "file_id": "test-id",
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            },
            "members": members,
        },
    }
    return content, record


class _Publisher(ThreadingHTTPServer):
    data: bytes
    etag: str
    requests: list[tuple[str, str | None]]
    interrupt: bool


class _Handler(BaseHTTPRequestHandler):
    server: _Publisher

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_HEAD(self) -> None:
        self.server.requests.append(("HEAD", None))
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.server.data)))
        self.send_header("ETag", self.server.etag)
        self.end_headers()

    def do_GET(self) -> None:
        range_header = self.headers.get("Range")
        self.server.requests.append(("GET", range_header))
        offset = (
            int(range_header.removeprefix("bytes=").removesuffix("-"))
            if range_header
            else 0
        )
        body = self.server.data[offset:]
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", self.server.etag)
        if range_header:
            self.send_header(
                "Content-Range",
                f"bytes {offset}-{len(self.server.data) - 1}/{len(self.server.data)}",
            )
        self.end_headers()
        if self.server.interrupt:
            self.wfile.write(body[: max(1, len(body) // 2)])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.server.interrupt = False
        else:
            self.wfile.write(body)


@pytest.fixture
def publisher(monkeypatch: pytest.MonkeyPatch) -> Any:
    def connect_loopback(sock: socket.socket, address: Any) -> None:
        if not isinstance(address, tuple) or address[0] != "127.0.0.1":
            raise AssertionError("Only local HTTP tests may connect")
        _SOCKET_CONNECT(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect_loopback)
    monkeypatch.setattr(socket, "create_connection", _CREATE_CONNECTION)
    server = _Publisher(("127.0.0.1", 0), _Handler)
    server.requests = []
    server.etag = '"test-archive-v1"'
    server.interrupt = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}/file"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_resume_then_verified_cache_hit_without_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: Any
) -> None:
    server, url = publisher
    server.data, record = _fixture()
    monkeypatch.setattr(acquisition, "load_registration", lambda: record)
    root = StorageRoot(tmp_path)
    server.interrupt = True
    result = acquisition.bootstrap(root, url=url)
    assert result["files"] == 2
    assert result["bytes"] == 620
    assert any(
        value is not None for method, value in server.requests if method == "GET"
    )
    requests_before_hit = len(server.requests)
    hit = acquisition.bootstrap(root, url=url)
    assert hit["cache_hit"] is True
    assert hit["downloaded_bytes"] == 0
    assert len(server.requests) == requests_before_hit
    assert acquisition.verify(root)["receipt_sha256"] == result["receipt_sha256"]


def test_concurrent_bootstrap_downloads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: Any
) -> None:
    server, url = publisher
    server.data, record = _fixture()
    monkeypatch.setattr(acquisition, "load_registration", lambda: record)
    root = StorageRoot(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _: acquisition.bootstrap(root, url=url), range(2))
        )
    assert sorted(item["cache_hit"] for item in results) == [False, True]
    assert sum(method == "GET" for method, _ in server.requests) == 1


def test_changed_content_and_corruption_require_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: Any
) -> None:
    server, url = publisher
    server.data, record = _fixture()
    monkeypatch.setattr(acquisition, "load_registration", lambda: record)
    root = StorageRoot(tmp_path)
    server.data = server.data[:-1] + bytes([server.data[-1] ^ 1])
    with pytest.raises(acquisition.AcquisitionError, match="SHA-256"):
        acquisition.bootstrap(root, url=url)
    server.data, _ = _fixture()
    acquisition.bootstrap(root, url=url)
    destination = root.namespace("datasets") / "mendeley/bjy7vr4xkt/v1"
    target = destination / "Data/video/test.mp4"
    target.write_bytes(b"X" + target.read_bytes()[1:])
    with pytest.raises(acquisition.AcquisitionError, match="corrupt"):
        acquisition.bootstrap(root, url=url)
    assert acquisition.bootstrap(root, url=url, repair=True)["cache_hit"] is False
    assert acquisition.verify(root)["state"] == "verified"


@pytest.mark.parametrize("entry", ["../escape", "Data/video/other.mp4"])
def test_unregistered_or_traversing_archive_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: Any, entry: str
) -> None:
    server, url = publisher
    server.data, record = _fixture(extra=entry)
    monkeypatch.setattr(acquisition, "load_registration", lambda: record)
    with pytest.raises(acquisition.AcquisitionError, match="unsafe|differs"):
        acquisition.bootstrap(StorageRoot(tmp_path), url=url)
    assert not (tmp_path / "escape").exists()


def test_wrong_version_receipt_rejected_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: Any
) -> None:
    server, url = publisher
    server.data, record = _fixture()
    monkeypatch.setattr(acquisition, "load_registration", lambda: record)
    root = StorageRoot(tmp_path)
    acquisition.bootstrap(root, url=url)
    receipt = root.namespace("datasets") / "mendeley/bjy7vr4xkt/v1/acquisition.json"
    payload = json.loads(receipt.read_text())
    payload["version"] = 2
    receipt.write_text(json.dumps(payload))
    with pytest.raises(acquisition.AcquisitionError, match="version"):
        acquisition.verify(root)


def test_symlink_archive_entry_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: Any
) -> None:
    server, url = publisher
    server.data, record = _fixture(extra="Data/video/link.mp4", symlink=True)
    monkeypatch.setattr(acquisition, "load_registration", lambda: record)
    with pytest.raises(acquisition.AcquisitionError, match="unsafe"):
        acquisition.bootstrap(StorageRoot(tmp_path), url=url)
