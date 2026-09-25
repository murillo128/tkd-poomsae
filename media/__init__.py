"""Read-only, timestamp-preserving access to local camera recordings."""

from .reader import (
    DecodedFrame,
    DecodeError,
    DecodeWindowExceeded,
    FrameRef,
    IngestError,
    IngestManifest,
    MediaReader,
    Recording,
    ingest,
)

__all__ = [
    "DecodeError",
    "DecodeWindowExceeded",
    "DecodedFrame",
    "FrameRef",
    "IngestManifest",
    "IngestError",
    "MediaReader",
    "Recording",
    "ingest",
]
