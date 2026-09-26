"""Native-time evidence for automatic recording synchronization."""

from .cues import CueConfig, CueEvent, CueSample, CueSet, CueStream, extract_cues
from .solver import (
    SyncSource,
    TimelineFailure,
    publish_offsets,
    solve_offsets,
    sources_from_manifest,
)

__all__ = [
    "CueConfig", "CueEvent", "CueSample", "CueSet", "CueStream", "extract_cues",
    "SyncSource", "TimelineFailure", "publish_offsets", "solve_offsets",
    "sources_from_manifest",
]
