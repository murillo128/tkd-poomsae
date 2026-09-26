"""Detailed local frames, digit bends and body-relative spatial relations."""

from .artifact import load_detailed_geometry, publish_detailed_geometry
from .core import (
    DetailedSample,
    GeometryConfig,
    body_frame,
    derive_sample,
    direction,
    foot_geometry,
    forearm_crossing,
    hand_geometry,
    head_geometry,
    point_relations,
)

__all__ = [
    "DetailedSample",
    "GeometryConfig",
    "body_frame",
    "derive_sample",
    "direction",
    "foot_geometry",
    "forearm_crossing",
    "hand_geometry",
    "head_geometry",
    "load_detailed_geometry",
    "point_relations",
    "publish_detailed_geometry",
]
