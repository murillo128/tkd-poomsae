"""Offline reversible overlays on exact immutable automatic semantics."""

from .core import EditError, IncompatibleAutomaticBase, StaleRevision
from .session import EffectiveSemanticView, SemanticEditor

__all__ = [
    "EditError",
    "EffectiveSemanticView",
    "IncompatibleAutomaticBase",
    "SemanticEditor",
    "StaleRevision",
]
