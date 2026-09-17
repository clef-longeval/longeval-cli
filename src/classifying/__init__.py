"""Pluggable query/interaction classifiers and their DB-backed verdicts."""

from __future__ import annotations

from .builtins import IsBooleanClassifier, WordCountClassifier
from .records import (
    Classification,
    InteractionClassifier,
    InteractionView,
    QueryClassifier,
)
from .runner import (
    ClassificationStats,
    ClassifierConfig,
    classify_interactions,
    classify_queries,
)

__all__ = [
    "Classification",
    "ClassificationStats",
    "ClassifierConfig",
    "InteractionClassifier",
    "InteractionView",
    "IsBooleanClassifier",
    "QueryClassifier",
    "WordCountClassifier",
    "classify_interactions",
    "classify_queries",
]
