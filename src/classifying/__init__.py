"""Pluggable classifiers for queries, sessions, results and interactions."""

from __future__ import annotations

from .builtins import (
    ClickDelayClassifier,
    ClicksClassifier,
    IsBooleanClassifier,
    ManualLabelsClassifier,
    SearchBurstClassifier,
    WordCountClassifier,
)
from .records import Classification, Classifier, RowClassifier
from .rows import InteractionView, QueryView, ResultView, SessionView
from .runner import ClassifierConfig, classify

__all__ = [
    "Classification",
    "Classifier",
    "ClassifierConfig",
    "ClickDelayClassifier",
    "ClicksClassifier",
    "InteractionView",
    "IsBooleanClassifier",
    "ManualLabelsClassifier",
    "QueryView",
    "ResultView",
    "RowClassifier",
    "SearchBurstClassifier",
    "SessionView",
    "WordCountClassifier",
    "classify",
]
