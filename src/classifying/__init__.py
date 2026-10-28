"""Pluggable classifiers for queries, sessions, results, interactions and documents."""

from __future__ import annotations

from .builtins import (
    AbstractLengthClassifier,
    ClickDelayClassifier,
    ClicksClassifier,
    IsBooleanClassifier,
    ManualLabelsClassifier,
    SearchBurstClassifier,
    WordCountClassifier,
)
from .records import Classification, Classifier, RowClassifier
from .rows import DocumentView, InteractionView, QueryView, ResultView, SessionView
from .runner import ClassifierConfig, classify

__all__ = [
    "AbstractLengthClassifier",
    "Classification",
    "Classifier",
    "ClassifierConfig",
    "ClickDelayClassifier",
    "ClicksClassifier",
    "DocumentView",
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
