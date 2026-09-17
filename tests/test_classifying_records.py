from __future__ import annotations

import pytest

from src.classifying.records import (
    Classification,
    InteractionClassifier,
    InteractionView,
    QueryClassifier,
)


def test_query_classifier_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        QueryClassifier()  # type: ignore[abstract]


def test_interaction_classifier_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        InteractionClassifier()  # type: ignore[abstract]


def test_classification_confidence_defaults_to_none():
    assert Classification(label="spam").confidence is None


def test_query_classifier_subclass_implements_classify():
    class AlwaysSpam(QueryClassifier):
        name = "always-spam"

        def classify(self, query: str) -> Classification | None:
            return Classification(label="spam", confidence=1.0)

    result = AlwaysSpam().classify("anything")

    assert result == Classification(label="spam", confidence=1.0)


def test_interaction_classifier_subclass_receives_an_interaction_view():
    class EchoTarget(InteractionClassifier):
        name = "echo-target"

        def classify(self, interaction: InteractionView) -> Classification | None:
            return Classification(label=interaction.type or "unknown")

    view = InteractionView(
        id=1, document_id="d1", type="get-pdf", session_id="s1", date=None
    )

    result = EchoTarget().classify(view)

    assert result == Classification(label="get-pdf")
