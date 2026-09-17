"""Fixture module exercising the loader's `module.path:name` import-spec
branch: an already-instantiated object, as opposed to a bare class."""

from __future__ import annotations

from src.classifying.records import Classification, QueryClassifier


class UpperQueryClassifier(QueryClassifier):
    name = "upper"

    def classify(self, query: str) -> Classification | None:
        return Classification(label=query.upper())


upper_query_classifier = UpperQueryClassifier()
