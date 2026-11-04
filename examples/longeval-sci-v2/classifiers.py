"""Snapshot-independent classifiers for LongEval Sci 2026 (paper §3.1, §4.1).

They don't depend on a date range, so run them once, globally::

    uv run longeval-cli classify --classifier classifiers.py --database-url "$DB"

The query flags label every query `0`/`1`, so the snapshot filters can
`require` them to be `0`. `has-doi` and `is-structured` are CORE-specific;
`is-boolean` and `word-count` are built-ins.
"""

import re

from src.classifying.builtins import (
    ClickDelayClassifier,
    IsBooleanClassifier,
    WordCountClassifier,
)
from src.classifying.records import Classification, RowClassifier


class HasDoiClassifier(RowClassifier):
    """`1` if the query contains a DOI (a lookup, navigational intent)."""

    name = "has-doi"
    level = "queries"

    doi_regex = re.compile(r"(^|[^0-9])10\.[0-9]{4,9}/[-._;()/:a-zA-Z0-9]+")

    def classify(self, row):
        return Classification.flag(bool(self.doi_regex.search(row.text)))


class IsStructuredClassifier(RowClassifier):
    """`1` if the query names a CORE search field, e.g. `title:` or `authors:`."""

    name = "is-structured"
    level = "queries"

    core_fields = (
        "abstract", "acceptedDate", "arxivId", "authors", "citationCount",
        "contributors", "createdDate", "dataProviders", "depositedDate",
        "documentType", "doi", "downloadUrl", "fieldOfStudy", "fullText",
        "identifiers", "journals", "links", "magId", "oaiIds", "outputs",
        "publishedDate", "publisher", "pubmedId", "references",
        "sourceFulltextUrls", "title", "updatedDate", "yearPublished",
        "language", "_exists_", "id",
    )  # fmt: skip
    field_lookup = re.compile(
        r"(^|\s|[^a-zA-Z0-9])(" + "|".join(core_fields) + r"):", re.IGNORECASE
    )

    def classify(self, row):
        return Classification.flag(bool(self.field_lookup.search(row.text)))


classifiers = [
    IsBooleanClassifier(),  # queries
    WordCountClassifier(),  # queries
    HasDoiClassifier(),  # queries
    IsStructuredClassifier(),  # queries
    # interactions: "late" if more than 24 hours after the search
    ClickDelayClassifier("click-delay", max_hours=24),
]
