"""QueryClassifier flagging queries that use CORE's field-lookup syntax
(e.g. ``yearPublished:``, ``documentType:``), ported from the project's
legacy Postgres classification script.

The original SQL's alternation had a leading empty branch
(``(|abstract|...)``), which would match a bare ``:`` preceded by
whitespace/start/punctuation even with no field name at all — almost
certainly an accidental artifact of building the pattern from a joined list
rather than intended behavior. It's dropped here.
"""

from __future__ import annotations

import re

from src.classifying.records import Classification, QueryClassifier

_FIELDS = (
    "abstract",
    "acceptedDate",
    "arxivId",
    "authors",
    "citationCount",
    "contributors",
    "createdDate",
    "dataProviders",
    "depositedDate",
    "documentType",
    "doi",
    "downloadUrl",
    "fieldOfStudy",
    "fullText",
    "identifiers",
    "journals",
    "links",
    "magId",
    "oaiIds",
    "outputs",
    "publishedDate",
    "publisher",
    "pubmedId",
    "references",
    "sourceFulltextUrls",
    "title",
    "updatedDate",
    "yearPublished",
    "language",
    "_exists_",
    "id",
)
_FIELD_LOOKUP = re.compile(
    r"(^|\s|[^a-zA-Z0-9])(" + "|".join(_FIELDS) + r"):", re.IGNORECASE
)


class IsStructuredClassifier(QueryClassifier):
    name = "is-structured"

    def classify(self, query: str) -> Classification | None:
        label = "structured" if _FIELD_LOOKUP.search(query) else "not-structured"
        return Classification(label=label)


query_classifier = IsStructuredClassifier()
