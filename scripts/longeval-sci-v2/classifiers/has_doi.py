"""QueryClassifier flagging queries that embed a DOI, ported from the
project's legacy Postgres classification script.
"""

from __future__ import annotations

import re

from src.classifying.records import Classification, QueryClassifier

_DOI = re.compile(r"(^|[^0-9])10\.[0-9]{4,9}/[-._;()/:a-zA-Z0-9]+")


class HasDoiClassifier(QueryClassifier):
    name = "has-doi"

    def classify(self, query: str) -> Classification | None:
        label = "doi" if _DOI.search(query) else "no-doi"
        return Classification(label=label)


query_classifier = HasDoiClassifier()
