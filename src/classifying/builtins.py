"""General-purpose ``QueryClassifier``s that don't depend on any particular
log format or search backend — unlike, say, a CORE-specific field-lookup
classifier, these apply to query text from any source.
"""

from __future__ import annotations

import re

from .records import Classification, QueryClassifier

_BOOLEAN_OPERATOR = re.compile(r'(AND|OR|NOT|" or "|" and ")')


class IsBooleanClassifier(QueryClassifier):
    """Flags queries that use a boolean operator (AND/OR/NOT), quoted or not.

    Known limitation: this only catches queries that literally spell out an
    operator (or a quoted ``" and "``/``" or "``); a query like "combo meals
    option and (yearpublished>=1977 and yearpublished<=2025)" that
    lowercases its operators without quotes still slips through, since
    there's no reliable way to tell it apart from a plain multi-word query.
    """

    name = "is-boolean"

    def classify(self, query: str) -> Classification | None:
        label = "boolean" if _BOOLEAN_OPERATOR.search(query) else "not-boolean"
        return Classification(label=label)


class WordCountClassifier(QueryClassifier):
    """Records a query's word count as its ``label``.

    The count is stored as a string, not as ``confidence``: there's no
    per-query uncertainty here, and a plain label lets later filtering
    (``select-queries``) match on it the same way as any other classifier
    (``WHERE label = ...``). Returns ``None`` (no opinion) for an empty
    query rather than a count of zero or one.
    """

    name = "word-count"

    def classify(self, query: str) -> Classification | None:
        words = query.split()
        if not words:
            return None
        return Classification(label=str(len(words)))
