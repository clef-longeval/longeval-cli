"""General-purpose classifiers that don't depend on any particular log format
or search backend — unlike, say, a CORE-specific field-lookup classifier.
"""

from __future__ import annotations

import csv
import re
from collections import deque
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..filtering import SnapshotConfig, rule_clauses
from ..models import Interaction, SearchResult
from .records import Classification, Classifier, RowClassifier
from .rows import DocumentView, InteractionView, QueryView

_BOOLEAN_OPERATOR = re.compile(r'(AND|OR|NOT|" or "|" and ")')


class IsBooleanClassifier(RowClassifier):
    """Flags queries that use a boolean operator (AND/OR/NOT), quoted or not.

    Known limitation: this only catches queries that literally spell out an
    operator (or a quoted ``" and "``/``" or "``); a query like "combo meals
    option and (yearpublished>=1977 and yearpublished<=2025)" that
    lowercases its operators without quotes still slips through, since
    there's no reliable way to tell it apart from a plain multi-word query.
    """

    name = "is-boolean"
    level = "queries"

    def classify(self, row: QueryView) -> Classification | None:
        return Classification.flag(bool(_BOOLEAN_OPERATOR.search(row.text)))


class WordCountClassifier(RowClassifier):
    """Records a query's word count as its ``label``.

    The count is stored as a string, not as ``confidence``: there's no
    per-query uncertainty here, and a plain label lets later filtering
    (a filter) match on it the same way as any other classifier
    (``WHERE label = ...``). Returns ``None`` (no opinion) for an empty
    query rather than a count of zero or one.
    """

    name = "word-count"
    level = "queries"

    def classify(self, row: QueryView) -> Classification | None:
        words = row.text.split()
        if not words:
            return None
        return Classification(label=str(len(words)), value=float(len(words)))


class ManualLabelsClassifier(RowClassifier):
    """Applies labels a human added to an ``export queries`` TSV.

    The reviewer adds a ``manual_label`` column to the exported file. Matches
    on the exact ``query`` text, so only queries present in the file are
    classified; a query that's absent, or whose label cell is blank, gets
    ``None`` (no opinion) and keeps any earlier manual verdict untouched.
    Needs a file path, so wrap it in a small file defining
    ``classifiers``, e.g.
    ``classifiers = ManualLabelsClassifier("review/annotated.tsv")``.
    """

    level = "queries"

    def __init__(
        self,
        path: Path | str,
        *,
        name: str = "manual",
        label_column: str = "manual_label",
        delimiter: str = "\t",
    ) -> None:
        self.name = name
        self._labels: dict[str, str] = {}
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            missing = {"query", label_column} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} is missing column(s): {', '.join(sorted(missing))}"
                )
            for row in reader:
                label = (row[label_column] or "").strip()
                if label:
                    self._labels[row["query"]] = label

    def classify(self, row: QueryView) -> Classification | None:
        label = self._labels.get(row.text)
        return Classification(label=label) if label else None


class ClicksClassifier(Classifier):
    """Counts each query's clicks in the snapshot.

    A click is what the snapshot config calls one: an interaction of one of
    its ``click_types`` (any type if none are set) that passes its
    ``interactions`` rules, on a result whose session and own verdicts pass
    the ``sessions``/``results`` rules, so e.g. bot clicks don't count. The
    ``from``/``to`` range bounds the search date, as in the query-session export;
    accepted clicks on those searches count even if they happened later.
    Stores the count as both ``label`` and ``value`` (so a filter can threshold
    on it); queries without a click get no verdict.
    """

    level = "queries"

    def __init__(self, name: str = "clicks") -> None:
        self.name = name

    def classify_all(
        self, db: Session, config: SnapshotConfig
    ) -> Iterator[tuple[str, Classification]]:
        clicks = rule_clauses(config, "interactions", Interaction.id)
        if config.click_types:  # without click_types, every type counts
            clicks.append(Interaction.type.in_(sorted(config.click_types)))
        statement = (
            sa.select(SearchResult.query_id, sa.func.count())
            .select_from(Interaction)
            .join(SearchResult, Interaction.result_id == SearchResult.id)
            .where(
                SearchResult.query_id.is_not(None),
                *clicks,
                *rule_clauses(config, "sessions", SearchResult.session_id),
                *rule_clauses(config, "results", SearchResult.id),
            )
            .group_by(SearchResult.query_id)
        )
        if config.start is not None:
            statement = statement.where(SearchResult.queried_at >= config.start)
        if config.end is not None:
            statement = statement.where(SearchResult.queried_at < config.end)

        for query_id, count in db.execute(statement).all():
            yield query_id, Classification(label=str(count), value=float(count))


class ClickDelayClassifier(RowClassifier):
    """Labels an interaction ``late`` if it came more than ``max_hours`` after
    the search that showed it (``value``: the delay in hours).

    Interactions in time, or without both timestamps, get no verdict.
    """

    level = "interactions"

    def __init__(self, name: str = "click-delay", *, max_hours: float = 24) -> None:
        self.name = name
        self.max_delay = timedelta(hours=max_hours)

    def classify(self, row: InteractionView) -> Classification | None:
        if row.date is None or row.searched_at is None:
            return None
        delay = row.date - row.searched_at
        if delay <= self.max_delay:
            return None
        return Classification(label="late", value=delay.total_seconds() / 3600)


class SearchBurstClassifier(Classifier):
    """Labels a session ``bot`` if it made ``searches`` searches within
    ``within_seconds`` at least once (``value``: its most searches in one
    such window).

    Only searches inside the snapshot's ``from``/``to`` range count. Other
    sessions get no verdict.
    """

    level = "sessions"

    def __init__(
        self, name: str = "bot", *, searches: int = 10, within_seconds: float = 60
    ) -> None:
        self.name = name
        self.searches = searches
        self.window = timedelta(seconds=within_seconds)

    def classify_all(
        self, db: Session, config: SnapshotConfig
    ) -> list[tuple[str, Classification]]:
        statement = (
            sa.select(SearchResult.session_id, SearchResult.queried_at)
            .where(
                SearchResult.session_id.is_not(None),
                SearchResult.queried_at.is_not(None),
            )
            .order_by(SearchResult.session_id, SearchResult.queried_at)
        )
        if config.start is not None:
            statement = statement.where(SearchResult.queried_at >= config.start)
        if config.end is not None:
            statement = statement.where(SearchResult.queried_at < config.end)

        most: dict[str, int] = {}
        window: deque = deque()
        current = None
        for session_id, queried_at in db.execute(statement):
            if session_id != current:
                current, window = session_id, deque()
            window.append(queried_at)
            while queried_at - window[0] > self.window:
                window.popleft()
            most[session_id] = max(most.get(session_id, 0), len(window))
        # Collected first: the runner writes while consuming the verdicts.
        return [
            (session_id, Classification(label="bot", value=float(count)))
            for session_id, count in most.items()
            if count >= self.searches
        ]


class AbstractLengthClassifier(RowClassifier):
    """Labels a document ``short`` if its abstract, stripped, has fewer than
    ``min_chars`` characters; a missing abstract counts as empty (``value``:
    the length).

    Documents with a long enough abstract get no verdict, so a filter reads
    ``documents: {exclude: {abstract-length: short}}``.
    """

    level = "documents"

    def __init__(
        self, name: str = "abstract-length", *, min_chars: int = 100
    ) -> None:
        self.name = name
        self.min_chars = min_chars

    def classify(self, row: DocumentView) -> Classification | None:
        length = len((row.abstract or "").strip())
        if length >= self.min_chars:
            return None
        return Classification(label="short", value=float(length))
