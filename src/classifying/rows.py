"""Read-only views of the rows a :class:`RowClassifier` labels, with context.

Each level has one view with the context that is cheap to join in (e.g. an
interaction's rank and search time), read page by page in id order. Anything
not in a view is better computed by an aggregate ``Classifier`` than looked up
per row.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..clicks.query_session import parse_items
from ..models import Interaction, Query, SearchResult, SearchSession

PAGE_SIZE = 1_000


@dataclass(frozen=True, slots=True)
class QueryView:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class SessionView:
    id: str
    started_at: datetime | None
    ended_at: datetime | None


@dataclass(frozen=True, slots=True)
class ResultView:
    id: int
    query_id: str | None
    query: str | None
    session_id: str | None
    queried_at: datetime | None
    doc_ids: tuple[str, ...]  # in rank order; empty if items are malformed


@dataclass(frozen=True, slots=True)
class InteractionView:
    id: int
    type: str | None
    date: datetime | None
    document_id: str
    session_id: str | None
    result_id: int
    query_id: str | None
    searched_at: datetime | None  # when the result was shown
    rank: int | None  # the document's 1-based position in that result


def _doc_ids(items: Any) -> tuple[str, ...]:
    return tuple(doc_id for _, doc_id in parse_items(items) or ())


def _query(row) -> QueryView:
    return QueryView(row.id, row.query or "")


def _session(row) -> SessionView:
    return SessionView(row.id, row.started_at, row.ended_at)


def _result(row) -> ResultView:
    return ResultView(
        row.id,
        row.query_id,
        row.query,
        row.session_id,
        row.queried_at,
        _doc_ids(row.items),
    )


def _interaction(row) -> InteractionView:
    ranks = {}
    for position, doc_id in parse_items(row.items) or ():
        ranks.setdefault(doc_id, position)
    return InteractionView(
        row.id,
        row.type,
        row.date,
        row.document_id,
        row.session_id,
        row.result_id,
        row.query_id,
        row.queried_at,
        ranks.get(row.document_id),
    )


#: level -> (id column, statement selecting the view's columns, row -> view)
_VIEWS: dict[str, tuple[Any, Callable[[], sa.Select], Callable]] = {
    "queries": (Query.id, lambda: sa.select(Query.id, Query.query), _query),
    "sessions": (
        SearchSession.id,
        lambda: sa.select(
            SearchSession.id, SearchSession.started_at, SearchSession.ended_at
        ),
        _session,
    ),
    "results": (
        SearchResult.id,
        lambda: sa.select(
            SearchResult.id,
            SearchResult.query_id,
            Query.query,
            SearchResult.session_id,
            SearchResult.queried_at,
            SearchResult.items,
        ).outerjoin(Query, Query.id == SearchResult.query_id),
        _result,
    ),
    "interactions": (
        Interaction.id,
        lambda: sa.select(
            Interaction.id,
            Interaction.type,
            Interaction.date,
            Interaction.document_id,
            Interaction.session_id,
            Interaction.result_id,
            SearchResult.query_id,
            SearchResult.queried_at,
            SearchResult.items,
        ).join(SearchResult, SearchResult.id == Interaction.result_id),
        _interaction,
    ),
}


def iter_rows(db: Session, level: str) -> Iterator[Any]:
    """Yield a view of every row of ``level``, in id order, a page at a time.

    Each page is read completely before it is yielded, so the caller may
    write and commit between rows.
    """
    id_column, select, view = _VIEWS[level]
    last = None
    while True:
        statement = select().order_by(id_column).limit(PAGE_SIZE)
        if last is not None:
            statement = statement.where(id_column > last)
        page = db.execute(statement).all()
        yield from map(view, page)
        if len(page) < PAGE_SIZE:
            return
        last = page[-1].id
