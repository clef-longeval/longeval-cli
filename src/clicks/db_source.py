"""Bounded-memory, deterministic streaming of query sessions from the database."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..filtering import SnapshotConfig, rule_clauses
from ..models import Interaction, SearchResult
from .query_session import QuerySession, QuerySessionStats, build_query_session

#: SQLite builds before 3.32 allow only 999 bound parameters per statement.
MAX_BATCH_SIZE = 900


def iter_query_sessions_from_db(
    session: Session,
    config: SnapshotConfig,
    stats: QuerySessionStats | None = None,
    *,
    batch_size: int = 500,
) -> Iterator[QuerySession]:
    """Yield the snapshot's query sessions ordered by ``(queried_at, result id)``.

    ``config`` supplies the date range ``[start, end)`` (UTC, half-open, on
    ``SearchResult.queried_at``), the ``click_types`` that count as a click and
    the verdict rules: a result whose query, session or own verdicts fail them
    is left out; an interaction that fails them is not a click. The
    interaction date is deliberately not snapshotted: a click belongs to the
    query session it was logged against, whenever it happened.

    Memory is bounded by ``batch_size`` results: the results are
    read with keyset pagination, and each page's clicks come from one extra
    query (not one per result). Nothing is collected beyond the current page.

    ``stats`` (optional) is filled as the iterator advances and is complete
    once it is exhausted. Reads are not wrapped in a single snapshot, so the
    stream is reproducible only while the source rows do not change.
    """
    if not config.click_types:
        raise ValueError("the snapshot config must list `click_types`")
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be in 1..{MAX_BATCH_SIZE}")
    stats = stats if stats is not None else QuerySessionStats()
    columns = (
        SearchResult.id,
        SearchResult.query_id,
        SearchResult.session_id,
        SearchResult.queried_at,
        SearchResult.items,
    )
    base = sa.select(*columns).where(
        SearchResult.query_id.is_not(None), SearchResult.queried_at.is_not(None)
    )
    base = base.where(*result_clauses(config))
    if config.start is not None:
        base = base.where(SearchResult.queried_at >= config.start)
    if config.end is not None:
        base = base.where(SearchResult.queried_at < config.end)
    base = base.order_by(SearchResult.queried_at, SearchResult.id).limit(batch_size)

    last: tuple[datetime, int] | None = None
    while True:
        statement = base
        if last is not None:
            statement = base.where(
                sa.tuple_(SearchResult.queried_at, SearchResult.id) > sa.tuple_(*last)
            )
        page = session.execute(statement).all()
        if not page:
            break

        clicked = _clicked_documents(session, [row.id for row in page], config)
        for row in page:
            query_session = build_query_session(
                result_id=row.id,
                query_id=row.query_id,
                search_session_id=row.session_id,
                timestamp=row.queried_at,
                items=row.items,
                clicked=clicked.get(row.id, set()),
                stats=stats,
            )
            if query_session is not None:
                stats.count(query_session)
                yield query_session

        last = (page[-1].queried_at, page[-1].id)
        if len(page) < batch_size:
            break


def result_clauses(config: SnapshotConfig) -> list:
    """The query, session and result rules as conditions on ``SearchResult``."""
    return [
        *rule_clauses(config, "queries", SearchResult.query_id),
        *rule_clauses(config, "sessions", SearchResult.session_id),
        *rule_clauses(config, "results", SearchResult.id),
    ]


def click_clauses(config: SnapshotConfig) -> list:
    """What makes an interaction a click: its type and the interaction rules."""
    return [
        Interaction.type.in_(sorted(config.click_types)),
        *rule_clauses(config, "interactions", Interaction.id),
    ]


def _clicked_documents(
    session: Session, result_ids: Iterable[int], config: SnapshotConfig
) -> dict[int, set[str]]:
    """Accepted-click documents per result; repeated clicks collapse in the set."""
    rows = session.execute(
        sa.select(Interaction.result_id, Interaction.document_id)
        .where(Interaction.result_id.in_(list(result_ids)), *click_clauses(config))
        .distinct()
    )
    clicked: dict[int, set[str]] = defaultdict(set)
    for result_id, document_id in rows:
        clicked[result_id].add(document_id)
    return clicked
