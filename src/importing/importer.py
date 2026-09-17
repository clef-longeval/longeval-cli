"""Two-command import: search events (+ rankings), then interactions.

``import_queries`` and ``import_clicks`` can run as separate CLI
invocations, potentially far apart in time. What durably joins a click's raw
``search_id`` back to the ``SearchResult`` ``import_queries`` created for it
is the ``SearchReference`` table (``models.py``) — written in the same
transaction as its ``SearchResult``, so (unlike an external mapping store)
there's no separate crash-recovery step: either both rows commit, or
neither does. ``InteractionReference`` plays the same role for click-line
idempotency.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import sqlalchemy as sa

from ..database import create_database_engine, create_schema, make_session_factory
from ..models import (
    Interaction,
    InteractionReference,
    Query,
    SearchReference,
    SearchResult,
    SearchSession,
)
from .records import LogParser, RejectedRecord
from .sources import SourceLine, discover_inputs, stream_source

DEFAULT_PAGE = 1
DEFAULT_RESULTS_PER_PAGE = 10


class ImportAbort(Exception):
    """Raised to stop an import early (``strict`` mode, ``max_errors``)."""


@dataclass
class ImportConfig:
    inputs: list[Path]
    database_url: str
    parser: LogParser
    batch_size: int = 5_000
    strict: bool = False
    max_errors: int | None = None
    limit: int | None = None


@dataclass
class ImportStats:
    searches_seen: int = 0
    interactions_seen: int = 0
    rejected_records: int = 0
    conflicts: int = 0
    unresolved_searches: int = 0
    missing_documents: int = 0
    session_mismatches: int = 0
    known_mappings: int = 0


def import_queries(config: ImportConfig) -> ImportStats:
    """Phase 1: ingest search events (queries + their rankings)."""
    engine = create_database_engine(config.database_url)
    create_schema(engine)
    session_factory = make_session_factory(engine)

    files = discover_inputs(config.inputs)
    stats = ImportStats()
    with session_factory() as session:
        _run_query_phase(config, files, session, stats)
        stats.known_mappings = session.scalar(
            sa.select(sa.func.count()).select_from(SearchReference)
        )
    return stats


def import_clicks(config: ImportConfig) -> ImportStats:
    """Phase 2: ingest click events, linking each to its ``SearchResult``."""
    engine = create_database_engine(config.database_url)
    create_schema(engine)
    session_factory = make_session_factory(engine)

    files = discover_inputs(config.inputs)
    stats = ImportStats()
    with session_factory() as session:
        _run_click_phase(config, files, session, stats)
    return stats


def _stream_all(files: list[Path]) -> Iterator[SourceLine]:
    for file in files:
        yield from stream_source(file)


def _record_rejection(config: ImportConfig, stats: ImportStats) -> None:
    stats.rejected_records += 1
    if config.strict:
        raise ImportAbort("strict mode: rejected record encountered")
    if config.max_errors is not None and stats.rejected_records > config.max_errors:
        raise ImportAbort("max_errors exceeded")


def _utc_aware(timestamp):
    """SQLite's ``DateTime`` doesn't round-trip tzinfo, so a value just read
    back from the database compares unsafely against a freshly parsed one
    unless both are coerced to aware datetimes first."""
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _touch_session(session, cache: dict, session_key: str, timestamp) -> SearchSession:
    search_session = cache.get(session_key) or session.get(SearchSession, session_key)
    if search_session is None:
        search_session = SearchSession(
            id=session_key, started_at=timestamp, ended_at=timestamp
        )
        session.add(search_session)
        session.flush()
    else:
        timestamp = _utc_aware(timestamp)
        started_at = (
            _utc_aware(search_session.started_at) if search_session.started_at else None
        )
        ended_at = (
            _utc_aware(search_session.ended_at) if search_session.ended_at else None
        )
        if started_at is None or timestamp < started_at:
            search_session.started_at = timestamp
        if ended_at is None or timestamp > ended_at:
            search_session.ended_at = timestamp
    cache[session_key] = search_session
    return search_session


def _fingerprint(session_key: str, query: str, queried_at, document_ids) -> str:
    """A content fingerprint for a search event, to tell a conflicting
    duplicate (same search_id, different content) from a genuine repeat.

    ``queried_at`` is normalized to UTC-aware first: SQLite's ``DateTime``
    doesn't round-trip tzinfo, so a value freshly parsed from a log line and
    the same value read back from the database must fingerprint identically.
    """
    payload = "\x1f".join(
        (session_key, query, _utc_aware(queried_at).isoformat(), *document_ids)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ordered_document_ids(items: dict | None) -> list[str]:
    if not items:
        return []
    return [items[key]["docid"] for key in sorted(items, key=int)]


def _existing_search_fingerprint(session, result_id: int) -> str:
    row = session.get(SearchResult, result_id)
    return _fingerprint(
        row.session_id or "",
        row.query.query if row.query else "",
        row.queried_at,
        _ordered_document_ids(row.items),
    )


def _run_query_phase(config: ImportConfig, files, session, stats: ImportStats) -> None:
    known_queries: set[str] = set(session.scalars(sa.select(Query.id)))
    seen: dict[str, int] = {}
    sessions_cache: dict[str, SearchSession] = {}
    processed = 0
    pending = 0

    for result in config.parser.iter_searches(_stream_all(files)):
        if isinstance(result, RejectedRecord):
            _record_rejection(config, stats)
            continue

        if config.limit is not None and processed >= config.limit:
            break
        processed += 1
        stats.searches_seen += 1

        record = result.record
        fingerprint = _fingerprint(
            record.session_id, record.query, record.timestamp, record.document_ids
        )

        result_id = seen.get(record.external_search_id)
        if result_id is None:
            existing_ref = session.get(SearchReference, record.external_search_id)
            result_id = existing_ref.result_id if existing_ref else None

        if result_id is not None:
            if _existing_search_fingerprint(session, result_id) != fingerprint:
                stats.conflicts += 1
                if config.strict:
                    raise ImportAbort(
                        f"conflicting duplicate search {record.external_search_id!r}"
                    )
            continue

        query_id = config.parser.query_id(record.query)
        if query_id not in known_queries:
            session.add(Query(id=query_id, query=record.query))
            known_queries.add(query_id)

        search_session = _touch_session(
            session, sessions_cache, record.session_id, record.timestamp
        )

        result_row = SearchResult(
            session_id=search_session.id,
            query_id=query_id,
            queried_at=record.timestamp,
            page=DEFAULT_PAGE,
            results_per_page=DEFAULT_RESULTS_PER_PAGE,
            items={
                str(rank): {"docid": doc_id}
                for rank, doc_id in enumerate(record.document_ids, start=1)
            },
        )
        session.add(result_row)
        session.flush()

        session.add(
            SearchReference(
                search_id=record.external_search_id, result_id=result_row.id
            )
        )
        seen[record.external_search_id] = result_row.id

        pending += 1
        if pending >= config.batch_size:
            session.commit()
            pending = 0

    session.commit()


def _run_click_phase(config: ImportConfig, files, session, stats: ImportStats) -> None:
    known_this_run: set[str] = set()
    result_cache: dict[int, tuple[dict, str | None]] = {}
    sessions_cache: dict[str, SearchSession] = {}
    processed = 0
    pending = 0

    for result in config.parser.iter_interactions(_stream_all(files)):
        if isinstance(result, RejectedRecord):
            _record_rejection(config, stats)
            continue

        if config.limit is not None and processed >= config.limit:
            break
        processed += 1
        stats.interactions_seen += 1

        record = result.record
        interaction_key = f"{result.line.source_reference}#{result.line.record_number}"
        if interaction_key in known_this_run or (
            session.get(InteractionReference, interaction_key) is not None
        ):
            continue
        known_this_run.add(interaction_key)

        ref = session.get(SearchReference, record.external_search_id)
        if ref is None:
            stats.unresolved_searches += 1
            continue
        result_id = ref.result_id

        if result_id not in result_cache:
            row = session.get(SearchResult, result_id)
            result_cache[result_id] = (row.items or {}, row.session_id)
        items, result_session_id = result_cache[result_id]

        document_ids = {entry.get("docid") for entry in items.values()}
        if record.document_id not in document_ids:
            stats.missing_documents += 1
            continue

        if (
            record.observed_session_id is not None
            and record.observed_session_id != result_session_id
        ):
            stats.session_mismatches += 1

        if result_session_id is not None:
            _touch_session(session, sessions_cache, result_session_id, record.timestamp)

        interaction_row = Interaction(
            session_id=result_session_id,
            result_id=result_id,
            document_id=record.document_id,
            type=record.interaction_type,
            date=record.timestamp,
        )
        session.add(interaction_row)
        session.flush()

        session.add(
            InteractionReference(
                interaction_key=interaction_key, interaction_id=interaction_row.id
            )
        )

        pending += 1
        if pending >= config.batch_size:
            session.commit()
            pending = 0

    session.commit()
