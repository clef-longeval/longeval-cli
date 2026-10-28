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
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite
from tqdm.auto import tqdm

from ..database import (
    create_database_engine,
    create_schema,
    create_secondary_indexes,
    drop_secondary_indexes,
    make_session_factory,
)
from ..models import (
    Interaction,
    InteractionReference,
    Query,
    SearchReference,
    SearchResult,
    SearchSession,
)
from .records import (
    InteractionLogEntry,
    LogParser,
    RejectedRecord,
    SearchLogEntry,
)
from .sources import SourceLine, discover_inputs, stream_source

DEFAULT_PAGE = 1
DEFAULT_RESULTS_PER_PAGE = 10
# Each batch is looked up with one ``IN (...)`` of up to ``batch_size`` keys, and
# SQLite caps bound variables per statement at 32766 (its default).
MAX_BATCH_SIZE = 32_766


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
    # A tqdm counter of records processed (the total line count of a streamed
    # .tar.gz is unknown, so it shows a rate unless ``limit`` bounds it).
    progress: bool = True
    # Drop the secondary indexes for the load and rebuild them at the end: with
    # tens of millions of rows, maintaining them per insert dominates. If a run
    # aborts they stay missing until the next import (or `init-db`-style) rebuild.
    defer_indexes: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size must be between 1 and {MAX_BATCH_SIZE} (SQLite's "
                f"bound-variable limit), got {self.batch_size}; use `limit` to "
                "cap the number of records imported"
            )


def import_queries(config: ImportConfig) -> Counter[str]:
    """Phase 1: ingest search events (queries + their rankings)."""
    tables = [Query.__table__, SearchResult.__table__, SearchReference.__table__]

    def run(session, stats, files, bar):
        _run_query_phase(config, files, session, stats, bar)
        stats["known_mappings"] = session.scalar(
            sa.select(sa.func.count()).select_from(SearchReference)
        )

    return _import(config, tables, run, "queries")


def import_clicks(config: ImportConfig) -> Counter[str]:
    """Phase 2: ingest click events, linking each to its ``SearchResult``."""
    tables = [Interaction.__table__, InteractionReference.__table__]

    def run(session, stats, files, bar):
        _run_click_phase(config, files, session, stats, bar)

    return _import(config, tables, run, "clicks")


def _import(config: ImportConfig, tables, run, desc: str) -> Counter[str]:
    engine = create_database_engine(config.database_url)
    create_schema(engine)
    session_factory = make_session_factory(engine)

    files = discover_inputs(config.inputs)  # fail on bad paths before touching indexes
    stats: Counter[str] = Counter()
    if config.defer_indexes:
        drop_secondary_indexes(engine, tables)
    with tqdm(
        total=config.limit,
        desc=desc,
        unit=" records",
        unit_scale=True,
        disable=not config.progress,
    ) as bar:
        with session_factory() as session:
            run(session, stats, files, bar)
        if config.defer_indexes:
            bar.set_postfix_str("building indexes")
            create_secondary_indexes(engine, tables)
    return stats


def _stream_all(files: list[Path]) -> Iterator[SourceLine]:
    for file in files:
        yield from stream_source(file)


def _record_rejection(config: ImportConfig, stats: Counter[str]) -> None:
    stats["rejected_records"] += 1
    if config.strict:
        raise ImportAbort("strict mode: rejected record encountered")
    if config.max_errors is not None and stats["rejected_records"] > config.max_errors:
        raise ImportAbort("max_errors exceeded")


def _utc_aware(timestamp):
    """SQLite's ``DateTime`` doesn't round-trip tzinfo, so a value just read
    back from the database compares unsafely against a freshly parsed one
    unless both are coerced to aware datetimes first. Postgres returns aware
    values in the server's timezone, so those are converted to UTC."""
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


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


def _insert_rows(session, model, rows: list[dict], on_conflict=None) -> None:
    """Insert ``rows`` (dicts keyed by ``model`` attribute) into its table.

    ``on_conflict(stmt)`` adds an ``ON CONFLICT`` clause to a dialect INSERT.
    On Postgres the rows are streamed with ``COPY`` instead: an executemany
    there runs one INSERT per row, which made imports several times slower
    than on SQLite. ``COPY`` can't resolve conflicts, so rows that need
    ``on_conflict`` are copied into a temporary table and upserted from it.
    """
    if not rows:
        return
    keys = list(rows[0])
    columns = [sa.inspect(model).columns[key] for key in keys]
    table = model.__table__
    if session.get_bind().dialect.name != "postgresql":
        stmt = sqlite.insert(table)
        if on_conflict is not None:
            stmt = on_conflict(stmt)
        session.execute(
            stmt,
            [{c.name: row[k] for k, c in zip(keys, columns)} for row in rows],
        )
        return

    from psycopg import sql

    names = [column.name for column in columns]
    target = table.name
    cursor = session.connection().connection.driver_connection.cursor()
    if on_conflict is not None:
        target = f"_import_{table.name}"
        cursor.execute(
            sql.SQL(
                "DROP TABLE IF EXISTS {tmp}; "
                "CREATE TEMPORARY TABLE {tmp} (LIKE {table}) ON COMMIT DROP"
            ).format(tmp=sql.Identifier(target), table=sql.Identifier(table.name))
        )
    copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier(target), sql.SQL(", ").join(map(sql.Identifier, names))
    )
    encode = [json.dumps if isinstance(c.type, sa.JSON) else None for c in columns]
    with cursor, cursor.copy(copy_sql) as copy:
        for row in rows:
            copy.write_row(
                [
                    row[key] if dump is None or row[key] is None else dump(row[key])
                    for key, dump in zip(keys, encode)
                ]
            )
    if on_conflict is not None:
        staged = sa.table(target, *(sa.column(name) for name in names))
        stmt = postgresql.insert(table).from_select(names, sa.select(*staged.c))
        session.execute(on_conflict(stmt))


def _upsert_sessions(session, times: dict[str, list]) -> None:
    """Insert sessions, widening ``start``/``end`` of ones that already exist."""
    table = SearchSession.__table__
    start, end = table.c.start, table.c.end

    def widen(stmt):
        return stmt.on_conflict_do_update(
            index_elements=[table.c.id],
            set_={
                "start": sa.case(
                    (
                        sa.or_(start.is_(None), stmt.excluded.start < start),
                        stmt.excluded.start,
                    ),
                    else_=start,
                ),
                "end": sa.case(
                    (
                        sa.or_(end.is_(None), stmt.excluded.end > end),
                        stmt.excluded.end,
                    ),
                    else_=end,
                ),
            },
        )

    _insert_rows(
        session,
        SearchSession,
        [
            {"id": k, "started_at": lo, "ended_at": hi}
            for k, (lo, hi) in times.items()
        ],
        widen,
    )


def _widen(times: dict[str, list], session_id: str, timestamp) -> None:
    span = times.get(session_id)
    if span is None:
        times[session_id] = [timestamp, timestamp]
    elif timestamp < span[0]:
        span[0] = timestamp
    elif timestamp > span[1]:
        span[1] = timestamp


def _max_id(session, model) -> int:
    return session.scalar(sa.select(sa.func.max(model.id))) or 0


def _run_query_phase(
    config: ImportConfig, files, session, stats: Counter[str], bar: tqdm
) -> None:
    # Only ever holds one batch: at ~60M rows, per-row dicts of ids/fingerprints
    # outgrow RAM. Cross-batch lookups go to the primary-key indexes instead.
    next_result_id = _max_id(session, SearchResult) + 1
    processed = 0
    batch: list[SearchLogEntry] = []

    def flush() -> None:
        nonlocal next_result_id
        next_result_id = _flush_searches(
            config, session, batch, stats, next_result_id
        )
        session.commit()
        bar.update(processed - bar.n)
        batch.clear()

    for result in config.parser.iter_searches(_stream_all(files)):
        if isinstance(result, RejectedRecord):
            _record_rejection(config, stats)
            continue

        if config.limit is not None and processed >= config.limit:
            break
        processed += 1
        stats["searches_seen"] += 1
        batch.append(result.record)

        if len(batch) >= config.batch_size:
            flush()

    flush()


def _stored_fingerprints(session, search_ids) -> dict[str, str]:
    """Fingerprints of the already-imported searches among ``search_ids``."""
    rows = session.execute(
        sa.select(
            SearchReference.search_id,
            SearchResult.session_id,
            Query.query,
            SearchResult.queried_at,
            SearchResult.items,
        )
        .join(SearchResult, SearchResult.id == SearchReference.result_id)
        .outerjoin(Query, Query.id == SearchResult.query_id)
        .where(SearchReference.search_id.in_(search_ids))
    )
    return {
        search_id: _fingerprint(
            session_id or "", query or "", queried_at, _ordered_document_ids(items)
        )
        for search_id, session_id, query, queried_at, items in rows
    }


def _flush_searches(config, session, batch, stats, next_id: int) -> int:
    known = _stored_fingerprints(session, {r.external_search_id for r in batch})
    queries: dict[str, str] = {}
    session_times: dict[str, list] = {}
    results: list[dict] = []
    references: list[dict] = []

    for record in batch:
        fingerprint = _fingerprint(
            record.session_id, record.query, record.timestamp, record.document_ids
        )
        prior = known.get(record.external_search_id)
        if prior is not None:
            if prior != fingerprint:
                stats["conflicts"] += 1
                if config.strict:
                    raise ImportAbort(
                        f"conflicting duplicate search {record.external_search_id!r}"
                    )
            continue
        known[record.external_search_id] = fingerprint

        query_id = config.parser.query_id(record.query)
        queries[query_id] = record.query
        _widen(session_times, record.session_id, record.timestamp)
        results.append(
            {
                "id": next_id,
                "session_id": record.session_id,
                "query_id": query_id,
                "queried_at": record.timestamp,
                "page": DEFAULT_PAGE,
                "results_per_page": DEFAULT_RESULTS_PER_PAGE,
                "items": {
                    str(rank): {"docid": doc_id}
                    for rank, doc_id in enumerate(record.document_ids, start=1)
                },
            }
        )
        references.append(
            {"search_id": record.external_search_id, "result_id": next_id}
        )
        next_id += 1

    if results:
        _insert_rows(
            session,
            Query,
            [{"id": k, "query": v} for k, v in queries.items()],
            lambda stmt: stmt.on_conflict_do_nothing(index_elements=["id"]),
        )
        _upsert_sessions(session, session_times)
        _insert_rows(session, SearchResult, results)
        _insert_rows(session, SearchReference, references)
    return next_id


def _run_click_phase(
    config: ImportConfig, files, session, stats: Counter[str], bar: tqdm
) -> None:
    next_interaction_id = _max_id(session, Interaction) + 1
    processed = 0
    pending: list[tuple[str, InteractionLogEntry]] = []

    def flush() -> None:
        nonlocal next_interaction_id
        next_interaction_id = _flush_clicks(
            session, pending, stats, next_interaction_id
        )
        session.commit()
        bar.update(processed - bar.n)
        pending.clear()

    for result in config.parser.iter_interactions(_stream_all(files)):
        if isinstance(result, RejectedRecord):
            _record_rejection(config, stats)
            continue

        if config.limit is not None and processed >= config.limit:
            break
        processed += 1
        stats["interactions_seen"] += 1

        interaction_key = f"{result.line.source_reference}#{result.line.record_number}"
        pending.append((interaction_key, result.record))

        if len(pending) >= config.batch_size:
            flush()

    flush()


def _flush_clicks(session, pending, stats: Counter[str], next_id: int) -> int:
    """Resolve a batch of clicks against their results and bulk-insert them."""
    imported = set(
        session.scalars(
            sa.select(InteractionReference.interaction_key).where(
                InteractionReference.interaction_key.in_([key for key, _ in pending])
            )
        )
    )
    unique: dict[str, InteractionLogEntry] = {}
    for key, record in pending:
        if key not in imported:
            unique.setdefault(key, record)
    pending = list(unique.items())
    search_ids = {record.external_search_id for _, record in pending}
    resolved = {
        search_id: (result_id, session_id, items)
        for search_id, result_id, session_id, items in session.execute(
            sa.select(
                SearchReference.search_id,
                SearchResult.id,
                SearchResult.session_id,
                SearchResult.items,
            )
            .join(SearchResult, SearchResult.id == SearchReference.result_id)
            .where(SearchReference.search_id.in_(search_ids))
        )
    }
    doc_ids: dict[str, set] = {}
    session_times: dict[str, list] = {}
    interactions: list[dict] = []
    references: list[dict] = []

    for interaction_key, record in pending:
        hit = resolved.get(record.external_search_id)
        if hit is None:
            stats["unresolved_searches"] += 1
            continue
        result_id, result_session_id, items = hit

        documents = doc_ids.get(record.external_search_id)
        if documents is None:
            documents = doc_ids[record.external_search_id] = {
                entry.get("docid") for entry in (items or {}).values()
            }
        if record.document_id not in documents:
            stats["missing_documents"] += 1
            continue

        if (
            record.observed_session_id is not None
            and record.observed_session_id != result_session_id
        ):
            stats["session_mismatches"] += 1

        if result_session_id is not None:
            _widen(session_times, result_session_id, record.timestamp)

        interactions.append(
            {
                "id": next_id,
                "session_id": result_session_id,
                "result_id": result_id,
                "document_id": record.document_id,
                "type": record.interaction_type,
                "date": record.timestamp,
            }
        )
        references.append(
            {"interaction_key": interaction_key, "interaction_id": next_id}
        )
        next_id += 1

    if interactions:
        _upsert_sessions(session, session_times)
        _insert_rows(session, Interaction, interactions)
        _insert_rows(session, InteractionReference, references)
    return next_id
