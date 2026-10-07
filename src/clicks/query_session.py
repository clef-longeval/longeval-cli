"""The shared, versioned representation of one *query session*.

A query session is one :class:`~src.models.SearchResult` as the user saw it: the
query, the search session, when it was issued, and every displayed document with its
1-based position and a binary click flag. Unclicked documents are kept — they
are the negative examples a click model needs.

Both ways of feeding a click model (streaming from the database, replaying an
export) produce exactly these objects, in the same order.

Handling of imperfect data (each case is counted in :class:`QuerySessionStats`):

* ``items`` is not a non-empty JSON object, or any item has a non-integer /
  non-positive position key, a missing / non-string / empty ``docid``, or two
  keys that normalise to the same position (``"1"`` and ``"01"``): the whole
  query session is skipped (``malformed_items``). Dropping just the bad item
  would silently shift the ranking the model learns from.
* ``query_id`` or ``queried_at`` is NULL: the result is not selectable and is
  silently left out of the stream.
* The same document is displayed more than once: the query session is kept when
  none of its copies was clicked (all flags 0 are unambiguous). If an
  accepted interaction names that document, the click cannot be attributed to
  one position, so the query session is skipped (``ambiguous_click``) rather than
  guessing.
* Several accepted interactions on one displayed document count once.
* An accepted interaction on a document that is not displayed in its result is
  ignored and counted (``unmatched_interactions``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: Bump when the record layout changes; the export manifest carries it and
#: readers reject other versions.
FORMAT_VERSION = 1


class QuerySessionFormatError(ValueError):
    """A serialized query session or export does not match ``FORMAT_VERSION``."""


@dataclass(frozen=True)
class QuerySession:
    result_id: int
    query_id: str
    search_session_id: str | None
    timestamp: datetime  # timezone-aware, UTC
    doc_ids: tuple[str, ...]  # ordered by position
    positions: tuple[int, ...]  # 1-based as logged, strictly increasing
    clicks: tuple[int, ...]  # 0/1 per displayed item


@dataclass
class QuerySessionStats:
    """Counters filled while an iterator runs; complete once it is exhausted."""

    query_sessions: int = 0
    items: int = 0
    clicks: int = 0
    unmatched_interactions: int = 0
    skipped: Counter[str] = field(default_factory=Counter)

    def count(self, query_session: QuerySession) -> None:
        self.query_sessions += 1
        self.items += len(query_session.doc_ids)
        self.clicks += sum(query_session.clicks)


def to_utc(value: datetime) -> datetime:
    """Interpret naive datetimes as UTC (that is how the database stores them)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_items(items: Any) -> list[tuple[int, str]] | None:
    """Return ``(position, doc_id)`` sorted by position, or ``None`` if malformed."""
    if not isinstance(items, dict) or not items:
        return None
    parsed: dict[int, str] = {}
    for key, item in items.items():
        try:
            position = int(key)
        except (TypeError, ValueError):
            return None
        doc_id = item.get("docid") if isinstance(item, dict) else None
        if position < 1 or position in parsed:
            return None
        if not isinstance(doc_id, str) or not doc_id:
            return None
        parsed[position] = doc_id
    return sorted(parsed.items())


def build_query_session(
    *,
    result_id: int,
    query_id: str,
    search_session_id: str | None,
    timestamp: datetime,
    items: Any,
    clicked: set[str],
    stats: QuerySessionStats,
) -> QuerySession | None:
    """Combine one result row with its clicked documents, or skip it (counted)."""
    parsed = parse_items(items)
    if parsed is None:
        stats.skipped["malformed_items"] += 1
        return None

    displayed = Counter(doc_id for _, doc_id in parsed)
    if any(displayed[doc_id] > 1 for doc_id in clicked):
        stats.skipped["ambiguous_click"] += 1
        return None
    stats.unmatched_interactions += sum(
        1 for doc_id in clicked if doc_id not in displayed
    )

    return QuerySession(
        result_id=result_id,
        query_id=query_id,
        search_session_id=search_session_id,
        timestamp=to_utc(timestamp),
        doc_ids=tuple(doc_id for _, doc_id in parsed),
        positions=tuple(position for position, _ in parsed),
        clicks=tuple(int(doc_id in clicked) for _, doc_id in parsed),
    )


def to_record(query_session: QuerySession) -> dict[str, Any]:
    return {
        "result_id": query_session.result_id,
        "query_id": query_session.query_id,
        "search_session_id": query_session.search_session_id,
        "timestamp": query_session.timestamp.isoformat(timespec="microseconds"),
        "doc_ids": list(query_session.doc_ids),
        "positions": list(query_session.positions),
        "clicks": list(query_session.clicks),
    }


def from_record(record: Any) -> QuerySession:
    if not isinstance(record, dict):
        raise QuerySessionFormatError("a query session record must be a JSON object")
    try:
        query_session = QuerySession(
            result_id=int(record["result_id"]),
            query_id=str(record["query_id"]),
            search_session_id=record["search_session_id"],
            timestamp=to_utc(datetime.fromisoformat(record["timestamp"])),
            doc_ids=tuple(record["doc_ids"]),
            positions=tuple(int(p) for p in record["positions"]),
            clicks=tuple(int(c) for c in record["clicks"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QuerySessionFormatError(f"malformed query_session record: {exc}") from exc
    lengths = {len(query_session.doc_ids), len(query_session.positions)}
    lengths.add(len(query_session.clicks))
    if len(lengths) != 1:
        raise QuerySessionFormatError("doc_ids/positions/clicks differ in length")
    return query_session
