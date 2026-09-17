"""Normalized records produced by parsing one raw :class:`SourceLine`.

A format parser (``importing.formats``) never raises on a bad line — it
wraps every outcome as a :class:`SourceRecord` (successfully parsed) or a
:class:`RejectedRecord` (malformed, unrecognized, or filtered out), always
carrying the originating :class:`~src.importing.sources.SourceLine` for
provenance and error reporting.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .sources import SourceLine


@dataclass(frozen=True, slots=True)
class SearchLogEntry:
    """A search event: a query and the ranked list of documents it returned."""

    external_search_id: str
    session_id: str
    query: str
    document_ids: tuple[str, ...]
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class InteractionLogEntry:
    """A click/interaction against a previously logged search result."""

    external_search_id: str
    document_id: str
    timestamp: datetime
    interaction_type: str | None = None
    observed_session_id: str | None = None


LogEntry = SearchLogEntry | InteractionLogEntry


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """A line that parsed successfully into a normalized ``record``."""

    line: SourceLine
    record: LogEntry


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """A line that could not be turned into a record, and why."""

    line: SourceLine
    reason: str


ParseResult = SourceRecord | RejectedRecord


class LogParser(Protocol):
    """What a format parser must provide — the "configurable parser"
    extension point. A user's ``parser.py`` can implement this from
    scratch, or subclass a built-in parser and override just ``query_id``
    to change query normalization."""

    def iter_searches(self, lines: Iterable[SourceLine]) -> Iterator[ParseResult]: ...

    def iter_interactions(
        self, lines: Iterable[SourceLine]
    ) -> Iterator[ParseResult]: ...

    def query_id(self, query: str) -> str: ...
