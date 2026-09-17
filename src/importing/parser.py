"""Base class for log parsers — the "configurable parser" extension point.

Nothing here knows about any particular log format. Subclass this and
implement four hooks to support a new one:

- ``_is_search`` / ``_is_interaction`` — recognize a decoded line's shape
  (e.g. by which keys it has).
- ``_parse_search`` / ``_parse_interaction`` — turn a recognized line into a
  ``SearchLogEntry``/``InteractionLogEntry``.

Override ``query_id`` too if a format needs different query normalization.
JSON decoding and dispatch (``iter_searches``/``iter_interactions``, which
silently skip lines belonging to the other pass and reject lines matching
neither shape) are shared here, along with generic text/timestamp helpers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from typing import Any

from src import identity
from src.importing.records import ParseResult, RejectedRecord
from src.importing.sources import SourceLine


class BaseParser:
    """Shared plumbing for a ``LogParser`` (see ``records.py``)."""

    def __init__(self, *, accepted_interaction_types: frozenset[str] | None = None):
        self.accepted_interaction_types = accepted_interaction_types

    def query_id(self, query: str) -> str:
        """Return a stable id for ``query``, used to dedupe/group queries."""
        return identity.query_id(identity.normalize_query(query))

    def iter_searches(self, lines: Iterable[SourceLine]) -> Iterator[ParseResult]:
        for line in lines:
            raw = self._decode(line)
            if raw is None:
                yield RejectedRecord(line, "invalid-json")
                continue
            if self._is_search(raw):
                yield self._parse_search(line, raw)
            elif self._is_interaction(raw):
                continue  # belongs to the interaction pass
            else:
                yield RejectedRecord(line, "unknown-record-shape")

    def iter_interactions(self, lines: Iterable[SourceLine]) -> Iterator[ParseResult]:
        for line in lines:
            raw = self._decode(line)
            if raw is None or not self._is_interaction(raw):
                continue  # malformed / belongs to the search pass / unknown
            yield self._parse_interaction(line, raw)

    @staticmethod
    def _decode(line: SourceLine) -> dict[str, Any] | None:
        try:
            raw = json.loads(line.text)
        except (json.JSONDecodeError, TypeError):
            return None
        return raw if isinstance(raw, dict) else None

    def _is_search(self, raw: dict[str, Any]) -> bool:
        raise NotImplementedError

    def _is_interaction(self, raw: dict[str, Any]) -> bool:
        raise NotImplementedError

    def _parse_search(self, line: SourceLine, raw: dict[str, Any]) -> ParseResult:
        raise NotImplementedError

    def _parse_interaction(self, line: SourceLine, raw: dict[str, Any]) -> ParseResult:
        raise NotImplementedError


def parse_timestamp(date: Any) -> datetime | None:
    """Parse a Unix-epoch ``date`` field; ``None`` if it isn't one."""
    try:
        return datetime.fromtimestamp(float(date), tz=timezone.utc)
    except (TypeError, ValueError):
        return None
