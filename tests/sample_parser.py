"""A test-only ``--parser`` file, so the tests don't depend on any example.

It reads the JSONL shape the tests' synthetic logs use: a search line has a
``search_id`` key (plus ``date``, ``query``, ``serp``, ``uid``), an interaction
line a ``trackId`` key (``<search_id>-<document_id>``, plus ``date``, ``type``,
``uid``). Load it by path (``SAMPLE_PARSER_PATH`` in ``conftest.py``), like any
user-supplied parser.
"""

from __future__ import annotations

from typing import Any

from src.importing.parser import BaseParser, parse_timestamp
from src.importing.records import (
    InteractionLogEntry,
    ParseResult,
    RejectedRecord,
    SearchLogEntry,
    SourceRecord,
)
from src.importing.sources import SourceLine


def _text(value: Any) -> str | None:
    """Strip NUL characters from a string field; ``None`` if absent."""
    if not isinstance(value, str):
        return None
    return value.replace("\x00", "")


class SampleParser(BaseParser):
    def _is_search(self, raw: dict[str, Any]) -> bool:
        return "search_id" in raw

    def _is_interaction(self, raw: dict[str, Any]) -> bool:
        return "trackId" in raw

    def _parse_search(self, line: SourceLine, raw: dict[str, Any]) -> ParseResult:
        timestamp = parse_timestamp(raw.get("date"))
        if timestamp is None:
            return RejectedRecord(line, "invalid-timestamp")

        search_id = raw.get("search_id")
        query = raw.get("query")
        if not search_id or not query:
            return RejectedRecord(line, "invalid-search-record")

        return SourceRecord(
            line,
            SearchLogEntry(
                external_search_id=str(search_id),
                session_id=_text(raw.get("uid")) or f"anonymous:{search_id}",
                query=_text(str(query)) or "",
                document_ids=tuple(str(doc) for doc in raw.get("serp") or ()),
                timestamp=timestamp,
            ),
        )

    def _parse_interaction(self, line: SourceLine, raw: dict[str, Any]) -> ParseResult:
        timestamp = parse_timestamp(raw.get("date"))
        if timestamp is None:
            return RejectedRecord(line, "invalid-timestamp")

        search_id, separator, document_id = str(raw.get("trackId") or "").rpartition(
            "-"
        )
        if not separator:
            return RejectedRecord(line, "malformed-track-id")
        if search_id == "undefined":
            return RejectedRecord(line, "undefined-search-id")
        if document_id == "undefined":
            return RejectedRecord(line, "undefined-document-id")

        interaction_type = raw.get("type")
        if (
            self.accepted_interaction_types is not None
            and interaction_type not in self.accepted_interaction_types
        ):
            return RejectedRecord(line, "interaction-type-not-accepted")

        return SourceRecord(
            line,
            InteractionLogEntry(
                external_search_id=search_id,
                document_id=document_id,
                timestamp=timestamp,
                interaction_type=interaction_type,
                observed_session_id=_text(raw.get("uid")) or None,
            ),
        )


parser = SampleParser()
