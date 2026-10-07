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


def clean_text(value: Any) -> str | None:
    """Strip NUL characters from a raw log string field; ``None`` if absent."""
    if not isinstance(value, str):
        return None
    return value.replace("\x00", "")


class LongEvalSciParser(BaseParser):
    """Parser for the LongEval-Sci frontend log format.

    A search event has a ``search_id`` key; a click/interaction event has a
    ``trackId`` key instead.
    """

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

        uid = clean_text(raw.get("uid"))
        session_id = uid or f"anonymous:{search_id}"
        document_ids = tuple(str(doc_id) for doc_id in raw.get("serp") or ())

        return SourceRecord(
            line,
            SearchLogEntry(
                external_search_id=str(search_id),
                session_id=session_id,
                query=clean_text(str(query)) or "",
                document_ids=document_ids,
                timestamp=timestamp,
            ),
        )

    def _parse_interaction(self, line: SourceLine, raw: dict[str, Any]) -> ParseResult:
        timestamp = parse_timestamp(raw.get("date"))
        if timestamp is None:
            return RejectedRecord(line, "invalid-timestamp")

        track_id = str(raw.get("trackId") or "")
        search_id, separator, document_id = track_id.rpartition("-")
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
                observed_session_id=clean_text(raw.get("uid")) or None,
            ),
        )


parser = LongEvalSciParser()
