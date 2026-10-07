"""Click-model training data: query sessions from the database or a frozen export."""

from .db_source import iter_query_sessions_from_db
from .file_source import iter_query_sessions_from_file, read_manifest
from .query_doc_ids import QueryDocIdMap
from .query_session import (
    FORMAT_VERSION,
    QuerySession,
    QuerySessionFormatError,
    QuerySessionStats,
)

__all__ = [
    "FORMAT_VERSION",
    "QuerySession",
    "QuerySessionFormatError",
    "QuerySessionStats",
    "QueryDocIdMap",
    "iter_query_sessions_from_file",
    "iter_query_sessions_from_db",
    "read_manifest",
]
