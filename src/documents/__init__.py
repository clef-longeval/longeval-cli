"""Document collection: sources, fetching, candidate pools and selection."""

from .candidates import (
    CANDIDATE_REASONS,
    collect_candidates,
    collect_negative_candidates,
    collect_published_candidates,
)
from .fetching import (
    FetchReport,
    fetch_documents,
    find_snapshot_document_ids,
    read_document_ids,
)
from .loader import load_document_source
from .source import BaseDocumentSource, DocumentRecord, DocumentSourceError

__all__ = [
    "CANDIDATE_REASONS",
    "BaseDocumentSource",
    "DocumentRecord",
    "DocumentSourceError",
    "FetchReport",
    "collect_candidates",
    "collect_negative_candidates",
    "collect_published_candidates",
    "fetch_documents",
    "find_snapshot_document_ids",
    "load_document_source",
    "read_document_ids",
]
