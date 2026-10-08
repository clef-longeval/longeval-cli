"""Document collection: sources, fetching, candidate pools and selection."""

from .fetching import (
    FetchReport,
    fetch_documents,
    find_snapshot_document_ids,
    read_document_ids,
)
from .loader import load_document_source
from .source import BaseDocumentSource, DocumentRecord, DocumentSourceError

__all__ = [
    "BaseDocumentSource",
    "DocumentRecord",
    "DocumentSourceError",
    "FetchReport",
    "fetch_documents",
    "find_snapshot_document_ids",
    "load_document_source",
    "read_document_ids",
]
