"""Document collection: sources, fetching, candidate pools and selection."""

from .loader import load_document_source
from .source import BaseDocumentSource, DocumentRecord, DocumentSourceError

__all__ = [
    "BaseDocumentSource",
    "DocumentRecord",
    "DocumentSourceError",
    "load_document_source",
]
