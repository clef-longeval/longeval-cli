"""Base class for document sources — the "configurable source" extension point.

Nothing here knows about any particular document provider. Subclass
:class:`BaseDocumentSource` in a source file to support one:

- ``fetch_batch`` (required) — fetch up to ``batch_size`` documents by id.
- ``iter_published_ids`` (optional) — list the ids of documents published in
  a time window, for a snapshot's ``"published"`` candidates.
- ``negative_candidate_ids`` (optional) — derive ids of non-relevant documents
  from relevant ones, for a snapshot's ``"negative"`` candidates; without it,
  negatives are sampled at random from the documents the logs already know.

Batching, parallel requests, recording outcomes and storage are shared and
live in ``fetching.py``; a source only talks to its provider.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class DocumentSourceError(RuntimeError):
    """A source request failed (network error, bad response, rate limit, ...)."""


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """One document as a source returns it, ready to store as a ``Document``.

    ``id`` must be the id the logs use for the document. ``published_at`` must
    be timezone-aware. ``details`` holds every other field the source provides
    (authors, DOI, links, ...); it is stored and released unchanged.
    """

    id: str
    title: str | None = None
    abstract: str | None = None
    full_text: str | None = None
    published_at: datetime | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(f"document id must be a non-empty string: {self.id!r}")
        if self.published_at is not None and self.published_at.tzinfo is None:
            raise ValueError(
                f"document {self.id!r}: published_at must be timezone-aware"
            )


class BaseDocumentSource:
    """Shared interface of a document source (see the module docstring)."""

    #: The most ids one ``fetch_batch`` call is given.
    batch_size: int = 10

    def fetch_batch(self, document_ids: list[str]) -> list[DocumentRecord]:
        """Return the documents the provider has for ``document_ids``.

        An id without a record in the result counts as not found, which is
        final. Raise :class:`DocumentSourceError` when the request itself
        failed: the whole batch is then recorded as an error and retried on
        the next run.
        """
        raise NotImplementedError

    def iter_published_ids(self, start: datetime, end: datetime) -> Iterator[str]:
        """Yield the ids of documents published in ``[start, end)``."""
        raise NotImplementedError

    def negative_candidate_ids(self, relevant_ids: Iterable[str]) -> Iterator[str]:
        """Yield ids of documents unlikely to be relevant, from ``relevant_ids``."""
        raise NotImplementedError
