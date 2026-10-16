from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.documents.source import BaseDocumentSource, DocumentRecord

UTC_START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_document_record_defaults_to_empty_fields():
    record = DocumentRecord(id="42")

    assert record.title is None
    assert record.published_at is None
    assert record.details == {}


def test_document_record_requires_a_non_empty_string_id():
    with pytest.raises(ValueError):
        DocumentRecord(id="")
    with pytest.raises(ValueError):
        DocumentRecord(id=42)  # type: ignore[arg-type]


def test_document_record_requires_an_aware_published_at():
    with pytest.raises(ValueError, match="timezone-aware"):
        DocumentRecord(id="42", published_at=datetime(2025, 1, 1))

    assert DocumentRecord(id="42", published_at=UTC_START).published_at == UTC_START


def test_base_document_source_hooks_are_not_implemented():
    source = BaseDocumentSource()

    with pytest.raises(NotImplementedError):
        source.fetch_batch(["1"])
    with pytest.raises(NotImplementedError):
        source.iter_published_ids(UTC_START, UTC_START)
    with pytest.raises(NotImplementedError):
        source.negative_candidate_ids(["1"])
    with pytest.raises(NotImplementedError):
        source.iter_ranked_ids("query", 10)
