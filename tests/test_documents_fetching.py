from __future__ import annotations

import gzip
from datetime import datetime, timezone

import sqlalchemy as sa

from src.database import create_database_engine, create_schema, make_session_factory
from src.documents.fetching import (
    fetch_documents,
    find_snapshot_document_ids,
    read_document_ids,
)
from src.documents.source import (
    BaseDocumentSource,
    DocumentRecord,
    DocumentSourceError,
)
from src.models import Document, DocumentFetch, SnapshotDocument
from tests.conftest import sqlite_url

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


class FakeSource(BaseDocumentSource):
    batch_size = 2

    def __init__(self, records, failing_ids=()):
        self.record_map = {record.id: record for record in records}
        self.failing_ids = set(failing_ids)

    def fetch_batch(self, document_ids):
        if self.failing_ids & set(document_ids):
            raise DocumentSourceError("service unavailable")
        return [self.record_map[i] for i in document_ids if i in self.record_map]


class ExtraRecordSource(BaseDocumentSource):
    def fetch_batch(self, document_ids):
        return [DocumentRecord(id=i) for i in document_ids] + [
            DocumentRecord(id="extra")
        ]


def _session_factory(tmp_path):
    engine = create_database_engine(sqlite_url(tmp_path))
    create_schema(engine)
    return make_session_factory(engine)


def _statuses(session_factory):
    with session_factory() as session:
        rows = session.execute(
            sa.select(DocumentFetch.document_id, DocumentFetch.status)
        ).all()
    return dict(rows)


def test_fetch_documents_stores_documents_and_missing_ids(tmp_path):
    session_factory = _session_factory(tmp_path)
    source = FakeSource(
        [
            DocumentRecord(
                id="1", title="One", published_at=NOW, details={"doi": "10.1/one"}
            ),
            DocumentRecord(id="3", title="Three"),
        ]
    )

    report = fetch_documents(session_factory, source, ["1", "2", "3"])

    assert (report.requested, report.fetched, report.not_found) == (3, 2, 1)
    assert report.failed == 0
    assert _statuses(session_factory) == {"1": "ok", "2": "not_found", "3": "ok"}
    with session_factory() as session:
        document = session.get(Document, "1")
    assert document.title == "One"
    assert document.details == {"doi": "10.1/one"}
    assert document.published_at.date() == NOW.date()


def test_fetch_documents_skips_settled_ids_and_retries_errors(tmp_path):
    session_factory = _session_factory(tmp_path)
    records = [DocumentRecord(id="1"), DocumentRecord(id="3")]
    ids = ["1", "2", "3"]  # batches: [1, 2] and [3]

    first = fetch_documents(
        session_factory, FakeSource(records, failing_ids={"3"}), ids
    )
    assert _statuses(session_factory)["3"] == "error"
    second = fetch_documents(session_factory, FakeSource(records), ids)

    assert (first.fetched, first.not_found, first.failed) == (1, 1, 1)
    assert (second.skipped, second.fetched, second.failed) == (2, 1, 0)
    assert _statuses(session_factory) == {"1": "ok", "2": "not_found", "3": "ok"}


def test_fetch_documents_drops_too_long_full_texts(tmp_path):
    session_factory = _session_factory(tmp_path)
    source = FakeSource(
        [
            DocumentRecord(id="1", full_text="x" * 11),
            DocumentRecord(id="2", full_text="short"),
        ]
    )

    report = fetch_documents(
        session_factory, source, ["1", "2"], full_text_max_chars=10
    )

    assert report.full_texts_dropped == 1
    with session_factory() as session:
        assert session.get(Document, "1").full_text is None
        assert session.get(Document, "2").full_text == "short"


def test_fetch_documents_in_parallel_fetches_each_id_once(tmp_path):
    session_factory = _session_factory(tmp_path)
    ids = [str(number) for number in range(25)]
    source = FakeSource([DocumentRecord(id=i) for i in ids])

    report = fetch_documents(session_factory, source, ids + ids[:5], max_workers=3)

    assert (report.requested, report.fetched) == (25, 25)
    assert set(_statuses(session_factory).values()) == {"ok"}


def test_fetch_documents_ignores_unrequested_records(tmp_path):
    session_factory = _session_factory(tmp_path)

    fetch_documents(session_factory, ExtraRecordSource(), ["1"])

    assert _statuses(session_factory) == {"1": "ok"}


def test_read_document_ids_skips_blank_lines_and_repeats(tmp_path):
    plain = tmp_path / "ids.txt"
    plain.write_text("2\n\n1\n2\n")
    packed = tmp_path / "ids.txt.gz"
    with gzip.open(packed, "wt", encoding="utf-8") as handle:
        handle.write("2\n\n1\n2\n")

    assert read_document_ids(plain) == ["2", "1"]
    assert read_document_ids(packed) == ["2", "1"]


def test_find_snapshot_document_ids(tmp_path):
    session_factory = _session_factory(tmp_path)
    with session_factory() as session:
        session.add_all(
            [
                SnapshotDocument(
                    snapshot="s1", document_id="b", reason="relevant", selected_at=NOW
                ),
                SnapshotDocument(
                    snapshot="s2", document_id="b", reason="inherited", selected_at=NOW
                ),
                SnapshotDocument(
                    snapshot="s2", document_id="a", reason="negative", selected_at=NOW
                ),
            ]
        )
        session.commit()

        assert find_snapshot_document_ids(session, "s1") == ["b"]
        assert find_snapshot_document_ids(session) == ["a", "b"]
        

def test_fetch_documents_can_retry_not_found_ids(tmp_path):
    session_factory = _session_factory(tmp_path)
    fetch_documents(session_factory, FakeSource([]), ["1"])

    skipped = fetch_documents(
        session_factory, FakeSource([DocumentRecord(id="1")]), ["1"]
    )
    retried = fetch_documents(
        session_factory,
        FakeSource([DocumentRecord(id="1")]),
        ["1"],
        retry_not_found=True,
    )

    assert (skipped.skipped, retried.fetched) == (1, 1)
    assert _statuses(session_factory) == {"1": "ok"}
