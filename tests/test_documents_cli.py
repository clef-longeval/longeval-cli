from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from click.testing import CliRunner

from src.cli import main
from src.database import create_database_engine, create_schema, make_session_factory
from src.models import DocumentCandidate, DocumentFetch
from tests.conftest import sqlite_url

_EVEN_SOURCE = (
    "from src.documents import BaseDocumentSource, DocumentRecord\n"
    "class EvenSource(BaseDocumentSource):\n"
    "    def fetch_batch(self, document_ids):\n"
    "        return [DocumentRecord(id=i) for i in document_ids if int(i) % 2 == 0]\n"
    "source = EvenSource()\n"
)


def _write_source(tmp_path):
    path = tmp_path / "source.py"
    path.write_text(_EVEN_SOURCE)
    return path


def test_cli_fetch_documents_from_an_ids_file(tmp_path):
    db_url = sqlite_url(tmp_path)
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("1\n2\n3\n4\n")

    result = CliRunner().invoke(
        main,
        [
            "fetch-documents",
            "--source",
            str(_write_source(tmp_path)),
            "--ids",
            str(ids_path),
            "--database-url",
            db_url,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "2  fetched" in result.output
    with make_session_factory(create_database_engine(db_url))() as session:
        rows = session.execute(
            sa.select(DocumentFetch.document_id, DocumentFetch.status)
        ).all()
    assert dict(rows) == {"1": "not_found", "2": "ok", "3": "not_found", "4": "ok"}


def test_cli_fetch_documents_retries_not_found_ids(tmp_path):
    db_url = sqlite_url(tmp_path)
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("1\n2\n")
    every_path = tmp_path / "every_source.py"
    every_path.write_text(
        "from src.documents import BaseDocumentSource, DocumentRecord\n"
        "class EverySource(BaseDocumentSource):\n"
        "    def fetch_batch(self, document_ids):\n"
        "        return [DocumentRecord(id=i) for i in document_ids]\n"
        "source = EverySource()\n"
    )

    def fetch(source_path, *extra):
        return CliRunner().invoke(
            main,
            [
                "fetch-documents",
                "--source",
                str(source_path),
                "--ids",
                str(ids_path),
                "--database-url",
                db_url,
                *extra,
            ],
        )

    fetch(_write_source(tmp_path))  # "1" is not found
    result = fetch(every_path, "--retry-not-found")

    assert result.exit_code == 0, result.output
    assert "1  fetched" in result.output
    with make_session_factory(create_database_engine(db_url))() as session:
        rows = session.execute(
            sa.select(DocumentFetch.document_id, DocumentFetch.status)
        ).all()
    assert dict(rows) == {"1": "ok", "2": "ok"}


def test_cli_fetch_documents_needs_ids(tmp_path):
    result = CliRunner().invoke(
        main,
        [
            "fetch-documents",
            "--source",
            str(_write_source(tmp_path)),
            "--database-url",
            sqlite_url(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "select-documents" in result.output


def test_cli_fetch_documents_rejects_snapshot_with_ids(tmp_path):
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("1\n")

    result = CliRunner().invoke(
        main,
        [
            "fetch-documents",
            "--source",
            str(_write_source(tmp_path)),
            "--ids",
            str(ids_path),
            "--snapshot",
            "s1",
            "--database-url",
            sqlite_url(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "can't be combined" in result.output


_LISTING_SOURCE = (
    "from src.documents import BaseDocumentSource\n"
    "class ListingSource(BaseDocumentSource):\n"
    "    def fetch_batch(self, document_ids):\n"
    "        return []\n"
    "    def iter_published_ids(self, start, end):\n"
    "        yield from ['7', '8']\n"
    "source = ListingSource()\n"
)
_FILTER = (
    "snapshot: s1\n"
    "from: 2025-03-01\n"
    "to: 2025-06-01\n"
    "click_types: [click]\n"
    "published_from: 2024-12-01\n"
    "published_to: 2025-06-01\n"
)


def _collect(tmp_path, *options):
    source_path = tmp_path / "listing_source.py"
    source_path.write_text(_LISTING_SOURCE)
    filter_path = tmp_path / "s1.yaml"
    filter_path.write_text(_FILTER)
    return CliRunner().invoke(
        main,
        [
            "collect-document-candidates",
            str(filter_path),
            "--source",
            str(source_path),
            "--database-url",
            sqlite_url(tmp_path),
            *options,
        ],
    )


def test_cli_collect_published_document_candidates(tmp_path):
    result = _collect(tmp_path, "--only", "published")

    assert result.exit_code == 0, result.output
    assert "2  published candidates" in result.output


def test_cli_collect_negative_candidates_needs_qrels(tmp_path):
    result = _collect(tmp_path)

    assert result.exit_code != 0
    assert "qrels" in result.output

def test_cli_select_documents(tmp_path):
    db_url = sqlite_url(tmp_path)
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        session.add_all(
            DocumentCandidate(
                snapshot="s1",
                reason="published",
                document_id=f"p{number}",
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
            for number in range(4)
        )
        session.commit()
    qrels_path = tmp_path / "qrels.trec"
    qrels_path.write_text("q1 0 d1 1\nq1 0 d2 0\n")
    filter_path = tmp_path / "s1.yaml"
    filter_path.write_text(_FILTER)

    result = CliRunner().invoke(
        main,
        [
            "select-documents",
            str(filter_path),
            "--qrels",
            str(qrels_path),
            "--database-url",
            db_url,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "2  relevant" in result.output
    assert "1  published" in result.output
    assert "0  negative (of 1 wanted" in result.output
    assert "Selected 3 documents for snapshot s1" in result.output
