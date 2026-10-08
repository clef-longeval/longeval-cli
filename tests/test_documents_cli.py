from __future__ import annotations

import sqlalchemy as sa
from click.testing import CliRunner

from src.cli import main
from src.database import create_database_engine, make_session_factory
from src.models import DocumentFetch
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
