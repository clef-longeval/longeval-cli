from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.documents.loader import load_document_source_module

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "longeval-sci-v2"
CORE = load_document_source_module(EXAMPLE / "core_source.py")
LEGACY = load_document_source_module(EXAMPLE / "legacy_sqlite_source.py")

LEGACY_SCHEMA = """
CREATE TABLE works (
  id TEXT PRIMARY KEY, title TEXT, abstract TEXT, createdDate TEXT,
  updatedDate TEXT, publishedDate TEXT, doi TEXT, arxivId TEXT,
  pubmedId TEXT, magId TEXT, fullText TEXT
);
CREATE TABLE authors (
  author_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE
);
CREATE TABLE work_authors (
  work_id TEXT NOT NULL, author_id INTEGER NOT NULL, author_pos INTEGER NOT NULL
);
CREATE TABLE oai_ids (work_id TEXT NOT NULL, oai_id TEXT NOT NULL);
CREATE TABLE links (work_id TEXT NOT NULL, type TEXT, url TEXT NOT NULL);
"""


@pytest.fixture
def legacy_db(tmp_path):
    path = tmp_path / "core_works.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(LEGACY_SCHEMA)
        connection.execute(
            "INSERT INTO works (id, title, abstract, publishedDate, fullText) "
            "VALUES ('42', 'T', 'A', '2025-04-01T00:00:00', 'F')"
        )
        connection.executemany(
            "INSERT INTO authors (author_id, name) VALUES (?, ?)",
            [(1, "Ada"), (2, "Bob")],
        )
        connection.executemany(
            "INSERT INTO work_authors VALUES (?, ?, ?)", [("42", 2, 1), ("42", 1, 0)]
        )
        connection.execute("INSERT INTO oai_ids VALUES ('42', 'oai:x:1')")
        connection.execute("INSERT INTO links VALUES ('42', 'download', 'https://x')")
        connection.commit()
    return path


def test_legacy_source_reads_works_in_the_api_shape(legacy_db):
    records = LEGACY.LegacySqliteSource(legacy_db).fetch_batch(["42", "43"])

    assert [record.id for record in records] == ["42"]
    record = records[0]
    assert record.full_text == "F"
    assert record.published_at == datetime(2025, 4, 1, tzinfo=timezone.utc)
    assert record.details["authors"] == [{"name": "Ada"}, {"name": "Bob"}]
    assert record.details["oaiIds"] == ["oai:x:1"]
    assert record.details["links"] == [{"type": "download", "url": "https://x"}]
    assert set(record.details) == set(CORE.make_core_record({"id": 1}).details)


def test_legacy_source_needs_a_database(monkeypatch, tmp_path):
    monkeypatch.delenv("LEGACY_CORE_DB", raising=False)

    with pytest.raises(ValueError, match="LEGACY_CORE_DB"):
        LEGACY.LegacySqliteSource().fetch_batch(["1"])
    with pytest.raises(ValueError, match="not found"):
        LEGACY.LegacySqliteSource(tmp_path / "missing.sqlite").fetch_batch(["1"])
