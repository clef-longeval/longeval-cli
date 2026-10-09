"""Tests for schema creation."""

from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from conftest import sqlite_url

from src.database import (
    create_database_engine,
    create_schema,
    make_session_factory,
    session_scope,
)
from src.models import Document


def test_create_schema_creates_all_tables(tmp_path):
    engine = create_database_engine(sqlite_url(tmp_path))
    create_schema(engine)

    tables = set(sa.inspect(engine).get_table_names())
    assert {
        "sessions",
        "queries",
        "results",
        "interactions",
        "query_classifications",
        "interaction_classifications",
        "documents",
        "document_fetches",
        "document_candidates",
        "snapshot_documents",
        "document_classifications",
    } <= tables


def test_create_schema_is_additive(tmp_path):
    engine = create_database_engine(sqlite_url(tmp_path))
    create_schema(engine)
    create_schema(engine)  # must not raise for tables that already exist

    assert "sessions" in sa.inspect(engine).get_table_names()


def test_document_keeps_source_details(tmp_path):
    engine = create_database_engine(sqlite_url(tmp_path))
    create_schema(engine)
    session_factory = make_session_factory(engine)
    details = {"doi": "10.1/x", "authors": [{"name": "A. Author"}]}

    with session_scope(session_factory) as session:
        session.add(
            Document(
                id="42",
                title="A title",
                details=details,
                fetched_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
        )
    with session_scope(session_factory) as session:
        document = session.get(Document, "42")

    assert document is not None
    assert document.details == details
    assert document.abstract is None
