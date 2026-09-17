"""Tests for schema creation."""

from __future__ import annotations

import sqlalchemy as sa
from conftest import sqlite_url

from src.database import create_database_engine, create_schema


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
    } <= tables


def test_create_schema_is_additive(tmp_path):
    engine = create_database_engine(sqlite_url(tmp_path))
    create_schema(engine)
    create_schema(engine)  # must not raise for tables that already exist

    assert "sessions" in sa.inspect(engine).get_table_names()
