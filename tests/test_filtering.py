from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.classifying import AbstractLengthClassifier, ClassifierConfig, classify
from src.database import create_database_engine, create_schema, make_session_factory
from src.filtering import (
    FilterError,
    document_filter_report,
    filter_documents,
    filter_queries,
    filter_report,
    load_snapshot_config,
)
from src.models import Document, Query, QueryClassification, SnapshotDocument
from tests.conftest import sqlite_url

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def session(tmp_path):
    """Four queries; is-boolean/manual are global, clicks is scoped.

    q0: 0, s1=40    q1: 1, s1=50
    q2: 0, manual=remove, s2=35    q3: 0, s1=5
    """
    engine = create_database_engine(sqlite_url(tmp_path))
    create_schema(engine)
    verdicts = [
        ("q0", "is-boolean", "", "0", None),
        ("q1", "is-boolean", "", "1", None),
        ("q2", "is-boolean", "", "0", None),
        ("q3", "is-boolean", "", "0", None),
        ("q2", "manual", "", "remove", None),
        ("q0", "clicks", "s1", "40", 40.0),
        ("q1", "clicks", "s1", "50", 50.0),
        ("q3", "clicks", "s1", "5", 5.0),
        ("q2", "clicks", "s2", "35", 35.0),
    ]
    with make_session_factory(engine)() as session:
        session.add_all([Query(id=f"q{i}", query=f"query {i}") for i in range(4)])
        session.add_all(
            QueryClassification(
                query_id=q, classifier=c, snapshot=sc, label=label, value=v,
                created_at=NOW,
            )
            for q, c, sc, label, v in verdicts
        )
        session.commit()
        yield session


def _filter(tmp_path, text: str):
    path = tmp_path / "criteria.yaml"
    path.write_text(text)
    return load_snapshot_config(path)


def _ids(session, criteria) -> list[str]:
    return [q for q, _ in filter_queries(session, criteria)]


def test_require_min_and_exclude_within_one_scope(session, tmp_path):
    criteria = _filter(
        tmp_path,
        "snapshot: s1\n"
        "queries:\n"
        "  require: {is-boolean: 0}\n"
        "  min: {clicks: 31}\n"
        "  exclude: {manual: remove}\n",
    )
    assert _ids(session, criteria) == ["q0"]


def test_other_scopes_verdicts_are_ignored(session, tmp_path):
    criteria = _filter(tmp_path, "snapshot: s1\nqueries: {min: {clicks: 31}}\n")
    # q2's 35 clicks are in s2, so it doesn't count here.
    assert _ids(session, criteria) == ["q0", "q1"]


def test_exclude_keeps_queries_without_a_verdict(session, tmp_path):
    criteria = _filter(tmp_path, "queries: {exclude: {manual: remove}}\n")
    assert _ids(session, criteria) == ["q0", "q1", "q3"]


def test_require_drops_queries_without_a_verdict(session, tmp_path):
    criteria = _filter(tmp_path, "queries: {require: {manual: [remove, keep]}}\n")
    assert _ids(session, criteria) == ["q2"]


def test_report_counts_after_each_rule(session, tmp_path):
    criteria = _filter(
        tmp_path,
        "snapshot: s1\nqueries: {require: {is-boolean: 0}, min: {clicks: 31}}\n",
    )

    report = filter_report(session, criteria)

    assert report.total == 4
    assert [n for _, n in report.steps] == [3, 1]
    assert report.kept == 1


@pytest.mark.parametrize(
    "text, message",
    [
        ("requir: {a: b}\n", "unknown key"),
        ("require: {a: b}\n", "unknown key"),  # rules belong to a level
        ("sessions: {requir: {a: b}}\n", "unknown key.* in `sessions`"),
        ("sessions: [a]\n", "require/min/exclude"),
        ("from: yesterday\n", "not a date"),
        ("from: 2025-06-01\nto: 2025-03-01\n", "before"),
        ("queries: {min: {clicks: lots}}\n", "number"),
        ("queries: {require: [a]}\n", "classifier -> label"),
    ],
)
def test_invalid_filter_files_are_rejected(tmp_path, text, message):
    with pytest.raises(FilterError, match=message):
        _filter(tmp_path, text)


def test_numeric_labels_are_read_as_strings(tmp_path):
    criteria = _filter(tmp_path, "queries: {require: {word-count: [2, 3]}}\n")
    assert criteria.queries.require["word-count"] == ("2", "3")

def test_snapshot_range_is_read_as_utc(tmp_path):
    criteria = _filter(
        tmp_path,
        "snapshot: s1\nfrom: 2025-03-01\nto: 2025-06-01T12:00:00+02:00\n",
    )
    assert criteria.start == datetime(2025, 3, 1, tzinfo=timezone.utc)
    assert criteria.end == datetime(2025, 6, 1, 10, tzinfo=timezone.utc)
    assert _filter(tmp_path, "snapshot: s1\n").start is None  # no rules needed

def test_published_window_is_read_as_utc(tmp_path):
    criteria = _filter(
        tmp_path, "published_from: 2024-12-01\npublished_to: 2025-06-01\n"
    )

    assert criteria.published_start == datetime(2024, 12, 1, tzinfo=timezone.utc)
    assert criteria.published_end == datetime(2025, 6, 1, tzinfo=timezone.utc)


def test_published_from_must_be_before_published_to(tmp_path):
    with pytest.raises(FilterError, match="published_from"):
        _filter(tmp_path, "published_from: 2025-06-01\npublished_to: 2025-03-01\n")


def _documents_session_factory(tmp_path):
    """Snapshot s1 selects five documents; one of them was never fetched."""
    db_url = sqlite_url(tmp_path, "documents.sqlite3")
    engine = create_database_engine(db_url)
    create_schema(engine)
    factory = make_session_factory(engine)
    long_abstract = "x" * 100
    with factory() as session:
        session.add_all(
            [
                Document(
                    id="ok",
                    abstract=long_abstract,
                    published_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    fetched_at=NOW,
                ),
                Document(id="short", abstract="tiny", fetched_at=NOW),
                Document(
                    id="new",
                    abstract=long_abstract,
                    published_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
                    fetched_at=NOW,
                ),
                Document(id="undated", abstract=long_abstract, fetched_at=NOW),
            ]
        )
        session.add_all(
            SnapshotDocument(
                snapshot="s1", document_id=d, reason="relevant", selected_at=NOW
            )
            for d in ("ok", "short", "new", "undated", "unfetched")
        )
        session.commit()
    classify(
        ClassifierConfig(
            database_url=db_url,
            classifiers=AbstractLengthClassifier(),
            progress=False,
        )
    )
    return factory


def test_filter_documents_applies_rules_and_the_publication_cutoff(tmp_path):
    factory = _documents_session_factory(tmp_path)
    criteria = _filter(
        tmp_path,
        "snapshot: s1\n"
        "published_to: 2025-06-01\n"
        "documents:\n"
        "  exclude: {abstract-length: short}\n",
    )

    with factory() as session:
        assert filter_documents(session, criteria) == ["ok", "undated"]
        report = document_filter_report(session, criteria)

    assert report.total == 5
    assert report.steps == [
        ("fetched", 4),
        ("exclude abstract-length in ['short']", 3),
        ("published before 2025-06-01T00:00:00+00:00", 2),
    ]


def test_filter_documents_needs_a_snapshot_name(tmp_path):
    factory = _documents_session_factory(tmp_path)

    with factory() as session, pytest.raises(FilterError, match="snapshot"):
        filter_documents(session, _filter(tmp_path, "published_to: 2025-06-01\n"))
