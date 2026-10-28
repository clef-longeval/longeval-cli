from __future__ import annotations

from datetime import datetime, timezone

import pytest
import sqlalchemy as sa

from src.database import create_database_engine, create_schema, make_session_factory
from src.documents.selecting import select_documents
from src.filtering import SnapshotConfig
from src.models import DocumentCandidate, SnapshotDocument
from tests.conftest import sqlite_url

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
S1 = SnapshotConfig(snapshot="s1")
RELEVANT = ["r1", "r2", "r3", "r4"]


@pytest.fixture
def session_factory(tmp_path):
    """Snapshot s1 with ten candidates per pool: p0..p9 and n0..n9."""
    engine = create_database_engine(sqlite_url(tmp_path))
    create_schema(engine)
    factory = make_session_factory(engine)
    with factory() as session:
        session.add_all(
            DocumentCandidate(
                snapshot="s1",
                reason=reason,
                document_id=f"{reason[0]}{number}",
                created_at=NOW,
            )
            for reason in ("published", "negative")
            for number in range(10)
        )
        session.commit()
    return factory


def _selection(session_factory, snapshot="s1"):
    with session_factory() as session:
        rows = session.execute(
            sa.select(SnapshotDocument.document_id, SnapshotDocument.reason).where(
                SnapshotDocument.snapshot == snapshot
            )
        ).all()
    return dict(rows)


def test_select_documents_samples_each_pool_by_ratio(session_factory):
    report = select_documents(session_factory, S1, RELEVANT, seed=1)

    assert report.selected == {
        "relevant": 4,
        "inherited": 0,
        "published": 2,
        "negative": 2,
    }
    assert report.total == 8
    selection = _selection(session_factory)
    assert {d for d, reason in selection.items() if reason == "relevant"} == set(
        RELEVANT
    )
    assert all(d[0] == "p" for d, r in selection.items() if r == "published")
    assert all(d[0] == "n" for d, r in selection.items() if r == "negative")


def test_select_documents_is_reproducible_and_replaces(session_factory):
    select_documents(session_factory, S1, RELEVANT, seed=1)
    first = _selection(session_factory)

    select_documents(session_factory, S1, RELEVANT, seed=2)
    select_documents(session_factory, S1, RELEVANT, seed=1)

    assert _selection(session_factory) == first
    assert len(first) == 8


def test_select_documents_never_samples_a_selected_document(session_factory):
    relevant = RELEVANT + [f"p{number}" for number in range(10)]

    report = select_documents(session_factory, S1, relevant, seed=1)

    assert (report.selected["published"], report.wanted["published"]) == (0, 7)
    assert (report.selected["negative"], report.wanted["negative"]) == (7, 7)


def test_select_documents_extends_an_earlier_snapshot(session_factory):
    with session_factory() as session:
        session.add_all(
            SnapshotDocument(
                snapshot="s0", document_id=d, reason="relevant", selected_at=NOW
            )
            for d in ("old", "r1")
        )
        session.commit()

    report = select_documents(
        session_factory,
        S1,
        RELEVANT,
        seed=1,
        extends="s0",
        published_ratio=0,
        negative_ratio=0,
    )

    assert _selection(session_factory) == {
        **dict.fromkeys(RELEVANT, "relevant"),
        "old": "inherited",
    }
    assert report.selected["inherited"] == 1


@pytest.mark.parametrize(
    ("config", "relevant", "options", "message"),
    [
        (SnapshotConfig(), RELEVANT, {}, "snapshot"),
        (S1, [], {}, "qrels"),
        (S1, RELEVANT, {"extends": "s1"}, "itself"),
        (S1, RELEVANT, {"extends": "missing"}, "no selected documents"),
        (S1, RELEVANT, {"negative_ratio": -1}, "ratio"),
    ],
)
def test_select_documents_rejects_bad_arguments(
    session_factory, config, relevant, options, message
):
    with pytest.raises(ValueError, match=message):
        select_documents(session_factory, config, relevant, seed=1, **options)

    assert _selection(session_factory) == {}  # nothing written
