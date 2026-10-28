from __future__ import annotations

from datetime import datetime, timezone

import pytest
import sqlalchemy as sa

from src.database import create_database_engine, create_schema, make_session_factory
from src.documents.candidates import (
    collect_candidates,
    collect_negative_candidates,
    collect_published_candidates,
)
from src.documents.source import BaseDocumentSource
from src.filtering import FilterError, SnapshotConfig
from src.models import DocumentCandidate, Query, SearchResult, SearchSession
from tests.conftest import sqlite_url

START = datetime(2025, 3, 1, tzinfo=timezone.utc)
END = datetime(2025, 6, 1, tzinfo=timezone.utc)
CONFIG = SnapshotConfig(
    snapshot="s1",
    start=START,
    end=END,
    click_types=("click",),
    published_start=START,
    published_end=END,
)


class ListingSource(BaseDocumentSource):
    """Lists fixed published ids; derives negatives as id + 1, like CORE."""

    def __init__(self, published_ids):
        self.published_ids = published_ids
        self.windows = []

    def fetch_batch(self, document_ids):
        return []

    def iter_published_ids(self, start, end):
        self.windows.append((start, end))
        yield from self.published_ids

    def negative_candidate_ids(self, relevant_ids):
        for document_id in relevant_ids:
            yield str(int(document_id) + 1)


class FetchOnlySource(BaseDocumentSource):
    def fetch_batch(self, document_ids):
        return []


@pytest.fixture
def session_factory(tmp_path):
    engine = create_database_engine(sqlite_url(tmp_path))
    create_schema(engine)
    return make_session_factory(engine)


def _candidates(session_factory, reason):
    with session_factory() as session:
        return set(
            session.scalars(
                sa.select(DocumentCandidate.document_id).where(
                    DocumentCandidate.snapshot == "s1",
                    DocumentCandidate.reason == reason,
                )
            )
        )


def test_published_candidates_are_listed_for_the_window_and_replaced(
    session_factory,
):
    source = ListingSource([7, "8", "8"])

    assert collect_published_candidates(session_factory, source, CONFIG) == 2
    assert source.windows == [(START, END)]
    assert _candidates(session_factory, "published") == {"7", "8"}

    collect_published_candidates(session_factory, ListingSource(["9"]), CONFIG)
    assert _candidates(session_factory, "published") == {"9"}


def test_published_candidates_need_a_window_and_the_hook(session_factory):
    with pytest.raises(FilterError, match="published_from"):
        collect_published_candidates(
            session_factory, ListingSource([]), SnapshotConfig(snapshot="s1")
        )
    with pytest.raises(ValueError, match="iter_published_ids"):
        collect_published_candidates(session_factory, FetchOnlySource(), CONFIG)


def test_candidates_need_a_snapshot_name(session_factory):
    with pytest.raises(FilterError, match="snapshot"):
        collect_negative_candidates(
            session_factory, FetchOnlySource(), SnapshotConfig(), []
        )


def test_negative_candidates_from_the_source_skip_relevant_ids(session_factory):
    count = collect_negative_candidates(
        session_factory, ListingSource([]), CONFIG, ["1", "2", "5"]
    )

    assert count == 2
    assert _candidates(session_factory, "negative") == {"3", "6"}


def test_negative_candidates_fall_back_to_displayed_documents(session_factory):
    with session_factory() as session:
        session.add(Query(id="q1", query="query"))
        session.add(SearchSession(id="u1", started_at=START))
        session.add_all(
            [
                SearchResult(
                    id=1,
                    session_id="u1",
                    query_id="q1",
                    queried_at=datetime(2025, 4, 1, tzinfo=timezone.utc),
                    items={"1": {"docid": "a"}, "2": {"docid": "b"}},
                ),
                SearchResult(  # after the snapshot's searches
                    id=2,
                    session_id="u1",
                    query_id="q1",
                    queried_at=datetime(2025, 7, 1, tzinfo=timezone.utc),
                    items={"1": {"docid": "late"}},
                ),
            ]
        )
        session.commit()

    collect_negative_candidates(session_factory, FetchOnlySource(), CONFIG, ["b"])

    assert _candidates(session_factory, "negative") == {"a"}


def test_collect_candidates_checks_arguments_before_collecting(session_factory):
    source = ListingSource(["7"])

    with pytest.raises(ValueError, match="qrels"):
        collect_candidates(session_factory, source, CONFIG)
    with pytest.raises(ValueError, match="unknown"):
        collect_candidates(
            session_factory, source, CONFIG, reasons=("random",), relevant_ids=[]
        )
    assert source.windows == []  # nothing was listed


def test_collect_candidates_collects_each_pool(session_factory):
    counts = collect_candidates(
        session_factory, ListingSource(["7"]), CONFIG, relevant_ids=["1"]
    )

    assert counts == {"published": 1, "negative": 1}
