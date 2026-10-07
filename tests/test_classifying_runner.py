from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa

from src.classifying import (
    Classification,
    ClassifierConfig,
    InteractionView,
    QueryView,
    RowClassifier,
    classify,
)
from src.database import create_database_engine, create_schema, make_session_factory
from src.filtering import SnapshotConfig
from src.models import (
    Interaction,
    InteractionClassification,
    Query,
    QueryClassification,
    ResultClassification,
    SearchResult,
    SearchSession,
    SessionClassification,
)
from tests.conftest import sqlite_url

T0 = datetime(2025, 3, 10, 12, tzinfo=timezone.utc)


class LengthClassifier(RowClassifier):
    """Labels a query "long"/"short"; no opinion on queries containing "skip"."""

    name = "length"
    level = "queries"

    def classify(self, row: QueryView) -> Classification | None:
        if "skip" in row.text:
            return None
        return Classification("long" if len(row.text) > 5 else "short", 0.5)


class EchoClassifier(RowClassifier):
    """Labels an interaction with its rank and remembers the views it saw."""

    name = "echo"
    level = "interactions"

    def __init__(self) -> None:
        self.seen: list[InteractionView] = []

    def classify(self, row: InteractionView) -> Classification | None:
        self.seen.append(row)
        return Classification(label=str(row.rank))


def _seed_queries(db_url: str, queries: list[str]) -> None:
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        session.add_all(Query(id=f"q{i}", query=text) for i, text in enumerate(queries))
        session.commit()


def _labels(db_url: str, model=QueryClassification) -> dict:
    with sa.orm.Session(create_database_engine(db_url)) as session:
        return {
            (row.classifier, row.snapshot): row.label
            for row in session.scalars(sa.select(model))
        } | {"count": session.scalar(sa.select(sa.func.count()).select_from(model))}


def test_writes_a_verdict_per_query_and_skips_no_opinion(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_queries(db_url, ["hi", "a much longer query", "please skip me"])

    stored = classify(ClassifierConfig(db_url, LengthClassifier(), progress=False))

    assert stored == {"length": 2}
    with sa.orm.Session(create_database_engine(db_url)) as session:
        assert session.get(QueryClassification, ("q0", "length", "")).label == "short"
        assert session.get(QueryClassification, ("q1", "length", "")).label == "long"
        assert session.get(QueryClassification, ("q2", "length", "")) is None


def test_rerunning_upserts_and_a_snapshot_gets_its_own_rows(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_queries(db_url, ["hi"])
    for _ in range(2):
        classify(ClassifierConfig(db_url, LengthClassifier(), progress=False))
    classify(
        ClassifierConfig(
            db_url,
            LengthClassifier(),
            snapshot=SnapshotConfig(snapshot="s1"),
            progress=False,
        )
    )

    assert _labels(db_url)["count"] == 2


class OpinionClassifier(RowClassifier):
    """Labels the queries in ``opinions``; no opinion on the rest."""

    name = "opinion"
    level = "queries"

    def __init__(self, opinions: set[str], fail: bool = False) -> None:
        self.opinions, self.fail = opinions, fail

    def classify(self, row: QueryView) -> Classification | None:
        if self.fail and row.id == "q1":
            raise RuntimeError("boom")
        return Classification("yes") if row.id in self.opinions else None


def _verdict_ids(db_url: str, snapshot: str = "") -> set[str]:
    with sa.orm.Session(create_database_engine(db_url)) as session:
        return set(
            session.scalars(
                sa.select(QueryClassification.query_id).where(
                    QueryClassification.classifier == "opinion",
                    QueryClassification.snapshot == snapshot,
                )
            )
        )


def test_a_complete_rerun_drops_verdicts_it_no_longer_produces(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_queries(db_url, ["a", "b", "c"])
    s1 = SnapshotConfig(snapshot="s1")
    for snapshot in (SnapshotConfig(), s1):
        classify(
            ClassifierConfig(
                db_url, OpinionClassifier({"q0", "q1"}), snapshot, progress=False
            )
        )

    classify(ClassifierConfig(db_url, OpinionClassifier({"q2"}), progress=False))

    assert _verdict_ids(db_url) == {"q2"}
    assert _verdict_ids(db_url, "s1") == {"q0", "q1"}  # other snapshots untouched


def test_a_limited_rerun_is_partial_and_keeps_older_verdicts(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_queries(db_url, ["a", "b", "c"])
    classify(ClassifierConfig(db_url, OpinionClassifier({"q0", "q1"}), progress=False))

    classify(
        ClassifierConfig(db_url, OpinionClassifier({"q2"}), limit=5, progress=False)
    )

    assert _verdict_ids(db_url) == {"q0", "q1", "q2"}


def test_a_failed_rerun_keeps_the_previous_verdicts(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_queries(db_url, ["a", "b", "c"])
    classify(ClassifierConfig(db_url, OpinionClassifier({"q0", "q1"}), progress=False))

    with pytest.raises(RuntimeError, match="boom"):
        classify(
            ClassifierConfig(
                db_url,
                OpinionClassifier({"q2"}, fail=True),
                batch_size=1,
                progress=False,
            )
        )

    assert _verdict_ids(db_url) == {"q0", "q1"}


def test_limit_and_batch_size(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_queries(db_url, ["a", "b", "c", "d", "e"])

    stored = classify(
        ClassifierConfig(
            db_url, [LengthClassifier()], limit=3, batch_size=2, progress=False
        )
    )

    assert stored == {"length": 3}
    assert _labels(db_url)["count"] == 3


def test_interaction_views_carry_rank_and_search_time(tmp_path):
    db_url = sqlite_url(tmp_path)
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        session.add(Query(id="q0", query="x"))
        session.add(
            SearchResult(
                id=1,
                query_id="q0",
                queried_at=T0,
                items={"1": {"docid": "A"}, "2": {"docid": "B"}},
            )
        )
        session.flush()
        session.add(
            Interaction(
                result_id=1,
                document_id="B",
                type="works",
                date=T0 + timedelta(minutes=1),
            )
        )
        session.commit()
    echo = EchoClassifier()

    classify(ClassifierConfig(db_url, echo, progress=False))

    (view,) = echo.seen
    assert (view.rank, view.query_id, view.type) == (2, "q0", "works")
    assert view.searched_at.replace(tzinfo=timezone.utc) == T0
    assert _labels(db_url, InteractionClassification)[("echo", "")] == "2"


def test_an_unknown_level_is_rejected(tmp_path):
    class Bad(LengthClassifier):
        level = "query"

    with pytest.raises(ValueError, match="level must be one of"):
        classify(ClassifierConfig(sqlite_url(tmp_path), Bad(), progress=False))


def test_shows_a_progress_bar_per_classifier(tmp_path, capsys):
    db_url = sqlite_url(tmp_path)
    _seed_queries(db_url, ["hi", "a much longer query"])

    classify(ClassifierConfig(db_url, LengthClassifier()))

    assert "length: 2" in capsys.readouterr().err


def test_flag_sets_label_and_value():
    assert Classification.flag(True) == Classification(label="1", value=1.0)
    assert Classification.flag(False) == Classification(label="0", value=0.0)


@pytest.mark.parametrize(
    "level, verdicts, key",
    [
        ("sessions", SessionClassification, ("s0", "seen", "")),
        ("results", ResultClassification, (1, "seen", "")),
    ],
)
def test_session_and_result_rows_are_labelled_too(tmp_path, level, verdicts, key):
    class Seen(RowClassifier):
        name = "seen"

        def classify(self, row) -> Classification:
            return Classification(label=type(row).__name__)

    Seen.level = level
    db_url = sqlite_url(tmp_path)
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        session.add(SearchSession(id="s0", started_at=T0))
        session.add(SearchResult(id=1, session_id="s0", queried_at=T0, items={}))
        session.commit()

    classify(ClassifierConfig(db_url, Seen(), progress=False))

    with sa.orm.Session(engine) as session:
        assert session.get(verdicts, key).label in {"SessionView", "ResultView"}
