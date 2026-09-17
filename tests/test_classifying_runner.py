from __future__ import annotations

import sqlalchemy as sa

from src.classifying.records import (
    Classification,
    InteractionClassifier,
    QueryClassifier,
)
from src.classifying.runner import (
    ClassifierConfig,
    classify_interactions,
    classify_queries,
)
from src.database import create_database_engine, create_schema, make_session_factory
from src.models import (
    Interaction,
    InteractionClassification,
    Query,
    QueryClassification,
    SearchResult,
)
from tests.conftest import sqlite_url


class LengthClassifier(QueryClassifier):
    """Labels a query "long"/"short"; ignores queries containing "skip"."""

    name = "length"

    def classify(self, query: str) -> Classification | None:
        if "skip" in query:
            return None
        label = "long" if len(query) > 5 else "short"
        return Classification(label=label, confidence=0.5)


class TargetEchoClassifier(InteractionClassifier):
    """Labels an interaction with its click target, unless it has none."""

    name = "target-echo"

    def classify(self, interaction) -> Classification | None:
        if interaction.type is None:
            return None
        return Classification(label=interaction.type)


def _seed_queries(db_url: str, queries: list[str]) -> None:
    engine = create_database_engine(db_url)
    create_schema(engine)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        for i, text in enumerate(queries):
            session.add(Query(id=f"q{i}", query=text))
        session.commit()


def _seed_interaction(db_url: str, *, type_: str | None) -> int:
    engine = create_database_engine(db_url)
    create_schema(engine)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        result = SearchResult(items={"1": {"docid": "d1"}})
        session.add(result)
        session.flush()
        interaction = Interaction(result_id=result.id, document_id="d1", type=type_)
        session.add(interaction)
        session.commit()
        return interaction.id


def test_classify_queries_writes_a_row_per_query(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_queries(db_url, ["hi", "a much longer query"])

    stats = classify_queries(ClassifierConfig(db_url, LengthClassifier()))

    assert stats.considered == 2
    assert stats.classified == 2
    assert stats.unlabeled == 0
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.get(QueryClassification, ("q0", "length")).label == "short"
        assert session.get(QueryClassification, ("q1", "length")).label == "long"


def test_classify_queries_skips_queries_the_classifier_declines(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_queries(db_url, ["please skip me"])

    stats = classify_queries(ClassifierConfig(db_url, LengthClassifier()))

    assert stats.unlabeled == 1
    assert stats.classified == 0
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.query(QueryClassification).count() == 0


def test_rerunning_a_classifier_upserts_instead_of_duplicating(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_queries(db_url, ["hi"])
    classify_queries(ClassifierConfig(db_url, LengthClassifier()))

    stats = classify_queries(ClassifierConfig(db_url, LengthClassifier()))

    assert stats.classified == 1
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.query(QueryClassification).count() == 1


def test_classify_queries_respects_limit(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_queries(db_url, ["a", "b", "c"])

    stats = classify_queries(ClassifierConfig(db_url, LengthClassifier(), limit=2))

    assert stats.considered == 2


def test_classify_interactions_writes_a_row_keyed_by_interaction_and_classifier(
    tmp_path,
):
    db_url = sqlite_url(tmp_path)
    interaction_id = _seed_interaction(db_url, type_="get-pdf")

    stats = classify_interactions(ClassifierConfig(db_url, TargetEchoClassifier()))

    assert stats.classified == 1
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        row = session.get(
            InteractionClassification, (interaction_id, "target-echo")
        )
        assert row.label == "get-pdf"


def test_classify_interactions_skips_untyped_interactions(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_interaction(db_url, type_=None)

    stats = classify_interactions(ClassifierConfig(db_url, TargetEchoClassifier()))

    assert stats.unlabeled == 1
    assert stats.classified == 0
