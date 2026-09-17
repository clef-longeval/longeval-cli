from __future__ import annotations

import sqlalchemy as sa
from click.testing import CliRunner

from src.cli import main
from src.database import create_database_engine, create_schema, make_session_factory
from src.models import (
    Interaction,
    InteractionClassification,
    Query,
    QueryClassification,
    SearchResult,
)
from tests.conftest import sqlite_url

_QUERY_CLASSIFIER_SOURCE = (
    "from src.classifying.records import Classification, QueryClassifier\n"
    "class UpperClassifier(QueryClassifier):\n"
    "    name = 'upper'\n"
    "    def classify(self, query):\n"
    "        return Classification(label=query.upper())\n"
    "query_classifier = UpperClassifier()\n"
)

_INTERACTION_CLASSIFIER_SOURCE = (
    "from src.classifying.records import Classification, InteractionClassifier\n"
    "class TargetEchoClassifier(InteractionClassifier):\n"
    "    name = 'target-echo'\n"
    "    def classify(self, interaction):\n"
    "        return Classification(label=interaction.type or 'unknown')\n"
    "interaction_classifier = TargetEchoClassifier()\n"
)


def _seed_query(db_url: str) -> None:
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        session.add(Query(id="q0", query="hello"))
        session.commit()


def test_cli_classify_queries_smoke(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_query(db_url)
    classifier_path = tmp_path / "classifier.py"
    classifier_path.write_text(_QUERY_CLASSIFIER_SOURCE)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "classify-queries",
            "--classifier",
            str(classifier_path),
            "--database-url",
            db_url,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "classified: 1" in result.output
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.get(QueryClassification, ("q0", "upper")).label == "HELLO"


def test_cli_classify_interactions_smoke(tmp_path):
    db_url = sqlite_url(tmp_path)
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        result = SearchResult(items={"1": {"docid": "d1"}})
        session.add(result)
        session.flush()
        interaction = Interaction(result_id=result.id, document_id="d1", type="get-pdf")
        session.add(interaction)
        session.commit()
        interaction_id = interaction.id

    classifier_path = tmp_path / "classifier.py"
    classifier_path.write_text(_INTERACTION_CLASSIFIER_SOURCE)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "classify-interactions",
            "--classifier",
            str(classifier_path),
            "--database-url",
            db_url,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "classified: 1" in result.output
    with sa.orm.Session(engine) as session:
        row = session.get(InteractionClassification, (interaction_id, "target-echo"))
        assert row.label == "get-pdf"


def test_cli_classify_queries_accepts_an_import_spec(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_query(db_url)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "classify-queries",
            "--classifier",
            "src.classifying.builtins:WordCountClassifier",
            "--database-url",
            db_url,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "classified: 1" in result.output
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.get(QueryClassification, ("q0", "word-count")).label == "1"


def test_cli_classify_queries_requires_classifier(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_query(db_url)
    runner = CliRunner()

    result = runner.invoke(
        main, ["classify-queries", "--database-url", db_url]
    )

    assert result.exit_code != 0
    assert "--classifier" in result.output
