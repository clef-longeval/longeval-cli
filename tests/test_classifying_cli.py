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
    "from src.classifying import Classification, RowClassifier\n"
    "class UpperClassifier(RowClassifier):\n"
    "    name = 'upper'\n"
    "    level = 'queries'\n"
    "    def classify(self, row):\n"
    "        return Classification(label=row.text.upper())\n"
    "classifiers = UpperClassifier()\n"
)

_INTERACTION_CLASSIFIER_SOURCE = (
    "from src.classifying import Classification, RowClassifier\n"
    "class TargetEchoClassifier(RowClassifier):\n"
    "    name = 'target-echo'\n"
    "    level = 'interactions'\n"
    "    def classify(self, row):\n"
    "        return Classification(label=row.type or 'unknown')\n"
    "classifiers = TargetEchoClassifier()\n"
)


def _seed_query(db_url: str) -> None:
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        session.add(Query(id="q0", query="hello"))
        session.commit()


def test_cli_classify_a_query_classifier(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_query(db_url)
    classifier_path = tmp_path / "classifier.py"
    classifier_path.write_text(_QUERY_CLASSIFIER_SOURCE)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "classify",
            "--classifier",
            str(classifier_path),
            "--database-url",
            db_url,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "upper: 1 verdicts (query_classifications)" in result.output
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.get(QueryClassification, ("q0", "upper", "")).label == "HELLO"


def test_cli_classify_an_interaction_classifier(tmp_path):
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
            "classify",
            "--classifier",
            str(classifier_path),
            "--database-url",
            db_url,
        ],
    )

    assert result.exit_code == 0, result.output
    stored = "target-echo: 1 verdicts (interaction_classifications)"
    assert stored in result.output
    with sa.orm.Session(engine) as session:
        key = (interaction_id, "target-echo", "")
        row = session.get(InteractionClassification, key)
        assert row.label == "get-pdf"


def test_cli_classify_runs_a_builtin_via_a_wrapper_file(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_query(db_url)
    wrapper = tmp_path / "word_count.py"
    wrapper.write_text(
        "from src.classifying.builtins import WordCountClassifier\n"
        "classifiers = WordCountClassifier()\n"
    )
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "classify",
            "--classifier",
            str(wrapper),
            "--database-url",
            db_url,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "word-count: 1 verdicts" in result.output
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.get(QueryClassification, ("q0", "word-count", "")).label == "1"


def test_cli_classify_requires_classifier(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed_query(db_url)
    runner = CliRunner()

    result = runner.invoke(
        main, ["classify", "--database-url", db_url]
    )

    assert result.exit_code != 0
    assert "--classifier" in result.output
