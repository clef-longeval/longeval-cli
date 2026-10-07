from __future__ import annotations

import csv
from datetime import datetime, timezone

import pytest
from click.testing import CliRunner

from src.classifying.builtins import ManualLabelsClassifier
from src.classifying.runner import ClassifierConfig, classify
from src.cli import main
from src.database import create_database_engine, create_schema, make_session_factory
from src.models import Query, QueryClassification
from tests.conftest import sqlite_url

_TRICKY = 'tab\tand\nnewline "quoted"'


def _seed(db_url: str) -> None:
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        session.add_all(
            [
                Query(id="q0", query="plain query"),
                Query(id="q1", query="boolean AND query"),
                Query(id="q2", query=_TRICKY),
            ]
        )
        session.add_all(
            QueryClassification(
                query_id=q,
                classifier="is-boolean",
                snapshot="",
                label=label,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            for q, label in [
                ("q0", "0"),
                ("q1", "1"),
                ("q2", "0"),
            ]
        )
        session.commit()


def _invoke(*args: str):
    return CliRunner().invoke(main, ["export", "queries", *args])


def _filter(tmp_path, text: str = "queries: {require: {is-boolean: 0}}\n"):
    spec = tmp_path / "filter.yaml"
    spec.write_text(text)
    return str(spec)


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_export_queries_writes_the_filtered_queries_as_tsv(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed(db_url)
    out = tmp_path / "work" / "queries.tsv"

    result = _invoke(_filter(tmp_path), "--output", str(out), "--database-url", db_url)

    assert result.exit_code == 0, result.output
    assert "Wrote 2 queries" in result.output
    assert "require is-boolean in ['0']" in result.output
    assert out.read_text().splitlines()[0] == "query_id\tquery"
    assert _read(out) == [
        {"query_id": "q0", "query": "plain query"},
        {"query_id": "q2", "query": _TRICKY},
    ]


def test_export_queries_requires_an_output(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed(db_url)

    result = _invoke(_filter(tmp_path), "--database-url", db_url)

    assert result.exit_code != 0
    assert "--output" in result.output


def test_export_queries_reports_a_malformed_filter(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed(db_url)

    result = _invoke(
        _filter(tmp_path, "requir: {is-boolean: 0}\n"),
        "--output",
        str(tmp_path / "q.tsv"),
        "--database-url",
        db_url,
    )

    assert result.exit_code != 0
    assert "unknown key" in result.output


def test_the_export_with_a_manual_label_column_round_trips_as_a_classifier(
    tmp_path,
):
    db_url = sqlite_url(tmp_path)
    _seed(db_url)
    out = tmp_path / "queries.tsv"
    _invoke(
        _filter(tmp_path, "queries: {exclude: {manual: remove}}\n"),
        "--output",
        str(out),
        "--database-url",
        db_url,
    )
    rows = _read(out)
    for row in rows:
        row["manual_label"] = "remove" if row["query"] == _TRICKY else ""
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    stored = classify(
        ClassifierConfig(db_url, ManualLabelsClassifier(out), progress=False)
    )

    assert stored == {"manual": 1}
    with make_session_factory(create_database_engine(db_url))() as session:
        assert session.get(QueryClassification, ("q2", "manual", "")).label == "remove"
        assert session.get(QueryClassification, ("q0", "manual", "")) is None


def test_manual_classifier_rejects_a_file_without_the_label_column(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("query_id\tquery\nq0\tcats\n")
    with pytest.raises(ValueError, match="manual_label"):
        ManualLabelsClassifier(path)
