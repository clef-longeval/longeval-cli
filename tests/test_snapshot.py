from __future__ import annotations

import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from src.cli import main
from src.database import create_database_engine, create_schema, make_session_factory
from src.models import Query, QueryClassification
from tests.conftest import sqlite_url

FILTER = """\
snapshot: snapshot-1
from: 2025-01-01
to: 2025-02-01
click_types: [click]
queries: {exclude: {manual: remove}}
"""


def _seed(db_url: str) -> None:
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        session.add_all(
            Query(id=q, query=text)
            for q, text in [
                ("q1", "first"),
                ("q2", 'tab\tand "quotes"'),
                ("q3", "no clicks"),
                ("q4", "removed"),
            ]
        )
        session.add(
            QueryClassification(
                query_id="q4",
                classifier="manual",
                snapshot="",
                label="remove",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
        session.commit()


def _scores(path: Path, rows) -> Path:
    queries, docs, ctr = zip(*rows)
    pq.write_table(
        pa.table({"query_id": queries, "doc_id": docs, "score": ctr}), path
    )
    return path


@pytest.fixture
def setup(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed(db_url)
    spec = tmp_path / "snapshot-1.yaml"
    spec.write_text(FILTER)
    # q3 has only a 0 label: dropped from the qrels, so from the queries too
    scores = _scores(
        tmp_path / "scores.parquet",
        [("q1", "d1", 0.5), ("q1", "d2", 0.01), ("q2", "d1", 0.95),
         ("q3", "d1", 0.0)],
    )
    return db_url, spec, scores


def _invoke(*args: str):
    return CliRunner().invoke(main, ["export", "snapshot", *args])


def _gunzip(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return handle.read()


def test_snapshot_writes_qrels_queries_and_manifest(tmp_path, setup):
    db_url, spec, scores = setup
    out = tmp_path / "release"

    result = _invoke(
        str(spec), str(out), "--scores", str(scores),
        "--discretizer", "threshold", "--database-url", db_url,
    )

    assert result.exit_code == 0, result.output
    assert "1  of them left out (no qrels)" in result.output
    target = out / "snapshot-1"
    assert sorted(p.name for p in target.iterdir()) == [
        "manifest.json", "qrels.trec.gz", "queries.tsv.gz",
    ]
    assert _gunzip(target / "qrels.trec.gz").splitlines() == [
        "q1 0 d1 1", "q1 0 d2 0", "q2 0 d1 2",
    ]
    rows = list(
        csv.reader(_gunzip(target / "queries.tsv.gz").splitlines(True),
                   delimiter="\t")
    )
    assert rows == [["query_id", "query"], ["q1", "first"],
                    ["q2", 'tab\tand "quotes"']]

    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["snapshot"] == "snapshot-1"
    assert manifest["selection"]["snapshot"]["from"].startswith("2025-01-01")
    assert manifest["relevance"]["discretizer"] == {
        "type": "ThresholdDiscretizer", "thresholds": [0.1, 0.9],
    }
    assert manifest["counts"] == {
        "queries_passing_filter": 3,
        "queries": 2,
        "qrels_pairs": 3,
        "scored_pairs": 4,
        "queries_without_positive_label": 1,
    }
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
    assert not list(out.glob(".*"))  # no staging directory left


def test_snapshot_is_byte_reproducible(tmp_path, setup):
    db_url, spec, scores = setup
    args = ["--scores", str(scores), "--discretizer", "chapelle",
            "--database-url", db_url]

    for name in ("a", "b"):
        assert _invoke(str(spec), str(tmp_path / name), *args).exit_code == 0

    a, b = tmp_path / "a/snapshot-1", tmp_path / "b/snapshot-1"
    for name in ("qrels.trec.gz", "queries.tsv.gz", "manifest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_snapshot_refuses_qrels_for_queries_the_filter_drops(tmp_path, setup):
    db_url, spec, _ = setup
    scores = _scores(tmp_path / "other.parquet", [("q4", "d1", 0.9)])
    out = tmp_path / "release"

    result = _invoke(
        str(spec), str(out), "--scores", str(scores),
        "--discretizer", "threshold", "--database-url", db_url,
    )

    assert result.exit_code != 0
    assert "don't pass the filter" in result.output
    assert list(out.iterdir()) == []  # nothing half-written


def test_snapshot_refuses_an_existing_directory(tmp_path, setup):
    db_url, spec, scores = setup
    (tmp_path / "release/snapshot-1").mkdir(parents=True)

    result = _invoke(
        str(spec), str(tmp_path / "release"), "--scores", str(scores),
        "--discretizer", "threshold", "--database-url", db_url,
    )

    assert result.exit_code != 0
    assert "already exists" in result.output


def test_snapshot_needs_a_discretizer_and_a_snapshot_name(tmp_path, setup):
    db_url, spec, scores = setup
    common = [str(tmp_path / "out"), "--scores", str(scores),
              "--database-url", db_url]

    assert "--discretizer" in _invoke(str(spec), *common).output

    unnamed = tmp_path / "unnamed.yaml"
    unnamed.write_text("click_types: [click]\n")
    result = _invoke(str(unnamed), *common, "--discretizer", "threshold")
    assert result.exit_code != 0
    assert "`snapshot` name" in result.output


def test_export_queries_and_qrels_gzip_by_suffix(tmp_path, setup):
    db_url, spec, scores = setup
    queries, qrels = tmp_path / "q.tsv.gz", tmp_path / "qrels.txt.gz"

    runner = CliRunner()
    for args in (
        ["queries", str(spec), "--output", str(queries), "--database-url", db_url],
        ["qrels", str(scores), "--output", str(qrels)],
    ):
        result = runner.invoke(main, ["export", *args])
        assert result.exit_code == 0, result.output

    assert _gunzip(queries).startswith("query_id\tquery\nq1\tfirst\n")
    assert _gunzip(qrels).splitlines()[0] == "q1 0 d1 0.5"
