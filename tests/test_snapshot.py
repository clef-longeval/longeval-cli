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
from src.models import Document, Query, QueryClassification, SnapshotDocument
from tests.conftest import sqlite_url

FILTER = """\
snapshot: snapshot-1
from: 2025-01-01
to: 2025-02-01
period: P1M
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
    _add_documents(db_url, ["d1", "d2"])
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
        "documents", "manifest.json", "qrels.trec.gz", "queries.tsv.gz",
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
    assert list(manifest)[:6] == [
        "format", "format_version", "snapshot", "from", "to", "period",
    ]
    assert manifest["format"] == "longeval-snapshot"
    assert manifest["format_version"] == 2
    assert manifest["snapshot"] == "snapshot-1"
    assert manifest["from"] == "2025-01-01T00:00:00+00:00"
    assert manifest["to"] == "2025-02-01T00:00:00+00:00"
    assert manifest["period"] == "P1M"
    assert manifest["selection"]["snapshot"] == {
        "from": manifest["from"], "to": manifest["to"],
    }
    assert "period" not in manifest["selection"]["filter"]
    assert manifest["documents"]["id_field"] == "id"
    assert manifest["documents"]["text_fields"] == ["title", "abstract", "fullText"]
    assert manifest["documents"]["path"] == "documents"
    assert manifest["relevance"]["discretizer"] == {
        "type": "ThresholdDiscretizer", "thresholds": [0.1, 0.9],
    }
    assert manifest["counts"] == {
        "queries_passing_filter": 3,
        "queries": 2,
        "qrels_pairs": 3,
        "scored_pairs": 4,
        "queries_without_positive_label": 1,
        "documents": 2,
        "qrels_pairs_without_document": 0,
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


def _add_documents(
    db_url: str,
    document_ids: list[str],
    snapshot: str = "snapshot-1",
    details: dict | None = None,
) -> None:
    """Select and 'fetch' ``document_ids`` (new ones) for ``snapshot``."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        for document_id in document_ids:
            if session.get(Document, document_id) is None:
                session.add(
                    Document(
                        id=document_id,
                        title=document_id.upper(),
                        abstract="abstract",
                        details=details or {"doi": f"10.1/{document_id}"},
                        fetched_at=now,
                    )
                )
            session.add(
                SnapshotDocument(
                    snapshot=snapshot,
                    document_id=document_id,
                    reason="relevant",
                    selected_at=now,
                )
            )
        session.commit()


def _invoke_release(spec, out, scores, db_url, *extra: str):
    return _invoke(
        str(spec), str(out), "--scores", str(scores),
        "--discretizer", "threshold", *extra, "--database-url", db_url,
    )


def test_snapshot_packages_documents(tmp_path, setup):
    db_url, spec, scores = setup
    _add_documents(db_url, ["d9"])  # selected, but judged by no qrels
    out = tmp_path / "release"

    result = _invoke_release(spec, out, scores, db_url)

    assert result.exit_code == 0, result.output
    documents = out / "snapshot-1" / "documents"
    lines = _gunzip(documents / "documents_000000.jsonl.gz").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["d1", "d2", "d9"]
    assert json.loads(lines[0]) == {
        "id": "d1", "title": "D1", "abstract": "abstract",
        "doi": "10.1/d1", "fullText": None,
    }
    manifest = json.loads((out / "snapshot-1" / "manifest.json").read_text())
    assert manifest["documents"]["path"] == "documents"
    assert manifest["counts"]["documents"] == 3
    assert manifest["counts"]["qrels_pairs_without_document"] == 0
    shard = manifest["documents"]["shards"][0]
    digest = hashlib.sha256((documents / shard["path"]).read_bytes()).hexdigest()
    assert digest == shard["sha256"]


def _scores_judging_d3(tmp_path) -> Path:
    """Scores that also judge d3, which is not in the collection."""
    return _scores(
        tmp_path / "with-d3.parquet", [("q1", "d1", 0.5), ("q1", "d3", 0.5)]
    )


def test_snapshot_refuses_qrels_documents_outside_the_collection(tmp_path, setup):
    db_url, spec, _ = setup
    out = tmp_path / "release"

    result = _invoke_release(spec, out, _scores_judging_d3(tmp_path), db_url)

    assert result.exit_code != 0
    assert "1 qrels documents are not in the collection" in result.output
    assert "--drop-missing-documents" in result.output
    assert not (out / "snapshot-1").exists()


def test_snapshot_can_drop_qrels_pairs_without_documents(tmp_path, setup):
    db_url, spec, _ = setup
    out = tmp_path / "release"

    result = _invoke_release(
        spec, out, _scores_judging_d3(tmp_path), db_url, "--drop-missing-documents"
    )

    assert result.exit_code == 0, result.output
    assert _gunzip(out / "snapshot-1" / "qrels.trec.gz").splitlines() == [
        "q1 0 d1 1",
    ]
    manifest = json.loads((out / "snapshot-1" / "manifest.json").read_text())
    assert manifest["counts"]["qrels_pairs_without_document"] == 1


@pytest.mark.parametrize(
    "missing, text",
    [
        ("period: P1M\n", "needs `from`, `to` and `period`"),
        ("to: 2025-02-01\n", "`period` needs both `from` and `to`"),
    ],
)
def test_snapshot_needs_a_time_slot(tmp_path, setup, missing, text):
    db_url, _, scores = setup
    spec = tmp_path / "partial.yaml"
    spec.write_text(FILTER.replace(missing, ""))

    result = _invoke_release(spec, tmp_path / "release", scores, db_url)

    assert result.exit_code != 0
    assert text in result.output


def test_snapshot_refuses_a_snapshot_without_qrels(tmp_path, setup):
    db_url, spec, _ = setup
    scores = _scores(tmp_path / "zeros.parquet", [("q3", "d1", 0.0)])
    out = tmp_path / "release"

    result = _invoke_release(spec, out, scores, db_url)

    assert result.exit_code != 0
    assert "has no qrels" in result.output
    assert not out.exists() or list(out.iterdir()) == []


def test_snapshot_refuses_reserved_document_fields(tmp_path, setup):
    db_url, spec, scores = setup
    _add_documents(db_url, ["d9"], details={"snapshot": "other"})

    result = _invoke_release(spec, tmp_path / "release", scores, db_url)

    assert result.exit_code != 0
    assert "'snapshot' field" in result.output
    assert not (tmp_path / "release" / "snapshot-1").exists()


def _release_second(tmp_path, db_url, scores, out, name, extra_yaml: str):
    """Export a second snapshot, ``name``, into ``out``; return the result."""
    _add_documents(db_url, ["d1", "d2"], snapshot=name)
    spec = tmp_path / f"{name}.yaml"
    spec.write_text(f"snapshot: {name}\n{extra_yaml}click_types: [click]\n")
    return _invoke_release(spec, out, scores, db_url)


@pytest.mark.parametrize(
    "slot, error",
    [
        # the next month: on the grid
        ("from: 2025-02-01\nto: 2025-03-01\nperiod: P1M\n", None),
        # a hole in between is allowed
        ("from: 2025-04-01\nto: 2025-05-01\nperiod: P1M\n", None),
        # earlier than the first one, still on the grid
        ("from: 2024-11-01\nto: 2024-12-01\nperiod: P1M\n", None),
        ("from: 2025-01-01\nto: 2025-02-01\nperiod: P1M\n", "overlaps"),
        ("from: 2025-02-15\nto: 2025-03-15\nperiod: P1M\n", "off the release's"),
        ("from: 2025-02-01\nto: 2025-02-08\nperiod: P1W\n", "share one period"),
    ],
)
def test_snapshot_must_fit_the_release_grid(tmp_path, setup, slot, error):
    db_url, spec, scores = setup
    out = tmp_path / "release"
    assert _invoke_release(spec, out, scores, db_url).exit_code == 0

    result = _release_second(tmp_path, db_url, scores, out, "snapshot-2", slot)

    if error is None:
        assert result.exit_code == 0, result.output
        manifest = json.loads((out / "snapshot-2" / "manifest.json").read_text())
        assert manifest["period"] == "P1M"
    else:
        assert result.exit_code != 0
        assert error in result.output
        assert not (out / "snapshot-2").exists()


def test_snapshot_refuses_a_released_snapshot_id(tmp_path, setup):
    db_url, spec, scores = setup
    out = tmp_path / "release"
    assert _invoke_release(spec, out, scores, db_url).exit_code == 0
    (out / "snapshot-1").rename(out / "first")

    result = _invoke_release(spec, out, scores, db_url)

    assert result.exit_code != 0
    assert "already released" in result.output


def test_snapshot_refuses_an_older_format_in_the_release(tmp_path, setup):
    db_url, spec, scores = setup
    old = tmp_path / "release" / "old"
    old.mkdir(parents=True)
    (old / "manifest.json").write_text(
        json.dumps({"format": "longeval-snapshot", "format_version": 1})
    )

    result = _invoke_release(spec, tmp_path / "release", scores, db_url)

    assert result.exit_code != 0
    assert "format version 1" in result.output
