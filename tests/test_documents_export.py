from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone

from click.testing import CliRunner

from src.cli import main
from src.database import create_database_engine, create_schema, make_session_factory
from src.models import Document, SnapshotDocument
from tests.conftest import sqlite_url

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
SHARDS = ("documents_000000.jsonl.gz", "documents_000001.jsonl.gz")


def _setup(tmp_path):
    """Snapshot s1 selects d1-d3 and one document that was never fetched."""
    db_url = sqlite_url(tmp_path)
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        for document_id in ("d1", "d2", "d3"):
            session.add(
                Document(
                    id=document_id,
                    title=document_id.upper(),
                    abstract="abstract",
                    full_text=f"text of {document_id}",
                    details={"authors": [{"name": "Ada"}]},
                    fetched_at=NOW,
                )
            )
        session.add_all(
            SnapshotDocument(
                snapshot="s1", document_id=d, reason="relevant", selected_at=NOW
            )
            for d in ("d1", "d2", "d3", "unfetched")
        )
        session.commit()
    spec = tmp_path / "s1.yaml"
    spec.write_text("snapshot: s1\n")
    return db_url, spec


def _export(db_url, spec, output_dir):
    return CliRunner().invoke(
        main,
        [
            "export",
            "documents",
            str(spec),
            str(output_dir),
            "--shard-size",
            "2",
            "--database-url",
            db_url,
        ],
    )


def _records(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_export_documents_writes_shards_and_a_manifest(tmp_path):
    db_url, spec = _setup(tmp_path)
    output_dir = tmp_path / "documents"

    result = _export(db_url, spec, output_dir)

    assert result.exit_code == 0, result.output
    assert "3  fetched" in result.output
    first, second = (_records(output_dir / name) for name in SHARDS)
    assert [record["id"] for record in first + second] == ["d1", "d2", "d3"]
    assert first[0] == {
        "id": "d1",
        "title": "D1",
        "abstract": "abstract",
        "authors": [{"name": "Ada"}],
        "fullText": "text of d1",
    }
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["counts"] == {"documents": 3}
    assert manifest["selection"]["selected"] == 4
    assert manifest["selection"]["steps"] == [{"rule": "fetched", "remaining": 3}]
    assert [shard["records"] for shard in manifest["shards"]] == [2, 1]


def test_export_documents_is_byte_reproducible(tmp_path):
    db_url, spec = _setup(tmp_path)

    _export(db_url, spec, tmp_path / "a")
    _export(db_url, spec, tmp_path / "b")

    for name in (*SHARDS, "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()


def test_export_documents_refuses_a_non_empty_directory(tmp_path):
    db_url, spec = _setup(tmp_path)
    output_dir = tmp_path / "documents"
    _export(db_url, spec, output_dir)

    result = _export(db_url, spec, output_dir)

    assert result.exit_code != 0
    assert "not empty" in result.output
