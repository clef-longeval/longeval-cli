from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from src.retrieval import INDEX_SOFTWARE, retrieve_snapshot


class FakeLocalExecution:
    def __init__(self, calls):
        self.calls = calls

    def run(self, image, command, input_dir, output_dir, input_run=None, **kwargs):
        self.calls.append((image, Path(input_dir), Path(output_dir), input_run))
        if image == "index-image":
            (Path(output_dir) / "index").mkdir()
            (Path(output_dir) / "index" / "data.properties").write_text("")
        else:
            (Path(output_dir) / "run.txt").write_text("q1 Q0 d1 1 1.0 x\n")


class FakeClient:
    def __init__(self):
        self.calls = []
        self.local_execution = FakeLocalExecution(self.calls)

    def public_system_details(self, team, software):
        assert team == "tira-ir-starter"
        image = "index-image" if software == INDEX_SOFTWARE else software
        return {"public_image_name": image, "command": "run"}


def _snapshot(tmp_path):
    snapshot = tmp_path / "snapshot-1"
    (snapshot / "documents").mkdir(parents=True)
    with gzip.open(snapshot / "documents" / "documents_000000.jsonl.gz", "wt") as out:
        for record in (
            {"id": "d1", "title": "A title", "abstract": "Text"},
            {"id": "d2", "title": None, "abstract": "Only"},
        ):
            out.write(json.dumps(record) + "\n")
    with gzip.open(snapshot / "queries.tsv.gz", "wt") as out:
        out.write("query_id\tquery\nq1\tsome <query> & more\n")
    return snapshot


def test_indexes_once_then_retrieves_per_model_with_the_index_as_input_run(tmp_path):
    client = FakeClient()
    output = tmp_path / "out"

    report = retrieve_snapshot(
        _snapshot(tmp_path), output, models=("BM25", "PL2"), client=client
    )

    assert (report.documents, report.queries) == (2, 1)
    assert [call[0] for call in client.calls] == [
        "index-image",
        "BM25 (tira-ir-starter-pyterrier)",
        "PL2 (tira-ir-starter-pyterrier)",
    ]
    assert all(call[3] == (output / "index").resolve() for call in client.calls[1:])
    assert set(report.runs) == {"BM25", "PL2"}
    documents = (output / "input/index/documents.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in documents] == [
        {"docno": "d1", "text": "A title Text"},
        {"docno": "d2", "text": "Only"},
    ]
    assert "some &lt;query&gt; &amp; more" in (
        output / "input/retrieval/queries.xml"
    ).read_text()


def test_existing_results_are_reused_unless_recomputed(tmp_path):
    snapshot, output, client = _snapshot(tmp_path), tmp_path / "out", FakeClient()
    retrieve_snapshot(snapshot, output, models=("BM25",), client=client)
    client.calls.clear()

    report = retrieve_snapshot(
        snapshot, output, models=("BM25", "DPH"), client=client
    )
    assert [call[0] for call in client.calls] == ["DPH (tira-ir-starter-pyterrier)"]
    assert report.reused == ["index", "BM25"]

    client.calls.clear()
    retrieve_snapshot(snapshot, output, models=("BM25",), recompute=True, client=client)
    assert len(client.calls) == 2


def test_needs_a_snapshot_with_documents(tmp_path):
    with pytest.raises(ValueError, match="export snapshot"):
        retrieve_snapshot(tmp_path, tmp_path / "out", client=FakeClient())
