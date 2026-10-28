"""Index a snapshot's documents and retrieve with several models, via TIRA.

The input is a released snapshot directory (see ``export snapshot --documents``):
``documents/`` and ``queries.tsv.gz``. Nothing is computed in this process: the
PyTerrier approaches of ``ir-benchmarks/tira-ir-starter`` run as Docker
containers through TIRA's Python client, first one that builds the index and
then one per retrieval model, which gets the index as its input run. Needs the
``tira`` extra, Docker (see ``tira-cli verify-installation --local-only``) and
network access to pull the images.

Layout of the output directory (an existing result is kept, see ``recompute``)::

    input/index/documents.jsonl    {"docno", "text"} per document
    input/retrieval/queries.xml    TREC topics
    index/index/                   the PyTerrier index (the index software's output)
    runs/<model>/run.txt           one TREC run per model
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import io
import json
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIRA_TEAM = "tira-ir-starter"
SOFTWARE_SUFFIX = "(tira-ir-starter-pyterrier)"
INDEX_SOFTWARE = f"Index {SOFTWARE_SUFFIX}"
DEFAULT_MODELS = ("BM25", "DirichletLM", "DPH", "Hiemstra_LM", "PL2")
MISSING_TIRA = "retrieval needs TIRA: install it with `uv sync --extra tira`"


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    documents: int
    queries: int
    index: Path
    runs: dict[str, Path]
    reused: list[str]


def iter_documents(documents_dir: Path) -> Iterator[dict[str, str]]:
    """Yield ``{"docno", "text"}`` per document of an ``export documents`` dir.

    The text is the title followed by the abstract; the full text is not indexed.
    """
    shards = sorted(documents_dir.glob("documents_*.jsonl*"))
    if not shards:
        raise ValueError(f"no documents_*.jsonl.gz shards in {documents_dir}")
    for shard in shards:
        opener = gzip.open if shard.suffix == ".gz" else open
        with opener(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                parts = (record.get("title"), record.get("abstract"))
                yield {
                    "docno": str(record["id"]),
                    "text": " ".join(part for part in parts if part),
                }


def write_documents_jsonl(documents_dir: Path, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for document in iter_documents(documents_dir):
            handle.write(json.dumps(document) + "\n")
            count += 1
    return count


def write_queries_xml(queries_tsv: Path, output: Path) -> int:
    """Convert the ``query_id<TAB>query`` TSV of ``export queries`` to TREC topics."""
    opener = gzip.open if queries_tsv.suffix == ".gz" else open
    with opener(queries_tsv, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    root = ET.Element("topics")
    for row in rows:
        topic = ET.SubElement(root, "topic", attrib={"number": row["query_id"]})
        ET.SubElement(topic, "query").text = row["query"]
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="unicode")
    return len(rows)


def _find_index(root: Path) -> Path | None:
    matches = sorted(root.rglob("data.properties"))
    return matches[0].parent if matches else None


def _run_approach(
    client: Any,
    software: str,
    input_dir: Path,
    output_dir: Path,
    input_run: Path | None = None,
) -> None:
    """Run one TIRA approach locally; its noisy output is shown only on failure."""
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            details = client.public_system_details(TIRA_TEAM, software)
            image = details.get("public_image_name") or details["tira_image_name"]
            arguments = {"input_run": input_run} if input_run else {}
            client.local_execution.run(
                image=image,
                command=details["command"],
                input_dir=input_dir,
                output_dir=output_dir,
                allow_network=False,
                forward_environment_variables=details.get(
                    "forward_environment_variable"
                ),
                **arguments,
            )
    except Exception as error:
        raise RuntimeError(
            f"Running '{software}' via TIRA failed: {error}\n{captured.getvalue()}"
        ) from error


def retrieve_snapshot(
    snapshot_dir: Path,
    output_dir: Path,
    models: tuple[str, ...] = DEFAULT_MODELS,
    recompute: bool = False,
    client: Any = None,
    progress: Callable[[str], None] = lambda message: None,
) -> RetrievalReport:
    """Build the index of the snapshot's documents, then run every model on it.

    An index or run that already exists in ``output_dir`` is reused unless
    ``recompute`` is set, so an interrupted call can be started again and
    models can be added later.
    """
    if not models:
        raise ValueError("give at least one retrieval model")
    documents_dir = snapshot_dir / "documents"
    queries_path = snapshot_dir / "queries.tsv.gz"
    for path in (documents_dir, queries_path):
        if not path.exists():
            raise ValueError(
                f"{path} not found: pass a directory written by "
                "`export snapshot --documents`"
            )
    if client is None:
        try:
            from tira.rest_api_client import Client
        except ImportError as error:
            raise ValueError(MISSING_TIRA) from error
        client = Client()

    output_dir = output_dir.resolve()  # Docker needs absolute host paths
    index_input = output_dir / "input" / "index"
    queries_input = output_dir / "input" / "retrieval"
    index_output = output_dir / "index"
    reused: list[str] = []

    documents = write_documents_jsonl(documents_dir, index_input / "documents.jsonl")
    queries = write_queries_xml(queries_path, queries_input / "queries.xml")

    index = _find_index(index_output)
    if index is not None and not recompute:
        reused.append("index")
    else:
        progress(f"Indexing {documents} documents")
        shutil.rmtree(index_output, ignore_errors=True)
        index_output.mkdir(parents=True)
        _run_approach(client, INDEX_SOFTWARE, index_input, index_output)
        index = _find_index(index_output)
        if index is None:
            raise RuntimeError("TIRA did not produce an index (no data.properties)")

    runs: dict[str, Path] = {}
    for model in models:
        model_output = output_dir / "runs" / model
        run = model_output / "run.txt"
        if run.is_file() and not recompute:
            reused.append(model)
        else:
            progress(f"Retrieving {queries} queries with {model}")
            shutil.rmtree(model_output, ignore_errors=True)
            model_output.mkdir(parents=True)
            # the software appends "/index" to the directory it is given
            _run_approach(
                client,
                f"{model} {SOFTWARE_SUFFIX}",
                queries_input,
                model_output,
                input_run=index.parent,
            )
            if not run.is_file():
                raise RuntimeError(f"TIRA did not produce a run for {model}")
        runs[model] = run
    return RetrievalReport(documents, queries, index, runs, reused)
