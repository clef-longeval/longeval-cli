"""Writing the test collection's files from the database.

Each exporter takes a :class:`~.filtering.SnapshotConfig` (which queries to keep) and
an output path, and is exposed as a ``longeval-cli export <what>`` subcommand. Text
outputs are gzipped when their name ends in ``.gz`` (deterministically: no
timestamp or file name in the header, so equal content gives equal bytes).
:func:`write_snapshot` assembles the released ``<snapshot>/`` directory from the
same writers.
"""

from __future__ import annotations

import csv
import dataclasses
import gzip
import hashlib
import io
import json
import shutil
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from .clicks.file_source import FORMAT_NAME, MANIFEST_NAME
from .clicks.query_session import (
    FORMAT_VERSION,
    QuerySession,
    QuerySessionStats,
    to_record,
)
from .discretizing import Discretizer
from .filtering import (
    FilterError,
    SnapshotConfig,
    document_filter_report,
    filter_documents,
    filter_queries,
)
from .models import Document

ORDERING = "queried_at, result_id"
SNAPSHOT_FORMAT = "longeval-snapshot"
SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_MANIFEST = "manifest.json"
QUERIES_FILE = "queries.tsv.gz"
QRELS_FILE = "qrels.trec.gz"
DOCUMENTS_FORMAT = "longeval-documents"
DOCUMENTS_FORMAT_VERSION = 1
DOCUMENTS_DIR = "documents"
_DOCUMENT_CHUNK = 900  # ids per `IN (...)`, under SQLite's oldest limit of 999


@contextmanager
def _text_output(path: Path) -> Iterator[io.TextIOBase]:
    """Open ``path`` for writing text, gzipped when it ends in ``.gz``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".gz":
        with path.open("w", encoding="utf-8", newline="") as handle:
            yield handle
        return
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed,
        io.TextIOWrapper(packed, encoding="utf-8", newline="") as handle,
    ):
        yield handle


def _text_input(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_queries_tsv(session: Session, criteria: SnapshotConfig, output: Path) -> int:
    """Write the queries passing ``criteria`` as a TSV; return how many.

    The header is ``query_id<TAB>query``. Fields are quoted like any CSV
    dialect would, so a query containing tabs, newlines or quotes survives a
    round trip through a spreadsheet and ``csv.DictReader``. The file doubles
    as the manual-review sheet: add a ``manual_label`` column and feed it to
    :class:`~.classifying.builtins.ManualLabelsClassifier`.
    """
    rows = filter_queries(session, criteria)
    _write_queries(rows, output)
    return len(rows)


def _write_queries(rows: list[tuple[str, str]], output: Path) -> None:
    with _text_output(output) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query_id", "query"])
        writer.writerows(rows)


def write_query_sessions(
    query_sessions: Iterable[QuerySession],
    output_dir: Path,
    config: SnapshotConfig,
    stats: QuerySessionStats,
    *,
    shard_size: int = 100_000,
) -> dict[str, Any]:
    """Write ``query_sessions`` as shards plus a manifest; return the manifest.

    ``stats`` must be the object the iterator feeding ``query_sessions`` fills,
    so the skip counters in the manifest are complete once this returns.
    Refuses a directory that already holds an export.
    """
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = output_dir.glob("query-sessions-*")
    if (output_dir / MANIFEST_NAME).exists() or any(shards):
        raise FileExistsError(f"{output_dir} already contains an export")

    shards = _write_shards(
        (to_record(query_session) for query_session in query_sessions),
        output_dir,
        _shard_name,
        shard_size,
    )

    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "selection": _selection_record(config),
        "counts": {
            "query_sessions": stats.query_sessions,
            "items": stats.items,
            "clicks": stats.clicks,
            "skipped": dict(sorted(stats.skipped.items())),
            "unmatched_interactions": stats.unmatched_interactions,
        },
        "shards": shards,
    }
    temporary = output_dir / (MANIFEST_NAME + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / MANIFEST_NAME)
    return manifest


class _ShardWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records = 0
        self._raw = path.open("wb")
        gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._text = io.TextIOWrapper(gz, encoding="utf-8", newline="\n")

    def write(self, record: dict[str, Any]) -> None:
        self._text.write(
            json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        )
        self.records += 1

    def close(self) -> dict[str, Any]:
        self._text.close()  # also closes the gzip stream
        self._raw.close()
        return {
            "path": self.path.name,
            "records": self.records,
            "sha256": _sha256(self.path),
        }


def _write_shards(
    records: Iterable[dict[str, Any]],
    output_dir: Path,
    shard_name: Callable[[int], str],
    shard_size: int,
) -> list[dict[str, Any]]:
    """Write ``records`` as gzipped JSONL shards of ``shard_size``; describe each."""
    shards: list[dict[str, Any]] = []
    writer: _ShardWriter | None = None
    for record in records:
        if writer is None:
            writer = _ShardWriter(output_dir / shard_name(len(shards)))
        writer.write(record)
        if writer.records == shard_size:
            shards.append(writer.close())
            writer = None
    if writer is not None:
        shards.append(writer.close())
    return shards


def _shard_name(index: int) -> str:
    return f"query-sessions-{index:05d}.jsonl.gz"


def _selection_record(config: SnapshotConfig) -> dict[str, Any]:
    rules = dataclasses.asdict(config)  # the range is recorded under "snapshot"
    # Neither the publication window nor document rules select query sessions.
    for key in (
        "start",
        "end",
        "published_start",
        "published_end",
        "click_types",
        "documents",
    ):
        del rules[key]

    def moment(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    return {
        "click_types": sorted(config.click_types),
        "filter": rules,
        "snapshot": {  # [from, to) on the result timestamp, UTC
            "from": moment(config.start),
            "to": moment(config.end),
        },
        "ordering": ORDERING,
    }


def write_documents(
    session: Session,
    config: SnapshotConfig,
    output_dir: Path,
    *,
    shard_size: int = 1_000,
) -> dict[str, Any]:
    """Write the snapshot's documents as shards plus a manifest; return it.

    The documents are those :func:`~.filtering.filter_documents` keeps, in id
    order, one :func:`to_document_record` per line, in gzipped JSONL shards of
    ``shard_size``. Refuses a non-empty ``output_dir``; the manifest, written
    last, records the filter's steps, the count and each shard's SHA-256.
    """
    document_ids = filter_documents(session, config)
    return _write_documents(session, config, document_ids, output_dir, shard_size)


def to_document_record(row: Any) -> dict[str, Any]:
    """A released document: ``id``, ``title``, ``abstract``, the source's
    details (e.g. ``authors``, ``doi``) and ``fullText``."""
    record = {"id": row.id, "title": row.title, "abstract": row.abstract}
    for key, value in (row.details or {}).items():
        record.setdefault(key, value)
    record["fullText"] = row.full_text
    return record


def _write_documents(
    session: Session,
    config: SnapshotConfig,
    document_ids: list[str],
    output_dir: Path,
    shard_size: int,
) -> dict[str, Any]:
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = document_filter_report(session, config)
    shards = _write_shards(
        map(to_document_record, _iter_document_rows(session, document_ids)),
        output_dir,
        _document_shard_name,
        shard_size,
    )
    published_end = config.published_end
    manifest = {
        "format": DOCUMENTS_FORMAT,
        "format_version": DOCUMENTS_FORMAT_VERSION,
        "snapshot": config.snapshot,
        "selection": {
            "rules": dataclasses.asdict(config.documents),
            "published_before": published_end.isoformat() if published_end else None,
            "selected": report.total,
            "steps": [
                {"rule": rule, "remaining": remaining}
                for rule, remaining in report.steps
            ],
        },
        "counts": {"documents": sum(shard["records"] for shard in shards)},
        "shards": shards,
    }
    temporary = output_dir / (MANIFEST_NAME + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / MANIFEST_NAME)
    return manifest


def _iter_document_rows(session: Session, document_ids: list[str]) -> Iterator[Any]:
    """The documents' exported columns, in ``document_ids`` order, by chunks.

    Plain rows, not ORM objects, so the session keeps no full text in memory.
    """
    columns = (
        Document.id,
        Document.title,
        Document.abstract,
        Document.full_text,
        Document.details,
    )
    for start in range(0, len(document_ids), _DOCUMENT_CHUNK):
        chunk = document_ids[start : start + _DOCUMENT_CHUNK]
        yield from session.execute(
            sa.select(*columns).where(Document.id.in_(chunk)).order_by(Document.id)
        )


def _document_shard_name(index: int) -> str:
    return f"documents_{index:06d}.jsonl.gz"


@dataclasses.dataclass(frozen=True)
class QrelsReport:
    pairs_read: int
    pairs_written: int
    queries_written: int
    queries_dropped: int  # queries left without any label > 0
    pairs_without_document: int = 0  # left out: document not in the collection


def write_qrels(
    scores: Path,
    output: Path,
    discretizer: Discretizer | None = None,
    *,
    score_column: str = "score",
    keep_all_queries: bool = False,
    document_ids: set[str] | None = None,
) -> QrelsReport:
    """Write TREC qrels (``query_id 0 doc_id label``, space separated).

    ``scores`` is a Parquet file with ``query_id``, ``doc_id`` and the
    ``score_column`` (e.g. what ``examples/minimal-example/train_dctr_from_export.py``
    writes). Without a ``discretizer`` the label column holds the score itself.
    With one, pairs it maps to ``None`` are left out, and — unless
    ``keep_all_queries`` — so is every query that has no label above 0 (it
    would carry no relevance information). Rows keep the file's order. The file
    is read in batches, twice when queries have to be dropped; only the set
    of queries with a positive label is held in memory. With ``document_ids``,
    pairs whose document is not among them are left out first (and counted),
    as if they had not been scored.
    """
    import pyarrow.parquet as pq

    source = pq.ParquetFile(scores)

    def rows():
        columns = ["query_id", "doc_id", score_column]
        for batch in source.iter_batches(columns=columns):
            yield from zip(
                *(batch.column(c).to_pylist() for c in columns), strict=True
            )

    def label(score: float) -> int | float | None:
        return score if discretizer is None else discretizer.label(score)

    def in_collection(doc_id: str) -> bool:
        return document_ids is None or doc_id in document_ids

    drop_queries = discretizer is not None and not keep_all_queries
    relevant: set[str] = set()
    if drop_queries:
        relevant = {
            q
            for q, d, score in rows()
            if in_collection(d) and (label(score) or 0) > 0
        }

    read = written = without_document = 0
    seen: set[str] = set()
    all_queries: set[str] = set()
    with _text_output(output) as handle:
        for query_id, doc_id, score in rows():
            read += 1
            all_queries.add(query_id)
            if not in_collection(doc_id):
                without_document += 1
                continue
            value = label(score)
            if value is None or (drop_queries and query_id not in relevant):
                continue
            text = f"{value:.8g}" if discretizer is None else str(value)
            handle.write(f"{query_id} 0 {doc_id} {text}\n")
            written += 1
            seen.add(query_id)
    dropped = len(all_queries - relevant) if drop_queries else 0
    return QrelsReport(read, written, len(seen), dropped, without_document)


def read_qrels_document_ids(path: Path) -> set[str]:
    """Return every document id a TREC qrels file judges, whatever its label."""
    document_id_set: set[str] = set()
    with _text_input(path) as handle:
        for number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 4:
                raise ValueError(
                    f"{path}:{number}: not a TREC qrels line: {line.rstrip()!r}"
                )
            document_id_set.add(fields[2])
    return document_id_set


@dataclasses.dataclass(frozen=True)
class SnapshotReport:
    path: Path
    qrels: QrelsReport
    queries_passing: int  # queries passing the filter's `queries` rules
    queries_written: int  # of those, the ones with qrels
    documents: int | None = None  # packaged documents; None without them


def write_snapshot(
    session: Session,
    config: SnapshotConfig,
    scores: Path,
    output_root: Path,
    discretizer: Discretizer,
    *,
    score_column: str = "score",
    keep_all_queries: bool = False,
    documents: bool = False,
    drop_missing_documents: bool = False,
    document_shard_size: int = 1_000,
) -> SnapshotReport:
    """Write the released ``output_root/<snapshot>/`` directory; return a report.

    It holds ``qrels.trec.gz`` (as :func:`write_qrels` writes it),
    ``queries.tsv.gz`` (as :func:`write_queries_tsv`, restricted to the queries
    with qrels) and a ``manifest.json`` recording the selection, the scores file
    and discretizer, the counts and every file's SHA-256. With ``documents`` it
    also holds ``documents/`` (as :func:`write_documents` writes it), and every
    qrels document has to be in it, unless ``drop_missing_documents`` leaves
    the other qrels pairs out. Every qrels query has to pass the filter's
    ``queries`` rules: otherwise the scores came from another selection. When a
    check fails nothing is written. The directory appears only once complete,
    and an existing one is refused.
    """
    if not config.snapshot:
        raise FilterError("the filter needs a `snapshot` name to export a snapshot")
    if drop_missing_documents and not documents:
        raise ValueError("dropping missing documents needs the documents exported")
    target = output_root / config.snapshot
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    staging = output_root / f".{config.snapshot}.partial"
    shutil.rmtree(staging, ignore_errors=True)  # left over from a failed run
    try:
        # 1. Qrels, restricted to the collection when asked to.
        document_ids = filter_documents(session, config) if documents else None
        qrels = write_qrels(
            scores,
            staging / QRELS_FILE,
            discretizer,
            score_column=score_column,
            keep_all_queries=keep_all_queries,
            document_ids=(
                set(document_ids)
                if document_ids is not None and drop_missing_documents
                else None
            ),
        )
        with _text_input(staging / QRELS_FILE) as handle:
            judged = {line.split(" ", 1)[0] for line in handle}

        # 2. Queries: every qrels query has to pass the filter.
        passing = filter_queries(session, config)
        unknown = judged - {query_id for query_id, _ in passing}
        if unknown:
            raise FilterError(
                f"{len(unknown)} qrels queries don't pass the filter (e.g. "
                f"{', '.join(sorted(unknown)[:3])}); were the scores computed "
                "for another selection?"
            )
        kept = [row for row in passing if row[0] in judged]
        _write_queries(kept, staging / QUERIES_FILE)

        # 3. Documents: every qrels document has to be in the collection.
        documents_manifest = None
        if document_ids is not None:
            missing = read_qrels_document_ids(staging / QRELS_FILE).difference(
                document_ids
            )
            if missing:
                raise FilterError(
                    f"{len(missing)} qrels documents are not in the collection "
                    f"(e.g. {', '.join(sorted(missing)[:3])}): select and fetch "
                    "them, or drop their qrels pairs with --drop-missing-documents"
                )
            documents_manifest = _write_documents(
                session,
                config,
                document_ids,
                staging / DOCUMENTS_DIR,
                document_shard_size,
            )

        # 4. The manifest, written last.
        counts: dict[str, int] = {
            "queries_passing_filter": len(passing),
            "queries": len(kept),
            "qrels_pairs": qrels.pairs_written,
            "scored_pairs": qrels.pairs_read,
            "queries_without_positive_label": qrels.queries_dropped,
        }
        if documents_manifest is not None:
            counts["documents"] = documents_manifest["counts"]["documents"]
            counts["qrels_pairs_without_document"] = qrels.pairs_without_document
        manifest = {
            "format": SNAPSHOT_FORMAT,
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "snapshot": config.snapshot,
            "selection": _selection_record(config),
            "relevance": {
                "scores": {"path": scores.name, "sha256": _sha256(scores)},
                "score_column": score_column,
                "discretizer": {
                    "type": type(discretizer).__name__,
                    **dataclasses.asdict(discretizer),
                },
                "keep_all_queries": keep_all_queries,
            },
            "counts": counts,
            "documents": (
                None
                if documents_manifest is None
                else {"path": DOCUMENTS_DIR, **documents_manifest}
            ),
            "files": {
                name: _sha256(staging / name) for name in (QRELS_FILE, QUERIES_FILE)
            },
        }
        (staging / SNAPSHOT_MANIFEST).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    packaged = None if documents_manifest is None else counts["documents"]
    return SnapshotReport(target, qrels, len(passing), len(kept), packaged)
