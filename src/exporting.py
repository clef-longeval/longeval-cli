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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from .clicks.file_source import FORMAT_NAME, MANIFEST_NAME
from .clicks.query_session import (
    FORMAT_VERSION,
    QuerySession,
    QuerySessionStats,
    to_record,
)
from .discretizing import Discretizer
from .filtering import FilterError, SnapshotConfig, filter_queries

ORDERING = "queried_at, result_id"
SNAPSHOT_FORMAT = "longeval-snapshot"
SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_MANIFEST = "manifest.json"
QUERIES_FILE = "queries.tsv.gz"
QRELS_FILE = "qrels.trec.gz"


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

    shards: list[dict[str, Any]] = []
    writer: _ShardWriter | None = None
    for query_session in query_sessions:
        if writer is None:
            writer = _ShardWriter(output_dir / _shard_name(len(shards)))
        writer.write(query_session)
        if writer.records == shard_size:
            shards.append(writer.close())
            writer = None
    if writer is not None:
        shards.append(writer.close())

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

    def write(self, query_session: QuerySession) -> None:
        self._text.write(
            json.dumps(
                to_record(query_session), separators=(",", ":"), ensure_ascii=False
            )
            + "\n"
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


@dataclasses.dataclass(frozen=True)
class QrelsReport:
    pairs_read: int
    pairs_written: int
    queries_written: int
    queries_dropped: int  # queries left without any label > 0


def write_qrels(
    scores: Path,
    output: Path,
    discretizer: Discretizer | None = None,
    *,
    score_column: str = "score",
    keep_all_queries: bool = False,
) -> QrelsReport:
    """Write TREC qrels (``query_id 0 doc_id label``, space separated).

    ``scores`` is a Parquet file with ``query_id``, ``doc_id`` and the
    ``score_column`` (e.g. what ``examples/minimal-example/train_dctr_from_export.py``
    writes). Without a ``discretizer`` the label column holds the score itself.
    With one, pairs it maps to ``None`` are left out, and — unless
    ``keep_all_queries`` — so is every query that has no label above 0 (it
    would carry no relevance information). Rows keep the file's order. The file
    is read in batches, twice when queries have to be dropped; only the set
    of queries with a positive label is held in memory.
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

    drop_queries = discretizer is not None and not keep_all_queries
    relevant: set[str] = set()
    if drop_queries:
        relevant = {q for q, _, score in rows() if (label(score) or 0) > 0}

    read = written = 0
    seen: set[str] = set()
    all_queries: set[str] = set()
    with _text_output(output) as handle:
        for query_id, doc_id, score in rows():
            read += 1
            all_queries.add(query_id)
            value = label(score)
            if value is None or (drop_queries and query_id not in relevant):
                continue
            text = f"{value:.8g}" if discretizer is None else str(value)
            handle.write(f"{query_id} 0 {doc_id} {text}\n")
            written += 1
            seen.add(query_id)
    dropped = len(all_queries - relevant) if drop_queries else 0
    return QrelsReport(read, written, len(seen), dropped)


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


def write_snapshot(
    session: Session,
    config: SnapshotConfig,
    scores: Path,
    output_root: Path,
    discretizer: Discretizer,
    *,
    score_column: str = "score",
    keep_all_queries: bool = False,
) -> SnapshotReport:
    """Write the released ``output_root/<snapshot>/`` directory; return a report.

    It holds ``qrels.trec.gz`` (as :func:`write_qrels` writes it),
    ``queries.tsv.gz`` (as :func:`write_queries_tsv`, restricted to the queries
    with qrels) and a ``manifest.json`` recording the selection, the scores file
    and discretizer, the counts and every file's SHA-256. Documents are not
    packaged yet. Every qrels query has to pass the filter's ``queries`` rules:
    otherwise the scores came from another selection and nothing is written.
    The directory appears only once complete, and an existing one is refused.
    """
    if not config.snapshot:
        raise FilterError("the filter needs a `snapshot` name to export a snapshot")
    target = output_root / config.snapshot
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    staging = output_root / f".{config.snapshot}.partial"
    shutil.rmtree(staging, ignore_errors=True)  # left over from a failed run
    try:
        qrels = write_qrels(
            scores,
            staging / QRELS_FILE,
            discretizer,
            score_column=score_column,
            keep_all_queries=keep_all_queries,
        )
        with _text_input(staging / QRELS_FILE) as handle:
            judged = {line.split(" ", 1)[0] for line in handle}

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
            "counts": {
                "queries_passing_filter": len(passing),
                "queries": len(kept),
                "qrels_pairs": qrels.pairs_written,
                "scored_pairs": qrels.pairs_read,
                "queries_without_positive_label": qrels.queries_dropped,
            },
            "documents": None,  # not packaged yet
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
    return SnapshotReport(target, qrels, len(passing), len(kept))
