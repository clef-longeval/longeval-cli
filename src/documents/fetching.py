"""Fetching documents from a source into the database.

``fetch_documents`` splits the ids into the source's batches, requests
several batches at once on worker threads, and stores each batch's outcome
in one transaction from the calling thread: a ``Document`` per record found,
and a ``DocumentFetch`` per id (``ok``, ``not_found`` or ``error``).
Ids whose last fetch was ``ok`` or ``not_found`` are skipped, so an
interrupted run can simply be started again; ``error`` ids are retried, and
so are ``not_found`` ones with ``retry_not_found`` (e.g. after switching to
another source).
"""

from __future__ import annotations

import gzip
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from tqdm.auto import tqdm

from ..database import session_scope
from ..models import Document, DocumentFetch, SnapshotDocument
from .source import BaseDocumentSource, DocumentRecord, DocumentSourceError

#: Fetch statuses a re-run does not request again.
FINAL_STATUSES = ("ok", "not_found")

_LOOKUP_CHUNK = 500  # ids per `IN (...)` lookup, well under SQLite's limit

BatchOutcome = list[DocumentRecord] | DocumentSourceError


@dataclass(frozen=True)
class FetchReport:
    requested: int  # distinct ids asked for
    skipped: int  # of those, already fetched or known to be missing
    fetched: int  # stored as documents
    not_found: int
    failed: int  # in batches the source could not answer; retried next run
    full_texts_dropped: int  # longer than `full_text_max_chars`


def read_document_ids(path: Path) -> list[str]:
    """Read one id per line; blank lines and repeated ids are skipped."""
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open(encoding="utf-8")
    with handle:
        return list(dict.fromkeys(line.strip() for line in handle if line.strip()))


def find_snapshot_document_ids(
    session: Session, snapshot: str | None = None
) -> list[str]:
    """Return the ids selected for ``snapshot``, or for any snapshot."""
    statement = sa.select(SnapshotDocument.document_id).distinct()
    if snapshot is not None:
        statement = statement.where(SnapshotDocument.snapshot == snapshot)
    return list(session.scalars(statement.order_by(SnapshotDocument.document_id)))


def fetch_documents(
    session_factory: sessionmaker[Session],
    source: BaseDocumentSource,
    document_ids: Iterable[str],
    *,
    max_workers: int = 1,
    full_text_max_chars: int | None = None,
    retry_not_found: bool = False,
) -> FetchReport:
    """Fetch the ``document_ids`` not settled yet and store every outcome.

    A full text longer than ``full_text_max_chars`` is stored as empty.
    Only :class:`DocumentSourceError` marks a batch as failed; any other
    exception from the source is a bug in it and stops the run.
    """
    # 1. Drop repeated ids and those a previous run settled.
    requested_list = list(dict.fromkeys(document_ids))
    settled_statuses = ("ok",) if retry_not_found else FINAL_STATUSES
    with session_factory() as session:
        pending_list = _find_unsettled_ids(session, requested_list, settled_statuses)
    size = source.batch_size
    batch_list = [
        pending_list[start : start + size]
        for start in range(0, len(pending_list), size)
    ]

    # 2. Request batches on worker threads; store each one as it completes.
    counts: Counter[str] = Counter()
    with (
        ThreadPoolExecutor(max_workers=max_workers) as executor,
        tqdm(total=len(batch_list), desc="Fetching documents", unit="batch") as bar,
    ):
        outcomes = _iter_batch_outcomes(
            executor, source, batch_list, max_in_flight=2 * max_workers
        )
        for batch, outcome in outcomes:
            with session_scope(session_factory) as session:
                counts.update(
                    _store_batch(session, batch, outcome, full_text_max_chars)
                )
            bar.update()

    return FetchReport(
        requested=len(requested_list),
        skipped=len(requested_list) - len(pending_list),
        fetched=counts["ok"],
        not_found=counts["not_found"],
        failed=counts["error"],
        full_texts_dropped=counts["full_texts_dropped"],
    )


def _find_unsettled_ids(
    session: Session, document_ids: list[str], settled_statuses: tuple[str, ...]
) -> list[str]:
    """Return the ``document_ids`` whose last fetch has none of ``settled_statuses``."""
    settled_set: set[str] = set()
    for start in range(0, len(document_ids), _LOOKUP_CHUNK):
        chunk = document_ids[start : start + _LOOKUP_CHUNK]
        settled_set.update(
            session.scalars(
                sa.select(DocumentFetch.document_id).where(
                    DocumentFetch.document_id.in_(chunk),
                    DocumentFetch.status.in_(settled_statuses),
                )
            )
        )
    return [
        document_id for document_id in document_ids if document_id not in settled_set
    ]


def _iter_batch_outcomes(
    executor: ThreadPoolExecutor,
    source: BaseDocumentSource,
    batch_list: list[list[str]],
    *,
    max_in_flight: int,
) -> Iterator[tuple[list[str], BatchOutcome]]:
    """Yield each batch with its records, or the error the source raised.

    At most ``max_in_flight`` batches are submitted at a time, so a long id
    list doesn't queue a future per batch up front.
    """
    batches = iter(batch_list)
    in_flight: dict[Future[list[DocumentRecord]], list[str]] = {}
    while True:
        while len(in_flight) < max_in_flight:
            batch = next(batches, None)
            if batch is None:
                break
            in_flight[executor.submit(source.fetch_batch, batch)] = batch
        if not in_flight:
            return
        done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
        for future in done:
            batch = in_flight.pop(future)
            try:
                outcome: BatchOutcome = future.result()
            except DocumentSourceError as error:
                outcome = error
            yield batch, outcome


def _store_batch(
    session: Session,
    batch: list[str],
    outcome: BatchOutcome,
    full_text_max_chars: int | None,
) -> Counter[str]:
    """Store one batch's documents and fetch statuses; count the outcomes."""
    now = datetime.now(timezone.utc)
    counts: Counter[str] = Counter()
    if isinstance(outcome, DocumentSourceError):
        for document_id in batch:
            session.merge(
                DocumentFetch(
                    document_id=document_id,
                    status="error",
                    message=str(outcome),
                    attempted_at=now,
                )
            )
        counts["error"] = len(batch)
        return counts

    record_map = {record.id: record for record in outcome}  # unrequested ids ignored
    for document_id in batch:
        record = record_map.get(document_id)
        if record is None:
            session.merge(
                DocumentFetch(
                    document_id=document_id,
                    status="not_found",
                    message=None,
                    attempted_at=now,
                )
            )
            counts["not_found"] += 1
            continue

        full_text = record.full_text
        if (
            full_text is not None
            and full_text_max_chars is not None
            and len(full_text) > full_text_max_chars
        ):
            full_text = None
            counts["full_texts_dropped"] += 1
        session.merge(
            Document(
                id=record.id,
                title=record.title,
                abstract=record.abstract,
                full_text=full_text,
                published_at=record.published_at,
                details=record.details,
                fetched_at=now,
            )
        )
        session.merge(
            DocumentFetch(
                document_id=document_id, status="ok", message=None, attempted_at=now
            )
        )
        counts["ok"] += 1
    return counts
