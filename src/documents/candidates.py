"""Collecting a snapshot's candidate documents, for ``select-documents`` to sample.

Two pools, stored as ``DocumentCandidate`` rows under the snapshot's name:

- ``published`` — the documents the source lists as published in the
  filter's ``[published_from, published_to)``.
- ``negative`` — documents unlikely to be relevant: what the source's
  ``negative_candidate_ids`` derives from the qrels documents if it has that
  hook, otherwise every document displayed in the snapshot's searches. The
  qrels documents themselves are never negatives.

Collecting a pool replaces it in one transaction, once the whole listing has
succeeded, so a pool always comes from one complete run.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from tqdm.auto import tqdm

from ..clicks import iter_query_sessions_from_db
from ..database import session_scope
from ..filtering import FilterError, SnapshotConfig
from ..models import DocumentCandidate
from .source import BaseDocumentSource

CANDIDATE_REASONS = ("published", "negative")

_INSERT_CHUNK = 5_000


def collect_candidates(
    session_factory: sessionmaker[Session],
    source: BaseDocumentSource,
    config: SnapshotConfig,
    *,
    reasons: Iterable[str] = CANDIDATE_REASONS,
    relevant_ids: Iterable[str] | None = None,
) -> dict[str, int]:
    """Collect each pool in ``reasons``; return the candidates stored per pool.

    ``relevant_ids`` (the snapshot's qrels documents) is needed for negatives.
    Arguments are checked before any pool is collected.
    """
    reason_list = list(dict.fromkeys(reasons))
    unknown = set(reason_list) - set(CANDIDATE_REASONS)
    if unknown:
        raise ValueError(f"unknown candidate reason(s): {sorted(unknown)!r}")
    if "negative" in reason_list and relevant_ids is None:
        raise ValueError("negative candidates need the relevant (qrels) document ids")

    counts: dict[str, int] = {}
    if "published" in reason_list:
        counts["published"] = collect_published_candidates(
            session_factory, source, config
        )
    if "negative" in reason_list:
        counts["negative"] = collect_negative_candidates(
            session_factory, source, config, relevant_ids or ()
        )
    return counts


def collect_published_candidates(
    session_factory: sessionmaker[Session],
    source: BaseDocumentSource,
    config: SnapshotConfig,
) -> int:
    """Store the documents published in the filter's window; return how many."""
    _require_snapshot(config)
    if config.published_start is None or config.published_end is None:
        raise FilterError(
            "the filter needs `published_from` and `published_to` "
            "to collect published candidates"
        )
    if not _overrides(source, "iter_published_ids"):
        raise ValueError(
            f"{type(source).__name__} can't list documents by publication date "
            "(it has no iter_published_ids)"
        )
    listing = source.iter_published_ids(config.published_start, config.published_end)
    document_id_set = {
        str(document_id)
        for document_id in tqdm(listing, desc="Listing published documents")
    }
    return _replace_candidates(
        session_factory, config.snapshot, "published", document_id_set
    )


def collect_negative_candidates(
    session_factory: sessionmaker[Session],
    source: BaseDocumentSource,
    config: SnapshotConfig,
    relevant_ids: Iterable[str],
) -> int:
    """Store the snapshot's negative candidates; return how many."""
    _require_snapshot(config)
    relevant_set = {str(document_id) for document_id in relevant_ids}
    if _overrides(source, "negative_candidate_ids"):
        candidate_set = {
            str(document_id)
            for document_id in source.negative_candidate_ids(sorted(relevant_set))
        }
    else:
        with session_factory() as session:
            candidate_set = _displayed_document_ids(session, config)
    return _replace_candidates(
        session_factory, config.snapshot, "negative", candidate_set - relevant_set
    )


def _require_snapshot(config: SnapshotConfig) -> None:
    if not config.snapshot:
        raise FilterError("the filter needs a `snapshot` name to collect candidates")


def _overrides(source: BaseDocumentSource, hook: str) -> bool:
    """Whether ``source``'s class implements the optional ``hook``."""
    return getattr(type(source), hook) is not getattr(BaseDocumentSource, hook)


def _displayed_document_ids(session: Session, config: SnapshotConfig) -> set[str]:
    """Every document displayed in the searches the snapshot's filter selects."""
    return {
        document_id
        for query_session in iter_query_sessions_from_db(session, config)
        for document_id in query_session.doc_ids
    }


def _replace_candidates(
    session_factory: sessionmaker[Session],
    snapshot: str,
    reason: str,
    document_ids: Iterable[str],
) -> int:
    now = datetime.now(timezone.utc)
    rows = [
        {
            "snapshot": snapshot,
            "reason": reason,
            "document_id": document_id,
            "created_at": now,
        }
        for document_id in sorted(document_ids)
    ]
    with session_scope(session_factory) as session:
        session.execute(
            sa.delete(DocumentCandidate).where(
                DocumentCandidate.snapshot == snapshot,
                DocumentCandidate.reason == reason,
            )
        )
        for start in range(0, len(rows), _INSERT_CHUNK):
            session.execute(
                sa.insert(DocumentCandidate), rows[start : start + _INSERT_CHUNK]
            )
    return len(rows)
