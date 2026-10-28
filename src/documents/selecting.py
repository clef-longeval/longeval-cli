"""Selecting a snapshot's documents from its qrels and candidate pools.

A snapshot's collection is, in this order of precedence:

1. ``relevant`` — every document its qrels judge;
2. ``inherited`` — every document of the snapshot it extends, so a sequence
   of snapshots only grows;
3. ``published`` and ``negative`` — random samples of its candidate pools
   (see ``candidates.py``), each ``ratio`` × the number of qrels documents.

A document selected for several reasons keeps the first. Samples never pick
a document already selected, so a sample only falls short of its size when
its pool runs out. Sampling is seeded and a run replaces the snapshot's
previous selection, so equal inputs give equal selections.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from ..database import session_scope
from ..filtering import FilterError, SnapshotConfig
from ..models import DocumentCandidate, SnapshotDocument
from .fetching import find_snapshot_document_ids

SELECTION_REASONS = ("relevant", "inherited", "published", "negative")

_INSERT_CHUNK = 5_000


@dataclass(frozen=True)
class SelectionReport:
    selected: dict[str, int]  # reason -> documents selected for it
    wanted: dict[str, int]  # sampled reason -> sample size asked for

    @property
    def total(self) -> int:
        return sum(self.selected.values())


def select_documents(
    session_factory: sessionmaker[Session],
    config: SnapshotConfig,
    relevant_ids: Iterable[str],
    *,
    seed: int,
    extends: str | None = None,
    published_ratio: float = 0.5,
    negative_ratio: float = 0.5,
) -> SelectionReport:
    """Select the documents of ``config``'s snapshot and store them.

    ``relevant_ids`` are the snapshot's qrels documents; ``extends`` names an
    earlier snapshot whose whole selection is carried over.
    """
    # 1. Check the arguments before reading or writing anything.
    if not config.snapshot:
        raise FilterError("the filter needs a `snapshot` name to select documents")
    if extends == config.snapshot:
        raise ValueError(f"snapshot {extends!r} can't extend itself")
    ratio_map = {"published": published_ratio, "negative": negative_ratio}
    for reason, ratio in ratio_map.items():
        if ratio < 0:
            raise ValueError(f"the {reason} ratio must be >= 0, not {ratio}")
    relevant_list = sorted({str(document_id) for document_id in relevant_ids})
    if not relevant_list:
        raise ValueError("no relevant documents: is the qrels file empty?")

    with session_factory() as session:
        # 2. Relevant documents, then those of the extended snapshot.
        reason_map = dict.fromkeys(relevant_list, "relevant")
        if extends is not None:
            inherited_list = find_snapshot_document_ids(session, extends)
            if not inherited_list:
                raise ValueError(
                    f"snapshot {extends!r} has no selected documents to extend"
                )
            for document_id in inherited_list:
                reason_map.setdefault(document_id, "inherited")

        # 3. Seeded samples of each pool, skipping documents already selected.
        rng = random.Random(seed)
        wanted_map: dict[str, int] = {}
        for reason, ratio in ratio_map.items():
            wanted_map[reason] = int(len(relevant_list) * ratio)
            candidate_set = _find_candidate_ids(session, config.snapshot, reason)
            pool_list = sorted(candidate_set.difference(reason_map))
            size = min(wanted_map[reason], len(pool_list))
            for document_id in rng.sample(pool_list, size):
                reason_map[document_id] = reason

    # 4. Replace the snapshot's previous selection.
    _replace_selection(session_factory, config.snapshot, reason_map)
    counts = Counter(reason_map.values())
    return SelectionReport(
        selected={reason: counts[reason] for reason in SELECTION_REASONS},
        wanted=wanted_map,
    )


def _find_candidate_ids(session: Session, snapshot: str, reason: str) -> set[str]:
    return set(
        session.scalars(
            sa.select(DocumentCandidate.document_id).where(
                DocumentCandidate.snapshot == snapshot,
                DocumentCandidate.reason == reason,
            )
        )
    )


def _replace_selection(
    session_factory: sessionmaker[Session],
    snapshot: str,
    reason_map: dict[str, str],
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        {
            "snapshot": snapshot,
            "document_id": document_id,
            "reason": reason,
            "selected_at": now,
        }
        for document_id, reason in sorted(reason_map.items())
    ]
    with session_scope(session_factory) as session:
        session.execute(
            sa.delete(SnapshotDocument).where(SnapshotDocument.snapshot == snapshot)
        )
        for start in range(0, len(rows), _INSERT_CHUNK):
            session.execute(
                sa.insert(SnapshotDocument), rows[start : start + _INSERT_CHUNK]
            )
