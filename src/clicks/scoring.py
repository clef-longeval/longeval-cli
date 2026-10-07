"""Apply a fitted CLAX click model to a selection's candidates.

The candidates are the ``(query_id, doc_id)`` pairs shown in the query sessions
of whichever stream is passed (database snapshot or frozen export, see
``db_source.py``/``file_source.py``). Each query session is run through the model's
``predict_relevance`` and the per-pair probabilities are averaged over the
query sessions the pair appeared in. The result is the *scored docs* file every
exporter takes: Parquet with ``query_id``, ``doc_id`` and one numeric score
column, one row per pair, plus an ``query_sessions`` count.

The model only has parameters for the pairs of its :class:`QueryDocIdMap` (the one
saved at training time). Pairs the map doesn't know are skipped and counted
rather than scored, and items displayed below ``max_positions`` are not scored
either (use the training value). The helper needs NumPy and, to write,
PyArrow; it never imports CLAX or JAX itself.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..filtering import SnapshotConfig
from .clax_loaders import iter_batches
from .db_source import MAX_BATCH_SIZE, iter_query_sessions_from_db
from .file_source import iter_query_sessions_from_file
from .query_doc_ids import QueryDocIdMap
from .query_session import QuerySession


@dataclasses.dataclass(frozen=True)
class ScoringReport:
    query_sessions: int  # query sessions that had at least one known pair
    pairs: int  # distinct pairs scored
    unseen_items: int  # displayed items whose pair the model has no ID for


@dataclasses.dataclass(frozen=True)
class ScoredPairs:
    query_ids: list[str]
    doc_ids: list[str]
    scores: np.ndarray  # float64, mean probability over the pair's query sessions
    query_sessions: np.ndarray  # int32, how many query sessions each score averages
    report: ScoringReport


def score_candidates(
    model: Any,
    query_sessions: Iterable[QuerySession],
    query_doc_ids: QueryDocIdMap,
    *,
    batch_size: int = 1024,
    max_positions: int = 10,
) -> ScoredPairs:
    """Score every known pair in ``query_sessions`` with ``model.predict_relevance``.

    ``query_doc_ids`` must be the (frozen) mapping the model was trained with.
    CLAX returns log-probabilities; the score is their exponential, averaged
    per pair, so it lies in ``[0, 1]``. For models whose relevance does not
    depend on position or context, every query session of a pair gives the same
    value and the mean is just that value.
    """
    model.eval()
    sums = np.zeros(len(query_doc_ids), dtype=np.float64)
    counts = np.zeros(len(query_doc_ids), dtype=np.int64)
    unseen = [0]
    seen_query_sessions = 0
    known = _known_pairs_only(query_sessions, query_doc_ids, unseen)
    for batch in iter_batches(
        known, query_doc_ids, batch_size=batch_size, max_positions=max_positions
    ):
        seen_query_sessions += len(batch["n"])
        mask = batch["mask"]
        probs = np.exp(np.asarray(model.predict_relevance(batch), dtype=np.float64))
        np.add.at(sums, batch["query_doc_ids"][mask], probs[mask])
        np.add.at(counts, batch["query_doc_ids"][mask], 1)

    scored = np.flatnonzero(counts)
    pairs = list(query_doc_ids)
    query_ids = [pairs[i][0] for i in scored]
    doc_ids = [pairs[i][1] for i in scored]
    return ScoredPairs(
        query_ids,
        doc_ids,
        sums[scored] / counts[scored],
        counts[scored].astype(np.int32),
        ScoringReport(seen_query_sessions, len(scored), unseen[0]),
    )


def score_from_db(
    model: Any,
    query_doc_ids: QueryDocIdMap,
    config: SnapshotConfig,
    *,
    database_url: str | None = None,
    batch_size: int = 1024,
    max_positions: int = 10,
) -> ScoredPairs:
    """:func:`score_candidates` over the snapshot's query sessions in the database.

    ``database_url`` defaults like the CLI (``LONGEVAL_DATABASE_URL``, else the
    local SQLite file). The session lives only as long as the scoring does.
    """
    from ..database import create_database_engine, make_session_factory

    with make_session_factory(create_database_engine(database_url))() as session:
        return score_candidates(
            model,
            iter_query_sessions_from_db(
                session, config, batch_size=min(batch_size, MAX_BATCH_SIZE)
            ),
            query_doc_ids,
            batch_size=batch_size,
            max_positions=max_positions,
        )


def score_from_file(
    model: Any,
    query_doc_ids: QueryDocIdMap,
    export_dir: Path | str,
    *,
    batch_size: int = 1024,
    max_positions: int = 10,
) -> ScoredPairs:
    """:func:`score_candidates` over a frozen ``export clicks`` directory."""
    return score_candidates(
        model,
        iter_query_sessions_from_file(Path(export_dir)),
        query_doc_ids,
        batch_size=batch_size,
        max_positions=max_positions,
    )


def _known_pairs_only(
    query_sessions: Iterable[QuerySession],
    query_doc_ids: QueryDocIdMap,
    unseen: list[int],
) -> Iterator[QuerySession]:
    """Drop items whose pair has no ID (and query sessions left with none)."""
    for query_session in query_sessions:
        keep = [
            i
            for i, doc in enumerate(query_session.doc_ids)
            if query_doc_ids.lookup(query_session.query_id, doc) is not None
        ]
        unseen[0] += len(query_session.doc_ids) - len(keep)
        if not keep:
            continue
        if len(keep) == len(query_session.doc_ids):
            yield query_session
            continue
        yield dataclasses.replace(
            query_session,
            doc_ids=tuple(query_session.doc_ids[i] for i in keep),
            positions=tuple(query_session.positions[i] for i in keep),
            clicks=tuple(query_session.clicks[i] for i in keep),
        )


def write_scored_docs(scored: ScoredPairs, path: Path | str) -> None:
    """Write ``query_id``, ``doc_id``, ``score`` and ``query_sessions`` as Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "query_id": pa.array(scored.query_ids, type=pa.string()),
            "doc_id": pa.array(scored.doc_ids, type=pa.string()),
            "score": pa.array(scored.scores, type=pa.float64()),
            "query_sessions": pa.array(scored.query_sessions, type=pa.int32()),
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
