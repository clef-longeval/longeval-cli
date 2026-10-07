"""The connection to CLAX: query sessions as the batches its click models consume.

``loaders_from_db`` and ``loaders_from_file`` return the
``(query_doc_ids, train, validation)`` CLAX's ``Trainer`` needs; ``iter_batches`` is the
batching they and the scoring helper (``scoring.py``) share.

Layout follows CLAX 0.1.1's own datasets (``clax.datasets.YandexDataset`` and
``SessionCollator``): one row per query session, ``positions`` 1-based ``int16``,
``query_doc_ids`` ``int32``, ``clicks`` ``float16``, ``mask`` ``bool`` and ``n``
``int16``, zero-padded on the right. This module needs only NumPy, so it works
with either query session source and does not import CLAX or JAX; install the
``clax`` extra to train.

Padding is deterministic: every batch is padded to the fixed ``max_positions``
(static shapes avoid JAX recompilation), padded cells are 0 / ``False``, and
only the final batch may have fewer than ``batch_size`` rows. Items displayed
at a position above ``max_positions`` are dropped (CLAX truncates likewise).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from ..filtering import SnapshotConfig
from .db_source import MAX_BATCH_SIZE, iter_query_sessions_from_db
from .file_source import iter_query_sessions_from_file
from .query_doc_ids import QueryDocIdMap
from .query_session import QuerySession


def iter_batches(
    query_sessions: Iterable[QuerySession],
    query_doc_ids: QueryDocIdMap,
    *,
    batch_size: int,
    max_positions: int = 10,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield CLAX batches, holding at most ``batch_size`` query sessions in memory.

    ``query_doc_ids`` assigns IDs as pairs appear (unless opened read-only, in
    which case an unseen pair raises ``KeyError``).
    """
    if batch_size < 1 or max_positions < 1:
        raise ValueError("batch_size and max_positions must be positive")

    buffer: list[QuerySession] = []
    for query_session in query_sessions:
        buffer.append(query_session)
        if len(buffer) == batch_size:
            yield _make_batch(buffer, query_doc_ids, max_positions)
            buffer = []
    if buffer:
        yield _make_batch(buffer, query_doc_ids, max_positions)


def _make_batch(
    query_sessions: list[QuerySession], query_doc_ids: QueryDocIdMap, max_positions: int
) -> dict[str, np.ndarray]:
    rows = len(query_sessions)
    batch = {
        "query_doc_ids": np.zeros((rows, max_positions), dtype=np.int32),
        "positions": np.zeros((rows, max_positions), dtype=np.int16),
        "clicks": np.zeros((rows, max_positions), dtype=np.float16),
        "mask": np.zeros((rows, max_positions), dtype=np.bool_),
        "n": np.zeros(rows, dtype=np.int16),
    }
    for row, query_session in enumerate(query_sessions):
        kept = [
            i for i, p in enumerate(query_session.positions) if p <= max_positions
        ]
        ids = query_doc_ids.assign(
            (query_session.query_id, query_session.doc_ids[i]) for i in kept
        )
        n = len(kept)
        batch["query_doc_ids"][row, :n] = ids
        batch["positions"][row, :n] = [query_session.positions[i] for i in kept]
        batch["clicks"][row, :n] = [query_session.clicks[i] for i in kept]
        batch["mask"][row, :n] = True
        batch["n"][row] = n
    return batch


class BatchLoader:
    """CLAX batches of one split, re-iterable: a fresh pass over the source each time.

    Not a ``DataLoader``: CLAX's ``Trainer`` only iterates, and ``__len__`` (the
    number of batches) is there for its progress bar.
    """

    def __init__(
        self,
        open_stream: Callable[[], Iterable[QuerySession]],
        query_doc_ids: QueryDocIdMap,
        *,
        validation: bool,
        holdout: int,
        query_sessions: int,
        batch_size: int,
        max_positions: int,
    ) -> None:
        self.open_stream, self.query_doc_ids = open_stream, query_doc_ids
        self.validation, self.holdout = validation, holdout
        self.batch_size, self.max_positions = batch_size, max_positions
        # indices 0, holdout, 2*holdout...
        in_validation = -(-query_sessions // holdout)
        self.size = in_validation if validation else query_sessions - in_validation

    def __len__(self) -> int:
        return -(-self.size // self.batch_size)

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        split = (
            query_session
            for i, query_session in enumerate(self.open_stream())
            if (i % self.holdout == 0) == self.validation
        )
        yield from iter_batches(
            split,
            self.query_doc_ids,
            batch_size=self.batch_size,
            max_positions=self.max_positions,
        )


def loaders_from_stream(
    open_stream: Callable[[], Iterable[QuerySession]],
    *,
    holdout: int = 10,
    batch_size: int = 1024,
    max_positions: int = 10,
) -> tuple[QueryDocIdMap, BatchLoader, BatchLoader]:
    """Return ``(query_doc_ids, train, validation)`` ready for CLAX's ``Trainer``.

    The general form of :func:`loaders_from_db`, for any query session source.
    ``open_stream`` must return the same query sessions in the same order every
    call (true for an export, and for a database that doesn't change); it is
    called once here and once per epoch and split. Every ``holdout``-th
    query session is validation. CLAX needs the embedding table size up front, so
    one first pass assigns every pair ID, then freezes ``query_doc_ids``.
    """
    if holdout < 2:
        raise ValueError("holdout must be at least 2")
    query_doc_ids = QueryDocIdMap()
    query_sessions = 0
    for batch in iter_batches(
        open_stream(), query_doc_ids, batch_size=batch_size, max_positions=max_positions
    ):
        query_sessions += len(batch["n"])
    if not len(query_doc_ids):
        raise ValueError("the selection contains no query_sessions")
    query_doc_ids.frozen = True
    
    def loader(validation: bool) -> BatchLoader:
        return BatchLoader(
            open_stream,
            query_doc_ids,
            validation=validation,
            holdout=holdout,
            query_sessions=query_sessions,
            batch_size=batch_size,
            max_positions=max_positions,
        )

    return query_doc_ids, loader(False), loader(True)


def loaders_from_db(
    session: Session,
    config: SnapshotConfig,
    *,
    holdout: int = 10,
    batch_size: int = 1024,
    max_positions: int = 10,
) -> tuple[QueryDocIdMap, BatchLoader, BatchLoader]:
    """:func:`loaders_from_stream` over the snapshot's query sessions in the database.

    ``session`` must stay open while the loaders are used (they read lazily),
    and the database must not change meanwhile. Every ``holdout``-th query session
    (about ``1/holdout``) is validation. The database is paged independently of
    ``batch_size``, so any batch size works.
    """
    page_size = min(batch_size, MAX_BATCH_SIZE)
    return loaders_from_stream(
        lambda: iter_query_sessions_from_db(session, config, batch_size=page_size),
        holdout=holdout,
        batch_size=batch_size,
        max_positions=max_positions,
    )


def loaders_from_file(
    export_dir: Path | str,
    *,
    holdout: int = 10,
    batch_size: int = 1024,
    max_positions: int = 10,
) -> tuple[QueryDocIdMap, BatchLoader, BatchLoader]:
    """:func:`loaders_from_stream` over a frozen ``export clicks`` directory.

    The export is read, and its checksums verified, on every pass, so the
    query sessions are identical and in the same order each time.
    """
    return loaders_from_stream(
        lambda: iter_query_sessions_from_file(Path(export_dir)),
        holdout=holdout,
        batch_size=batch_size,
        max_positions=max_positions,
    )
