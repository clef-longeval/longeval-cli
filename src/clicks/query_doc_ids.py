"""An in-memory ``(query_id, doc_id) -> integer`` mapping, saved as Parquet.

CLAX looks up one embedding row per query–document pair, so every pair needs a
stable integer. During training the mapping is a plain dict; afterwards
:meth:`QueryDocIdMap.save` writes it as Parquet next to the fitted model, and
:meth:`QueryDocIdMap.load` reads it back (frozen) so the same pairs can be scored.

IDs are dense and assigned in first-seen order starting at 0, so the same
query session stream always yields the same IDs, and ``len(mapping)`` is the
number of embedding rows the model needs (CLAX's ``query_doc_pairs``). Padding
positions reuse ID 0 but are masked out by CLAX.

The dict grows with the number of distinct pairs, so it must fit in RAM; for a
very large log use a hashed embedding or a different mapping. Needs
``pyarrow`` for ``save``/``load`` only.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

#: CLAX stores ``query_doc_ids`` as int32.
MAX_QUERY_DOC_ID = 2**31 - 2


class QueryDocIdMap:
    """Assigns IDs on first sight; a loaded map is frozen and raises on unseen pairs."""

    def __init__(self) -> None:
        self._ids: dict[tuple[str, str], int] = {}
        self.frozen = False

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """The ``(query_id, doc_id)`` pairs in ID order."""
        return iter(self._ids)

    def lookup(self, query_id: str, doc_id: str) -> int | None:
        """The pair's ID, or ``None`` if unseen. Never assigns."""
        return self._ids.get((query_id, doc_id))

    def assign(self, pairs: Iterable[tuple[str, str]]) -> list[int]:
        """IDs for ``pairs`` in order, creating unseen ones unless frozen."""
        ids = []
        for pair in pairs:
            known = self._ids.get(pair)
            if known is None:
                if self.frozen:
                    raise KeyError(f"unknown pair {pair!r}")
                if len(self._ids) > MAX_QUERY_DOC_ID:
                    raise OverflowError("more pairs than fit in an int32 ID")
                known = self._ids[pair] = len(self._ids)
            ids.append(known)
        return ids

    def save(self, path: Path | str) -> None:
        """Write ``id, query_id, doc_id`` rows (ordered by id) as Parquet."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        queries, docs = zip(*self._ids) if self._ids else ((), ())
        table = pa.table(
            {
                "id": pa.array(range(len(self._ids)), type=pa.int32()),
                "query_id": pa.array(queries, type=pa.string()),
                "doc_id": pa.array(docs, type=pa.string()),
            }
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd")

    @classmethod
    def load(cls, path: Path | str) -> QueryDocIdMap:
        """Read a saved mapping; IDs must be dense ``0..n-1`` as ``save`` writes."""
        import pyarrow.parquet as pq

        table = pq.read_table(path).to_pydict()
        mapping = cls()
        for id_, query_id, doc_id in zip(
            table["id"], table["query_id"], table["doc_id"], strict=True
        ):
            if id_ != len(mapping._ids):
                raise ValueError(f"{path}: pair IDs are not dense and ordered")
            mapping._ids[(query_id, doc_id)] = id_
        mapping.frozen = True
        return mapping
