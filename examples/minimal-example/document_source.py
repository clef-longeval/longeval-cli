"""A document source for the minimal example: no network, no provider.

``fetch-documents --source document_source.py`` asks it for documents by id. It
knows exactly the documents of ``ground_truth.tsv`` (the ones the synthetic
searches can show) and returns a made-up record for each: a title and abstract
built from the query the document belongs to, and a publication date. Any other
id is not found, like a document the provider does not have.

A real source subclasses ``BaseDocumentSource`` the same way and calls its
provider in ``fetch_batch`` (see ``examples/longeval-sci-v2/core_source.py``).
The optional hooks (``iter_published_ids``, ``negative_candidate_ids``) are left
out: the snapshot then has no ``published`` candidates, and its ``negative``
candidates are the documents the searches displayed.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.documents.source import BaseDocumentSource, DocumentRecord

GROUND_TRUTH = Path(__file__).parent / "ground_truth.tsv"
FIRST_PUBLISHED = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _load_topics() -> dict[str, str]:
    with GROUND_TRUTH.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {row["doc_id"]: row["query"] for row in rows}


class MinimalDocumentSource(BaseDocumentSource):
    batch_size = 5

    def __init__(self) -> None:
        self.topics = _load_topics()

    def fetch_batch(self, document_ids: list[str]) -> list[DocumentRecord]:
        records = []
        for document_id in document_ids:
            topic = self.topics.get(document_id)
            if topic is None:
                continue
            published = FIRST_PUBLISHED + timedelta(days=int(document_id) % 1800)
            records.append(
                DocumentRecord(
                    id=document_id,
                    title=f"A study of {topic} ({document_id})",
                    abstract=(
                        f"This synthetic paper is about {topic}. It exists only "
                        "to give the minimal example a document collection."
                    ),
                    published_at=published,
                    details={"doi": f"10.0000/minimal.{document_id}"},
                )
            )
        return records


source = MinimalDocumentSource()
