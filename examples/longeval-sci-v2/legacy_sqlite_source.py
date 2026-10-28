r"""Document source reading the CORE works database of the earlier scripts.

The scripts that built LongEval-Sci before this repository downloaded CORE
works into a SQLite file (``core_works.sqlite``: tables ``works``,
``authors``, ``work_authors``, ``oai_ids`` and ``links``). This source reads
documents from that file, so they need not be downloaded again::

    export LEGACY_CORE_DB=data/core_data/documents/core_works.sqlite
    uv run longeval-cli fetch-documents \
        --source examples/longeval-sci-v2/legacy_sqlite_source.py \
        --snapshot snapshot-1

Ids missing from the file are stored as not found, including those the old
scripts failed to download, so fetch the rest from the API afterwards with
``core_source.py`` and ``--retry-not-found``.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from src.documents import BaseDocumentSource, DocumentRecord
from src.documents.loader import load_document_source_module

# Loaded through the plugin loader: example files are not an importable package.
make_core_record = load_document_source_module(
    Path(__file__).with_name("core_source.py")
).make_core_record

WORK_COLUMNS = (
    "id",
    "title",
    "abstract",
    "createdDate",
    "updatedDate",
    "publishedDate",
    "doi",
    "arxivId",
    "pubmedId",
    "magId",
    "fullText",
)
_WORKS_SQL = f"SELECT {', '.join(WORK_COLUMNS)} FROM works WHERE id IN ({{}})"
_AUTHORS_SQL = (
    "SELECT wa.work_id, a.name FROM work_authors wa "
    "JOIN authors a ON a.author_id = wa.author_id "
    "WHERE wa.work_id IN ({}) ORDER BY wa.work_id, wa.author_pos"
)
_OAI_IDS_SQL = (
    "SELECT work_id, oai_id FROM oai_ids WHERE work_id IN ({}) "
    "ORDER BY work_id, oai_id"
)
_LINKS_SQL = (
    "SELECT work_id, type, url FROM links WHERE work_id IN ({}) "
    "ORDER BY work_id, url"
)


class LegacySqliteSource(BaseDocumentSource):
    """A legacy ``core_works.sqlite``; ``path`` defaults to ``LEGACY_CORE_DB``."""

    batch_size = 500  # ids per `IN (...)`, under SQLite's oldest limit of 999

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = path

    def fetch_batch(self, document_ids: list[str]) -> list[DocumentRecord]:
        path = self.path or os.environ.get("LEGACY_CORE_DB")
        if not path:
            raise ValueError("set LEGACY_CORE_DB to the legacy core_works.sqlite")
        if not Path(path).is_file():
            raise ValueError(f"legacy database not found: {path}")
        placeholders = ", ".join("?" * len(document_ids))
        uri = f"{Path(path).resolve().as_uri()}?mode=ro"
        # One read-only connection per call: batches run on worker threads.
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            work_map: dict[str, dict[str, Any]] = {}
            for row in connection.execute(
                _WORKS_SQL.format(placeholders), document_ids
            ):
                work = dict(zip(WORK_COLUMNS, row, strict=True))
                work.update(authors=[], oaiIds=[], links=[])
                work_map[str(row[0])] = work
            for work_id, name in connection.execute(
                _AUTHORS_SQL.format(placeholders), document_ids
            ):
                if str(work_id) in work_map:
                    work_map[str(work_id)]["authors"].append({"name": name})
            for work_id, oai_id in connection.execute(
                _OAI_IDS_SQL.format(placeholders), document_ids
            ):
                if str(work_id) in work_map:
                    work_map[str(work_id)]["oaiIds"].append(oai_id)
            for work_id, link_type, url in connection.execute(
                _LINKS_SQL.format(placeholders), document_ids
            ):
                if str(work_id) in work_map:
                    work_map[str(work_id)]["links"].append(
                        {"type": link_type, "url": url}
                    )
        return [make_core_record(work) for work in work_map.values()]


source = LegacySqliteSource()
