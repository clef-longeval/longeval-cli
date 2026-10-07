"""Reading frozen query session exports (writing lives in ``src/exporting.py``).

Layout of an export directory::

    manifest.json
    query-sessions-00000.jsonl.gz      # one JSON query session per line
    query-sessions-00001.jsonl.gz
    ...

The manifest is written last, so a directory without one is an incomplete
export. Shards are gzip with a zeroed timestamp, so identical input gives
byte-identical shards and stable checksums. The manifest records the selection
(never the database URL), the accepted click types, counts, and each shard's
record count and SHA-256.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .query_session import (
    FORMAT_VERSION,
    QuerySession,
    QuerySessionFormatError,
    from_record,
)

MANIFEST_NAME = "manifest.json"
FORMAT_NAME = "longeval-sci-query-sessions"


def read_manifest(directory: Path) -> dict[str, Any]:
    path = Path(directory) / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise QuerySessionFormatError(
            f"{directory} has no {MANIFEST_NAME}; the export is missing or incomplete"
        ) from exc
    except json.JSONDecodeError as exc:
        raise QuerySessionFormatError(f"{path} is not valid JSON: {exc}") from exc
    if manifest.get("format") != FORMAT_NAME:
        raise QuerySessionFormatError(f"{path} is not a {FORMAT_NAME} manifest")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise QuerySessionFormatError(
            f"unsupported export version {manifest.get('format_version')!r} "
            f"(this reader understands {FORMAT_VERSION})"
        )
    return manifest


def iter_query_sessions_from_file(
    directory: Path, *, verify: bool = True
) -> Iterator[QuerySession]:
    """Stream an export's query sessions in their original order, one line at a time.

    Validates the manifest and every record's format version. With ``verify``
    (default) each shard's SHA-256 and record count are checked against the
    manifest; a checksum mismatch raises before the shard's first record, a
    record-count mismatch after its last.
    """
    directory = Path(directory)
    for shard in read_manifest(directory)["shards"]:
        path = directory / shard["path"]
        if verify:
            _check_checksum(path, shard)
        records = 0
        try:
            with gzip.open(path, "rt", encoding="utf-8") as text:
                for line_number, line in enumerate(text, start=1):
                    try:
                        query_session = from_record(json.loads(line))
                    except (json.JSONDecodeError, QuerySessionFormatError) as exc:
                        raise QuerySessionFormatError(
                            f"{path}:{line_number}: {exc}"
                        ) from exc
                    records += 1
                    yield query_session
        except (OSError, EOFError, zlib.error, UnicodeDecodeError) as exc:
            raise QuerySessionFormatError(
                f"{path} is not a readable shard: {exc}"
            ) from exc
        if verify and records != shard["records"]:
            raise QuerySessionFormatError(
                f"{path} holds {records} records, the manifest says {shard['records']}"
            )


def _check_checksum(path: Path, shard: dict[str, Any]) -> None:
    try:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise QuerySessionFormatError(f"{path} cannot be read: {exc}") from exc
    if digest != shard["sha256"]:
        raise QuerySessionFormatError(f"{path} does not match its manifest checksum")
