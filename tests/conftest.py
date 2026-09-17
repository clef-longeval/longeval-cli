"""Shared test helpers for building small synthetic log archives."""

from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

from src.importing.loader import load_parser_module

#: scripts/longeval-sci-v2/parser.py isn't a normal package (the hyphen in
#: `longeval-sci-v2` makes it unimportable any other way), so tests reach
#: LongEvalSciParser via the same cached loader the CLI itself uses.
LongEvalSciParser = load_parser_module(None).LongEvalSciParser


def make_tar_gz(path: Path, members: dict[str, list[str]]) -> Path:
    """Write a streaming-friendly .tar.gz with one text member per key.

    ``members`` maps member name to a list of text lines (no trailing newlines).
    """
    with tarfile.open(path, "w:gz") as tar:
        for name, lines in members.items():
            content = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return path


def make_plain_gz(path: Path, lines: list[str]) -> Path:
    content = ("\n".join(lines) + "\n").encode("utf-8")
    with gzip.open(path, "wb") as handle:
        handle.write(content)
    return path


def sqlite_url(tmp_path: Path, name: str = "test.sqlite3") -> str:
    return f"sqlite+pysqlite:///{tmp_path / name}"
