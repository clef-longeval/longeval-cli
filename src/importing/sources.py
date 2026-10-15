"""Reading raw search-log inputs into plain lines.

Turns whatever's on disk — files, directories, ``.gz``, and multi-member
``.tar.gz`` archives — into an ordered stream of :class:`SourceLine`s,
independent of what the lines mean. Nothing here parses JSON or knows about
searches/interactions; that's a format's own parser (``importing/parser.py``).
"""

from __future__ import annotations

import gzip
import hashlib
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SourceLine:
    """One line read from a source, before any parsing."""

    source_reference: str
    record_number: int
    text: str


def discover_inputs(inputs: list[Path]) -> list[Path]:
    """Expand ``inputs`` (files or directories) into a sorted list of files."""
    files: list[Path] = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
        else:
            files.append(path)
    return sorted(files)


def fingerprint_file(path: Path) -> str:
    """A stable content fingerprint for ``path``, e.g. to detect whether a
    previously-seen file has changed since."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_source(path: Path) -> Iterator[SourceLine]:
    """Yield every line in ``path`` as a :class:`SourceLine`.

    Recognizes a plain text file, a single-member ``.gz``, or a
    multi-member ``.tar.gz`` archive (each member streamed under its own
    ``source_reference``, so a record's position stays traceable).
    """
    if path.name.endswith(".tar.gz"):
        yield from _stream_tar_gz(path)
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield from _stream_lines(str(path), handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield from _stream_lines(str(path), handle)


def _stream_tar_gz(path: Path) -> Iterator[SourceLine]:
    # Stream mode ("r|"): members are read in one forward pass. ``getmembers()``
    # on "r:gz" first decompresses the whole archive just to list them (~30 s
    # for a 4.6 GB archive), then decompresses it again to read them.
    with tarfile.open(path, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            lines = (line.decode("utf-8") for line in handle)
            yield from _stream_lines(f"{path}:{member.name}", lines)


def _stream_lines(source_reference: str, lines) -> Iterator[SourceLine]:
    for record_number, raw_line in enumerate(lines, start=1):
        yield SourceLine(source_reference, record_number, raw_line.rstrip("\n"))
