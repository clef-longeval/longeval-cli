from __future__ import annotations

import pytest

from src.documents.loader import load_document_source
from src.documents.source import BaseDocumentSource

ECHO_SOURCE = (
    "from src.documents.source import BaseDocumentSource, DocumentRecord\n"
    "\n"
    "\n"
    "class EchoSource(BaseDocumentSource):\n"
    "    def fetch_batch(self, document_ids):\n"
    "        return [DocumentRecord(id=i) for i in document_ids]\n"
    "\n"
    "\n"
    "source = EchoSource()\n"
)


def test_load_document_source_loads_the_source_object(tmp_path):
    path = tmp_path / "source.py"
    path.write_text(ECHO_SOURCE)

    source = load_document_source(path)

    assert isinstance(source, BaseDocumentSource)
    assert [record.id for record in source.fetch_batch(["1", "2"])] == ["1", "2"]
    assert load_document_source(path) is source  # cached by path


def test_load_document_source_rejects_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_document_source(tmp_path / "missing.py")


def test_load_document_source_requires_a_top_level_source(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("not_source = 1\n")

    with pytest.raises(ValueError, match="top-level `source`"):
        load_document_source(path)


def test_load_document_source_requires_a_base_document_source(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("source = object()\n")

    with pytest.raises(ValueError, match="BaseDocumentSource"):
        load_document_source(path)
