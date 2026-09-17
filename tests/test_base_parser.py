"""Tests for the shared BaseParser plumbing, independent of any one format.

BaseParser itself doesn't know what a "search" or "interaction" line looks
like — that's left to ``_is_search``/``_is_interaction``/``_parse_search``/
``_parse_interaction`` hooks a format subclasses. ``_FakeParser`` below is a
minimal concrete implementation used only to exercise the generic dispatch
loop (decoding, routing, rejection) in isolation from any real log format.
"""

from __future__ import annotations

import pytest

from src import identity
from src.importing.parser import BaseParser
from src.importing.records import RejectedRecord, SourceRecord
from src.importing.sources import SourceLine


class _FakeParser(BaseParser):
    def _is_search(self, raw):
        return raw.get("kind") == "search"

    def _is_interaction(self, raw):
        return raw.get("kind") == "interaction"

    def _parse_search(self, line, raw):
        return SourceRecord(line, raw)

    def _parse_interaction(self, line, raw):
        return SourceRecord(line, raw)


def test_query_id_normalizes_before_hashing():
    parser = BaseParser()

    assert parser.query_id("Hello  World") == parser.query_id("hello world")
    assert parser.query_id("Hello  World") == identity.query_id(
        identity.normalize_query("Hello  World")
    )


def test_unimplemented_hooks_raise():
    parser = BaseParser()
    line = SourceLine("ref", 1, "{}")

    with pytest.raises(NotImplementedError):
        parser._is_search({})
    with pytest.raises(NotImplementedError):
        parser._is_interaction({})
    with pytest.raises(NotImplementedError):
        parser._parse_search(line, {})
    with pytest.raises(NotImplementedError):
        parser._parse_interaction(line, {})


def test_iter_searches_dispatches_via_hooks():
    parser = _FakeParser()
    lines = [
        SourceLine("ref", 1, "not json"),
        SourceLine("ref", 2, '{"something_else": true}'),
        SourceLine("ref", 3, '{"kind": "interaction"}'),  # belongs to the other pass
        SourceLine("ref", 4, '{"kind": "search"}'),
    ]

    results = list(parser.iter_searches(lines))

    assert len(results) == 3
    assert [r.reason for r in results if isinstance(r, RejectedRecord)] == [
        "invalid-json",
        "unknown-record-shape",
    ]
    [accepted] = [r for r in results if isinstance(r, SourceRecord)]
    assert accepted.record == {"kind": "search"}


def test_iter_interactions_dispatches_via_hooks():
    parser = _FakeParser()
    lines = [
        SourceLine("ref", 1, "not json"),
        SourceLine("ref", 2, '{"kind": "search"}'),  # belongs to the other pass
        SourceLine("ref", 3, '{"kind": "interaction"}'),
    ]

    results = list(parser.iter_interactions(lines))

    # malformed JSON and lines belonging to the other pass are silently
    # skipped, not reported — the search pass already reports those once.
    [result] = results
    assert isinstance(result, SourceRecord)
    assert result.record == {"kind": "interaction"}
