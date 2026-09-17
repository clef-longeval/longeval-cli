from __future__ import annotations

import json
from datetime import datetime, timezone

from src.importing.records import RejectedRecord, SourceRecord
from src.importing.sources import SourceLine
from tests.conftest import LongEvalSciParser


def _line(data: dict | str, number: int = 1, ref: str = "ref") -> SourceLine:
    text = data if isinstance(data, str) else json.dumps(data)
    return SourceLine(ref, number, text)


def test_parses_search_record():
    parser = LongEvalSciParser()
    data = {
        "date": 1730213984,
        "search_id": "abc",
        "serp": [31347699, 85868561],
        "query": "example",
        "uid": "u1",
    }

    [result] = list(parser.iter_searches([_line(data)]))

    assert isinstance(result, SourceRecord)
    record = result.record
    assert record.external_search_id == "abc"
    assert record.session_id == "u1"
    assert record.query == "example"
    assert record.document_ids == ("31347699", "85868561")
    assert record.timestamp == datetime.fromtimestamp(1730213984, tz=timezone.utc)


def test_search_without_uid_uses_anonymous_session():
    parser = LongEvalSciParser()
    data = {"date": 1, "search_id": "abc", "serp": [], "query": "q"}

    [result] = list(parser.iter_searches([_line(data)]))

    assert result.record.session_id == "anonymous:abc"
    assert result.record.document_ids == ()


def test_empty_serp_permitted():
    parser = LongEvalSciParser()
    data = {"date": 1, "search_id": "abc", "serp": [], "query": "q"}

    [result] = list(parser.iter_searches([_line(data)]))

    assert result.record.document_ids == ()


def test_interaction_parses_trackid_and_type():
    parser = LongEvalSciParser()
    data = {"date": 5, "trackId": "search1-doc1", "type": "works", "uid": "u1"}

    [result] = list(parser.iter_interactions([_line(data)]))

    record = result.record
    assert record.external_search_id == "search1"
    assert record.document_id == "doc1"
    assert record.interaction_type == "works"
    assert record.observed_session_id == "u1"


def test_interaction_missing_type_and_uid_permitted():
    parser = LongEvalSciParser()
    data = {"date": 5, "trackId": "search1-doc1"}

    [result] = list(parser.iter_interactions([_line(data)]))

    assert isinstance(result, SourceRecord)
    assert result.record.interaction_type is None
    assert result.record.observed_session_id is None


def test_hyphenated_search_id_uses_rsplit():
    parser = LongEvalSciParser()
    data = {"date": 5, "trackId": "search-with-hyphens-doc1"}

    [result] = list(parser.iter_interactions([_line(data)]))

    assert result.record.external_search_id == "search-with-hyphens"
    assert result.record.document_id == "doc1"


def test_undefined_search_id_in_track_id_rejected():
    parser = LongEvalSciParser()
    data = {"date": 5, "trackId": "undefined-doc1"}

    [result] = list(parser.iter_interactions([_line(data)]))

    assert isinstance(result, RejectedRecord)


def test_undefined_document_id_in_track_id_rejected():
    parser = LongEvalSciParser()
    data = {"date": 5, "trackId": "search1-undefined"}

    [result] = list(parser.iter_interactions([_line(data)]))

    assert isinstance(result, RejectedRecord)


def test_unicode_and_nul_characters_handled():
    parser = LongEvalSciParser()
    query = "café \x00search"
    data = {
        "date": 1,
        "search_id": "abc",
        "serp": [1],
        "query": query,
        "uid": "u\x00id",
    }

    [result] = list(parser.iter_searches([_line(data)]))

    assert "\x00" not in result.record.query
    assert "café" in result.record.query
    assert result.record.session_id == "uid"


def test_invalid_json_rejected():
    parser = LongEvalSciParser()

    [result] = list(parser.iter_searches([_line("{not valid json")]))

    assert isinstance(result, RejectedRecord)
    assert result.reason == "invalid-json"


def test_invalid_timestamp_rejected():
    parser = LongEvalSciParser()
    data = {"date": "not-a-number", "search_id": "abc", "serp": []}

    [result] = list(parser.iter_searches([_line(data)]))

    assert isinstance(result, RejectedRecord)


def test_unknown_record_shape_rejected():
    parser = LongEvalSciParser()
    data = {"date": 1, "something_else": True}

    [result] = list(parser.iter_searches([_line(data)]))

    assert isinstance(result, RejectedRecord)
    assert result.reason == "unknown-record-shape"


def test_interaction_type_filtering_uses_parser_configuration():
    parser = LongEvalSciParser(accepted_interaction_types=frozenset({"works"}))
    accepted = {"date": 1, "trackId": "s-d", "type": "works"}
    rejected = {"date": 1, "trackId": "s-d", "type": "data-provider"}

    [result_accepted] = list(parser.iter_interactions([_line(accepted)]))
    [result_rejected] = list(parser.iter_interactions([_line(rejected)]))

    assert isinstance(result_accepted, SourceRecord)
    assert isinstance(result_rejected, RejectedRecord)


def test_search_and_interaction_lines_are_mutually_ignored():
    parser = LongEvalSciParser()
    search_line = _line({"date": 1, "search_id": "s1", "serp": []})
    interaction_line = _line({"date": 1, "trackId": "s1-d1"})

    assert list(parser.iter_interactions([search_line])) == []
    assert list(parser.iter_searches([interaction_line])) == []


def test_malformed_json_not_double_reported_in_interaction_pass():
    parser = LongEvalSciParser()
    bad = _line("{not valid json")

    assert list(parser.iter_interactions([bad])) == []
