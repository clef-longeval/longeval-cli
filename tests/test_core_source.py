from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.documents import DocumentSourceError
from src.documents.loader import load_document_source_module

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "longeval-sci-v2"
CORE = load_document_source_module(EXAMPLE / "core_source.py")

WORK = {
    "id": 42,
    "title": "T",
    "abstract": "A",
    "fullText": "F",
    "publishedDate": "2025-04-01T00:00:00",
    "magId": 7,
    "authors": [{"name": " Ada "}, {"name": ""}],
    "oaiIds": ["oai:x:1"],
    "links": [{"type": "download", "url": "https://x"}, {"type": "reader"}],
}
START = datetime(2025, 3, 1, tzinfo=timezone.utc)
END = datetime(2025, 6, 1, tzinfo=timezone.utc)


def _source(responses):
    """A source whose requests return ``responses`` in turn, recording params."""
    source = CORE.CoreDocumentSource(api_key="key")
    calls = []

    def get_json(params):
        calls.append(params)
        return responses.pop(0)

    source._get_json = get_json
    return source, calls


def test_make_core_record_has_the_release_shape():
    record = CORE.make_core_record(WORK)

    assert (record.id, record.full_text) == ("42", "F")
    assert record.published_at == datetime(2025, 4, 1, tzinfo=timezone.utc)
    assert record.details["authors"] == [{"name": "Ada"}]
    assert record.details["links"] == [{"type": "download", "url": "https://x"}]
    assert record.details["magId"] == "7"
    assert record.details["publishedDate"] == "2025-04-01T00:00:00"


def test_parse_core_date_tolerates_bad_values():
    assert CORE.parse_core_date(None) is None
    assert CORE.parse_core_date("not a date") is None


def test_fetch_batch_searches_the_ids():
    source, calls = _source([{"results": [WORK]}])

    records = source.fetch_batch(["42", "43"])

    assert [record.id for record in records] == ["42"]
    assert calls == [{"q": "id:42 OR id:43", "limit": "2"}]


def test_iter_published_ids_scrolls_and_keeps_the_half_open_window():
    on_end = {"id": 2, "publishedDate": "2025-06-01T00:00:00"}
    undated = {"id": 3}
    source, calls = _source(
        [
            {"results": [WORK, on_end], "scrollId": "s1"},
            {"results": [undated], "scrollId": "s1"},
            {"results": [], "scrollId": "s1"},
        ]
    )

    assert list(source.iter_published_ids(START, END)) == ["42"]
    assert calls[0]["q"] == "publishedDate>=2025-03-01 AND publishedDate<=2025-06-01"
    assert calls[1] == {"scroll": "true", "scrollId": "s1"}


def test_negative_candidate_ids_are_the_next_ids():
    source = CORE.CoreDocumentSource(api_key="key")

    assert list(source.negative_candidate_ids(["41", "abc"])) == ["42"]


def test_requests_retry_rate_limits(monkeypatch):
    responses = [
        urllib.error.HTTPError(
            "url", 429, "Too Many Requests", {"X-RateLimit-Retry-After": "3"}, None
        ),
        io.BytesIO(json.dumps({"results": []}).encode()),
    ]
    sleeps = []

    def urlopen(request, timeout):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", sleeps.append)

    assert CORE.CoreDocumentSource(api_key="key").fetch_batch(["1"]) == []
    assert sleeps == [3]


def test_requests_give_up_with_a_source_error(monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(DocumentSourceError, match="after 2 tries"):
        CORE.CoreDocumentSource(api_key="key", max_tries=2).fetch_batch(["1"])


def test_requests_need_an_api_key(monkeypatch):
    monkeypatch.delenv("CORE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="CORE_API_KEY"):
        CORE.CoreDocumentSource().fetch_batch(["1"])
