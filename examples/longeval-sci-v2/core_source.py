r"""Document source for CORE (https://core.ac.uk), the LongEval-Sci documents.

Fetches works from the CORE API v3 search endpoint, lists the works
published in a date window, and derives negative candidates as id + 1.
Set ``CORE_API_KEY`` first, e.g.::

    export CORE_API_KEY=...
    uv run longeval-cli fetch-documents \
        --source examples/longeval-sci-v2/core_source.py \
        --snapshot snapshot-1 --max-workers 5

Records from ``legacy_sqlite_source.py`` go through :func:`make_core_record`
too, so both sources give documents of the same shape.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from typing import Any

from src.documents import BaseDocumentSource, DocumentRecord, DocumentSourceError

API_URL = "https://api.core.ac.uk/v3/search/works"
USER_AGENT = "longeval-cli (CORE document source)"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
DETAIL_TEXT_KEYS = (
    "createdDate",
    "updatedDate",
    "publishedDate",
    "doi",
    "arxivId",
    "pubmedId",
    "magId",
)


def parse_core_date(value: Any) -> datetime | None:
    """Parse a CORE date (ISO 8601, naive means UTC); ``None`` if it isn't one."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def make_core_record(work: dict[str, Any]) -> DocumentRecord:
    """Turn a CORE work, as the API returns it, into a ``DocumentRecord``.

    ``details`` has the shape of the LongEval-Sci releases: authors keep only
    their name, links their type and URL, and ids and dates are strings.
    """
    details: dict[str, Any] = {
        key: _text(work.get(key)) for key in DETAIL_TEXT_KEYS
    }
    details["authors"] = [
        {"name": author["name"].strip()}
        for author in work.get("authors") or []
        if isinstance(author, dict)
        and isinstance(author.get("name"), str)
        and author["name"].strip()
    ]
    details["oaiIds"] = [
        str(oai_id).strip()
        for oai_id in work.get("oaiIds") or []
        if str(oai_id).strip()
    ]
    details["links"] = [
        {"type": link.get("type"), "url": link["url"].strip()}
        for link in work.get("links") or []
        if isinstance(link, dict)
        and isinstance(link.get("url"), str)
        and link["url"].strip()
    ]
    return DocumentRecord(
        id=str(work["id"]),
        title=_text(work.get("title")),
        abstract=_text(work.get("abstract")),
        full_text=_text(work.get("fullText")),
        published_at=parse_core_date(work.get("publishedDate")),
        details=details,
    )


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


class CoreDocumentSource(BaseDocumentSource):
    """The CORE API v3; ``api_key`` defaults to the ``CORE_API_KEY`` variable.

    ``batch_size`` is the number of ids per ``id:a OR id:b ...`` search.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        batch_size: int = 10,
        page_size: int = 100,
        max_tries: int = 5,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.batch_size = batch_size
        self.page_size = page_size
        self.max_tries = max_tries
        self.timeout = timeout

    def fetch_batch(self, document_ids: list[str]) -> list[DocumentRecord]:
        query = " OR ".join(f"id:{document_id}" for document_id in document_ids)
        response = self._get_json({"q": query, "limit": str(len(document_ids))})
        return [
            make_core_record(work)
            for work in response.get("results") or []
            if work.get("id") is not None
        ]

    def iter_published_ids(self, start: datetime, end: datetime) -> Iterator[str]:
        """Scroll through the works published in ``[start, end)``.

        CORE is asked for whole days, both ends included; works outside the
        exact window, or without a readable date, are left out here.
        """
        query = f"publishedDate>={start:%Y-%m-%d} AND publishedDate<={end:%Y-%m-%d}"
        response = self._get_json(
            {"q": query, "limit": str(self.page_size), "scroll": "true"}
        )
        while True:
            works = response.get("results") or []
            if not works:
                return
            for work in works:
                published_at = parse_core_date(work.get("publishedDate"))
                if work.get("id") is None or published_at is None:
                    continue
                if start <= published_at < end:
                    yield str(work["id"])
            scroll_id = response.get("scrollId")
            if not scroll_id:
                return
            response = self._get_json({"scroll": "true", "scrollId": scroll_id})

    def negative_candidate_ids(self, relevant_ids: Iterable[str]) -> Iterator[str]:
        """The next CORE id of each relevant work.

        CORE ids are assigned in ingestion order, not by topic, so the next id
        is in effect a random work. It may not exist: fetching then records
        it as not found.
        """
        for document_id in relevant_ids:
            if document_id.isdigit():
                yield str(int(document_id) + 1)

    def _get_json(self, params: dict[str, str]) -> dict[str, Any]:
        """GET the search endpoint, retrying rate limits and server errors."""
        api_key = self.api_key or os.environ.get("CORE_API_KEY")
        if not api_key:
            raise ValueError("set CORE_API_KEY to use the CORE document source")
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT},
        )
        problem = ""
        for attempt in range(1, self.max_tries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    raise ValueError(
                        f"CORE rejected the API key (HTTP {error.code})"
                    ) from error
                if error.code not in RETRY_STATUSES:
                    raise DocumentSourceError(
                        f"CORE answered HTTP {error.code}: {url}"
                    ) from error
                problem = f"HTTP {error.code}"
                wait = _retry_after(error) or 2 ** (attempt + 2)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                problem = f"{type(error).__name__}: {error}"
                wait = 2 ** (attempt + 2)
            if attempt < self.max_tries:
                time.sleep(wait)
        raise DocumentSourceError(
            f"CORE request failed after {self.max_tries} tries ({problem}): {url}"
        )


def _retry_after(error: urllib.error.HTTPError) -> int | None:
    """The seconds CORE asks to wait after a rate limit, if it says."""
    value = error.headers.get("X-RateLimit-Retry-After") if error.headers else None
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _error_body(error: urllib.error.HTTPError) -> str:
    """The start of an error response, to tell a bad key from a blocked client."""
    try:
        return error.read(300).decode("utf-8", "replace").strip()
    except OSError:
        return ""


# The settings LongEval-Sci-v2 was collected with: 100 ids per search, 2 tries.
source = CoreDocumentSource(batch_size=100, max_tries=2)
