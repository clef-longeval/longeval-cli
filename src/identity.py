"""Stable identities used across snapshots."""

import hashlib
import re
import unicodedata
from urllib.parse import SplitResult, urlsplit, urlunsplit

_WHITESPACE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Apply simple query normalization by convention."""
    normalized = unicodedata.normalize("NFKC", text)
    return _WHITESPACE.sub(" ", normalized).strip().casefold()


def normalize_url(url: str) -> str:
    """Normalize a logged URL without following redirects."""
    parts = urlsplit(url.strip())
    hostname = (parts.hostname or "").lower()
    port = parts.port
    if port is not None and not (
        (parts.scheme.lower() == "http" and port == 80)
        or (parts.scheme.lower() == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"
    if parts.username:
        credentials = parts.username
        if parts.password:
            credentials += f":{parts.password}"
        hostname = f"{credentials}@{hostname}"
    normalized = SplitResult(
        scheme=parts.scheme.lower(),
        netloc=hostname,
        path=parts.path or "/",
        query=parts.query,
        fragment="",
    )
    return urlunsplit(normalized)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def query_id(text: str) -> str:
    return _sha256(text)


def document_id(url: str) -> str:
    return _sha256(normalize_url(url))
