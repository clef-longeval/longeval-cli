"""Portable SQLAlchemy 2.x mappings for the interaction database.

The physical table and column names intentionally match the legacy live
database. The Python API uses clearer names where that does not alter storage.
Unmapped legacy columns are tolerated when connecting to an existing database.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Query(Base):
    """A unique search query"""
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str | None] = mapped_column("query", String(2048), index=True)

    results: Mapped[list[SearchResult]] = relationship(back_populates="query")


class SearchSession(Base):
    """A single participant session."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    started_at: Mapped[datetime] = mapped_column("start", DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column("end", DateTime(timezone=True))
    ranking_system_id: Mapped[int | None] = mapped_column("system_ranking", Integer)

    results: Mapped[list[SearchResult]] = relationship(back_populates="session")
    interactions: Mapped[list[Interaction]] = relationship(back_populates="session")


class SearchResult(Base):
    """One paginated search response recorded from a user's session.

    ``items`` maps a string result index to a dictionary containing the
    document ID and ranker type. Ranker type can be ignored for now.

    {
        '1': {'docid': 'M27622217', 'type': 'BASE'},
        '2': {'docid': 'M27251231', 'type': 'BASE'},
        '3': {'docid': 'M27692969', 'type': 'BASE'}
        ...
    }
    """

    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id"), index=True
    )
    query_id: Mapped[str | None] = mapped_column(ForeignKey("queries.id"), index=True)
    system_id: Mapped[int | None] = mapped_column(Integer, index=True)
    queried_at: Mapped[datetime | None] = mapped_column(
        "q_date", DateTime(timezone=True)
    )
    total_results: Mapped[int | None] = mapped_column("num_found", Integer)
    page: Mapped[int | None] = mapped_column(Integer)
    results_per_page: Mapped[int | None] = mapped_column("rpp", Integer)
    items: Mapped[Any | None] = mapped_column(JSON)

    query: Mapped[Query | None] = relationship(back_populates="results")
    session: Mapped[SearchSession | None] = relationship(back_populates="results")
    interactions: Mapped[list[Interaction]] = relationship(back_populates="result")


class Interaction(Base):
    """A click interaction associated with a session and displayed result item."""

    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[datetime | None] = mapped_column("date", DateTime(timezone=True))
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id"), index=True
    )
    result_id: Mapped[int] = mapped_column(ForeignKey("results.id"), index=True)
    document_id: Mapped[str] = mapped_column(String(255), index=True)
    type: Mapped[str | None] = mapped_column("type", String(64), index=True)

    session: Mapped[SearchSession | None] = relationship(back_populates="interactions")
    result: Mapped[SearchResult] = relationship(back_populates="interactions")


class SearchReference(Base):
    """Joins a raw log's own ``search_id`` to the ``SearchResult`` it produced.

    Has no legacy counterpart — it exists purely so a later, separate
    ``import-clicks`` run can resolve a click's search id back to a result,
    and so re-running ``import-queries`` over the same lines is a no-op.
    Written in the same transaction as its ``SearchResult``, so (unlike an
    external mapping store) it never needs separate crash recovery.
    """

    __tablename__ = "search_references"

    search_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    result_id: Mapped[int] = mapped_column(
        ForeignKey("results.id"), unique=True, index=True
    )


class InteractionReference(Base):
    """Records that one raw interaction *line* was already imported.

    Keyed by the line's position (``"{source_reference}#{record_number}"``),
    not its content: unlike searches, two lines with identical content are
    legitimately separate interaction events, so idempotency has to be about
    "have I read this exact line before", not "have I seen this content".
    """

    __tablename__ = "interaction_references"

    interaction_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    interaction_id: Mapped[int] = mapped_column(
        ForeignKey("interactions.id"), unique=True, index=True
    )


class Verdict:
    """The columns every ``*_classifications`` table shares (a mixin).

    One row is one classifier's verdict on one row of the labelled table,
    keyed by ``(<entity>_id, classifier, snapshot)``: re-running a classifier
    upserts its own rows, a different classifier (e.g. a ``"manual"`` one)
    gets its own. ``snapshot`` names the snapshot a verdict was computed over
    (e.g. ``"snapshot-1"`` for a date-range click count); the empty string is
    global. ``label`` is a plain string, so a filter is a ``WHERE label = ...``
    with no join; ``value`` carries a number (a count, a delay) a filter can
    threshold on.
    """

    classifier: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot: Mapped[str] = mapped_column(String(128), primary_key=True, default="")
    label: Mapped[str] = mapped_column(String(128), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    value: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


# ``sort_order`` puts the entity column first in each primary key.
class QueryClassification(Verdict, Base):
    __tablename__ = "query_classifications"

    query_id: Mapped[str] = mapped_column(
        ForeignKey("queries.id"), primary_key=True, sort_order=-1
    )


class SessionClassification(Verdict, Base):
    __tablename__ = "session_classifications"

    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id"), primary_key=True, sort_order=-1
    )


class ResultClassification(Verdict, Base):
    __tablename__ = "result_classifications"

    result_id: Mapped[int] = mapped_column(
        ForeignKey("results.id"), primary_key=True, sort_order=-1
    )


class InteractionClassification(Verdict, Base):
    __tablename__ = "interaction_classifications"

    interaction_id: Mapped[int] = mapped_column(
        ForeignKey("interactions.id"), primary_key=True, sort_order=-1
    )


class DocumentClassification(Verdict, Base):
    __tablename__ = "document_classifications"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id"), primary_key=True, sort_order=-1
    )


#: The tables a classifier can label ("levels"), each with its verdict table
#: and that table's column naming the labelled row.
LEVELS = {
    "queries": (QueryClassification, QueryClassification.query_id),
    "sessions": (SessionClassification, SessionClassification.session_id),
    "results": (ResultClassification, ResultClassification.result_id),
    "interactions": (
        InteractionClassification,
        InteractionClassification.interaction_id,
    ),
    "documents": (DocumentClassification, DocumentClassification.document_id),
}


class Document(Base):
    """A collection document, as a document source plugin fetched it.

    Has no legacy counterpart. ``id`` is the same string the logs use (an
    interaction's ``document_id``, a result item's ``docid``), so a snapshot's
    qrels join to it directly. The fields filters and exports need are
    columns; everything else a source returns (authors, DOI, links, ...) is
    kept as-is in ``details`` and released unchanged.
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    full_text: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DocumentFetch(Base):
    """The outcome of the latest attempt to fetch one document id.

    ``status`` is ``"ok"``, ``"not_found"`` (the source has no such document,
    e.g. a negative id that doesn't exist) or ``"error"`` (the request failed;
    ``message`` says why). A re-run of ``fetch-documents`` skips ``"ok"`` and
    ``"not_found"`` ids and retries ``"error"`` ones. Not-found ids never get
    a ``Document``, so there is no foreign key to ``documents``.
    """

    __tablename__ = "document_fetches"

    document_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    message: Mapped[str | None] = mapped_column(Text)
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DocumentCandidate(Base):
    """A document id that ``select-documents`` may sample for a snapshot.

    Written by ``collect-document-candidates``. ``reason`` names the pool the
    id belongs to: ``"published"`` (published in the snapshot's window) or
    ``"negative"`` (a random non-relevant document). The id need not be
    fetched yet, so there is no foreign key to ``documents``.
    """

    __tablename__ = "document_candidates"

    snapshot: Mapped[str] = mapped_column(String(128), primary_key=True)
    reason: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SnapshotDocument(Base):
    """One document selected for a snapshot's collection.

    Written by ``select-documents``. ``reason`` is why it was selected:
    ``"relevant"`` (in the snapshot's qrels), ``"inherited"`` (in the snapshot
    it extends), ``"published"`` or ``"negative"`` (sampled from its
    candidates); a document selected for several reasons keeps the first in
    that order. Ids are selected before they are fetched, so there is no
    foreign key to ``documents``.
    """

    __tablename__ = "snapshot_documents"

    snapshot: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    reason: Mapped[str] = mapped_column(String(32), index=True)
    selected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
