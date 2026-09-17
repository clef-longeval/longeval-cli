"""Portable SQLAlchemy 2.x mappings for the interaction database.

The physical table and column names intentionally match the legacy live
database. The Python API uses clearer names where that does not alter storage.
Unmapped legacy columns are tolerated when connecting to an existing database.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
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


class QueryClassification(Base):
    """One classifier's verdict on one query.

    Keyed by ``(query_id, classifier)`` so re-running the same classifier
    upserts its previous verdict instead of accumulating duplicate rows,
    while a different classifier — including a "manual" one recording
    human review — gets its own row for the same query. ``label`` is a
    plain string rather than a foreign key into a labels table: it's a
    tag, not an entity with attributes of its own, so filtering on it
    (``WHERE label = 'spam'``) needs no join.
    """

    __tablename__ = "query_classifications"

    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id"), primary_key=True)
    classifier: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(128), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class InteractionClassification(Base):
    """One classifier's verdict on one interaction (see ``QueryClassification``)."""

    __tablename__ = "interaction_classifications"

    interaction_id: Mapped[int] = mapped_column(
        ForeignKey("interactions.id"), primary_key=True
    )
    classifier: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(128), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
