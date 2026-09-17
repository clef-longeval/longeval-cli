"""Runs a classifier over the DB and upserts its verdicts.

Each row is keyed by ``(entity_id, classifier.name)`` (see ``models.py``), so
rerunning the same classifier — e.g. after import picks up new rows —
updates its previous verdict instead of accumulating duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

import sqlalchemy as sa

from ..database import create_database_engine, create_schema, make_session_factory
from ..models import (
    Interaction,
    InteractionClassification,
    Query,
    QueryClassification,
)
from .records import (
    Classification,
    InteractionClassifier,
    InteractionView,
    QueryClassifier,
)


@dataclass
class ClassifierConfig:
    database_url: str
    classifier: QueryClassifier | InteractionClassifier
    batch_size: int = 5_000
    limit: int | None = None


@dataclass
class ClassificationStats:
    considered: int = 0
    classified: int = 0
    unlabeled: int = 0


def classify_queries(config: ClassifierConfig) -> ClassificationStats:
    """Run ``config.classifier`` (a :class:`QueryClassifier`) over every query."""
    engine = create_database_engine(config.database_url)
    create_schema(engine)
    session_factory = make_session_factory(engine)
    stats = ClassificationStats()
    classifier = cast(QueryClassifier, config.classifier)

    with session_factory() as session:
        pending = 0
        for query in session.scalars(sa.select(Query)):
            if config.limit is not None and stats.considered >= config.limit:
                break
            stats.considered += 1

            result = classifier.classify(query.query or "")
            if result is None:
                stats.unlabeled += 1
                continue

            existing = session.get(QueryClassification, (query.id, classifier.name))
            _record(
                session,
                existing,
                QueryClassification,
                {"query_id": query.id, "classifier": classifier.name},
                result,
            )
            stats.classified += 1

            pending += 1
            if pending >= config.batch_size:
                session.commit()
                pending = 0

        session.commit()
    return stats


def classify_interactions(config: ClassifierConfig) -> ClassificationStats:
    """Run ``config.classifier`` (an :class:`InteractionClassifier`) over every
    interaction."""
    engine = create_database_engine(config.database_url)
    create_schema(engine)
    session_factory = make_session_factory(engine)
    stats = ClassificationStats()
    classifier = cast(InteractionClassifier, config.classifier)

    with session_factory() as session:
        pending = 0
        for interaction in session.scalars(sa.select(Interaction)):
            if config.limit is not None and stats.considered >= config.limit:
                break
            stats.considered += 1

            view = InteractionView(
                id=interaction.id,
                document_id=interaction.document_id,
                type=interaction.type,
                session_id=interaction.session_id,
                date=interaction.date,
            )
            result = classifier.classify(view)
            if result is None:
                stats.unlabeled += 1
                continue

            existing = session.get(
                InteractionClassification, (interaction.id, classifier.name)
            )
            _record(
                session,
                existing,
                InteractionClassification,
                {
                    "interaction_id": interaction.id,
                    "classifier": classifier.name,
                },
                result,
            )
            stats.classified += 1

            pending += 1
            if pending >= config.batch_size:
                session.commit()
                pending = 0

        session.commit()
    return stats


def _record(session, existing, model, key: dict, result: Classification) -> None:
    """Insert a new classification row, or update the existing one in place."""
    now = datetime.now(timezone.utc)
    if existing is None:
        session.add(
            model(
                **key,
                label=result.label,
                confidence=result.confidence,
                created_at=now,
            )
        )
    else:
        existing.label = result.label
        existing.confidence = result.confidence
        existing.created_at = now
