"""Runs classifiers and stores their verdicts in the level's verdict table.

Rows are keyed by ``(entity id, classifier.name, snapshot)`` (see
``models.Verdict``). A complete run replaces the classifier's verdicts under
the snapshot: the old ones are deleted and the new ones written in a single
transaction, so a row it no longer has an opinion on loses its old verdict, and
a run that fails leaves the previous set untouched (while it runs, the
classifier itself no longer sees its old verdicts). A run with ``limit`` is
partial: it only upserts what it produced and deletes nothing. Verdicts are
written in bulk (``INSERT ... ON CONFLICT DO UPDATE``, which SQLite and
PostgreSQL both support), ``batch_size`` rows per statement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite
from tqdm.auto import tqdm

from ..database import create_database_engine, create_schema, make_session_factory
from ..filtering import SnapshotConfig
from ..models import LEVELS
from .records import Classification, Classifier


@dataclass
class ClassifierConfig:
    """What to run: one classifier or a list, run in order.

    ``snapshot.snapshot`` is the name the verdicts are stored under (the empty
    string is global); the whole config is handed to each classifier.
    """

    database_url: str
    classifiers: Classifier | Sequence[Classifier]
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    batch_size: int = 5_000
    # stop each classifier after this many verdicts; such a run is partial
    limit: int | None = None
    progress: bool = True  # one tqdm bar per classifier


def classify(config: ClassifierConfig) -> dict[str, int]:
    """Run every classifier; return how many verdicts each one stored."""
    classifiers = config.classifiers
    if isinstance(classifiers, Classifier):
        classifiers = [classifiers]
    for classifier in classifiers:
        if classifier.level not in LEVELS:
            raise ValueError(
                f"{classifier.name}: level must be one of {', '.join(LEVELS)}, "
                f"not {classifier.level!r}"
            )

    engine = create_database_engine(config.database_url)
    create_schema(engine)
    stored: dict[str, int] = {}
    with make_session_factory(engine)() as db:
        for classifier in classifiers:
            stored[classifier.name] = _run(db, classifier, config)
    return stored


def _run(db, classifier: Classifier, config: ClassifierConfig) -> int:
    verdicts, entity = LEVELS[classifier.level]
    now = datetime.now(timezone.utc)
    batch: list[dict] = []
    stored = 0

    def flush() -> None:
        if batch:
            _upsert(db, verdicts.__table__, entity.name, batch)
            batch.clear()

    try:
        if config.limit is None:  # complete: replace the previous verdict set
            db.execute(
                sa.delete(verdicts).where(
                    verdicts.classifier == classifier.name,
                    verdicts.snapshot == config.snapshot.snapshot,
                )
            )
        with tqdm(
            desc=classifier.name,
            unit=f" {classifier.level}",
            disable=not config.progress,
        ) as bar:
            for entity_id, result in classifier.classify_all(db, config.snapshot):
                if config.limit is not None and stored >= config.limit:
                    break
                batch.append(
                    _row(entity.name, entity_id, classifier, config, result, now)
                )
                stored += 1
                bar.update()
                if len(batch) >= config.batch_size:
                    flush()
            flush()
    except BaseException:
        db.rollback()  # the previous verdict set stays as it was
        raise
    db.commit()
    return stored


def _row(entity: str, entity_id, classifier, config, result: Classification, now):
    return {
        entity: entity_id,
        "classifier": classifier.name,
        "snapshot": config.snapshot.snapshot,
        "label": result.label,
        "confidence": result.confidence,
        "value": result.value,
        "created_at": now,
    }


def _upsert(db, table, entity: str, rows: list[dict]) -> None:
    dialect = postgresql if db.bind.dialect.name == "postgresql" else sqlite
    insert = dialect.insert(table)
    db.execute(
        insert.on_conflict_do_update(
            index_elements=[entity, "classifier", "snapshot"],
            set_={
                name: insert.excluded[name]
                for name in ("label", "confidence", "value", "created_at")
            },
        ),
        rows,
    )
