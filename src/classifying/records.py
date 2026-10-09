"""The classification extension point: a verdict and the two kinds of classifier.

Every classifier labels the rows of one table, its ``level``: ``"queries"``,
``"sessions"``, ``"results"``, ``"interactions"`` or ``"documents"`` (see
``models.LEVELS``).
Its verdicts go to that level's ``*_classifications`` table, and a filter
file's section of the same name applies them.

* :class:`Classifier` — yields ``(row id, Classification)`` from the database
  itself, e.g. one ``GROUP BY`` for a click count, or a session's search rate.
* :class:`RowClassifier` — gets one read-only view of a row at a time (with
  the context ``rows.py`` joins in) and never touches the database.

Storage is handled entirely by ``runner.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from ..filtering import SnapshotConfig
from .rows import iter_rows


@dataclass(frozen=True, slots=True)
class Classification:
    """One classifier's verdict: a label, optionally with a confidence/value."""

    label: str
    confidence: float | None = None
    value: float | None = None

    @classmethod
    def flag(cls, is_true: bool) -> Classification:
        """A yes/no verdict: label ``"1"``/``"0"``, ``value`` 1.0/0.0.

        The classifier's name says what the flag means (``has-doi``), so a
        filter reads ``require: {has-doi: 0}`` or ``min: {has-doi: 1}``.
        """
        return cls(label="1" if is_true else "0", value=float(is_true))


class Classifier(ABC):
    """Labels rows of the table ``level``, reading the database itself.

    ``name`` identifies the classifier in the verdict table, so re-running it
    upserts its own rows. ``config`` is the run's snapshot config (name, date
    range, click types, rules); a classifier that aggregates over a date range
    should respect ``config.start``/``config.end``. Yield nothing for rows you
    have no opinion on: a missing verdict is kept by ``exclude`` rules, so
    storing only the exceptions (``late``, ``bot``) keeps tables small.
    """

    name: str
    level: str

    @abstractmethod
    def classify_all(
        self, db: Session, config: SnapshotConfig
    ) -> Iterable[tuple[Any, Classification]]:
        """Yield ``(row id, verdict)`` for every row this classifier labels."""


class RowClassifier(Classifier):
    """Labels one row at a time from its view (see ``rows.py``).

    Every row of ``level`` is visited, whatever the snapshot range.
    """

    @abstractmethod
    def classify(self, row: Any) -> Classification | None:
        """Return this classifier's verdict, or ``None`` for no opinion."""

    def classify_all(
        self, db: Session, config: SnapshotConfig
    ) -> Iterator[tuple[Any, Classification]]:
        for row in iter_rows(db, self.level):
            result = self.classify(row)
            if result is not None:
                yield row.id, result
