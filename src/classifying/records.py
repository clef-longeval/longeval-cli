"""Normalized types for the classification extension point.

A ``QueryClassifier``/``InteractionClassifier`` never has to know about the
database — it takes plain data in (a query's text, or an
:class:`InteractionView`) and returns a :class:`Classification`, or ``None``
for "no opinion". Storage is handled entirely by ``runner.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Classification:
    """One classifier's verdict: a label, optionally with a confidence."""

    label: str
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class InteractionView:
    """Read-only view of one interaction, independent of the ORM session."""

    id: int
    document_id: str
    type: str | None
    session_id: str | None
    date: datetime | None


class QueryClassifier(ABC):
    """Assigns a label to a query's text.

    ``name`` identifies this classifier in the ``query_classifications``
    table, so re-running it upserts its own row rather than colliding with a
    different classifier's verdict on the same query. Subclasses are
    instantiated however they need to be (a regex/keyword classifier might
    take patterns, a manual one a labels-file path, an LLM one a model
    name) — there's no shared factory machinery beyond ``__init__``.
    """

    name: str

    @abstractmethod
    def classify(self, query: str) -> Classification | None:
        """Return this classifier's verdict, or ``None`` for no opinion."""


class InteractionClassifier(ABC):
    """Assigns a label to one interaction (see :class:`QueryClassifier`)."""

    name: str

    @abstractmethod
    def classify(self, interaction: InteractionView) -> Classification | None:
        """Return this classifier's verdict, or ``None`` for no opinion."""
