"""Turning click scores into graded relevance labels for a qrels file.

A :class:`Discretizer` maps one score (a click probability, or a click count)
to an integer label, or to ``None`` to leave the pair out of the qrels. Exporting
qrels without a discretizer writes the scores themselves.

Three built-ins:

* :class:`ThresholdDiscretizer` — ``ThresholdDiscretizer((0.1, 0.9))`` labels a
  click-through rate 0 below 0.1, 1 below 0.9 and 2 from 0.9 up.
* :class:`AnyClickDiscretizer` — label 1 for every pair with at least one
  click, and nothing for the others; apply it to a click-count column.
* :class:`ChapelleDiscretizer` — inverts the DCG-discount mapping of Chapelle et
  al. from a relevance grade ``g`` to a probability, ``P = (2**g - 1) / 2**gmax``,
  so ``g = log2(P * 2**gmax + 1)``, then rounds to an integer grade.

``export qrels --discretizer NAME`` picks one by name (see
:func:`make_discretizer`); there is deliberately no plug-in mechanism.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol


class Discretizer(Protocol):
    def label(self, score: float) -> int | None:
        """The label for ``score``, or ``None`` to omit the pair."""


@dataclass(frozen=True)
class ThresholdDiscretizer:
    """Label = number of thresholds the score has reached (``score >= t``)."""

    thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        ordered = list(self.thresholds)
        if not ordered or ordered != sorted(set(ordered)):
            raise ValueError("thresholds must be non-empty and strictly increasing")

    def label(self, score: float) -> int:
        return bisect_right(self.thresholds, score)


@dataclass(frozen=True)
class AnyClickDiscretizer:
    """Label 1 when the score (a click count) is positive, else leave it out."""

    def label(self, score: float) -> int | None:
        return 1 if score > 0 else None


@dataclass(frozen=True)
class ChapelleDiscretizer:
    """Grade ``g = log2(P * 2**gmax + 1)`` for a click probability ``P``.

    The mapping from a grade to a probability, ``P = (2**g - 1) / 2**gmax``
    (Chapelle et al., CIKM 2009), is reversed. The result is real-valued and
    ranges up to ``log2(2**gmax + 1)``, slightly above ``gmax``, so it is
    rounded and capped at ``gmax``. The rounding is not fixed by the source of
    the formula: ``"nearest"`` (default) gives grade 1 from ``P`` ≈ 0.10 and
    grade 2 from ``P`` ≈ 0.46 for ``gmax=2``; ``"floor"`` gives 0.25 and 0.75.
    """

    gmax: int = 2
    rounding: str = "nearest"

    def __post_init__(self) -> None:
        if self.gmax < 1:
            raise ValueError("gmax must be at least 1")
        if self.rounding not in ("nearest", "floor"):
            raise ValueError("rounding must be 'nearest' or 'floor'")

    def label(self, score: float) -> int:
        if not 0 <= score <= 1:
            raise ValueError(f"a probability must be in [0, 1], got {score}")
        grade = math.log2(score * 2**self.gmax + 1)
        grade = math.floor(grade + 0.5) if self.rounding == "nearest" else int(grade)
        return min(grade, self.gmax)


DISCRETIZERS = ("threshold", "chapelle", "any-click")


def make_discretizer(
    name: str,
    *,
    thresholds: tuple[float, ...] | None = None,
    gmax: int | None = None,
    rounding: str | None = None,
) -> Discretizer:
    """Build a built-in discretizer by name; parameters it ignores are an error.

    ``threshold`` takes ``thresholds`` (default ``(0.1, 0.9)``), ``chapelle``
    takes ``gmax`` (default 2) and ``rounding`` (default ``"nearest"``),
    ``any-click`` takes none.
    """
    given = {
        key: value
        for key, value in (
            ("thresholds", thresholds),
            ("gmax", gmax),
            ("rounding", rounding),
        )
        if value is not None
    }
    accepted = {
        "threshold": {"thresholds"},
        "chapelle": {"gmax", "rounding"},
        "any-click": set(),
    }
    if name not in accepted:
        raise ValueError(f"unknown discretizer {name!r}; choose from {DISCRETIZERS}")
    unused = set(given) - accepted[name]
    if unused:
        raise ValueError(f"{name} does not take {', '.join(sorted(unused))}")

    if name == "threshold":
        return ThresholdDiscretizer(thresholds or (0.1, 0.9))
    if name == "chapelle":
        return ChapelleDiscretizer(**given)
    return AnyClickDiscretizer()
