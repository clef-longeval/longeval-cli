"""The minimal example's classifiers; run with --filter filter.yaml.

One file may mix levels: each classifier labels the rows of one table. They
run in this order, so `clicks` comes last: it counts what the filter calls a
click, which needs the `bot` and `click-delay` verdicts to exist already. All
verdicts are stored under the filter's snapshot.
"""

from src.classifying.builtins import (
    ClickDelayClassifier,
    ClicksClassifier,
    IsBooleanClassifier,
    SearchBurstClassifier,
    WordCountClassifier,
)

classifiers = [
    IsBooleanClassifier(),  # queries
    WordCountClassifier(),  # queries
    # sessions: "bot" if it made 10 searches within a minute
    SearchBurstClassifier("bot", searches=10, within_seconds=60),
    # interactions: "late" if more than 24 hours after the search
    ClickDelayClassifier("click-delay", max_hours=24),
    # queries: clicks in the snapshot, without bot sessions and late clicks
    ClicksClassifier("clicks"),
]
