"""The reviewer's labels from an `export queries` sheet with a `manual_label` column::

    uv run longeval-cli classify --classifier manual.py --database-url "$DB"

Queries are matched on their exact text; a blank label gives no verdict.
"""

from src.classifying.builtins import ManualLabelsClassifier

classifiers = ManualLabelsClassifier(
    "data/interim/queries-checked/candidate-queries-snapshot-3.tsv"
)
