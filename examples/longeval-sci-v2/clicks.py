"""Click counts per query, once per snapshot (they depend on its date range)::

    uv run longeval-cli classify --classifier clicks.py \
        --filter filters/snapshot-1.yaml --database-url "$DB"

Counts what the filter calls a click (its `click_types`, without the clicks its
`interactions` rules exclude), so run `classifiers.py` first.
"""

from src.classifying.builtins import ClicksClassifier

classifiers = ClicksClassifier("clicks")
