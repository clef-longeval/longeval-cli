"""Document classifiers for LongEval-Sci: drop documents with a short abstract.

Run once (verdicts are global), then exclude them in each snapshot filter::

    uv run longeval-cli classify \
        --classifier examples/longeval-sci-v2/document_classifiers.py

    documents:
      exclude: {abstract-length: short}
"""

from src.classifying import AbstractLengthClassifier

classifiers = AbstractLengthClassifier(min_chars=100)
