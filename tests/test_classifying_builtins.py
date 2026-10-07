from __future__ import annotations

from src.classifying.builtins import IsBooleanClassifier, WordCountClassifier
from src.classifying.rows import QueryView


def _q(text: str) -> QueryView:
    return QueryView("q", text)


def test_is_boolean_flags_and_or_not():
    classifier = IsBooleanClassifier()

    assert classifier.classify(_q("cats AND dogs")).label == "1"
    assert classifier.classify(_q("plain query")).label == "0"


def test_is_boolean_flags_quoted_and_or():
    assert IsBooleanClassifier().classify(_q('"cats" or "dogs"')).label == "1"


def test_is_boolean_misses_unquoted_lowercase_operators():
    """Documents the classifier's known limitation."""
    query = "combo meals option and (yearpublished>=1977 and yearpublished<=2025)"
    assert IsBooleanClassifier().classify(_q(query)).label == "0"


def test_word_count_labels_the_number_of_words():
    classifier = WordCountClassifier()

    assert classifier.classify(_q("one two three")).label == "3"
    assert classifier.classify(_q("  one   two  ")).label == "2"


def test_word_count_returns_none_for_an_empty_query():
    classifier = WordCountClassifier()

    assert classifier.classify(_q("")) is None
    assert classifier.classify(_q("   ")) is None
