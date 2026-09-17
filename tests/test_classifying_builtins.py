from __future__ import annotations

from src.classifying.builtins import IsBooleanClassifier, WordCountClassifier


def test_is_boolean_flags_and_or_not():
    classifier = IsBooleanClassifier()

    assert classifier.classify("cats AND dogs").label == "boolean"
    assert classifier.classify("plain query").label == "not-boolean"


def test_is_boolean_flags_quoted_and_or():
    classifier = IsBooleanClassifier()

    assert classifier.classify('"cats" or "dogs"').label == "boolean"


def test_is_boolean_misses_unquoted_lowercase_operators():
    """Documents the classifier's known limitation."""
    classifier = IsBooleanClassifier()

    query = "combo meals option and (yearpublished>=1977 and yearpublished<=2025)"
    assert classifier.classify(query).label == "not-boolean"


def test_word_count_labels_the_number_of_words():
    classifier = WordCountClassifier()

    assert classifier.classify("one two three").label == "3"
    assert classifier.classify("  one   two  ").label == "2"


def test_word_count_returns_none_for_an_empty_query():
    classifier = WordCountClassifier()

    assert classifier.classify("") is None
    assert classifier.classify("   ") is None
