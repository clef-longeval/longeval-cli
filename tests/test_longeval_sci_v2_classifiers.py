from __future__ import annotations

from pathlib import Path

from src.classifying.loader import load_query_classifier

_CLASSIFIERS_DIR = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "longeval-sci-v2"
    / "classifiers"
)


def test_has_doi_matches_a_doi_in_the_query():
    classifier = load_query_classifier(_CLASSIFIERS_DIR / "has_doi.py")

    assert classifier.classify("doi:(10.1016/j.bone.2011.08.025)").label == "doi"
    assert classifier.classify("cats and dogs").label == "no-doi"


def test_is_structured_matches_field_lookup_syntax():
    classifier = load_query_classifier(_CLASSIFIERS_DIR / "is_structured.py")

    query = "working AND documentType:\"thesis\" AND (yearPublished>=2007)"
    assert classifier.classify(query).label == "structured"
    assert classifier.classify("plain query").label == "not-structured"


def test_is_structured_is_case_insensitive():
    classifier = load_query_classifier(_CLASSIFIERS_DIR / "is_structured.py")

    assert classifier.classify('language:"en"').label == "structured"


def test_is_structured_requires_a_named_field_before_the_colon():
    """A bare colon alone (no field name) must not match — the legacy SQL's
    accidental empty alternative is not carried over."""
    classifier = load_query_classifier(_CLASSIFIERS_DIR / "is_structured.py")

    assert classifier.classify("time 12:30 meeting").label == "not-structured"
