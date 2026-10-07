from __future__ import annotations

import pytest

from src.classifying.loader import load_classifiers


def test_loads_one_classifier_as_a_list(tmp_path):
    path = tmp_path / "classifier.py"
    path.write_text(
        "from src.classifying.builtins import WordCountClassifier\n"
        "classifiers = WordCountClassifier()\n"
    )

    (classifier,) = load_classifiers(path)

    assert classifier.name == "word-count"


def test_loads_a_list_of_mixed_levels(tmp_path):
    path = tmp_path / "classifier.py"
    path.write_text(
        "from src.classifying.builtins import ClickDelayClassifier, "
        "WordCountClassifier\n"
        "classifiers = [WordCountClassifier(), ClickDelayClassifier()]\n"
    )

    levels = [c.level for c in load_classifiers(path)]

    assert levels == ["queries", "interactions"]


def test_requires_the_top_level_object(tmp_path):
    path = tmp_path / "classifier.py"
    path.write_text("query_classifier = 1\n")

    with pytest.raises(ValueError, match="`classifiers`"):
        load_classifiers(path)


def test_rejects_a_missing_file_or_an_import_spec(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_classifiers(tmp_path / "does-not-exist.py")
    with pytest.raises(ValueError, match="not found"):
        load_classifiers("src.classifying.builtins:IsBooleanClassifier")
