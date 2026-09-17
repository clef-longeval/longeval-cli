from __future__ import annotations

import pytest

from src.classifying.loader import load_interaction_classifier, load_query_classifier


def test_load_query_classifier_loads_a_custom_file(tmp_path):
    path = tmp_path / "classifier.py"
    path.write_text(
        "from src.classifying.records import Classification, QueryClassifier\n"
        "class UpperClassifier(QueryClassifier):\n"
        "    name = 'upper'\n"
        "    def classify(self, query):\n"
        "        return Classification(label=query.upper())\n"
        "query_classifier = UpperClassifier()\n"
    )

    classifier = load_query_classifier(path)

    assert classifier.classify("hi").label == "HI"


def test_load_query_classifier_requires_the_top_level_object(tmp_path):
    path = tmp_path / "classifier.py"
    path.write_text("not_it = 1\n")

    with pytest.raises(ValueError):
        load_query_classifier(path)


def test_load_interaction_classifier_loads_a_custom_file(tmp_path):
    path = tmp_path / "classifier.py"
    path.write_text(
        "from src.classifying.records import Classification, InteractionClassifier\n"
        "class TypeEchoClassifier(InteractionClassifier):\n"
        "    name = 'type-echo'\n"
        "    def classify(self, interaction):\n"
        "        return Classification(label=interaction.type or 'unknown')\n"
        "interaction_classifier = TypeEchoClassifier()\n"
    )

    classifier = load_interaction_classifier(path)

    assert classifier.name == "type-echo"


def test_load_interaction_classifier_requires_the_top_level_object(tmp_path):
    path = tmp_path / "classifier.py"
    path.write_text("not_it = 1\n")

    with pytest.raises(ValueError):
        load_interaction_classifier(path)


def test_a_single_file_can_define_both_classifiers(tmp_path):
    path = tmp_path / "classifier.py"
    path.write_text(
        "from src.classifying.records import Classification\n"
        "class QC:\n"
        "    name = 'qc'\n"
        "    def classify(self, query):\n"
        "        return Classification(label='q')\n"
        "class IC:\n"
        "    name = 'ic'\n"
        "    def classify(self, interaction):\n"
        "        return Classification(label='i')\n"
        "query_classifier = QC()\n"
        "interaction_classifier = IC()\n"
    )

    assert load_query_classifier(path).name == "qc"
    assert load_interaction_classifier(path).name == "ic"


def test_load_query_classifier_from_an_import_spec_instantiates_a_class():
    classifier = load_query_classifier("src.classifying.builtins:IsBooleanClassifier")

    assert classifier.classify("cats AND dogs").label == "boolean"


def test_load_query_classifier_from_an_import_spec_uses_an_existing_instance():
    classifier = load_query_classifier(
        "tests.sample_classifiers:upper_query_classifier"
    )

    assert classifier.classify("hi").label == "HI"


def test_load_query_classifier_rejects_a_spec_that_is_neither_file_nor_import_spec(
    tmp_path,
):
    with pytest.raises(ValueError):
        load_query_classifier(str(tmp_path / "does-not-exist.py"))


def test_load_query_classifier_surfaces_a_missing_import_spec_attribute():
    with pytest.raises(ValueError):
        load_query_classifier("src.classifying.builtins:NoSuchClassifier")
