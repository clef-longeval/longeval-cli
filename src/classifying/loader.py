"""Loading classifiers from a Python file.

Mirrors ``importing/loader.py``'s "configurable parser" extension point:
``--classifier`` always points at a file exposing a top-level ``classifiers``
object (one classifier, or a list of them, of any levels). Even a generic
built-in needs a two-line wrapper file, so there is exactly one way to load a
classifier and its parameters live in code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_module_cache: dict[Path, ModuleType] = {}


def load_classifier_module(path: Path | str) -> ModuleType:
    """Load and cache the module a classifier *file* defines.

    Caching by resolved path means two calls for the same file return
    objects from the *same* module, so classes from it compare with
    ``isinstance`` as expected.
    """
    path = Path(path).resolve()
    if path not in _module_cache:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"could not load a module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _module_cache[path] = module
    return _module_cache[path]


def load_classifiers(path: Path | str) -> list:
    """Load the ``classifiers`` a file defines, always as a list."""
    if not Path(path).is_file():
        raise ValueError(f"classifier file not found: {path}")
    module = load_classifier_module(path)
    try:
        classifiers = module.classifiers
    except AttributeError:
        raise ValueError(f"{path} must define a top-level `classifiers`") from None
    if isinstance(classifiers, (list, tuple)):
        return list(classifiers)
    return [classifiers]
