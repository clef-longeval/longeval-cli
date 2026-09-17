"""Loading of a classifier from either a file or a plain import spec.

Mirrors ``importing/loader.py``'s "configurable parser" extension point for
a classifier that needs per-project customization: point ``--classifier``
at your own file exposing a top-level ``query_classifier``/
``interaction_classifier`` object. But a classifier that needs no
customization at all — one of ``src.classifying.builtins``'s, say — doesn't
need a wrapper file just to be loadable, so ``--classifier`` also accepts a
plain ``module.path:name`` import spec resolving to a class (instantiated
with no arguments) or an already-constructed instance.
"""

from __future__ import annotations

import importlib
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


def load_query_classifier(spec: str | Path):
    """Load a ``QueryClassifier`` from ``spec`` (see module docstring)."""
    return _load(spec, "query_classifier")


def load_interaction_classifier(spec: str | Path):
    """Load an ``InteractionClassifier`` from ``spec`` (see module docstring)."""
    return _load(spec, "interaction_classifier")


def _load(spec: str | Path, attr_name: str):
    if Path(spec).is_file():
        module = load_classifier_module(spec)
        try:
            return getattr(module, attr_name)
        except AttributeError:
            raise ValueError(
                f"{spec} must define a top-level `{attr_name}` object"
            ) from None

    module_name, sep, attr = str(spec).partition(":")
    if not sep:
        raise ValueError(
            f"{spec!r} is neither an existing file nor a `module.path:name` "
            "import spec"
        )
    module = importlib.import_module(module_name)
    try:
        obj = getattr(module, attr)
    except AttributeError:
        raise ValueError(f"{module_name!r} has no attribute {attr!r}") from None
    return obj() if isinstance(obj, type) else obj
