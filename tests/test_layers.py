"""The dependency direction between layers, enforced rather than remembered.

The project has two layers and they change at different rates. The task layer defines
*what is measured and how* — readers, resampling, features, the split, the metrics — and
it changed twice this week, both times to fix a defect. The model layer defines *what
does the measuring*, and it changed four times in a single afternoon.

The arrow must point one way. If a metric ever depended on a model, results from two
models would stop being comparable, and nothing would fail loudly enough to notice. Right
now the boundary holds because everyone remembers it; these tests make it hold anyway.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "shopfloor"

TASK = frozenset({"data", "dataset", "arrays", "features", "splits", "metrics"})
MODELS = frozenset({"baseline", "net", "train", "anomaly"})
SHARED = frozenset({"config"})
"""Settings only. Depends on nothing in the package, so both layers may use it."""


def package_modules() -> set[str]:
    """Every module in the package except the empty __init__."""
    return {path.stem for path in PACKAGE.glob("*.py")} - {"__init__"}


def imported_modules(name: str) -> set[str]:
    """Which sibling modules `name` imports, read from the syntax tree rather than text."""
    tree = ast.parse((PACKAGE / f"{name}.py").read_text())
    found: set[str] = set()

    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            targets = [node.module]
        elif isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]

        for target in targets:
            parts = target.split(".")
            if parts[0] == "shopfloor" and len(parts) > 1:
                found.add(parts[1])
    return found


def test_every_module_is_assigned_to_a_layer() -> None:
    """A new module cannot appear without someone deciding which layer it belongs to."""
    assert package_modules() == TASK | MODELS | SHARED


def test_the_task_layer_never_imports_a_model() -> None:
    """What we measure must not depend on what measures it."""
    violations = {
        module: sorted(imported_modules(module) & MODELS)
        for module in sorted(TASK)
        if imported_modules(module) & MODELS
    }
    assert violations == {}


def test_the_shared_layer_depends_on_nothing_in_the_package() -> None:
    """Settings sit underneath both layers, so they must not reach into either."""
    for module in sorted(SHARED):
        assert imported_modules(module) == set()
