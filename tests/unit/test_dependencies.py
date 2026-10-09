"""Guard: everything the application imports at runtime must be a runtime dependency.

The production image installs only ``[project.dependencies]``. A package that is
imported by ``src/easyrag`` but declared only in the dev group works in tests and then
crashes the container on startup — this test catches that before a build does.
"""

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "src" / "easyrag"


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _runtime_dependencies() -> set[str]:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    names = (
        re.split(r"[\s\[<>=!~;]", spec, maxsplit=1)[0]
        for spec in pyproject["project"]["dependencies"]
    )
    return {_canonical(name) for name in names}


def _imported_top_level_modules() -> set[str]:
    modules: set[str] = set()
    for path in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules - set(sys.stdlib_module_names) - {"easyrag"}


def test_every_runtime_import_is_a_declared_runtime_dependency() -> None:
    distributions = packages_distributions()
    declared = _runtime_dependencies()

    undeclared = {
        module: sorted(distributions.get(module, [module]))
        for module in _imported_top_level_modules()
        if not {_canonical(d) for d in distributions.get(module, [module])} & declared
    }

    assert not undeclared, (
        f"imported by src/easyrag but not in [project.dependencies]: {undeclared}"
    )
