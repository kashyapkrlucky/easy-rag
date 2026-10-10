"""Layer boundaries (ARCHITECTURE §3): dependencies point inward only.

- ``domain`` is pure: standard library and other domain modules only.
- The pipeline layers (``ingestion``, ``retrieval``, ``rag``) never import web, database,
  HTTP or model-backend code; adapters and the composition root wire those in.
"""

import ast
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src" / "easyrag"

INFRASTRUCTURE = (
    "fastapi",
    "starlette",
    "sqlalchemy",
    "pgvector",
    "psycopg",
    "httpx2",
    "alembic",
    "fastembed",
    "onnxruntime",
    "watchfiles",
    "prometheus_client",
)
ADAPTERS = (
    "easyrag.ollama",
    "easyrag.embeddings.ollama",
    "easyrag.llm.ollama",
    "easyrag.rerankers",
    "easyrag.watcher",
    "easyrag.storage",
    "easyrag.api",
    "easyrag.container",
    "easyrag.main",
    "easyrag.cli",
    "easyrag.observability.metrics",
    "easyrag.observability.transport",
)


def imports_of(package: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for path in (SOURCE / package).rglob("*.py"):
        names: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module)
        result[str(path.relative_to(SOURCE))] = names
    return result


def violations(package: str, forbidden: tuple[str, ...]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for module, names in imports_of(package).items():
        bad = sorted(n for n in names if any(n == f or n.startswith(f + ".") for f in forbidden))
        if bad:
            result[module] = bad
    return result


def test_domain_is_pure() -> None:
    offending = {
        module: sorted(
            n
            for n in names
            if n.split(".")[0] not in sys.stdlib_module_names and not n.startswith("easyrag.domain")
        )
        for module, names in imports_of("domain").items()
    }

    assert not {m: n for m, n in offending.items() if n}


@pytest.mark.parametrize("package", ["ingestion", "retrieval", "rag"])
def test_pipeline_layers_do_not_depend_on_infrastructure(package: str) -> None:
    assert not violations(package, INFRASTRUCTURE + ADAPTERS)
