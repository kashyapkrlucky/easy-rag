"""Command-line interface.

    easyrag index            index the Markdown files in EASYRAG_DOCS_DIR
    easyrag index --rebuild  delete the existing index first (needed after changing
                             the embedding model)
    easyrag eval DATASET     evaluate retrieval and answers on a labelled dataset, in an
                             isolated database (never touches your index)
    easyrag models pull      pull the configured embedding and chat models into Ollama

Exit codes: 0 success, 1 some documents failed, 2 configuration problem,
3 a dependency (embedding model) is unavailable.
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx2

from easyrag.config import Settings
from easyrag.container import build_container
from easyrag.domain.errors import IndexIncompatibleError
from easyrag.embeddings.base import EmbeddingProviderUnavailableError
from easyrag.ingestion.indexer import IndexReport
from easyrag.observability.logging import configure_logging
from easyrag.ollama import OllamaError

EXIT_OK = 0
EXIT_DOCUMENTS_FAILED = 1
EXIT_CONFIGURATION = 2
EXIT_UNAVAILABLE = 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="easyrag", description="EasyRAG command-line tools")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="index the Markdown files in EASYRAG_DOCS_DIR")
    index.add_argument(
        "--rebuild",
        action="store_true",
        help="delete the existing index first (required after changing the embedding model)",
    )
    evaluate = commands.add_parser("eval", help="evaluate retrieval and answer quality")
    evaluate.add_argument("dataset", type=Path, help="dataset directory (corpus/ + questions.yaml)")
    evaluate.add_argument(
        "--retrieval-only", action="store_true", help="skip answer generation (fast)"
    )
    evaluate.add_argument("--output", type=Path, help="report path (default: eval/reports/...)")
    evaluate.add_argument(
        "--keep-database", action="store_true", help="keep the isolated evaluation database"
    )
    models = commands.add_parser("models", help="manage the Ollama models")
    models.add_argument("action", choices=["pull"], help="pull missing configured models")
    arguments = parser.parse_args(argv)

    settings = Settings()
    configure_logging(settings.log_level)
    if arguments.command == "models":
        return asyncio.run(_pull_models(settings))
    if arguments.command == "eval":
        return asyncio.run(_evaluate(settings, arguments))
    return asyncio.run(_index(settings, rebuild=arguments.rebuild))


async def _pull_models(settings: Settings) -> int:
    container = build_container(settings)
    try:
        await container.models.ensure_all()
    except (OllamaError, httpx2.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    finally:
        await container.aclose()
    print(f"models present: {settings.embedding_model}, {settings.llm_model}")
    return EXIT_OK


async def _evaluate(settings: Settings, arguments: argparse.Namespace) -> int:
    from easyrag.evaluation.dataset import DatasetError, load_dataset
    from easyrag.evaluation.report import format_summary, write_report
    from easyrag.evaluation.runner import run_evaluation

    try:
        dataset = load_dataset(arguments.dataset)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIGURATION
    try:
        report = await run_evaluation(
            settings,
            dataset,
            retrieval_only=arguments.retrieval_only,
            keep_database=arguments.keep_database,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except EmbeddingProviderUnavailableError as exc:
        print(f"error: embedding model unavailable: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = arguments.output or Path("eval/reports") / f"{dataset.name}-{stamp}.json"
    write_report(report, output)
    print(format_summary(report))
    print(f"\nReport: {output}")
    return EXIT_OK


async def _index(settings: Settings, *, rebuild: bool) -> int:
    container = build_container(settings)
    try:
        report = await container.indexer.run(rebuild=rebuild)
    except (IndexIncompatibleError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIGURATION
    except EmbeddingProviderUnavailableError as exc:
        print(f"error: embedding model unavailable: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    finally:
        await container.aclose()

    _print_report(report)
    return EXIT_DOCUMENTS_FAILED if report.failed else EXIT_OK


def _print_report(report: IndexReport) -> None:
    print(
        f"Indexed in {report.duration_seconds:.1f}s: {report.documents_added} added, "
        f"{report.documents_updated} updated, {report.documents_unchanged} unchanged, "
        f"{report.documents_deleted} deleted; {report.chunks_embedded} chunks embedded, "
        f"{report.chunks_reused} reused"
    )
    if report.deletion_guarded:
        print(
            "  warning: the docs directory is empty, so indexed documents were NOT deleted. "
            "Check the docs path, or run `easyrag index --rebuild` to empty the index."
        )
    for failure in report.failed:
        print(f"  failed:  {failure.source_path}: {failure.reason}")
    for skipped in report.skipped:
        print(f"  skipped: {skipped.source_path}: {skipped.reason}")


if __name__ == "__main__":
    sys.exit(main())
