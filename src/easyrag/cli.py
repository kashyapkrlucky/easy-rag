"""Command-line interface.

    easyrag index            index the Markdown files in EASYRAG_DOCS_DIR
    easyrag index --rebuild  delete the existing index first (needed after changing
                             the embedding model)

Exit codes: 0 success, 1 some documents failed, 2 configuration problem,
3 a dependency (embedding model) is unavailable.
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence

from easyrag.config import Settings
from easyrag.container import build_container
from easyrag.embeddings.base import EmbeddingProviderUnavailableError
from easyrag.ingestion.indexer import IndexIncompatibleError, IndexReport
from easyrag.observability.logging import configure_logging

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
    arguments = parser.parse_args(argv)

    settings = Settings()
    configure_logging(settings.log_level)
    return asyncio.run(_index(settings, rebuild=arguments.rebuild))


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
        f"Indexed {report.documents_indexed} documents ({report.chunks_indexed} chunks) "
        f"in {report.duration_seconds:.1f}s"
    )
    for failure in report.failed:
        print(f"  failed:  {failure.source_path}: {failure.reason}")
    for skipped in report.skipped:
        print(f"  skipped: {skipped.source_path}: {skipped.reason}")


if __name__ == "__main__":
    sys.exit(main())
