import logging
import argparse
from pathlib import Path
from typing import Sequence

from repolens.config import config, logger
from repolens.ingestion.loader import(
    ingest_repository,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    
    parser = argparse.ArgumentParser(
        prog="repolens",
        description="Repository ingestion and retrieval tools.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Discover and ingest repository text files.",
    )

    ingest_parser.add_argument(
        "repository_path",
        type=Path,
        help="Path to the repository to ingest.",
    )

    ingest_parser.add_argument(
        "--repository-name",
        help="Optional name for the repository. If not provided, the name will be derived from the repository path.",
    )

    ingest_parser.add_argument(
        "--max-file-size",
        type=int,
        default=int(config["INGESTION"]["MAX_FILE_SIZE"]),
        help="Maximum file size in bytes to ingest. Files larger than this will be skipped.",
    )

    ingest_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging level.",
    )

    return parser


def run_ingest_command(args: argparse.Namespace) -> int:
    """Run the repository ingestion and print its summary."""

    try:
        documents, summary = ingest_repository(
            args.repository_path,
            repository_name=args.repository_name,
            max_file_size=args.max_file_size,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        logger.error(f"Error during ingestion: {error}")
        return 1
    except OSError as error:
        logger.error(f"Unable to ingest repository: {error}")
        return 1
    
    repository_name = (
        args.repository_name 
        or args.repository_path.resolve().name
    )

    logger.info("Repository ingestion completed.\n")
    logger.info(f"Repository: {repository_name}")
    logger.info(f"Discovered files: {summary.discovered_files}")
    logger.info(f"Ingested files: {summary.ingested_files}")
    logger.info(f"Empty files: {summary.empty_files}")
    logger.info(f"Failed files: {summary.failed_files}")

    if summary.failures:
        logger.info("\nFailed files:")

        for failure in summary.failures:
            logger.info(f"- {failure.relative_path}: {failure.reason}")
    
    # Documents remain in memory for the later chunking stage.
    logger.debug(
        "Created %d ingested documents",
        len(documents),
    )

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    if args.command == "ingest":
        return run_ingest_command(args)
    
    logger.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())