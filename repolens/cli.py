import logging
import argparse
from pathlib import Path
from typing import Sequence

from repolens.config import config, logger
from repolens.ingestion.chunker import chunk_documents
from repolens.ingestion.loader import(
    ingest_repository,
)


_PREVIEW_LENGTH = 120


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

    inspect_parser = subparsers.add_parser(
        "inspect-chunks",
        help="Ingest a repository and display its generated chunks.",
    )

    inspect_parser.add_argument(
        "repository_path",
        type=Path,
        help="Path to the repository to inspect.",
    )

    inspect_parser.add_argument(
        "--repository-name",
        help=(
            "Optional repository name. If omitted, the directory "
            "name is used."
        ),
    )

    inspect_parser.add_argument(
        "--max-file-size",
        type=int,
        default=int(config["INGESTION"]["MAX_FILE_SIZE"]),
        help="Maximum file size in bytes to ingest.",
    )

    inspect_parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(config["CHUNKING"]["CHUNK_SIZE"]),
        help="Maximum target size of each chunk in tokens.",
    )

    inspect_parser.add_argument(
        "--overlap",
        type=int,
        default=int(config["CHUNKING"]["OVERLAP"]),
        help="Target token overlap between adjacent chunks.",
    )

    inspect_parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=int(config["CHUNKING"]["MIN_CHUNK_SIZE"]),
        help="Minimum preferred chunk size in tokens.",
    )

    inspect_parser.add_argument(
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


def _build_preview(content: str) -> str:
    """Return a compact, single-line chunk preview."""

    preview = " ".join(content.split())

    if len(preview) <= _PREVIEW_LENGTH:
        return preview

    return f"{preview[:_PREVIEW_LENGTH - 3].rstrip()}..."


def run_inspect_chunks_command(args: argparse.Namespace) -> int:
    """Ingest, chunk, and display repository text without embedding it."""

    try:
        documents, summary = ingest_repository(
            args.repository_path,
            repository_name=args.repository_name,
            max_file_size=args.max_file_size,
        )

        chunks = chunk_documents(
            documents,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            min_chunk_size=args.min_chunk_size,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        logger.error("Unable to inspect chunks: %s", error)
        return 1
    except OSError as error:
        logger.error("Unable to inspect repository: %s", error)
        return 1

    repository_name = (
        args.repository_name
        or args.repository_path.resolve().name
    )

    print(f"Repository: {repository_name}")
    print(f"Ingested files: {summary.ingested_files}")
    print(f"Failed files: {summary.failed_files}")
    print(f"Chunks: {len(chunks)}")

    if not chunks:
        print("No chunks were produced.")
        return 0

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata

        print(f"\nChunk {index}")
        print(f"File: {metadata.relative_path}")
        print(
            f"Lines: {metadata.start_line}-{metadata.end_line}"
        )

        if metadata.heading is not None:
            print(f"Heading: {metadata.heading}")

        print(f"Tokens: {metadata.token_count}")
        print(f"Chunk ID: {metadata.chunk_id}")
        print(f"Preview: {_build_preview(chunk.content)}")

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

    if args.command == "inspect-chunks":
        return run_inspect_chunks_command(args)
    
    logger.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())