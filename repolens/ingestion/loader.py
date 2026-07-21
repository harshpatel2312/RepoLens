import fnmatch
import os
import io
import hashlib
import tokenize
import unicodedata
from pathlib import Path

from repolens.config import config, logger
from .models import (
    FileMetadata, 
    IngestedDocument, 
    IngestionFailure, 
    IngestionSummary,
)

def is_generated_file(file_path: Path) -> bool:
    """
    Check if the given file is a generated file based on its name.

    Args:
        file_path (Path): The path to the file.

    Returns:
        bool: True if the file is generated, False otherwise.
    """
    filename = file_path.name.lower()

    return any(fnmatch.fnmatch(file_path.name, pattern.lower()) for pattern in config["INGESTION"]["GENERATED_FILE_PATTERNS"])

def discover_files(
    repository_path: Path,
    max_file_size: int = int(config["INGESTION"]["MAX_FILE_SIZE"]),
) -> list[Path]:
    """Discover supported repository files in deterministic order."""
    
    repository_path = repository_path.resolve()

    if not repository_path.exists():
        raise FileNotFoundError(f"Repository does not exist: '{repository_path}'")

    if not repository_path.is_dir():
        raise NotADirectoryError(f"Repository is not a directory: '{repository_path}'")
    
    discovered_files: list[Path] = []
    
    for current_directory, directory_names, file_names in os.walk(
        repository_path, 
        topdown=True, 
        followlinks=False
     ):
        current_path = Path(current_directory)

        # Filter ignored directories
        directory_names[:] = sorted(
            directory_name
            for directory_name in directory_names
            if directory_name not in config["INGESTION"]["IGNORE_DIRECTORIES"]
            and not (current_path / directory_name).is_symlink()
        )

        for file_name in sorted(file_names):
            file_path = current_path / file_name
            relative_path = file_path.relative_to(repository_path).as_posix()

            # Do not follow file symlinks
            if file_path.is_symlink():
                logger.debug(f"Skipping symlink: {relative_path}")
                continue

            # Only [".py", ".txt", ".md"] files are supported
            if file_path.suffix.lower() not in config["INGESTION"]["EXTENSIONS"]:
                logger.debug(f"Skipping unsupported file type: {relative_path}")
                continue

            # Skip generated files
            if is_generated_file(file_path):
                logger.debug(f"Skipping generated file: {relative_path}")
                continue

            try:
                file_size = file_path.stat().st_size
            except OSError as error:
                logger.warning(f"Cannot inspect file {relative_path}: {error}")
                continue

            if file_size > max_file_size:
                logger.info(f"Skipping oversized file: {relative_path} ({file_size} bytes)")
                continue

            discovered_files.append(file_path)
    
    return sorted(
        discovered_files, 
        key=lambda path: path.relative_to(repository_path).as_posix(),
    )

def normalize_content(content: str) -> str:
    """Normalize text by applying Unicode NFC normalization"""
    content = content.replace("\r\n", "\n")  # Normalize line endings
    content = content.replace("\r", "\n")  # Normalize line endings

    return unicodedata.normalize("NFC", content)

def decode_content(path: Path, raw_content: bytes) -> str:
    """Decode supported repository files"""

    if b"\x00" in raw_content:
        raise ValueError(f"Binary content detected: {path}")
    
    if path.suffix.lower() == ".py":
        # Decode Python files using the encoding specified in the file
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw_content).readline)
        return raw_content.decode(encoding)
    
    return raw_content.decode("utf-8-sig")  # Decode all other files with BOM handling

def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of the content"""
    encoded_content = content.encode("utf-8")
    return hashlib.sha256(encoded_content).hexdigest()

def load_file(
    path: Path, 
    repository_root: Path, 
    repository_name: str | None = None,
) -> IngestedDocument | None:
    """Load a file and return its normalized content and metadata"""

    repository_root = repository_root.resolve()
    raw_content = path.read_bytes()

    decoded_content = decode_content(path, raw_content)
    normalized_content = normalize_content(decoded_content)

    if not normalized_content.strip():
        logger.info(f"Skipping empty file: {path.relative_to(repository_root).as_posix()}")
        return None
    
    metadata = FileMetadata(
        repository=repository_name or repository_root.name,
        relative_path=path.relative_to(repository_root).as_posix(),
        file_type=config["INGESTION"]["FILE_TYPES"].get(path.suffix.lower(), "unknown"),
        size_bytes=len(raw_content),
        content_hash=compute_content_hash(normalized_content),
    )

    return IngestedDocument(
        content=normalized_content,
        metadata=metadata,
    )

def ingest_repository(
    repository_path: Path,
    *,
    repository_name: str | None = None,
    max_file_size: int = int(config["INGESTION"]["MAX_FILE_SIZE"]),
) -> tuple[list[IngestedDocument], IngestionSummary]:
    """Discover and ingest supported files from a repository."""

    repository_root = repository_path.resolve()

    discovered_files = discover_files(
        repository_root, 
        max_file_size=max_file_size,
    )

    documents: list[IngestedDocument] = []
    summary = IngestionSummary(
        discovered_files=len(discovered_files),
    )

    for file_path in discovered_files:
        relative_path = file_path.relative_to(repository_root).as_posix()

        try:
            document = load_file(
                file_path, 
                repository_root, 
                repository_name=repository_name,
            )
        except (OSError, UnicodeError, SyntaxError, ValueError) as error:
            summary.failed_files += 1
            summary.failures.append(
                IngestionFailure(
                    relative_path=relative_path,
                    reason=str(error),
                )
            )

            logger.warning(f"Failed to ingest {relative_path}: {error}")
            continue

        if document is None:
            summary.empty_files += 1
            logger.info(f"Skipping empty file: {relative_path}")
            continue

        documents.append(document)
        summary.ingested_files += 1

        logger.info(f"Ingested file: {relative_path}")
    
    logger.info(
        f"Repository ingestion completed: "
        f"discovered={summary.discovered_files} "
        f"ingested={summary.ingested_files} "
        f"empty={summary.empty_files} "
        f"failed={summary.failed_files}"
    )

    return documents, summary








    