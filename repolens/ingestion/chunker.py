import hashlib
import json
from collections.abc import Sequence

import tiktoken

from repolens.config import config, logger
from repolens.models import (
    ChunkMetadata,
    TextChunk,
    IngestedDocument,
)

_TOKEN_ENCODER = tiktoken.get_encoding(config["TOKENIZER"]["ENCODING_NAME"])

def count_tokens(content: str) -> int:
    """Return the approximate number of tokens in text."""
    return len(_TOKEN_ENCODER.encode_ordinary(content))

def compute_chunk_hash(content: str) -> str:
    """Compute SHA-256 hash of the chunk's content"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def generate_chunk_id(
    *,
    repository: str,
    relative_path: str,
    start_line: int,
    end_line: int,
    content_hash: str,
) -> str:
    """Generate a deterministic chunk ID based on repository, file path, line range, and content hash."""

    identity = json.dumps(
        {
            "repository": repository,
            "relative_path": relative_path,
            "start_line": start_line,
            "end_line": end_line,
            "content_hash": content_hash,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()

def validate_chunking_settings(
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> None:
    """Validate chunking configuration values."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if min_chunk_size < 0:
        raise ValueError("min_chunk_size cannot be negative")
    if min_chunk_size > chunk_size:
        raise ValueError("min_chunk_size cannot be greater than chunk_size")

def _find_chunk_end(
    lines: Sequence[str],
    start_index: int,
    chunk_size: int,
) -> int:
    """Find the exclusive ending line index for a chunk."""

    end_index = start_index
    current_tokens = 0

    while end_index < len(lines):
        line_tokens = count_tokens(lines[end_index])

        if end_index > start_index and current_tokens + line_tokens > chunk_size:
            break

        current_tokens += line_tokens
        end_index += 1

        if current_tokens >= chunk_size:
            break

    return end_index

def _find_next_start(
    lines: Sequence[str],
    start_index: int,
    end_index: int,
    overlap: int,
) -> int:
    """Find the next starting line while retaining overlap."""

    if overlap == 0:
        return end_index

    next_start = end_index
    overlap_tokens = 0

    while next_start > start_index:
        previous_line_tokens = count_tokens(lines[next_start - 1])

        if overlap_tokens > 0 and overlap_tokens + previous_line_tokens > overlap:
            break

        next_start -= 1
        overlap_tokens += previous_line_tokens

        if overlap_tokens >= overlap:
            break

    # A chunk containing only one line (or one line larger than the
    # overlap budget) cannot overlap with itself because that would
    # prevent chunk_document's outer loop from progressing.
    if next_start <= start_index:
        return end_index

    return next_start

def _create_chunk(
    document: IngestedDocument,
    content: str,
    start_line: int,
    end_line: int,
    *,
    heading: str | None = None,
) -> TextChunk:
    """Create a text chunk and its deterministic metadata."""

    content_hash = compute_chunk_hash(content)

    chunk_id = generate_chunk_id(
        repository=document.metadata.repository,
        relative_path=document.metadata.relative_path,
        start_line=start_line,
        end_line=end_line,
        content_hash=content_hash,
    )

    metadata = ChunkMetadata(
        repository=document.metadata.repository,
        relative_path=document.metadata.relative_path,
        file_type=document.metadata.file_type,
        start_line=start_line,
        end_line=end_line,
        token_count=count_tokens(content),
        content_hash=content_hash,
        chunk_id=chunk_id,
        heading=heading,
    )

    return TextChunk(
        content=content,
        metadata=metadata,
    )

def chunk_document(
    document: IngestedDocument,
    *,
    chunk_size: int = int(config["CHUNKING"]["CHUNK_SIZE"]),
    overlap: int = int(config["CHUNKING"]["OVERLAP"]),
    min_chunk_size: int = int(config["CHUNKING"]["MIN_CHUNK_SIZE"]),
) -> list[TextChunk]:
    """Split an ingested document into line-aware chunks."""

    validate_chunking_settings(chunk_size, overlap, min_chunk_size)

    if not document.content.strip():
        return []

    lines = document.content.splitlines(keepends=True)
    chunks: list[TextChunk] = []
    start_index = 0

    while start_index < len(lines):
        end_index = _find_chunk_end(lines, start_index, chunk_size)
        content = "".join(lines[start_index:end_index])

        if content.strip():
            chunk = _create_chunk(
                document=document,
                content=content,
                start_line=start_index + 1,
                end_line=end_index,
            )

            if chunk.metadata.token_count >= min_chunk_size or not chunks:
                chunks.append(chunk)
            else:
                # Keep a small trailing chunk when dropping it
                # would remove source content.
                chunks.append(chunk)

        if end_index >= len(lines):
            break

        start_index = _find_next_start(lines, start_index, end_index, overlap)

    logger.debug(
        "Chunked %s into %d chunks",
        document.metadata.relative_path,
        len(chunks),
    )

    return chunks

def chunk_documents(
    documents: list[IngestedDocument],
    *,
    chunk_size: int = int(config["CHUNKING"]["CHUNK_SIZE"]),
    overlap: int = int(config["CHUNKING"]["OVERLAP"]),
    min_chunk_size: int = int(config["CHUNKING"]["MIN_CHUNK_SIZE"]),
) -> list[TextChunk]:
    """Chunk multiple ingested documents in input order."""

    chunks: list[TextChunk] = []

    for document in documents:
        chunks.extend(
                chunk_document(
                document,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_size=min_chunk_size
                )
            )

    total_characters = sum(len(chunk.content) for chunk in chunks)
    total_tokens = sum(chunk.metadata.token_count for chunk in chunks)

    logger.info(
        "Chunking completed: files=%d chunks=%d "
        "characters=%d tokens=%d",
        len(documents),
        len(chunks),
        total_characters,
        total_tokens,
    )

    return chunks