from collections.abc import Sequence
from functools import lru_cache
from time import perf_counter
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from repolens.models import TextChunk, EmbeddedChunk
from repolens.retrieval.embedding_cache import load_embedding_cache, save_embedding_cache
from repolens.config import config, logger


def validate_embedding_settings(
    model_name: str,
    batch_size: int,
) -> None:
    """Validate embedding configuration settings."""

    if not model_name.strip():
        raise ValueError("model_name cannot be empty.")
    
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")


def resolve_device(device: str = str(config["EMBEDDING"]["DEVICE"])) -> str:
    """Select suitable/available device"""

    normalized_device = device.strip().lower()

    if normalized_device not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"Invalid device '{device}'. Must be 'cpu', 'cuda', or 'auto'.")

    if normalized_device == "auto":
        return (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if normalized_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available")

    return normalized_device


@lru_cache(maxsize=4)
def load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    """Load the embedding model with caching to avoid reloading."""

    started_at = perf_counter()

    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception:
        logger.exception("Failed to load embedding model '%s' on device '%s'.", model_name, device)
        raise

    duration = perf_counter() - started_at
    logger.info("Loaded embedding model '%s' on device '%s' in %.2f seconds.", model_name, device, duration)

    return model


def _deduplicate_chunks(
    chunks: Sequence[TextChunk],
) -> tuple[list[TextChunk], int]:
    """Remove repeated chunk IDs while preserving order."""

    unique_chunks: dict[str, TextChunk] = {}

    for chunk in chunks:
        chunk_id = chunk.metadata.chunk_id
        existing_chunk = unique_chunks.get(chunk_id)

        if existing_chunk is None:
            unique_chunks[chunk_id] = chunk
            continue

        if existing_chunk != chunk:
            raise ValueError(
                "Conflicting chunks share the same chunk ID: "
                f"{chunk_id}"
            )

    # These lines must be outside the for-loop.
    duplicate_count = len(chunks) - len(unique_chunks)

    return list(unique_chunks.values()), duplicate_count


def _convert_vectors(
    vectors: Sequence[Sequence[float]],
    expected_count: int,
) -> list[list[float]]:
    """Convert and validate model output vectors."""

    if len(vectors) != expected_count:
        raise RuntimeError(
            "Embedding model returned an unexpected "
            "number of vectors"
        )

    vector_lists = [
        [
            float(value)
            for value in vector
        ]
        for vector in vectors
    ]

    dimensions = {len(vector) for vector in vector_lists}

    if len(dimensions) != 1 or 0 in dimensions:
        raise RuntimeError(
            "Embedding vectors have invalid or "
            "inconsistent dimensions"
        )

    return vector_lists


def embed_chunks(
    chunks: Sequence[TextChunk],
    model_name: str = str(
        config["EMBEDDING"]["MODEL_NAME"]
    ),
    batch_size: int = int(
        config["EMBEDDING"]["BATCH_SIZE"]
    ),
    normalize: bool = bool(
        config["EMBEDDING"]["NORMALIZE"]
    ),
    device: str = str(
        config["EMBEDDING"]["DEVICE"]
    ),
    embedding_model: SentenceTransformer | None = None,
    cache_path: str | Path | None = None,
) -> list[EmbeddedChunk]:
    """Generate embeddings while preserving chunk metadata."""

    validate_embedding_settings(
        model_name,
        batch_size,
    )

    if not chunks:
        logger.debug(
            "No chunks provided for embedding"
        )
        return []

    empty_chunk_ids = [
        chunk.metadata.chunk_id
        for chunk in chunks
        if not chunk.content.strip()
    ]

    if empty_chunk_ids:
        raise ValueError(
            "Cannot embed empty chunks. "
            f"Found {len(empty_chunk_ids)} empty chunks "
            f"with IDs: {empty_chunk_ids}"
        )

    unique_chunks, duplicate_count = (
        _deduplicate_chunks(chunks)
    )

    resolved_cache_path = (
        Path(cache_path)
        if cache_path is not None
        else None
    )

    vectors_by_id: dict[str, list[float]] = {}

    if resolved_cache_path is not None:
        vectors_by_id = load_embedding_cache(
            resolved_cache_path,
            model_name=model_name,
            normalize=normalize,
        )

    missing_chunks = [
        chunk
        for chunk in unique_chunks
        if chunk.metadata.chunk_id not in vectors_by_id
    ]

    reused_count = (
        len(unique_chunks) - len(missing_chunks)
    )

    duration = 0.0
    resolved_device: str | None = None

    if missing_chunks:
        resolved_device = resolve_device(device)

        model = (
            embedding_model
            if embedding_model is not None
            else load_embedding_model(
                model_name,
                resolved_device,
            )
        )

        # Embed only chunks that were not found in the cache.
        contents = [
            chunk.content
            for chunk in missing_chunks
        ]

        started_at = perf_counter()

        try:
            vectors = model.encode(
                contents,
                batch_size=batch_size,
                device=resolved_device,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception:
            duration = perf_counter() - started_at

            logger.exception(
                "Embedding failed: chunks=%d "
                "model=%s device=%s duration=%.3fs",
                len(missing_chunks),
                model_name,
                resolved_device,
                duration,
            )
            raise

        duration = perf_counter() - started_at

        new_vectors = _convert_vectors(
            vectors,
            len(missing_chunks),
        )

        for chunk, vector in zip(
            missing_chunks,
            new_vectors,
            strict=True,
        ):
            vectors_by_id[
                chunk.metadata.chunk_id
            ] = vector

    selected_vectors = [
        vectors_by_id[chunk.metadata.chunk_id]
        for chunk in unique_chunks
    ]

    # This must be a set of unique dimensions.
    dimensions = {
        len(vector)
        for vector in selected_vectors
    }

    if len(dimensions) != 1 or 0 in dimensions:
        raise RuntimeError(
            "Cached and generated embeddings have "
            "inconsistent dimensions"
        )

    if (
        resolved_cache_path is not None
        and missing_chunks
    ):
        save_embedding_cache(
            resolved_cache_path,
            vectors_by_id,
            model_name=model_name,
            normalize=normalize,
        )

    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            metadata=chunk.metadata,
            embedding=vector,
        )
        for chunk, vector in zip(
            unique_chunks,
            selected_vectors,
            strict=True,
        )
    ]

    device_used = (
        resolved_device
        if resolved_device is not None
        else "not-used-cache-only"
    )

    logger.info(
        "Embedding completed: chunks=%d embedded=%d "
        "reused=%d duplicates=%d dimensions=%d "
        "model=%s device=%s duration=%.3fs",
        len(embedded_chunks),
        len(missing_chunks),
        reused_count,
        duplicate_count,
        next(iter(dimensions)),
        model_name,
        device_used,
        duration,
    )

    return embedded_chunks

