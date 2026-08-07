from collections.abc import Sequence
from functools import lru_cache
from time import perf_counter

import torch
from sentence_transformers import SentenceTransformer

from repolens.models import TextChunk, EmbeddedChunk
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


def embed_chunks(
    chunks: Sequence[TextChunk],
    model_name: str = str(config["EMBEDDING"]["MODEL_NAME"]),
    batch_size: int = int(config["EMBEDDING"]["BATCH_SIZE"]),
    normalize: bool = bool(config["EMBEDDING"]["NORMALIZE"]),
    device: str = str(config["EMBEDDING"]["DEVICE"]),
    embedding_model: SentenceTransformer | None = None,
) -> list[EmbeddedChunk]:
    """Generate embeddings while preserving chunk metadata"""

    validate_embedding_settings(model_name, batch_size)

    if not chunks:
        logger.debug("No chunks provided for embedding")
        return []

    empty_chunk_ids = [chunk.metadata.chunk_id for chunk in chunks if not chunk.content.strip()]

    if empty_chunk_ids:
        raise ValueError(f"Cannot embed empty chunks. Found {len(empty_chunk_ids)} empty chunks with IDs: {empty_chunk_ids}")

    device = resolve_device(device)

    model = (
        embedding_model 
        if embedding_model is not None 
        else load_embedding_model(model_name, device)
    )

    contents = [chunk.content for chunk in chunks]

    started_at = perf_counter()

    try:
        vectors = model.encode(
            contents,
            batch_size=batch_size,
            device=device,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception:
        duration = perf_counter() - started_at
        logger.exception(
            "Embedding failed: chunks=%d "
            "model=%s device=%s duration=%.3fs",
            len(chunks),
            model_name,
            device,
            duration,
        )
        raise

    duration = perf_counter() - started_at

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Mismatch between number of embeddings ({len(vectors)}) and number of chunks ({len(chunks)})."
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
            f"Unexpected embedding dimensions: {dimensions}. "
            "All embeddings should have the same non-zero dimension."
        )

    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            metadata=chunk.metadata,
            embedding=vector,
        )
        for chunk, vector in zip(chunks, vector_lists, strict=True)
    ]

    embedding_dimension = next(iter(dimensions))

    logger.info(
        "Embedding completed: chunks=%d "
        "dimensions=%d model=%s device=%s "
        "normalized=%s duration=%.3fs",
        len(embedded_chunks),
        embedding_dimension,
        model_name,
        device,
        normalize,
        duration,
    )

    return embedded_chunks

    

