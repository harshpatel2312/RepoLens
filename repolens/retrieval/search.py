"""Semantic query embedding and retrieval orchestration."""

from typing import Any

from qdrant_client import QdrantClient

from repolens.config import config
from repolens.models import SearchResult
from repolens.retrieval.embedder import load_embedding_model, resolve_device
from repolens.retrieval.qdrant_store import search_points


def embed_query(
    query: str,
    *,
    model_name: str = str(config["EMBEDDING"]["MODEL_NAME"]),
    normalize: bool = bool(config["EMBEDDING"]["NORMALIZE"]),
    device: str = str(config["EMBEDDING"]["DEVICE"]),
    query_prefix: str = str(config["EMBEDDING"]["QUERY_PREFIX"]),
    embedding_model: Any | None = None,
) -> list[float]:
    """Embed one non-empty search query with the configured model."""

    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    if not model_name.strip():
        raise ValueError("model_name cannot be empty.")

    resolved_device = resolve_device(device)
    model = (
        embedding_model
        if embedding_model is not None
        else load_embedding_model(model_name, resolved_device)
    )

    vectors =  model.encode(
        [f"{query_prefix}{query}"],
        device=resolved_device,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    if len(vectors) != 1:
        raise RuntimeError("Embedding model did not return one query vector.")

    vector = [float(value) for value in vectors[0]]
    if not vector:
        raise RuntimeError("Embedding model returned an empty query vector.")

    return vector


def search_repository(
    query: str,
    client: QdrantClient,
    *,
    collection_name: str = str(config["QDRANT"]["COLLECTION_NAME"]),
    top_k: int = int(config["RETRIEVAL"]["TOP_K"]),
    repository: str | None = None,
    file_type: str | None = None,
    model_name: str = str(config["EMBEDDING"]["MODEL_NAME"]),
    normalize: bool = bool(config["EMBEDDING"]["NORMALIZE"]),
    device: str = str(config["EMBEDDING"]["DEVICE"]),
    query_prefix: str = str(config["EMBEDDING"]["QUERY_PREFIX"]),
    embedding_model: Any | None = None,
) -> list[SearchResult]:
    """Embed a question and retrieve its most relevant code chunks."""
    
    query_vector = embed_query(
        query,
        model_name=model_name,
        normalize=normalize,
        device=device,
        query_prefix=query_prefix,
        embedding_model=embedding_model,
    )

    return search_points(
        client,
        query_vector,
        collection_name=collection_name,
        top_k=top_k,
        repository=repository,
        file_type=file_type,
    )