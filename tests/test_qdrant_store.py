from qdrant_client import QdrantClient, models
import pytest

from repolens.models import ChunkMetadata, EmbeddedChunk
from repolens.retrieval.qdrant_store import (
    build_search_filter,
    ensure_collection,
    point_id_for_chunk,
    search_points,
    upsert_embedded_chunks,
)


def create_embedded_chunk(
    *,
    chunk_id: str,
    content: str,
    vector: list[float],
    repository: str = "RepoLens",
    relative_path: str = "repolens/example.py",
    file_type: str = "python",
) -> EmbeddedChunk:
    return EmbeddedChunk(
        content=content,
        metadata=ChunkMetadata(
            repository=repository,
            relative_path=relative_path,
            file_type=file_type,
            start_line=1,
            end_line=3,
            token_count=5,
            content_hash=f"hash-{chunk_id}",
            chunk_id=chunk_id,
            heading=None,
        ),
        embedding=vector,
    )


def test_point_id_is_deterministic_uuid() -> None:
    first = point_id_for_chunk("stable-chunk-id")
    second = point_id_for_chunk("stable-chunk-id")

    assert first == second
    assert first != point_id_for_chunk("different-chunk-id")
    assert len(first) == 36


def test_ensure_collection_creates_and_reuses_collection() -> None:
    client = QdrantClient(":memory:")

    assert ensure_collection(client, 3, collection_name="chunks") is True
    assert ensure_collection(client, 3, collection_name="chunks") is False


def test_incompatible_collection_is_rejected() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="chunks",
        vectors_config=models.VectorParams(
            size=2,
            distance=models.Distance.COSINE,
        ),
    )

    with pytest.raises(ValueError, match="incompatible"):
        ensure_collection(client, 3, collection_name="chunks")


def test_repeated_upsert_does_not_create_duplicates() -> None:
    client = QdrantClient(":memory:")
    chunk = create_embedded_chunk(
        chunk_id="chunk-1",
        content="Embedding caching is implemented here.",
        vector=[1.0, 0.0, 0.0],
    )

    upsert_embedded_chunks(client, [chunk], collection_name="chunks")
    upsert_embedded_chunks(client, [chunk], collection_name="chunks")

    assert client.count("chunks", exact=True).count == 1

    stored = client.retrieve(
        "chunks",
        ids=[point_id_for_chunk("chunk-1")],
        with_payload=True,
    )[0]
    assert stored.payload["content"] == chunk.content
    assert stored.payload["chunk_id"] == "chunk-1"
    assert stored.payload["relative_path"] == "repolens/example.py"


def test_search_returns_ranked_results_and_applies_filters() -> None:
    client = QdrantClient(":memory:")
    chunks = [
        create_embedded_chunk(
            chunk_id="python-result",
            content="Python embedding cache implementation",
            vector=[1.0, 0.0],
            repository="RepoLens",
            relative_path="repolens/retrieval/embedding_cache.py",
            file_type="python",
        ),
        create_embedded_chunk(
            chunk_id="markdown-result",
            content="Embedding cache documentation",
            vector=[0.8, 0.2],
            repository="RepoLens",
            relative_path="README.md",
            file_type="markdown",
        ),
        create_embedded_chunk(
            chunk_id="other-repository",
            content="Another repository",
            vector=[0.99, 0.01],
            repository="OtherRepo",
        ),
    ]
    upsert_embedded_chunks(client, chunks, collection_name="chunks")

    results = search_points(
        client,
        [1.0, 0.0],
        collection_name="chunks",
        top_k=5,
        repository="RepoLens",
        file_type="python",
    )

    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].metadata.chunk_id == "python-result"
    assert results[0].metadata.repository == "RepoLens"
    assert results[0].metadata.file_type == "python"


def test_filter_is_optional_and_validates_empty_values() -> None:
    assert build_search_filter() is None

    with pytest.raises(ValueError, match="repository filter"):
        build_search_filter(repository="   ")


def test_empty_and_inconsistent_vectors_are_rejected() -> None:
    client = QdrantClient(":memory:")
    valid = create_embedded_chunk(
        chunk_id="valid",
        content="valid",
        vector=[1.0, 0.0],
    )
    invalid = create_embedded_chunk(
        chunk_id="invalid",
        content="invalid",
        vector=[1.0, 0.0, 0.0],
    )

    assert upsert_embedded_chunks(client, [], collection_name="chunks") == 0

    with pytest.raises(ValueError, match="one dimension"):
        upsert_embedded_chunks(
            client,
            [valid, invalid],
            collection_name="chunks",
        )