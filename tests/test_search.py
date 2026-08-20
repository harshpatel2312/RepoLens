from qdrant_client import QdrantClient
import pytest

from repolens.models import ChunkMetadata, EmbeddedChunk
from repolens.retrieval.qdrant_store import upsert_embedded_chunks
from repolens.retrieval.search import embed_query, search_repository


class FakeEmbeddingModel:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.contents: list[str] = []
        self.arguments: dict[str, object] = {}

    def encode(
        self,
        contents: list[str],
        **kwargs: object,
    ) -> list[list[float]]:
        self.contents = contents
        self.arguments = kwargs
        return [self.vector]


def test_embed_query_validates_and_embeds_one_query() -> None:
    model = FakeEmbeddingModel([0.25, 0.75])

    vector = embed_query(
        "  where is caching implemented?  ",
        model_name="test-model",
        device="cpu",
        embedding_model=model,
    )

    assert vector == [0.25, 0.75]
    assert model.contents == [
        "Represent this sentence for searching relevant passages: "
        "where is caching implemented?"
    ]
    assert model.arguments == {
        "device": "cpu",
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }

    with pytest.raises(ValueError, match="cannot be empty"):
        embed_query("  ", embedding_model=model)


def test_search_repository_runs_end_to_end_with_in_memory_qdrant() -> None:
    client = QdrantClient(":memory:")
    chunk = EmbeddedChunk(
        content="load_embedding_cache reads cached vectors from disk",
        metadata=ChunkMetadata(
            repository="RepoLens",
            relative_path="repolens/retrieval/embedding_cache.py",
            file_type="python",
            start_line=1,
            end_line=10,
            token_count=8,
            content_hash="content-hash",
            chunk_id="cache-chunk",
        ),
        embedding=[1.0, 0.0],
    )
    upsert_embedded_chunks(client, [chunk], collection_name="chunks")

    results = search_repository(
        "Where is embedding caching implemented?",
        client,
        collection_name="chunks",
        model_name="test-model",
        device="cpu",
        embedding_model=FakeEmbeddingModel([1.0, 0.0]),
    )

    assert len(results) == 1
    assert results[0].metadata.relative_path == (
        "repolens/retrieval/embedding_cache.py"
    )