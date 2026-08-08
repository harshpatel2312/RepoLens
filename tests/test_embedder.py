import logging
from unittest.mock import patch
from pathlib import Path

import pytest

from repolens.models import (
    ChunkMetadata,
    TextChunk,
)
from repolens.retrieval.embedder import (
    embed_chunks,
    load_embedding_model,
    resolve_device,
    validate_embedding_settings,
)


class FakeEmbeddingModel:
    def __init__(
        self,
        vectors: list[list[float]],
    ) -> None:
        self.vectors = vectors
        self.contents: list[str] = []
        self.arguments: dict[str, object] = {}

    def encode(
        self,
        contents: list[str],
        **kwargs: object,
    ) -> list[list[float]]:
        self.contents = list(contents)
        self.arguments = kwargs

        return self.vectors


class FailingEmbeddingModel:
    def encode(
        self,
        contents: list[str],
        **kwargs: object,
    ) -> list[list[float]]:
        raise RuntimeError("test embedding failure")


def create_chunk(
    content: str,
    *,
    chunk_id: str = "chunk-1",
) -> TextChunk:
    return TextChunk(
        content=content,
        metadata=ChunkMetadata(
            repository="example-repository",
            relative_path="src/example.py",
            file_type="python",
            start_line=1,
            end_line=1,
            token_count=4,
            content_hash="content-hash",
            chunk_id=chunk_id,
        ),
    )


def test_resolve_device_uses_cpu_when_cuda_unavailable() -> None:
    with patch(
        "repolens.retrieval.embedder."
        "torch.cuda.is_available",
        return_value=False,
    ):
        assert resolve_device("auto") == "cpu"


def test_explicit_unavailable_cuda_is_rejected() -> None:
    with patch(
        "repolens.retrieval.embedder."
        "torch.cuda.is_available",
        return_value=False,
    ):
        with pytest.raises(
            RuntimeError,
            match="CUDA is not available",
        ):
            resolve_device("cuda")


def test_embedding_settings_are_validated() -> None:
    with pytest.raises(
        ValueError,
        match="model_name cannot be empty",
    ):
        validate_embedding_settings("", 16)

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        validate_embedding_settings(
            "test-model",
            0,
        )


def test_embed_chunks_preserves_content_and_metadata() -> None:
    chunks = [
        create_chunk(
            "first chunk",
            chunk_id="chunk-1",
        ),
        create_chunk(
            "second chunk",
            chunk_id="chunk-2",
        ),
    ]

    model = FakeEmbeddingModel(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )

    with patch(
        "repolens.retrieval.embedder."
        "torch.cuda.is_available",
        return_value=False,
    ):
        results = embed_chunks(
            chunks,
            model_name="test-model",
            batch_size=2,
            normalize=True,
            device="auto",
            embedding_model=model,
        )

    assert len(results) == 2

    assert results[0].content == "first chunk"
    assert (
        results[0].metadata.chunk_id
        == "chunk-1"
    )
    assert results[0].embedding == [
        0.1,
        0.2,
        0.3,
    ]

    assert (
        results[1].metadata.chunk_id
        == "chunk-2"
    )

    assert model.contents == [
        "first chunk",
        "second chunk",
    ]

    assert model.arguments == {
        "batch_size": 2,
        "device": "cpu",
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }


def test_empty_input_does_not_call_model() -> None:
    results = embed_chunks(
        [],
        device="cpu",
        embedding_model=FailingEmbeddingModel(),
    )

    assert results == []


def test_empty_chunk_is_rejected() -> None:
    chunk = create_chunk("   \n")

    with pytest.raises(
        ValueError,
        match="Cannot embed empty chunks",
    ):
        embed_chunks(
            [chunk],
            device="cpu",
            embedding_model=FakeEmbeddingModel([]),
        )


def test_embedding_failure_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunk = create_chunk("example content")

    with caplog.at_level(
        logging.ERROR,
        logger="repolens",
    ):
        with pytest.raises(
            RuntimeError,
            match="test embedding failure",
        ):
            embed_chunks(
                [chunk],
                model_name="test-model",
                device="cpu",
                embedding_model=FailingEmbeddingModel(),
            )

    assert "Embedding failed" in caplog.text


def test_model_loader_caches_model() -> None:
    fake_model = FakeEmbeddingModel(
        [[0.1, 0.2]]
    )

    load_embedding_model.cache_clear()

    with patch(
        "repolens.retrieval.embedder."
        "SentenceTransformer",
        return_value=fake_model,
    ) as model_class:
        first_model = load_embedding_model(
            "test-model",
            "cpu",
        )
        second_model = load_embedding_model(
            "test-model",
            "cpu",
        )

    assert first_model is fake_model
    assert second_model is fake_model

    model_class.assert_called_once_with(
        "test-model",
        device="cpu",
    )

    load_embedding_model.cache_clear()


def test_duplicate_chunk_ids_are_embedded_once() -> None:
    chunk = create_chunk(
        "duplicate content",
        chunk_id="duplicate-id",
    )

    model = FakeEmbeddingModel(
        [[0.1, 0.2, 0.3]]
    )

    results = embed_chunks(
        [chunk, chunk],
        model_name="test-model",
        device="cpu",
        embedding_model=model,
    )

    assert len(results) == 1
    assert model.contents == ["duplicate content"]


def test_cached_chunk_is_not_embedded_again(
    tmp_path: Path,
) -> None:
    cache_path = (
        tmp_path / "embedding-cache.json"
    )

    chunk = create_chunk(
        "cached content",
        chunk_id="cached-id",
    )

    first_model = FakeEmbeddingModel(
        [[0.1, 0.2, 0.3]]
    )

    embed_chunks(
        [chunk],
        model_name="test-model",
        device="cpu",
        embedding_model=first_model,
        cache_path=cache_path,
    )

    results = embed_chunks(
        [chunk],
        model_name="test-model",
        device="cpu",
        embedding_model=FailingEmbeddingModel(),
        cache_path=cache_path,
    )

    assert results[0].embedding == [
        0.1,
        0.2,
        0.3,
    ]


def test_only_new_chunk_is_embedded(
    tmp_path: Path,
) -> None:
    cache_path = (
        tmp_path / "embedding-cache.json"
    )

    unchanged_chunk = create_chunk(
        "unchanged content",
        chunk_id="unchanged-id",
    )

    first_model = FakeEmbeddingModel(
        [[0.1, 0.2]]
    )

    embed_chunks(
        [unchanged_chunk],
        model_name="test-model",
        device="cpu",
        embedding_model=first_model,
        cache_path=cache_path,
    )

    changed_chunk = create_chunk(
        "changed content",
        chunk_id="changed-id",
    )

    second_model = FakeEmbeddingModel(
        [[0.8, 0.9]]
    )

    results = embed_chunks(
        [unchanged_chunk, changed_chunk],
        model_name="test-model",
        device="cpu",
        embedding_model=second_model,
        cache_path=cache_path,
    )

    assert second_model.contents == [
        "changed content"
    ]

    assert results[0].embedding == [0.1, 0.2]
    assert results[1].embedding == [0.8, 0.9]


def test_different_model_invalidates_cache(
    tmp_path: Path,
) -> None:
    cache_path = (
        tmp_path / "embedding-cache.json"
    )

    chunk = create_chunk(
        "example content",
        chunk_id="chunk-id",
    )

    embed_chunks(
        [chunk],
        model_name="model-a",
        device="cpu",
        embedding_model=FakeEmbeddingModel(
            [[0.1, 0.2]]
        ),
        cache_path=cache_path,
    )

    replacement_model = FakeEmbeddingModel(
        [[0.7, 0.8]]
    )

    results = embed_chunks(
        [chunk],
        model_name="model-b",
        device="cpu",
        embedding_model=replacement_model,
        cache_path=cache_path,
    )

    assert replacement_model.contents == [
        "example content"
    ]
    assert results[0].embedding == [0.7, 0.8]