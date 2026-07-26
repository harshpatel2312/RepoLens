from repolens.ingestion.chunker import (
    chunk_document,
    generate_chunk_id,
)
from repolens.models import (
    FileMetadata,
    IngestedDocument,
)


def create_document(content: str) -> IngestedDocument:
    return IngestedDocument(
        content=content,
        metadata=FileMetadata(
            repository="example-repository",
            relative_path="src/example.py",
            file_type="python",
            size_bytes=len(content.encode("utf-8")),
            content_hash="source-file-hash",
        ),
    )


def test_chunk_document_preserves_metadata() -> None:
    document = create_document(
        "first = 1\n"
        "second = 2\n"
        "third = 3\n"
        "fourth = 4\n"
    )

    chunks = chunk_document(
        document,
        chunk_size=6,
        overlap=2,
        min_chunk_size=1,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.content.strip()
        assert chunk.metadata.repository == "example-repository"
        assert chunk.metadata.relative_path == "src/example.py"
        assert chunk.metadata.file_type == "python"
        assert chunk.metadata.start_line >= 1
        assert (
            chunk.metadata.end_line
            >= chunk.metadata.start_line
        )
        assert chunk.metadata.token_count > 0
        assert len(chunk.metadata.content_hash) == 64
        assert len(chunk.metadata.chunk_id) == 64
        assert chunk.metadata.heading is None


def test_chunk_ids_are_deterministic() -> None:
    document = create_document(
        "alpha = 1\n"
        "beta = 2\n"
        "gamma = 3\n"
    )

    first_result = chunk_document(
        document,
        chunk_size=5,
        overlap=1,
        min_chunk_size=1,
    )
    second_result = chunk_document(
        document,
        chunk_size=5,
        overlap=1,
        min_chunk_size=1,
    )

    first_ids = [
        chunk.metadata.chunk_id
        for chunk in first_result
    ]
    second_ids = [
        chunk.metadata.chunk_id
        for chunk in second_result
    ]

    assert first_ids == second_ids


def test_content_change_produces_different_chunk_id() -> None:
    first_document = create_document("value = 1\n")
    second_document = create_document("value = 2\n")

    first_chunk = chunk_document(
        first_document,
        min_chunk_size=1,
    )[0]
    second_chunk = chunk_document(
        second_document,
        min_chunk_size=1,
    )[0]

    assert (
        first_chunk.metadata.content_hash
        != second_chunk.metadata.content_hash
    )
    assert (
        first_chunk.metadata.chunk_id
        != second_chunk.metadata.chunk_id
    )


def test_chunk_line_ranges_include_overlap() -> None:
    document = create_document(
        "alpha\n"
        "beta\n"
        "gamma\n"
        "delta\n"
        "epsilon\n"
    )

    chunks = chunk_document(
        document,
        chunk_size=4,
        overlap=2,
        min_chunk_size=1,
    )

    assert len(chunks) > 1
    assert (
        chunks[1].metadata.start_line
        <= chunks[0].metadata.end_line
    )


def test_empty_document_produces_no_chunks() -> None:
    document = create_document("   \n")

    chunks = chunk_document(
        document,
        min_chunk_size=1,
    )

    assert chunks == []


def test_chunk_identity_includes_source_location() -> None:
    first_id = generate_chunk_id(
        repository="repo",
        relative_path="first.py",
        start_line=1,
        end_line=2,
        content_hash="same-hash",
    )
    second_id = generate_chunk_id(
        repository="repo",
        relative_path="second.py",
        start_line=1,
        end_line=2,
        content_hash="same-hash",
    )

    assert first_id != second_id