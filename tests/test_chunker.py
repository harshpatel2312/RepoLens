from repolens.ingestion.chunker import (
    chunk_document,
    generate_chunk_id,
)
from repolens.models import (
    FileMetadata,
    IngestedDocument,
)


def create_document(
    content: str,
    *,
    relative_path: str = "src/example.py",
    file_type: str = "python",
) -> IngestedDocument:
    return IngestedDocument(
        content=content,
        metadata=FileMetadata(
            repository="example-repository",
            relative_path=relative_path,
            file_type=file_type,
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


def test_markdown_headings_stay_with_their_sections() -> None:
    document = create_document(
        "Project overview.\n"
        "\n"
        "# Installation\n"
        "Install the package.\n"
        "\n"
        "## Configuration\n"
        "Set the environment variables.\n",
        relative_path="README.md",
        file_type="markdown",
    )

    chunks = chunk_document(
        document,
        chunk_size=100,
        overlap=0,
        min_chunk_size=1,
    )

    assert len(chunks) == 3

    assert chunks[0].metadata.heading is None

    assert chunks[1].metadata.heading == "Installation"
    assert chunks[1].content.startswith("# Installation\n")
    assert chunks[1].metadata.start_line == 3

    assert chunks[2].metadata.heading == "Configuration"
    assert chunks[2].content.startswith("## Configuration\n")
    assert chunks[2].metadata.start_line == 6


def test_oversized_markdown_section_is_split() -> None:
    document = create_document(
        "# Installation\n"
        "Run the first command.\n"
        "Run the second command.\n"
        "Run the third command.\n"
        "Run the fourth command.\n",
        relative_path="README.md",
        file_type="markdown",
    )

    chunks = chunk_document(
        document,
        chunk_size=8,
        overlap=2,
        min_chunk_size=1,
    )

    assert len(chunks) > 1
    assert chunks[0].content.startswith("# Installation\n")

    assert all(
        chunk.metadata.heading == "Installation"
        for chunk in chunks
    )


def test_markdown_heading_inside_code_fence_is_ignored() -> None:
    document = create_document(
        "# Example\n"
        "```python\n"
        "# This is a Python comment\n"
        "print('hello')\n"
        "```\n"
        "More explanation.\n",
        relative_path="README.md",
        file_type="markdown",
    )

    chunks = chunk_document(
        document,
        chunk_size=100,
        overlap=0,
        min_chunk_size=1,
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.heading == "Example"

    assert (
        "# This is a Python comment"
        in chunks[0].content
    )


def test_markdown_chunk_ids_are_deterministic() -> None:
    document = create_document(
        "# Usage\n"
        "Run the application.\n",
        relative_path="README.md",
        file_type="markdown",
    )

    first_chunks = chunk_document(
        document,
        chunk_size=10,
        overlap=2,
        min_chunk_size=1,
    )

    second_chunks = chunk_document(
        document,
        chunk_size=10,
        overlap=2,
        min_chunk_size=1,
    )

    first_ids = [
        chunk.metadata.chunk_id
        for chunk in first_chunks
    ]

    second_ids = [
        chunk.metadata.chunk_id
        for chunk in second_chunks
    ]

    assert first_ids == second_ids