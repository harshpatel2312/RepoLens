"""Acceptance coverage for REP-9 structure-aware Python chunking."""

import logging

import pytest

from repolens.ingestion.chunker import chunk_document
from repolens.models import FileMetadata, IngestedDocument


def create_python_document(
    content: str,
    *,
    relative_path: str = "src/example.py",
) -> IngestedDocument:
    return IngestedDocument(
        content=content,
        metadata=FileMetadata(
            repository="example-repository",
            relative_path=relative_path,
            file_type="python",
            size_bytes=len(content.encode("utf-8")),
            content_hash="source-file-hash",
        ),
    )


def test_module_statements_receive_module_metadata() -> None:
    document = create_python_document(
        '"""Repository configuration."""\n'
        "from pathlib import Path\n"
        "DEFAULT_ROOT = Path('.')\n"
    )

    chunks = chunk_document(document, min_chunk_size=1)

    assert len(chunks) == 1
    assert chunks[0].metadata.symbol_type == "module"
    assert chunks[0].metadata.symbol_name == "src.example"
    assert "DEFAULT_ROOT" in chunks[0].content


def test_package_initializer_uses_package_module_name() -> None:
    document = create_python_document(
        "VERSION = '1.0'\n",
        relative_path="repolens/retrieval/__init__.py",
    )

    chunk = chunk_document(document, min_chunk_size=1)[0]

    assert chunk.metadata.symbol_name == "repolens.retrieval"


def test_complete_function_preserves_decorator_signature_and_docstring() -> None:
    document = create_python_document(
        "from functools import lru_cache\n\n"
        "@lru_cache(maxsize=4)\n"
        "def load_model(name: str) -> str:\n"
        '    """Load and cache one model."""\n'
        "    return name.strip()\n"
    )

    chunks = chunk_document(document, chunk_size=100, min_chunk_size=1)
    function_chunks = [
        chunk for chunk in chunks
        if chunk.metadata.symbol_type == "function"
    ]

    assert len(function_chunks) == 1
    chunk = function_chunks[0]
    assert chunk.metadata.symbol_name == "load_model"
    assert chunk.metadata.signature == "def load_model(name: str) -> str:"
    assert chunk.metadata.start_line == 3
    assert chunk.metadata.end_line == 6
    assert "@lru_cache(maxsize=4)" in chunk.content
    assert '"""Load and cache one model."""' in chunk.content
    assert "return name.strip()" in chunk.content


def test_complete_class_and_individual_methods_are_retrievable() -> None:
    document = create_python_document(
        "class RepositoryIndex:\n"
        '    """Index repository chunks."""\n\n'
        "    def search(self, query: str) -> list[str]:\n"
        '        """Return matching files."""\n'
        "        return [query]\n"
    )

    chunks = chunk_document(document, chunk_size=150, min_chunk_size=1)

    class_chunk = next(
        chunk for chunk in chunks
        if chunk.metadata.symbol_type == "class"
    )
    method_chunk = next(
        chunk for chunk in chunks
        if chunk.metadata.symbol_type == "method"
    )

    assert class_chunk.metadata.symbol_name == "RepositoryIndex"
    assert class_chunk.metadata.start_line == 1
    assert class_chunk.metadata.end_line == 6
    assert "def search(self, query: str)" in class_chunk.content

    assert method_chunk.metadata.symbol_name == "RepositoryIndex.search"
    assert method_chunk.metadata.parent_symbol == "RepositoryIndex"
    assert method_chunk.metadata.start_line == 4
    assert method_chunk.metadata.end_line == 6
    assert method_chunk.content.startswith("class RepositoryIndex:\n")
    assert '"""Return matching files."""' in method_chunk.content


def test_async_methods_and_decorators_keep_class_context() -> None:
    document = create_python_document(
        "class Client(BaseClient):\n"
        "    @staticmethod\n"
        "    async def fetch(identifier: str) -> str:\n"
        "        return identifier\n"
    )

    chunks = chunk_document(document, chunk_size=150, min_chunk_size=1)
    method = next(
        chunk for chunk in chunks
        if chunk.metadata.symbol_type == "method"
    )

    assert method.metadata.symbol_name == "Client.fetch"
    assert method.metadata.signature == (
        "async def fetch(identifier: str) -> str:"
    )
    assert method.metadata.start_line == 2
    assert method.content.startswith("class Client(BaseClient):\n")
    assert "@staticmethod" in method.content


def test_nested_classes_preserve_qualified_parent_context() -> None:
    document = create_python_document(
        "class Outer:\n"
        "    class Inner:\n"
        "        def run(self) -> None:\n"
        "            return None\n"
    )

    chunks = chunk_document(document, chunk_size=150, min_chunk_size=1)
    method = next(
        chunk for chunk in chunks
        if chunk.metadata.symbol_type == "method"
    )

    assert method.metadata.symbol_name == "Outer.Inner.run"
    assert method.metadata.parent_symbol == "Outer.Inner"
    assert method.content.startswith("class Outer:\n    class Inner:\n")


def test_oversized_function_splits_without_losing_source_lines() -> None:
    document = create_python_document(
        "def calculate(value: int) -> int:\n"
        "    first = value + 1\n"
        "    second = first + 2\n"
        "    third = second + 3\n"
        "    return third\n"
    )

    chunks = chunk_document(
        document,
        chunk_size=14,
        overlap=3,
        min_chunk_size=1,
    )
    function_chunks = [
        chunk for chunk in chunks
        if chunk.metadata.symbol_type == "function"
    ]

    assert len(function_chunks) > 1
    assert all(
        chunk.metadata.symbol_name == "calculate"
        for chunk in function_chunks
    )
    assert function_chunks[0].metadata.start_line == 1
    assert function_chunks[-1].metadata.end_line == 5
    assert any("return third" in chunk.content for chunk in function_chunks)


def test_module_content_before_between_and_after_symbols_is_retained() -> None:
    document = create_python_document(
        "BEFORE = 1\n\n"
        "def first():\n"
        "    return BEFORE\n\n"
        "BETWEEN = 2\n\n"
        "def second():\n"
        "    return BETWEEN\n\n"
        "AFTER = 3\n"
    )

    chunks = chunk_document(document, chunk_size=100, min_chunk_size=1)
    module_content = "".join(
        chunk.content for chunk in chunks
        if chunk.metadata.symbol_type == "module"
    )

    assert "BEFORE = 1" in module_content
    assert "BETWEEN = 2" in module_content
    assert "AFTER = 3" in module_content


def test_syntax_errors_use_documented_baseline_fallback(caplog) -> None:
    document = create_python_document(
        "def broken(:\n"
        "    return 1\n"
    )

    with caplog.at_level(logging.WARNING, logger="repolens"):
        chunks = chunk_document(document, chunk_size=100, min_chunk_size=1)

    assert len(chunks) == 1
    assert "def broken(:" in chunks[0].content
    assert chunks[0].metadata.symbol_type is None
    assert chunks[0].metadata.symbol_name is None
    assert "falling back to baseline line-aware chunking" in caplog.text


def test_ast_chunks_have_deterministic_ids() -> None:
    document = create_python_document(
        "class Searcher:\n"
        "    def find(self, query: str) -> str:\n"
        "        return query\n"
    )

    first = chunk_document(document, min_chunk_size=1)
    second = chunk_document(document, min_chunk_size=1)

    assert [chunk.metadata.chunk_id for chunk in first] == [
        chunk.metadata.chunk_id for chunk in second
    ]


@pytest.mark.parametrize(
    "source, expected_name",
    [
        ("def synchronous():\n    return 1\n", "synchronous"),
        ("async def asynchronous():\n    return 1\n", "asynchronous"),
    ],
)
def test_synchronous_and_async_functions_are_supported(
    source: str,
    expected_name: str,
) -> None:
    chunks = chunk_document(
        create_python_document(source),
        min_chunk_size=1,
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.symbol_type == "function"
    assert chunks[0].metadata.symbol_name == expected_name
