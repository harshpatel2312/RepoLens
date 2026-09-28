from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import PurePosixPath

from repolens.config import logger
from repolens.ingestion.chunker import (
    _chunk_line_range,
    _create_chunk,
    count_tokens,
    validate_chunking_settings,
)
from repolens.models import IngestedDocument, TextChunk


SymbolNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _module_name(relative_path: str) -> str:
    """Return a stable import-style name for a Python source path."""

    path = PurePosixPath(relative_path).with_suffix("")  # Remove the file extension
    parts = list(path.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or "__init__"


def _symbol_start(node: SymbolNode) -> int:
    """Include decorators in the source range for decorated symbols."""

    decorator_lines = [decorator.lineno for decorator in node.decorator_list]
    return min([node.lineno, *decorator_lines])


def _signature(node: SymbolNode) -> str:
    """Build a readable signature without decorators or implementation."""

    signature_node = copy.copy(node)
    signature_node.decorator_list = []
    return ast.unparse(signature_node).splitlines()[0]


def _class_context(parents: tuple[ast.ClassDef, ...]) -> str:
    """Return enclosing class headers at their original indentation levels."""

    return "".join(
        f"{' ' * parent.col_offset}{_signature(parent)}\n"
        for parent in parents
    )


def _with_symbol_metadata(
    chunk: TextChunk,
    *,
    symbol_type: str,
    symbol_name: str,
    parent_symbol: str | None = None,
    signature: str | None = None,
) -> TextChunk:
    """Return a copy of the chunk with symbol metadata attached."""

    return replace(
        chunk,
        metadata=replace(
            chunk.metadata,
            symbol_type=symbol_type,
            symbol_name=symbol_name,
            parent_symbol=parent_symbol,
            signature=signature,
        )
    )


def _module_chunks(
    document: IngestedDocument,
    lines: list[str],
    start: int,
    end: int,
    *,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> list[TextChunk]:
    """Keep imports, constants, module docstrings, and executable statements."""

    if start >= end:
        return []

    chunks = _chunk_line_range(
        document,
        lines,
        start,
        end,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
    )

    return [
        _with_symbol_metadata(
            chunk,
            symbol_type="module",
            symbol_name=_module_name(document.metadata.relative_path),
        )
        for chunk in chunks
    ]


def _symbol_chunks(
    document: IngestedDocument,
    lines: list[str],
    node: SymbolNode,
    *,
    parents: tuple[ast.ClassDef, ...],
    symbol_type: str,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> list[TextChunk]:
    """Keep a complete symbol when possible, otherwise split its source."""

    start_line = _symbol_start(node)
    end_line = node.end_lineno
    if end_line is None:
        raise ValueError(f"Python symbol {node.name!r} has no ending line.")

    parent_symbol = ".".join(parent.name for parent in parents) or None
    symbol_name = (
        f"{parent_symbol}.{node.name}"
        if parent_symbol is not None
        else node.name
    )
    signature = _signature(node)
    context = _class_context(parents)
    source = "".join(lines[start_line - 1:end_line])
    complete_content = f"{context}{source}"

    if count_tokens(complete_content) <= chunk_size:
        complete_chunk = _create_chunk(
            document,
            complete_content,
            start_line,
            end_line,
        )
        return [
            _with_symbol_metadata(
                complete_chunk,
                symbol_type=symbol_type,
                symbol_name=symbol_name,
                parent_symbol=parent_symbol,
                signature=signature,
            )
        ]

    context_tokens = count_tokens(context) if context else 0
    source_budget = max(1, chunk_size - context_tokens)
    source_overlap = min(overlap, max(0, source_budget - 1))
    source_minimum = min(min_chunk_size, source_budget)
    source_chunks = _chunk_line_range(
        document,
        lines,
        start_line - 1,
        end_line,
        chunk_size=source_budget,
        overlap=source_overlap,
        min_chunk_size=source_minimum,
    )

    chunks: list[TextChunk] = []
    for source_chunk in source_chunks:
        content = f"{context}{source_chunk.content}"
        contextual_chunk = _create_chunk(
            document,
            content,
            source_chunk.metadata.start_line,
            source_chunk.metadata.end_line,
        )
        chunks.append(
            _with_symbol_metadata(
                contextual_chunk,
                symbol_type=symbol_type,
                symbol_name=symbol_name,
                parent_symbol=parent_symbol,
                signature=signature,
            )
        )

    return chunks


def _visit_symbol(
    document: IngestedDocument,
    lines: list[str],
    node: SymbolNode,
    *,
    parents: tuple[ast.ClassDef, ...],
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> list[TextChunk]:
    """Emit a class/function and recursively expose class-owned symbols."""

    if isinstance(node, ast.ClassDef):
        symbol_type = "class"
    elif parents:
        symbol_type = "method"
    else:
        symbol_type = "function"

    chunks = _symbol_chunks(
        document,
        lines,
        node,
        parents=parents,
        symbol_type=symbol_type,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
    )

    if isinstance(node, ast.ClassDef):
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                chunks.extend(
                    _visit_symbol(
                        document,
                        lines,
                        child,
                        parents=(*parents, node),
                        chunk_size=chunk_size,
                        overlap=overlap,
                        min_chunk_size=min_chunk_size,
                    )
                )

    return chunks


def chunk_python_document(
    document: IngestedDocument,
    *,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> list[TextChunk]:
    """Chunk Python syntax trees; fall back to baseline on syntax errors.

    Syntax-invalid files remain searchable through the existing line-aware
    splitter. Fallback chunks intentionally have no symbol metadata because
    their source cannot be parsed into trustworthy Python definitions.
    """

    validate_chunking_settings(chunk_size, overlap, min_chunk_size)
    if not document.content.strip():
        return []

    lines = document.content.splitlines(keepends=True)

    try:
        tree = ast.parse(
            document.content,
            filename=document.metadata.relative_path,
            type_comments=True,
        )
    except SyntaxError as error:
        logger.warning(
            "AST parsing failed for %s at line %s; falling back to "
            "baseline line-aware chunking: %s",
            document.metadata.relative_path,
            error.lineno,
            error.msg,
        )
        return _chunk_line_range(
            document,
            lines,
            0,
            len(lines),
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )

    symbols = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not symbols:
        return _module_chunks(
            document,
            lines,
            0,
            len(lines),
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )

    chunks: list[TextChunk] = []
    next_module_line = 0

    for symbol in symbols:
        symbol_start = _symbol_start(symbol) - 1
        chunks.extend(
            _module_chunks(
                document,
                lines,
                next_module_line,
                symbol_start,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_size=min_chunk_size,
            )
        )
        chunks.extend(
            _visit_symbol(
                document,
                lines,
                symbol,
                parents=(),
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_size=min_chunk_size,
            )
        )
        next_module_line = symbol.end_lineno or symbol.lineno

    chunks.extend(
        _module_chunks(
            document,
            lines,
            next_module_line,
            len(lines),
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )
    )

    logger.debug(
        "AST-chunked %s into %d chunks from %d top-level symbols",
        document.metadata.relative_path,
        len(chunks),
        len(symbols),
    )

    return chunks