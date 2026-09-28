# REP-9: baseline versus AST-aware Python chunking

## Why structure-aware chunks improve retrieval

The baseline splitter follows source lines and token budgets. It preserves
traceability but can separate a Python decorator, signature, docstring, and
implementation across unrelated chunks. It also has no reliable way to tell
whether a matching fragment belongs to a module, class, function, or method.

The AST-aware splitter parses Python files with the standard-library `ast`
module before applying the existing token-aware fallback splitter. It retains
whole symbols whenever they fit the configured limit and records their source
identity throughout indexing and retrieval.

## Representative source file

```python
from functools import lru_cache


@lru_cache(maxsize=4)
def load_model(name: str) -> str:
    """Load and cache an embedding model."""
    return name.strip()


class RepositoryIndex:
    """Search indexed repository chunks."""

    def search(self, query: str) -> list[str]:
        """Return source files matching a natural-language query."""
        return [query]
```

## Chunk comparison

| Concern | Baseline line-aware splitter | AST-aware Python splitter |
| --- | --- | --- |
| Primary boundary | Token budget and source lines | Module statements, classes, functions, and methods |
| Decorators | May be separated from their definitions | Included in the owning symbol's source range |
| Function signature | Can appear separately from its body | Preserved with the complete function when it fits |
| Docstrings | Ordinary text with no structural meaning | Retained inside their owning function/class/method |
| Classes | Can split at arbitrary lines | Complete class chunk when it fits the token budget |
| Methods | No explicit parent information | Dedicated method chunk includes enclosing class headers |
| Symbol metadata | File and line range only | Symbol type, qualified name, parent, signature, file, and lines |
| Large symbols | Split by source lines | Split by source lines while preserving structural metadata |
| Invalid Python | Processed as ordinary text | Logged and safely processed by the existing baseline splitter |

For the representative file, AST-aware chunking produces searchable chunks for
the module imports, the complete `load_model` function, the complete
`RepositoryIndex` class, and the specific `RepositoryIndex.search` method.

The class and method intentionally overlap. The complete class supports broad
questions about the class, while the method-specific chunk supports precise
questions about `search` and includes its enclosing class context.

## Metadata carried into Qdrant

Python symbol chunks add the following optional fields to the existing payload:

```json
{
  "symbol_type": "method",
  "symbol_name": "RepositoryIndex.search",
  "parent_symbol": "RepositoryIndex",
  "signature": "def search(self, query: str) -> list[str]:"
}
```

Markdown and plain-text chunks retain `null` structural fields. Previously
indexed Qdrant payloads without these fields remain readable.

## Syntax-error fallback

Files containing invalid Python are not discarded. If `ast.parse` raises
`SyntaxError`, RepoLens logs the file and failing line, then invokes the
existing line-aware splitter. These fallback chunks deliberately omit symbol
metadata because their Python structure cannot be trusted.

## Manual inspection

```bash
python -m repolens.cli inspect-chunks /path/to/RepoLens \
  --chunk-size 512 \
  --overlap 64 \
  --min-chunk-size 30
```

Run the focused tests with:

```bash
python -m pytest tests/test_ast_chunker.py -v
```

Run the regression suite with:

```bash
python -m pytest -v
```
