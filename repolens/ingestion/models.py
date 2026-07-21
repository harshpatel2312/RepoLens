from dataclasses import dataclass, field

@dataclass(frozen=True)
class Filemetadata:
    repository: str
    relative_path: str
    file_type: str
    size_bytes: int
    content_hash: str

@dataclass(frozen=True)
class IngestedDocument:
    content: str
    metadata: Filemetadata

@dataclass(frozen=True)
class IngestionFailure:
    relative_path: str
    reason: str

@dataclass
class IngestionSummary:
    files_visited: int = 0
    supported_files: int = 0
    files_ingested: int = 0
    files_ignored: int = 0
    files_unsupported: int = 0
    files_oversized: int = 0
    files_empty: int = 0
    files_failed: int = 0
    failures: list[IngestionFailure] = field(default_factory=list)
    