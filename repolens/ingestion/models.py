from dataclasses import dataclass, field

@dataclass(frozen=True)
class FileMetadata:
    repository: str
    relative_path: str
    file_type: str
    size_bytes: int
    content_hash: str

@dataclass(frozen=True)
class IngestedDocument:
    content: str
    metadata: FileMetadata

@dataclass(frozen=True)
class IngestionFailure:
    relative_path: str
    reason: str

@dataclass
class IngestionSummary:
    discovered_files: int = 0
    ingested_files: int = 0
    empty_files: int = 0
    failed_files: int = 0
    failures: list[IngestionFailure] = field(default_factory=list)
    