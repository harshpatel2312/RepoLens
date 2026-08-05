from pathlib import Path
import pytest

from repolens.ingestion.loader import (
    discover_files,
    ingest_repository,
    load_file,
)


def test_discovers_supported_files_in_sorted_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text(
        "print('hello')",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Example",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text(
        "Some notes",
        encoding="utf-8",
    )
    (tmp_path / "data.csv").write_text(
        "name,value",
        encoding="utf-8",
    )

    discovered = discover_files(tmp_path)

    relative_paths = [
        path.relative_to(tmp_path).as_posix()
        for path in discovered
    ]

    assert relative_paths == [
        "README.md",
        "main.py",
        "notes.txt",
    ]


def test_ignored_directories_are_not_scanned(
    tmp_path: Path,
) -> None:
    git_directory = tmp_path / ".git"
    git_directory.mkdir()
    (git_directory / "hidden.py").write_text(
        "secret = True",
        encoding="utf-8",
    )

    venv_directory = tmp_path / ".venv"
    venv_directory.mkdir()
    (venv_directory / "dependency.py").write_text(
        "dependency = True",
        encoding="utf-8",
    )

    (tmp_path / "visible.py").write_text(
        "visible = True",
        encoding="utf-8",
    )

    discovered = discover_files(tmp_path)

    relative_paths = [
        path.relative_to(tmp_path).as_posix()
        for path in discovered
    ]

    assert relative_paths == ["visible.py"]


def test_missing_repository_raises_error(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        discover_files(missing_path)


def test_load_file_returns_content_and_metadata(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "example.py"
    file_path.write_text(
        "print('RepoLens')\n",
        encoding="utf-8",
    )

    document = load_file(file_path, tmp_path)

    assert document is not None
    assert document.content == "print('RepoLens')\n"
    assert document.metadata.repository == tmp_path.name
    assert document.metadata.relative_path == "example.py"
    assert document.metadata.file_type == "python"
    assert document.metadata.size_bytes > 0
    assert len(document.metadata.content_hash) == 64


def test_load_file_normalizes_newlines(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_bytes(b"First line\r\nSecond line\r\n")

    document = load_file(file_path, tmp_path)

    assert document is not None
    assert document.content == "First line\nSecond line\n"


def test_load_file_skips_empty_content(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n", encoding="utf-8")

    document = load_file(file_path, tmp_path)

    assert document is None


def test_load_file_rejects_binary_content(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "invalid.txt"
    file_path.write_bytes(b"text\x00binary")

    with pytest.raises(ValueError, match="Binary content detected"):
        load_file(file_path, tmp_path)


def test_ingest_repository_returns_documents_and_summary(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text(
        "print('RepoLens')\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# RepoLens\n",
        encoding="utf-8",
    )
    (tmp_path / "data.csv").write_text(
        "name,value\n",
        encoding="utf-8",
    )

    documents, summary = ingest_repository(
        tmp_path,
        repository_name="example-repository",
    )

    relative_paths = [
        document.metadata.relative_path
        for document in documents
    ]

    assert relative_paths == [
        "README.md",
        "main.py",
    ]

    assert summary.discovered_files == 2
    assert summary.ingested_files == 2
    assert summary.empty_files == 0
    assert summary.failed_files == 0
    assert summary.failures == []

    assert all(
        document.metadata.repository == "example-repository"
        for document in documents
    )


def test_ingest_repository_records_empty_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "valid.txt").write_text(
        "Valid content\n",
        encoding="utf-8",
    )
    (tmp_path / "empty.txt").write_text(
        "   \n",
        encoding="utf-8",
    )

    documents, summary = ingest_repository(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata.relative_path == "valid.txt"

    assert summary.discovered_files == 2
    assert summary.ingested_files == 1
    assert summary.empty_files == 1
    assert summary.failed_files == 0


def test_invalid_file_does_not_stop_repository_ingestion(
    tmp_path: Path,
) -> None:
    (tmp_path / "valid.txt").write_text(
        "Valid content\n",
        encoding="utf-8",
    )

    (tmp_path / "invalid.txt").write_bytes(
        b"text\x00binary"
    )

    documents, summary = ingest_repository(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata.relative_path == "valid.txt"

    assert summary.discovered_files == 2
    assert summary.ingested_files == 1
    assert summary.empty_files == 0
    assert summary.failed_files == 1

    assert len(summary.failures) == 1
    assert summary.failures[0].relative_path == "invalid.txt"
    assert "Binary content detected" in summary.failures[0].reason