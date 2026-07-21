from pathlib import Path

import pytest

from repolens.ingestion.loader import discover_files


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