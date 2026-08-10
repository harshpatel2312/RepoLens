from pathlib import Path

from repolens.cli import main


def test_ingest_command_logs_summary(
    tmp_path: Path,
    capsys,
) -> None:
    (tmp_path / "main.py").write_text(
        "print('RepoLens')\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# RepoLens\n",
        encoding="utf-8",
    )

    exit_code = main([
        "ingest",
        str(tmp_path),
        "--repository-name",
        "sample-repository",
        "--log-level",
        "INFO",
    ])

    captured = capsys.readouterr()
    output = captured.err

    assert exit_code == 0
    assert "Repository ingestion completed." in output
    assert "Repository: sample-repository" in output
    assert "Discovered files: 2" in output
    assert "Ingested files: 2" in output
    assert "Empty files: 0" in output
    assert "Failed files: 0" in output


def test_ingest_command_handles_invalid_repository(
    tmp_path: Path,
    capsys,
) -> None:
    missing_path = tmp_path / "missing"

    exit_code = main([
        "ingest",
        str(missing_path),
    ])

    captured = capsys.readouterr()
    output = captured.err

    assert exit_code == 1
    assert "Error during ingestion:" in output
    assert str(missing_path) in output


def test_inspect_chunks_prints_chunk_metadata(
    tmp_path: Path,
    capsys,
) -> None:
    (tmp_path / "README.md").write_text(
        "# Overview\n\nRepoLens searches repositories.\n\n"
        "# Running tests\n\nRun python -m pytest -v.\n",
        encoding="utf-8",
    )

    exit_code = main([
        "inspect-chunks",
        str(tmp_path),
        "--repository-name",
        "sample-repository",
    ])

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Repository: sample-repository" in output
    assert "Ingested files: 1" in output
    assert "Chunks: 2" in output
    assert "File: README.md" in output
    assert "Heading: Running tests" in output
    assert "Tokens:" in output
    assert "Chunk ID:" in output
    assert "Preview: # Running tests" in output


def test_inspect_chunks_handles_empty_repository(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main([
        "inspect-chunks",
        str(tmp_path),
    ])

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Ingested files: 0" in output
    assert "Chunks: 0" in output
    assert "No chunks were produced." in output


def test_inspect_chunks_respects_chunk_settings(
    tmp_path: Path,
    capsys,
) -> None:
    (tmp_path / "notes.txt").write_text(
        "alpha beta gamma\n"
        "delta epsilon zeta\n"
        "eta theta iota\n",
        encoding="utf-8",
    )

    exit_code = main([
        "inspect-chunks",
        str(tmp_path),
        "--chunk-size",
        "4",
        "--overlap",
        "1",
        "--min-chunk-size",
        "0",
    ])

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Chunks: 3" in output


def test_inspect_chunks_handles_invalid_repository(
    tmp_path: Path,
    capsys,
) -> None:
    missing_path = tmp_path / "missing"

    exit_code = main([
        "inspect-chunks",
        str(missing_path),
    ])

    output = capsys.readouterr().err

    assert exit_code == 1
    assert "Unable to inspect chunks:" in output
    assert str(missing_path) in output


def test_inspect_chunks_output_is_deterministic(
    tmp_path: Path,
    capsys,
) -> None:
    (tmp_path / "b.txt").write_text(
        "second file\n",
        encoding="utf-8",
    )
    (tmp_path / "a.txt").write_text(
        "first file\n",
        encoding="utf-8",
    )

    arguments = [
        "inspect-chunks",
        str(tmp_path),
    ]

    first_exit_code = main(arguments)
    first_output = capsys.readouterr().out

    second_exit_code = main(arguments)
    second_output = capsys.readouterr().out

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert first_output == second_output
    assert first_output.index("File: a.txt") < first_output.index(
        "File: b.txt"
    )