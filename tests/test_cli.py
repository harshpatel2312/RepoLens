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