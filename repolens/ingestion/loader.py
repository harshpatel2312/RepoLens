import fnmatch
import logging
import os
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# Load config file
config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

def is_generated_file(file_path: Path) -> bool:
    """
    Check if the given file is a generated file based on its name.

    Args:
        file_path (Path): The path to the file.

    Returns:
        bool: True if the file is generated, False otherwise.
    """
    filename = file_path.name.lower()

    return any(fnmatch.fnmatch(file_path.name, pattern.lower()) for pattern in config["INGESTION"]["GENERATED_FILE_PATTERNS"])

def discover_files(
    repository_path: Path,
    max_file_size: int = int(config["INGESTION"]["MAX_FILE_SIZE"]),
) -> list[Path]:
    """Discover supported repository files in deterministic order."""
    
    repository_path = repository_path.resolve()

    if not repository_path.exists():
        raise FileNotFoundError(f"Repository does not exist: '{repository_path}'")

    if not repository_path.is_dir():
        raise NotADirectoryError(f"Repository is not a directory: '{repository_path}'")
    
    discovered_files: list[Path] = []
    
    for current_directory, directory_names, file_names in os.walk(
        repository_path, 
        topdown=True, 
        followlinks=False
     ):
        current_path = Path(current_directory)

        # Filter ignored directories
        directory_names[:] = sorted(
            directory_name
            for directory_name in directory_names
            if directory_name not in config["INGESTION"]["IGNORE_DIRECTORIES"]
            and not (current_path / directory_name).is_symlink()
        )

        for file_name in sorted(file_names):
            file_path = current_path / file_name
            relative_path = file_path.relative_to(repository_path).as_posix()

            # Do not follow file symlinks
            if file_path.is_symlink():
                logger.debug(f"Skipping symlink: {relative_path}")
                continue

            # Only [".py", ".txt", ".md"] files are supported
            if file_path.suffix.lower() not in config["INGESTION"]["EXTENSIONS"]:
                logger.debug(f"Skipping unsupported file type: {relative_path}")
                continue

            # Skip generated files
            if is_generated_file(file_path):
                logger.debug(f"Skipping generated file: {relative_path}")
                continue

            try:
                file_size = file_path.stat().st_size
            except OSError as error:
                logger.warning(f"Cannot inspect file {relative_path}: {error}")
                continue

            if file_size > max_file_size:
                logger.info(f"Skipping oversized file: {relative_path} ({file_size} bytes)")
                continue

            discovered_files.append(file_path)
    
    return sorted(
        discovered_files, 
        key=lambda path: path.relative_to(repository_path).as_posix(),
    )