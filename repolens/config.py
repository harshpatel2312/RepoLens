import logging
import yaml
from pathlib import Path

# Load config file once at import time; other modules just import `config`.
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

with open(_CONFIG_PATH, "r") as _config_file:
    config = yaml.safe_load(_config_file)

# Shared package logger
logger = logging.getLogger("repolens")
