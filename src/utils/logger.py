"""Utility functions for consistent logging configuration across ReViision.

This module centralises logging setup so that both the main application and
support utilities can share the same behaviour.  It borrows the rotating-file
pattern already used in the Pi-test-bench implementation.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

__all__ = [
    "setup_logging",
    "get_logger",
    "set_log_level",
]


DEFAULT_FORMAT = "% (asctime)s - %(name)s - %(levelname)s - %(message)s"
ROOT_LOGGER_NAME = "reviision"


def _parse_size(size_str: str) -> int:
    """Parse human-friendly sizes (e.g. ``"10MB"``) to bytes.

    Parameters
    ----------
    size_str:
        Input size string – supports the units *B*, *KB*, *MB*, *GB*.

    Returns
    -------
    int
        Size in bytes.  Falls back to ``10 MiB`` if parsing fails.
    """
    size_str = size_str.upper().strip()
    units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}

    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                number = float(size_str[: -len(unit)])
                return int(number * multiplier)
            except ValueError:
                break
    # Default – 10 MiB
    return 10 * 1024 * 1024


def setup_logging(
    level: str | int = "INFO",
    log_file: Optional[str] | Path = None,
    max_size: str = "10MB",
    backup_count: int = 5,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """Configure the root *reviision* logger.

    If *log_file* is given, a rotating file handler is added alongside a console
    handler.  All pre-existing handlers on the root *reviision* logger are
    replaced so repeated calls are harmless.
    """
    numeric_level = (
        level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
    )
    fmt = fmt or DEFAULT_FORMAT
    formatter = logging.Formatter(fmt)

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(numeric_level)

    # clear existing
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=_parse_size(max_size), backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the *reviision* namespace."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def set_log_level(level: str | int) -> None:
    """Change the level for all configured handlers uniformly."""
    numeric_level = (
        level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
    )
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    root_logger.setLevel(numeric_level)
    for h in root_logger.handlers:
        h.setLevel(numeric_level) 