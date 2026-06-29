"""
Centralized logging factory.

Usage (in any module):
    from surveillance.core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Detector initialized")
    logger.warning("Low confidence: %.3f", score)
"""

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from surveillance.core.constants import LOGS_DIR

_CONFIGURED: bool = False


def configure_logging(
    log_level: int = logging.DEBUG,
    log_file: Path | None = None,
    enable_rich: bool = True,
) -> None:
    """
    Configure the root logger. Call once at application startup.

    Args:
        log_level:   Minimum level to emit.
        log_file:    If provided, also write logs to this file.
        enable_rich: Use Rich's colored console handler.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if enable_rich:
        console_handler = RichHandler(
            level=log_level,
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "requests", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Auto-configures root logger on first call.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    if not _CONFIGURED:
        configure_logging(log_file=LOGS_DIR / "surveillance.log")
    return logging.getLogger(name)
