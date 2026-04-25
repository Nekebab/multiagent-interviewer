"""Logging configuration using Loguru."""

import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """Configure the global Loguru logger."""
    logger.remove()  # remove the default handler
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,  # show full traceback on exceptions
        diagnose=True,  # show variable values in tracebacks (turn off in prod)
    )
