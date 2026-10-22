import logging
import sys
from typing import Any

from app.core.config import settings


def setup_logger(name: str = "policylens") -> logging.Logger:
    """Set up and return a structured logger for the application."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return logger


logger = setup_logger()


def log_event(event_name: str, **kwargs: Any) -> None:
    """Helper to log structured key-value events."""
    payload = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"event={event_name} {payload}".strip())
