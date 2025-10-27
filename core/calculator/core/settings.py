"""Configuration settings shared by the calculator components."""

from __future__ import annotations

from typing import Any, Dict


log_format_: str = "%(asctime)s - [%(levelname)s] - %(name)s - %(filename)s.%(funcName)s:%(lineno)d - %(message)s"


class Settings:
    """Central repository for logging configuration."""

    LOGGING_CONFIG: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default_formatter": {"format": log_format_},
        },
        "handlers": {
            "file_handler": {
                "level": "DEBUG",
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default_formatter",
                "filename": "calculator.log",
                "backupCount": 10,
                "maxBytes": 5000000,
                "encoding": "utf8",
            }
        },
        "loggers": {
            "core": {"handlers": ["file_handler"], "level": "DEBUG", "propagate": True}
        },
    }
