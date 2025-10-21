log_format_ = "%(asctime)s - [%(levelname)s] - %(name)s - %(filename)s.%(funcName)s:%(lineno)d - %(message)s"


class Settings:
    LOGGING_CONFIG = {
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
