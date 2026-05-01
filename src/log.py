import logging
import os


def setup() -> None:
    level = logging.getLevelName(os.getenv("LOG_LEVEL", "INFO").upper())
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    logging.basicConfig(level=level, format=fmt)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
