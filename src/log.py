import logging
import os


def setup() -> None:
    level = logging.getLevelName(os.getenv("LOG_LEVEL", "INFO").upper())
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    logging.basicConfig(level=level, format=fmt)
    for noisy in ("httpx", "httpcore", "telegram.ext.ExtBot"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    aiosqlite_level = logging.getLevelName(os.getenv("AIOSQLITE_LOG_LEVEL", "WARNING").upper())
    logging.getLogger("aiosqlite").setLevel(aiosqlite_level)
    tg_app_level = logging.getLevelName(os.getenv("TELEGRAM_APP_LOG_LEVEL", "WARNING").upper())
    logging.getLogger("telegram.ext.Application").setLevel(tg_app_level)
    tg_updater_level = logging.getLevelName(os.getenv("TELEGRAM_UPDATER_LOG_LEVEL", "WARNING").upper())
    logging.getLogger("telegram.ext.Updater").setLevel(tg_updater_level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
