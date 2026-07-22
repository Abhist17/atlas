"""Project-wide logger. Logs to console + rotating file in logs/."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config.settings import LOG_DIR

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(console)

    fileh = RotatingFileHandler(
        LOG_DIR / "atlas.log", maxBytes=5_000_000, backupCount=3
    )
    fileh.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(fileh)

    logger.propagate = False
    return logger
