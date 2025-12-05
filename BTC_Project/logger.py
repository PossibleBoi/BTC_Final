import logging
import os
from datetime import datetime


LOG_DIR = "logs"
LOG_FILE = f"app_{datetime.now().strftime('%Y_%m_%d')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Create log directory if not exists
os.makedirs(LOG_DIR, exist_ok=True)


logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(funcName)s - Line %(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with custom formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # If handlers already exist, avoid duplicating them
    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_PATH)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(funcName)s - Line %(lineno)d: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
