"""Function for adding and deleting of logger."""
import logging


def add_logger(logging_file_path: str):
    """Set up logging through the project."""
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : %(levelname)s : %(message)s",
        handlers=[logging.FileHandler(logging_file_path, mode="w"), stream_handler],
    )
    return logging.getLogger()


def close_logger(logger):
    """Close existing logger."""
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
