import logging
import threading
from datetime import datetime
import os

# Module-level variable: set before importing this module to use a custom log name.
# When set, the log file will be named "<LOG_NAME>.log" instead of date-based.
LOG_NAME = None

def get_filename():
    log_dir = './logs'  # Directory where logs are stored
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create log directory if it doesn't exist

    if LOG_NAME:
        return f"{LOG_NAME}.log"

    date_prefix = datetime.now().strftime("%m-%d")  # Today's date in MM-DD format

    # List all .log files that start with today's date
    files = [f for f in os.listdir(log_dir) if f.startswith(date_prefix) and f.endswith('.log')]
    indices = [int(f.split('-')[-1].split('.')[0]) for f in files]  # Extract indices from filenames
    next_index = max(indices) + 1 if indices else 1  # Determine the next index

    return f"{date_prefix}-{next_index:02d}.log"  # Return filename with next index

def setup_logging(log_file='default.log'):
    log_filename =  get_filename()
    # Create a logger object
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Set the minimum level of messages to log

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(f'./logs/{log_filename}')
    fh.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# Lazy-initialized logger: created on first access so that LOG_NAME
# can be set before the log file is opened.
_logger = None
_logger_lock = threading.Lock()

def get_logger():
    global _logger
    if _logger is None:
        with _logger_lock:
            if _logger is None:
                _logger = setup_logging()
    return _logger


class _LazyLogger:
    """Proxy that defers setup_logging() until the first logging call."""
    def __getattr__(self, name):
        return getattr(get_logger(), name)

logger = _LazyLogger()
