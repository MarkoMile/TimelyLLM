import logging
from datetime import datetime
import os

def get_filename():
    date_prefix = datetime.now().strftime("%m-%d")  # Today's date in MM-DD format
    log_dir = './logs'  # Directory where logs are stored
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create log directory if it doesn't exist

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

    # Add the handlers to the logger (avoid duplicates if setup is called multiple times)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

# Initialize logger
logger = setup_logging()
