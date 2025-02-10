from transformer.model import Transformer
# from transformer.test import test
from transformer.train import train
# from transformer.inference import inference
# import alexnet.utils as utils
# import logging
# from rich.logging import RichHandler
import sys

# Check if the code is running in a Jupyter notebook
def is_notebook():
    try:
        # Check if IPython is installed and if we are in a Jupyter environment
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

# Set up the logging configuration
def setup_logging():

    # Get the root logger
    logger = logging.getLogger()

    # Remove all existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler for logging to a file
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)

    # Create a formatter for the file handler
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Conditionally choose a console handler based on whether we are in a notebook
    if is_notebook():
        # Use a simple StreamHandler for notebook environments
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        # Use RichHandler for terminals (assuming RichHandler is imported and available)
        from rich.logging import RichHandler
        console_handler = RichHandler()

    # Set the logging format for console output (simplified for clarity)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,  # Set logging level
        handlers=[console_handler, file_handler]  # Log to both console and file
    )

__all__ = ['train', 'Transformer']
