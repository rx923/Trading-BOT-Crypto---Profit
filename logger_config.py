# logger_config.py

import logging
from logging import LoggerAdapter
from colorama import Fore, Back, Style
import colorama

# Initialize colorama
colorama.init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """
    Formatter to apply colored formatting to log messages based on their level.
    """
    def format(self, record):
        color = {
            logging.INFO: Fore.GREEN,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR: Fore.RED
        }.get(record.levelno, Style.RESET_ALL)

        # Add background color if custom attribute is present
        if getattr(record, 'blue_background', False):
            color = Back.BLUE + Fore.WHITE

        formatted_message = f"{color}{record.levelname} - {record.getMessage()}{Style.RESET_ALL}"
        return formatted_message


class CustomLoggerAdapter(LoggerAdapter):
    """
    Custom LoggerAdapter to add extra context and attributes for log messages.
    """
    def process(self, msg, kwargs):
        # Ensure self.extra is a dictionary or an empty dictionary
        extra = dict(self.extra) if self.extra else {}

        # Update with any additional 'extra' from kwargs
        if 'extra' in kwargs:
            extra.update(kwargs['extra'])

        # Pass the updated extra dict back in kwargs
        kwargs['extra'] = extra

        return msg, kwargs


def get_logger(name: str = __name__, blue_background: bool = False) -> CustomLoggerAdapter:
    """
    Get a custom logger with colored output and optional background color.

    :param name: The name of the logger (usually the module's __name__).
    :param blue_background: Boolean flag to enable blue background for messages.
    :return: CustomLoggerAdapter with colored formatter.
    """
    logger = logging.getLogger(name)

    # Check if logger already has handlers, to avoid adding multiple handlers
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Create stream handler with the colored formatter
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        
        formatter = ColoredFormatter()
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)

    # Create and return a CustomLoggerAdapter
    return CustomLoggerAdapter(logger, {'blue_background': blue_background})
