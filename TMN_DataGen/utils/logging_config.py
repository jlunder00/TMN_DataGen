# TMN_DataGen/TMN_DataGen/utils/logging_config.py
import logging
import sys
from typing import Optional

def get_logger(name: str, verbose: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with appropriate level based on verbosity setting.
    
    Args:
        name: Logger name
        verbose: Verbosity level - None/"normal"/"debug"
    
    Example:
        >>> logger = get_logger("MyParser", "debug")
        >>> logger.debug("Detailed parsing info...")  # Will print
        >>> 
        >>> logger = get_logger("MyParser", "normal")
        >>> logger.debug("Detailed parsing info...")  # Won't print
        >>> logger.info("Processing complete")  # Will print
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist
    if not logger.handlers:
        logger.propagate = False  # Don't propagate to root logger
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Set levels based on verbosity
        if verbose == 'debug':
            logger.setLevel(logging.DEBUG)
            console_handler.setLevel(logging.DEBUG)
        elif verbose == 'normal':
            logger.setLevel(logging.INFO)
            console_handler.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
            console_handler.setLevel(logging.WARNING)
        
        logger.addHandler(console_handler)
    
    return logger

# Create root logger
logger = get_logger("TMN_DataGen")
