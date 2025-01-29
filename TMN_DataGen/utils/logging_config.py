# TMN_DataGen/utils/logging_config.py
import logging
import sys

def setup_logger(name: str, verbosity: str = 'normal') -> logging.Logger:
    """
    Set up logger with consistent format and verbosity levels.
    
    Args:
        name: Logger name
        verbosity: 'quiet', 'normal', or 'debug'
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Only configure if not already set up
        logger.propagate = False
        
        handler = logging.StreamHandler(sys.stdout)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Simplified format
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Set levels based on verbosity
        level_map = {
            'quiet': logging.WARNING,
            'normal': logging.INFO,
            'debug': logging.DEBUG
        }
        level = level_map.get(verbosity, logging.INFO)
        
        logger.setLevel(level)
        handler.setLevel(level)
        
        logger.addHandler(handler)
        logger.debug(f"Created logger '{name}' with level {logging.getLevelName(level)}")
    
    return logger

