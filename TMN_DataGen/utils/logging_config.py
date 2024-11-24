# TMN_DataGen/TMN_DataGen/utils/logging_config.py
import logging
import sys

def setup_logging(level=logging.INFO):
    # Create root logger
    root_logger = logging.getLogger("TMN_DataGen")
    if not root_logger.handlers:  # Only setup if not already configured
        root_logger.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(message)s')  # Simple format for tree visualization
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        root_logger.addHandler(console_handler)
    
    return root_logger

# Create logger instance
logger = setup_logging()
