# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# TMN_DataGen/utils/gpu_scheduler.py
import time
import logging
import threading

class GPUScheduler:
    """
    Coordinates GPU access across multiple processes using a semaphore.
    Designed to be used as a context manager with the 'with' statement.
    """
    
    def __init__(self, max_concurrent=4, timeout=None, logger=None):
        """
        Initialize the GPU scheduler.
        
        Args:
            max_concurrent: Maximum number of processes that can use the GPU simultaneously
            timeout: Maximum seconds to wait for GPU access (None = wait indefinitely)
            logger: Optional logger instance
        """
        self.semaphore = threading.Semaphore(max_concurrent)
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
    
    def __enter__(self):
        """Acquire the semaphore before GPU operations."""
        start_time = time.time()
        acquired = False
        
        while not acquired:
            acquired = self.semaphore.acquire(block=False, timeout=0.1)
            if acquired:
                self.logger.debug("GPU access acquired")
                return self
            
            if self.timeout and (time.time() - start_time > self.timeout):
                raise TimeoutError("Timed out waiting for GPU access")
            
            time.sleep(0.1)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the semaphore after GPU operations."""
        self.semaphore.release()
        self.logger.debug("GPU access released")
