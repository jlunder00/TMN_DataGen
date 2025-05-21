# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# TMN_DataGen/utils/gpu_coordinator.py
import threading
import time
import logging

class GPUCoordinator:
    """
    Thread-safe coordinator for GPU access.
    Designed to be used as a context manager with the 'with' statement.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, max_concurrent=4, logger=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GPUCoordinator, cls).__new__(cls)
                cls._instance._initialize(max_concurrent, logger)
            return cls._instance
    
    def _initialize(self, max_concurrent, logger):
        self.max_concurrent = max_concurrent
        self.logger = logger or logging.getLogger(__name__)
        self.semaphore = threading.Semaphore(max_concurrent)
        self.active_users = 0  # For logging only
    
    def __enter__(self):
        """Acquire the semaphore before GPU operations."""
        self.semaphore.acquire()
        with self._lock:
            self.active_users += 1
            self.logger.debug(f"GPU access acquired. Active users: {self.active_users}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the semaphore after GPU operations."""
        self.semaphore.release()
        with self._lock:
            self.active_users -= 1
            self.logger.debug(f"GPU access released. Active users: {self.active_users}")


# # Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# # TMN_DataGen/utils/gpu_coordinator.py
# import os
# import time
# import logging
# import tempfile
# import atexit
# import json
# from pathlib import Path
# import fcntl  # For file locking

# class GPUCoordinator:
#     """
#     Coordinates GPU access across multiple processes using file-based locking.
#     """
#     _instance = None
#     
#     def __new__(cls, max_concurrent=4, logger=None):
#         if cls._instance is None:
#             cls._instance = super(GPUCoordinator, cls).__new__(cls)
#             cls._instance._initialize(max_concurrent, logger)
#         return cls._instance
#     
#     def _initialize(self, max_concurrent, logger):
#         self.max_concurrent = max_concurrent
#         self.logger = logger or logging.getLogger(__name__)
#         
#         # Create a temporary file to track GPU usage
#         self.lock_dir = Path(tempfile.gettempdir()) / "gpu_coordinator"
#         self.lock_dir.mkdir(exist_ok=True)
#         self.state_file = self.lock_dir / "gpu_state.json"
#         
#         # Initialize the state file if it doesn't exist
#         if not self.state_file.exists():
#             with open(self.state_file, 'w') as f:
#                 json.dump({"active_users": 0}, f)
#         
#         # Register cleanup
#         atexit.register(self._cleanup)
#     
#     def _cleanup(self):
#         """Remove temporary files on exit."""
#         try:
#             if self.state_file.exists():
#                 self.state_file.unlink()
#             if self.lock_dir.exists():
#                 self.lock_dir.rmdir()
#         except:
#             pass
#     
#     def acquire_gpu(self, timeout=60):
#         """
#         Acquire GPU access with timeout. Returns True if successful.
#         
#         This method is safe to call from any process.
#         """
#         start_time = time.time()
#         
#         while True:
#             try:
#                 # Use file locking to ensure atomic read/write
#                 with open(self.state_file, 'r+') as f:
#                     # Get exclusive lock
#                     fcntl.flock(f, fcntl.LOCK_EX)
#                     
#                     # Read current state
#                     state = json.load(f)
#                     
#                     # Check if we can acquire
#                     if state["active_users"] < self.max_concurrent:
#                         # Acquire and update state
#                         state["active_users"] += 1
#                         
#                         # Write back the updated state
#                         f.seek(0)
#                         f.truncate()
#                         json.dump(state, f)
#                         
#                         self.logger.debug(f"GPU access acquired. Active users: {state['active_users']}")
#                         return True
#                     
#                     # Release lock
#                     fcntl.flock(f, fcntl.LOCK_UN)
#             
#             except Exception as e:
#                 self.logger.warning(f"Error acquiring GPU lock: {e}")
#             
#             # Check timeout
#             if timeout and (time.time() - start_time > timeout):
#                 self.logger.warning("Timed out waiting for GPU access")
#                 return False
#             
#             # Sleep before retrying
#             time.sleep(0.5)
#     
#     def release_gpu(self):
#         """
#         Release GPU access.
#         
#         This method is safe to call from any process.
#         """
#         try:
#             # Use file locking to ensure atomic read/write
#             with open(self.state_file, 'r+') as f:
#                 # Get exclusive lock
#                 fcntl.flock(f, fcntl.LOCK_EX)
#                 
#                 # Read current state
#                 state = json.load(f)
#                 
#                 # Release and update state
#                 state["active_users"] = max(0, state["active_users"] - 1)
#                 
#                 # Write back the updated state
#                 f.seek(0)
#                 f.truncate()
#                 json.dump(state, f)
#                 
#                 self.logger.debug(f"GPU access released. Active users: {state['active_users']}")
#                 
#                 # Release lock
#                 fcntl.flock(f, fcntl.LOCK_UN)
#         
#         except Exception as e:
#             self.logger.warning(f"Error releasing GPU lock: {e}")
