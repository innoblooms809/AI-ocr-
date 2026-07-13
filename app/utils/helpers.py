import time
from functools import wraps
from app.utils.logger import setup_logger

logger = setup_logger("helpers")

def time_it(func):
    """Decorator to log execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"Function '{func.__name__}' took {elapsed:.4f} seconds to execute.")
        return result
    return wrapper
