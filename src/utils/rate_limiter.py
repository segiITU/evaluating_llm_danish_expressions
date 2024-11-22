import time
from functools import wraps

def rate_limit(calls_per_minute: int = 60):
    """Decorator to limit API calls per minute."""
    min_interval = 60.0 / calls_per_minute
    last_call = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get current time
            current_time = time.time()
            
            # Check if we need to wait
            if func.__name__ in last_call:
                elapsed = current_time - last_call[func.__name__]
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            
            # Make the call
            result = func(*args, **kwargs)
            
            # Update last call time
            last_call[func.__name__] = time.time()
            
            return result
        return wrapper
    return decorator 