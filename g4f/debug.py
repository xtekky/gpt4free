import sys
from typing import Callable, List, Optional, Any

# Warning: name could conflict with Python's built-in logging module
logging: bool = False
version_check: bool = True
version: Optional[str] = None
log_handler: Callable = print  # More specifically: Callable[[Any, Optional[Any]], None]
logs: List[str] = []

def log(text: Any, file: Optional[Any] = None) -> None:
    """Log a message if logging is enabled."""
    if logging:
        log_handler(text, file=file)

def error(error: Any, name: Optional[str] = None) -> None:
    """Log an error message to stderr."""
    log(
        error if isinstance(error, str) else f"{type(error).__name__ if name is None else name}: {error}",
        file=sys.stderr
    )
