import sys
from typing import Callable, List, Optional, Any

logging: bool = False
version_check: bool = True
version: Optional[str] = None
log_handler: Callable = print  # More specifically: Callable[[Any, Optional[Any]], None]
logs: List[str] = []

def log(*text: Any, file: Optional[Any] = None) -> None:
    """Log a message if logging is enabled."""
    if logging:
        log_handler(*text, file=file)

def error(*error: Any, name: Optional[str] = None) -> None:
    """Log an error message to stderr."""
    error = [e if isinstance(e, str) else f"{type(e).__name__ if name is None else name}: {e}" for e in error]
    log(*error, file=sys.stderr)
