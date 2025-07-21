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
    error_messages = []
    for e in error:
        if isinstance(e, str):
            error_messages.append(e)
        elif isinstance(e, Exception):
            error_class = type(e).__name__ if name is None else name
            error_messages.append(f"{error_class}: {e}")
        else:
            error_messages.append(str(e))
    log(*error_messages, file=sys.stderr)
