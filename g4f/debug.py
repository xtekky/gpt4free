import sys
from typing import Callable, List, Optional, Any

logging: bool = False
version_check: bool = True
version: Optional[str] = None
log_handler: Callable[..., None] = print
logs: List[str] = []


def enable_logging(handler: Callable[..., None] = print) -> None:
    """Enable debug logging with optional handler."""
    global logging, log_handler
    logging = True
    log_handler = handler


def disable_logging() -> None:
    """Disable debug logging."""
    global logging
    logging = False


def log(*text: Any, file: Optional[Any] = None) -> None:
    """Log a message if logging is enabled."""
    if logging:
        message = " ".join(map(str, text))
        logs.append(message)
        log_handler(*text, file=file)


def error(*error_args: Any, name: Optional[str] = None) -> None:
    """Log an error message to stderr."""
    formatted_errors = [
        e if isinstance(e, str) else f"{name or type(e).__name__}: {e}"
        for e in error_args
    ]
    log(*formatted_errors, file=sys.stderr)