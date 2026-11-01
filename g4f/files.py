from __future__ import annotations

import re
import os
from urllib.parse import unquote

from .cookies import get_cookies_dir


# Device names that Windows reserves regardless of extension (e.g. "CON.txt").
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(10)}
    | {f"LPT{i}" for i in range(10)}
)


def secure_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize a filename for safe filesystem storage."""
    if filename is None:
        return None

    # Keep letters, numbers, basic punctuation, underscores
    filename = re.sub(r"[^\w.,_+\-]+", "_", unquote(filename).strip(), flags=re.UNICODE)
    encoding = "utf-8"
    encoded = filename.encode(encoding)
    if len(encoded) > max_length:
        encoded = encoded[:max_length-4] + encoded[-4:]
    decoded = encoded.decode(encoding, "ignore")
    decoded = decoded.strip(".,_+-")
    # Prefix reserved device names so they cannot refer to a Windows device.
    if decoded.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        decoded = f"_{decoded}"
    return decoded


def get_bucket_dir(*parts: str) -> str:
    """Return a path under the cookies 'buckets' directory with sanitized parts."""
    return os.path.join(
        get_cookies_dir(), "buckets", *[secure_filename(part) for part in parts if part]
    )
