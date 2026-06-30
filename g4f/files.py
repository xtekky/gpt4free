from __future__ import annotations

import re
import os
from urllib.parse import unquote

from .cookies import get_cookies_dir


def secure_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize a filename for safe filesystem storage."""
    if filename is None:
        return None

    # Keep letters, numbers, basic punctuation, underscores
    filename = re.sub(
        r"[^\w.,_+\-]+",
        "_",
        unquote(filename).strip(),
        flags=re.UNICODE
    )
    encoding = "utf-8"
    encoded = filename.encode(encoding)[:max_length]
    decoded = encoded.decode(encoding, "ignore")
    return decoded.strip(".,_+-")


def get_bucket_dir(*parts: str) -> str:
    """Return a path under the cookies 'buckets' directory with sanitized parts."""
    return os.path.join(
        get_cookies_dir(),
        "buckets",
        *[secure_filename(part) for part in parts if part]
    )