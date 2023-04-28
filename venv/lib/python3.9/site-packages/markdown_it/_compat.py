from __future__ import annotations

from collections.abc import Mapping
import sys
from typing import Any

if sys.version_info >= (3, 10):
    DATACLASS_KWARGS: Mapping[str, Any] = {"slots": True}
else:
    DATACLASS_KWARGS: Mapping[str, Any] = {}
