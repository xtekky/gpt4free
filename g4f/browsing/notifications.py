"""Small in-process notification store for browsing/trigger events.

Clients poll :meth:`NotificationStore.list` or use the HTTP endpoints.  The store
is intentionally dependency-free and bounded so a long-running trigger cannot
grow memory without limit.  Chat clients also use the browser Notification API
for completed assistant messages; these server notifications cover events that
arrive while no chat request is open (feeds, triggers, scheduled browsing).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, List, Optional

MAX_NOTIFICATIONS = 200


class NotificationStore:
    def __init__(self, max_items: int = MAX_NOTIFICATIONS):
        self._items: List[Dict] = []
        self._lock = threading.RLock()
        self.max_items = max_items

    def add(
        self,
        title: str,
        body: str = "",
        kind: str = "info",
        source: Optional[str] = None,
        url: Optional[str] = None,
        trigger_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        item = {
            "id": uuid.uuid4().hex[:16],
            "title": title or "Notification",
            "body": body or "",
            "kind": kind,
            "source": source or "",
            "url": url or "",
            "trigger_id": trigger_id or "",
            "metadata": metadata or {},
            "created": time.time() * 1000,
            "read": False,
        }
        with self._lock:
            self._items.insert(0, item)
            del self._items[self.max_items :]
        return dict(item)

    def list(self, unread_only: bool = False, limit: int = 50) -> List[Dict]:
        with self._lock:
            items = [item for item in self._items if not unread_only or not item["read"]]
            return [dict(item) for item in items[: max(1, limit)]]

    def unread_count(self) -> int:
        with self._lock:
            return sum(1 for item in self._items if not item["read"])

    def mark_read(self, notification_id: Optional[str] = None) -> int:
        with self._lock:
            changed = 0
            for item in self._items:
                if notification_id is None or item["id"] == notification_id:
                    if not item["read"]:
                        item["read"] = True
                        changed += 1
            return changed

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


notifications = NotificationStore()
