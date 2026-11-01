"""Prompt-created browsing triggers.

A trigger watches a source (RSS feed, web page or search query), remembers what
it has already seen, and emits notifications for new matching items.  The
notification carries the trigger's ``action_prompt`` so a chat client can hand
it straight to an LLM ("summarise the new items", "open a chat", ...).

This is deliberately a small, durable engine rather than a scheduler: the HTTP
layer can run triggers on demand (and a future worker can call the same
function periodically).  State is stored as JSON in the g4f cache directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

from ..config import get_cache_dir
from .browse import fast_browse, search
from .feeds import fetch_feed
from .notifications import notifications

_STORE_LOCK = threading.RLock()
_MAX_SEEN = 500
_URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
_INTERVAL_RE = re.compile(
    r"\bevery\s+(\d+)\s*(second|minute|hour|day|week)s?\b", re.IGNORECASE
)
_QUOTED_RE = re.compile(r"[\"“']([^\"”']{2,80})[\"”']")
_ABOUT_RE = re.compile(
    r"\b(?:about|mentioning|containing|matching|with|for)\s+(.+?)(?:[.!?]|$)",
    re.IGNORECASE,
)
_INTERVALS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800}


def _store_path() -> Path:
    override = os.environ.get("G4F_BROWSING_STORE")
    if override:
        return Path(override)
    root = Path(get_cache_dir()) / "browsing"
    root.mkdir(parents=True, exist_ok=True)
    return root / "triggers.json"


def _load() -> List[Dict]:
    path = _store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(data, dict):
        data = data.get("triggers", [])
    return data if isinstance(data, list) else []


def _save(triggers: List[Dict]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"triggers": triggers}, indent=2), encoding="utf-8")
    tmp.replace(path)


def _normalise_source(source: Optional[Dict], prompt: str) -> Dict:
    source = dict(source or {})
    if not source.get("url"):
        match = _URL_RE.search(prompt or "")
        if match:
            source["url"] = match.group(0).rstrip(".,;")
    url = str(source.get("url") or "").strip()
    source_type = (source.get("type") or "").lower()
    if not source_type:
        if url:
            path = urlparse(url).path.lower()
            source_type = "rss" if any(
                token in path for token in ("rss", "feed", "atom", ".xml")
            ) else "web"
        elif source.get("query"):
            source_type = "search"
        else:
            source_type = "search"
    source["type"] = source_type
    if url:
        source["url"] = url
    if source.get("query"):
        source["query"] = str(source["query"]).strip()
    return source


def _extract_keywords(prompt: str) -> List[str]:
    quoted = [value.strip() for value in _QUOTED_RE.findall(prompt or "") if value.strip()]
    if quoted:
        return quoted[:8]
    match = _ABOUT_RE.search(prompt or "")
    if match:
        phrase = match.group(1).strip(" ,.;:-")
        phrase = re.split(r"\b(?:every|each|and|then|until)\b", phrase, maxsplit=1)[0].strip(" ,.;:-")
        # Keep it short; the full phrase is still used as the search query.
        if phrase and len(phrase) <= 80:
            return [phrase]
    return []


def _extract_interval(prompt: str) -> int:
    lower = (prompt or "").lower()
    match = _INTERVAL_RE.search(lower)
    if match:
        return max(60, int(match.group(1)) * _INTERVALS[match.group(2).lower()])
    if "hourly" in lower:
        return 3600
    if "daily" in lower or "every day" in lower:
        return 86400
    if "weekly" in lower:
        return 604800
    return 900


def parse_trigger_prompt(prompt: str, source: Optional[Dict] = None) -> Dict:
    """Turn a natural-language request into a trigger definition.

    This is intentionally heuristic and side-effect free: the caller can show
    the result for review before creating the trigger.  An LLM can refine the
    same object later.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("A prompt is required")
    source = _normalise_source(source, prompt)
    keywords = _extract_keywords(prompt)
    interval = _extract_interval(prompt)
    query = source.get("query") or (keywords[0] if keywords else prompt)
    if source["type"] == "search":
        source["query"] = query

    name = re.sub(
        r"^(please\s+)?(when|whenever|if|notify me when|tell me when|watch)\s+",
        "",
        prompt,
        flags=re.IGNORECASE,
    ).strip()
    name = name[:80].rstrip(" .!?") or "Browsing trigger"
    action_type = "prompt"
    action_prompt = prompt
    if re.search(r"\b(summar|digest|brief|roundup)\w*", prompt, re.IGNORECASE):
        action_type = "summary"
        action_prompt = (
            "Summarise the new items reported by this trigger and cite their links. "
            f"Original request: {prompt}"
        )
    elif re.search(r"\b(open|start)\s+(a\s+)?chat\b", prompt, re.IGNORECASE):
        action_type = "chat"

    return {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "prompt": prompt,
        "source": source,
        "condition": {
            "keywords": keywords,
            "match": "any",
        },
        "action": {
            "type": action_type,
            "prompt": action_prompt,
        },
        "interval": interval,
        "enabled": True,
        "created": time.time() * 1000,
        "updated": time.time() * 1000,
        "last_run": None,
        "last_error": "",
        "seen": [],
    }


def _sanitise_trigger(data: Dict) -> Dict:
    if not isinstance(data, dict):
        raise ValueError("Trigger must be an object")
    prompt = str(data.get("prompt") or data.get("name") or "").strip()
    if not prompt:
        raise ValueError("A trigger prompt is required")
    base = parse_trigger_prompt(prompt, data.get("source"))
    base.update({key: data[key] for key in ("id", "name", "interval", "enabled") if key in data})
    if data.get("condition"):
        base["condition"].update(data["condition"])
    if data.get("action"):
        base["action"].update(data["action"])
    base["seen"] = list(data.get("seen") or base.get("seen") or [])[-_MAX_SEEN:]
    base["last_run"] = data.get("last_run")
    base["last_error"] = data.get("last_error", "")
    return base


def list_triggers() -> List[Dict]:
    with _STORE_LOCK:
        return _load()


def get_trigger(trigger_id: str) -> Optional[Dict]:
    for trigger in list_triggers():
        if trigger.get("id") == trigger_id:
            return trigger
    return None


def create_trigger(data: Dict) -> Dict:
    trigger = _sanitise_trigger(data)
    # Guard against duplicate ids when a client re-submits one.
    with _STORE_LOCK:
        triggers = _load()
        existing = {item.get("id") for item in triggers}
        if trigger["id"] in existing:
            trigger["id"] = uuid.uuid4().hex[:12]
        triggers.append(trigger)
        _save(triggers)
    return trigger


def update_trigger(trigger_id: str, patch: Dict) -> Optional[Dict]:
    with _STORE_LOCK:
        triggers = _load()
        for index, trigger in enumerate(triggers):
            if trigger.get("id") != trigger_id:
                continue
            merged = {**trigger, **(patch or {})}
            merged["id"] = trigger_id
            triggers[index] = _sanitise_trigger(merged)
            triggers[index]["created"] = trigger.get("created", triggers[index]["created"])
            _save(triggers)
            return triggers[index]
    return None


def delete_trigger(trigger_id: str) -> bool:
    with _STORE_LOCK:
        triggers = _load()
        remaining = [trigger for trigger in triggers if trigger.get("id") != trigger_id]
        if len(remaining) == len(triggers):
            return False
        _save(remaining)
        return True


def _matches(trigger: Dict, item: Dict) -> bool:
    keywords = [
        str(keyword).strip().lower()
        for keyword in (trigger.get("condition") or {}).get("keywords") or []
        if str(keyword).strip()
    ]
    if not keywords:
        return True
    haystack = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    mode = ((trigger.get("condition") or {}).get("match") or "any").lower()
    if mode == "all":
        return all(keyword in haystack for keyword in keywords)
    return any(keyword in haystack for keyword in keywords)


async def _source_items(trigger: Dict, timeout: float = 20.0) -> List[Dict]:
    source = trigger.get("source") or {}
    source_type = (source.get("type") or "").lower()
    if source_type == "rss" and source.get("url"):
        feed = await fetch_feed(source["url"], limit=50, timeout=timeout)
        return list(feed.get("items") or [])
    if source_type == "web" and source.get("url"):
        pages = await fast_browse([source["url"]], max_words=1200, timeout=timeout)
        page = pages[0] if pages else {}
        if page.get("error"):
            raise ValueError(page["error"])
        return [
            {
                "id": uuid.uuid4().hex[:12],
                "title": page.get("title") or source["url"],
                "summary": page.get("content") or "",
                "link": page.get("url") or source["url"],
                "published": None,
            }
        ]
    query = source.get("query") or trigger.get("prompt") or ""
    result = await search(
        query,
        apis=[source["api"]] if source.get("api") else None,
        limit=20,
        timeout=timeout,
    )
    return [
        {
            "id": uuid.uuid4().hex[:12],
            "title": row.get("title") or "",
            "summary": row.get("snippet") or "",
            "link": row.get("url") or "",
            "published": row.get("published") or None,
        }
        for row in result.get("results") or []
    ]


def _item_key(item: Dict) -> str:
    return str(item.get("id") or item.get("link") or item.get("title") or "")


async def run_trigger(trigger_id: str, timeout: float = 20.0) -> Dict:
    """Fetch a trigger's source and emit notifications for new matches."""
    trigger = get_trigger(trigger_id)
    if trigger is None:
        raise ValueError("Trigger not found")
    try:
        items = await _source_items(trigger, timeout=timeout)
    except Exception as exc:
        _update_run_state(trigger_id, error=f"{type(exc).__name__}: {exc}")
        return {
            "trigger": get_trigger(trigger_id),
            "new_items": [],
            "notifications": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    seen = set(trigger.get("seen") or [])
    candidates = [item for item in items if _item_key(item) and _item_key(item) not in seen]
    matched = [item for item in candidates if _matches(trigger, item)]
    emitted = []
    for item in matched[:20]:
        emitted.append(
            notifications.add(
                title=item.get("title") or trigger.get("name") or "Trigger update",
                body=item.get("summary") or trigger.get("prompt") or "",
                kind="trigger",
                source=(trigger.get("source") or {}).get("url")
                or (trigger.get("source") or {}).get("query")
                or "trigger",
                url=item.get("link") or "",
                trigger_id=trigger_id,
                metadata={
                    "item_id": item.get("id"),
                    "published": item.get("published"),
                    "action_prompt": (trigger.get("action") or {}).get("prompt"),
                    "action_type": (trigger.get("action") or {}).get("type"),
                },
            )
        )

    new_seen = list(dict.fromkeys([_item_key(item) for item in items if _item_key(item)] + list(seen)))
    _update_run_state(trigger_id, seen=new_seen[:_MAX_SEEN])
    return {
        "trigger": get_trigger(trigger_id),
        "new_items": matched,
        "notifications": emitted,
        "action_prompt": (trigger.get("action") or {}).get("prompt"),
        "error": "",
    }


def _update_run_state(trigger_id: str, seen: Optional[List[str]] = None, error: str = "") -> None:
    with _STORE_LOCK:
        triggers = _load()
        for trigger in triggers:
            if trigger.get("id") == trigger_id:
                trigger["last_run"] = time.time() * 1000
                trigger["last_error"] = error
                if seen is not None:
                    trigger["seen"] = seen
                trigger["updated"] = time.time() * 1000
                break
        _save(triggers)


def run_trigger_sync(trigger_id: str, timeout: float = 20.0) -> Dict:
    return asyncio.run(run_trigger(trigger_id, timeout=timeout))


def due_triggers(now: Optional[float] = None) -> List[Dict]:
    """Return enabled triggers whose interval has elapsed."""
    now_ms = (now if now is not None else time.time()) * 1000
    due = []
    for trigger in list_triggers():
        if not trigger.get("enabled", True):
            continue
        last = trigger.get("last_run") or trigger.get("created") or 0
        if now_ms - last >= int(trigger.get("interval") or 900) * 1000:
            due.append(trigger)
    return due


async def run_due_triggers(timeout: float = 20.0) -> List[Dict]:
    return [await run_trigger(trigger["id"], timeout=timeout) for trigger in due_triggers()]
