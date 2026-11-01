"""RSS/Atom feeds rendered as live g4f conversations.

No third-party XML dependency is required.  The parser accepts RSS 2.0 and
Atom 1.0 (including common namespace variants) and normalises every entry to a
small JSON-safe dict.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

_MAX_FEED_BYTES = 4 * 1024 * 1024
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _child(elem: ET.Element, *names: str) -> Optional[ET.Element]:
    wanted = {name.lower() for name in names}
    for child in list(elem):
        if _local_name(child.tag) in wanted:
            return child
    return None


def _child_text(elem: ET.Element, *names: str) -> str:
    child = _child(elem, *names)
    if child is None:
        return ""
    return clean_text("".join(child.itertext()))


def _child_attr(elem: ET.Element, name: str, attr: str = "href") -> str:
    child = _child(elem, name)
    if child is None:
        return ""
    return (child.get(attr) or clean_text("".join(child.itertext()))).strip()


def clean_text(value: Optional[str]) -> str:
    """Strip HTML/markup and collapse whitespace."""
    if not value:
        return ""
    text = html.unescape(str(value))
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def parse_date(value: str) -> Optional[str]:
    """Parse RFC 822 or ISO-8601 dates into an ISO string (UTC when known)."""
    if not value:
        return None
    value = value.strip()
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def date_to_ms(value: Optional[str], fallback: Optional[float] = None) -> int:
    if value:
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            pass
    return int((fallback if fallback is not None else time.time()) * 1000)


def _entry_id(entry: ET.Element, link: str, title: str) -> str:
    guid = _child_text(entry, "guid", "id")
    raw = guid or link or f"{title}"
    if raw:
        return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:20]
    return hashlib.sha1(ET.tostring(entry, encoding="utf-8")).hexdigest()[:20]


def _entry_link(entry: ET.Element) -> str:
    link = _child_attr(entry, "link")
    if link:
        return link
    # Atom entries sometimes carry the link as an ``id`` URL.
    ident = _child_text(entry, "id")
    if ident.startswith("http://") or ident.startswith("https://"):
        return ident
    return ""


def parse_feed(xml_text: str, source_url: str = "", limit: int = 50) -> Dict:
    """Parse RSS/Atom *xml_text* and return a feed envelope.

    The returned dict contains ``title``, ``description``, ``link``,
    ``source_url`` and ``items``.  Each item has ``id``, ``title``,
    ``summary``, ``link``, ``published``, ``author`` and ``source``.
    """
    if not xml_text or not xml_text.strip():
        raise ValueError("Feed is empty")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid RSS/Atom feed: {exc}") from exc

    root_name = _local_name(root.tag)
    if root_name == "feed":
        feed_elem = root
        entry_elems = [child for child in list(root) if _local_name(child.tag) == "entry"]
    else:
        channel = _child(root, "channel")
        feed_elem = channel if channel is not None else root
        entry_elems = [child for child in list(feed_elem) if _local_name(child.tag) == "item"]
        if not entry_elems:
            entry_elems = [child for child in list(root) if _local_name(child.tag) == "item"]

    feed_title = _child_text(feed_elem, "title") or (urlparse(source_url).netloc if source_url else "Feed")
    feed_description = _child_text(feed_elem, "description", "subtitle")
    feed_link = _child_attr(feed_elem, "link")

    if (
        not entry_elems
        and root_name not in ("rss", "rdf", "feed", "channel")
        and _child(root, "channel") is None
    ):
        raise ValueError("Document is not an RSS/Atom feed")

    items: List[Dict] = []
    seen = set()
    for entry in entry_elems:
        link = _entry_link(entry)
        title = _child_text(entry, "title") or (link or "Untitled")
        summary = _child_text(entry, "description", "summary", "content", "encoded")
        published = parse_date(
            _child_text(entry, "pubdate", "published", "updated", "date", "created")
        )
        author = _child_text(entry, "author", "creator", "name")
        item_id = _entry_id(entry, link, title)
        if item_id in seen:
            continue
        seen.add(item_id)
        items.append(
            {
                "id": item_id,
                "title": clean_text(title),
                "summary": summary[:4000],
                "link": link,
                "published": published,
                "author": author,
                "source": source_url,
            }
        )
        if limit and len(items) >= limit:
            break

    items.sort(key=lambda item: date_to_ms(item.get("published"), 0), reverse=True)
    return {
        "title": clean_text(feed_title),
        "description": feed_description,
        "link": feed_link or source_url,
        "source_url": source_url,
        "items": items,
    }


def _validate_url(url: str, allow_local: bool = False) -> None:
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        raise ValueError("Feed URL must be an absolute http(s) URL")
    try:
        from ..image import is_safe_url

        if not is_safe_url(url, allow_local=allow_local):
            raise ValueError("Feed URL is not a safe public address")
    except ImportError:
        pass


async def fetch_feed(
    url: str,
    limit: int = 50,
    timeout: float = 20.0,
    allow_local: Optional[bool] = None,
) -> Dict:
    """Download and parse a remote RSS/Atom feed."""
    allow_local = (
        os.environ.get("G4F_BROWSING_ALLOW_LOCAL", "").lower() in ("1", "true", "yes")
        if allow_local is None
        else allow_local
    )
    _validate_url(url, allow_local=allow_local)
    import aiohttp

    headers = {
        "User-Agent": os.environ.get(
            "G4F_BROWSING_USER_AGENT",
            "g4f-browsing/1.0 (+https://g4f.dev)",
        ),
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
    }
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.get(url, headers=headers, allow_redirects=True) as response:
            if response.status >= 400:
                raise ValueError(f"Feed request failed with HTTP {response.status}")
            raw = await response.content.read(_MAX_FEED_BYTES + 1)
            if len(raw) > _MAX_FEED_BYTES:
                raise ValueError("Feed is too large")
            charset = response.charset or "utf-8"
    text = raw.decode(charset, errors="replace")
    return parse_feed(text, source_url=url, limit=limit)


def read_feed(url: str, limit: int = 50, timeout: float = 20.0) -> Dict:
    """Synchronous convenience wrapper around :func:`fetch_feed`."""
    return asyncio.run(fetch_feed(url, limit=limit, timeout=timeout))


def feed_to_conversation(
    feed: Dict,
    conversation_id: Optional[str] = None,
    max_items: int = 50,
) -> Dict:
    """Convert a parsed feed into a g4f chat conversation.

    The newest item is the last message, so chats scroll to the latest update
    exactly like a normal conversation.
    """
    source_url = feed.get("source_url") or feed.get("link") or ""
    items = list(feed.get("items") or [])[:max_items]
    items.sort(key=lambda item: date_to_ms(item.get("published"), 0))
    conversation_items = []
    for item in items:
        title = item.get("title") or "Untitled"
        link = item.get("link") or ""
        summary = item.get("summary") or ""
        published = item.get("published") or ""
        lines = [f"## {title}"]
        if summary:
            lines.append("")
            lines.append(summary)
        footer = []
        if published:
            footer.append(f"_{published}_")
        if link:
            footer.append(f"[Read more]({link})")
        if footer:
            lines.append("")
            lines.append(" · ".join(footer))
        conversation_items.append(
            {
                "id": item.get("id"),
                "role": "assistant",
                "content": "\n".join(lines),
                "ts": date_to_ms(published, 0),
                "source": source_url,
                "feed": True,
            }
        )

    now_ms = int(time.time() * 1000)
    conversation_id = conversation_id or hashlib.sha1(
        (source_url or feed.get("title") or "feed").encode("utf-8", "ignore")
    ).hexdigest()[:20]
    return {
        "id": conversation_id,
        "type": "feed",
        "title": feed.get("title") or source_url or "Live feed",
        "feedUrl": source_url,
        "description": feed.get("description") or "",
        "items": conversation_items,
        "added": now_ms,
        "updated": now_ms,
        "feedSeen": [item.get("id") for item in items if item.get("id")],
    }


async def fetch_feed_conversation(
    url: str,
    conversation_id: Optional[str] = None,
    limit: int = 50,
    timeout: float = 20.0,
) -> Dict:
    feed = await fetch_feed(url, limit=limit, timeout=timeout)
    return feed_to_conversation(feed, conversation_id=conversation_id, max_items=limit)


def new_feed_item_ids(conversation: Dict, items: Iterable[Dict]) -> List[str]:
    """Return ids present in *items* that the conversation has not seen."""
    seen = set(conversation.get("feedSeen") or [])
    return [item["id"] for item in items if item.get("id") and item["id"] not in seen]
