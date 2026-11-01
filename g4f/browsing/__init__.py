"""Browsing APIs, live feeds, triggers and notifications for g4f.

Public surface::

    from g4f.browsing import list_apis, search, fast_browse
    from g4f.browsing import fetch_feed, fetch_feed_conversation
    from g4f.browsing import create_trigger, parse_trigger_prompt, run_trigger
    from g4f.browsing import notifications
"""

from .catalog import API_CATALOG, get_api, list_apis, resolve_apis
from .feeds import (
    feed_to_conversation,
    fetch_feed,
    fetch_feed_conversation,
    parse_feed,
    read_feed,
)
from .browse import browse, execute_api, extract_text, fast_browse, fetch_page, search
from .notifications import NotificationStore, notifications
from .triggers import (
    create_trigger,
    delete_trigger,
    due_triggers,
    get_trigger,
    list_triggers,
    parse_trigger_prompt,
    run_due_triggers,
    run_trigger,
    run_trigger_sync,
    update_trigger,
)

__all__ = [
    "API_CATALOG",
    "NotificationStore",
    "browse",
    "create_trigger",
    "delete_trigger",
    "due_triggers",
    "execute_api",
    "extract_text",
    "fast_browse",
    "feed_to_conversation",
    "fetch_feed",
    "fetch_feed_conversation",
    "fetch_page",
    "get_api",
    "get_trigger",
    "list_apis",
    "list_triggers",
    "notifications",
    "parse_feed",
    "parse_trigger_prompt",
    "read_feed",
    "resolve_apis",
    "run_due_triggers",
    "run_trigger",
    "run_trigger_sync",
    "search",
    "update_trigger",
]
