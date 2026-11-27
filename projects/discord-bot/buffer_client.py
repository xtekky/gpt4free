"""
Async client for posting Discord announcements (text + images) to Buffer.com.

Uses Buffer's GraphQL API (https://api.buffer.com) with a Bearer API key.
If explicit ``channel_ids`` are not provided, the client automatically
discovers all active (non-paused) channels across the account's
organizations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import aiohttp

log = logging.getLogger("g4f-discord.buffer")

BUFFER_GRAPHQL_URL = "https://api.buffer.com"

_GET_ORGANIZATIONS_QUERY = """
query GetOrganizations {
  account {
    organizations {
      id
      name
    }
  }
}
"""

_GET_CHANNELS_QUERY = """
query GetChannels($organizationId: OrganizationId!) {
  channels(input: { organizationId: $organizationId }) {
    id
    name
    service
    type
    isQueuePaused
  }
}
"""

_CREATE_POST_MUTATION = """
mutation CreatePost($input: CreatePostInput!) {
  createPost(input: $input) {
    ... on PostActionSuccess {
      post {
        id
        text
        dueAt
      }
    }
    ... on MutationError {
      message
    }
  }
}
"""


TWITTER_CHAR_LIMIT = 280


def _split_text(text: str, limit: int = TWITTER_CHAR_LIMIT) -> List[str]:
    """Split text into chunks of at most ``limit`` characters on natural boundaries."""
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        window = remaining[:limit]
        split_at = -1
        for sep in ("\n\n", "\n", " "):
            idx = window.rfind(sep)
            if idx > 0:
                split_at = idx
                break

        if split_at <= 0:
            split_at = limit

        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    return chunks or [text[:limit]]


def _truncate_text(text: str, limit: int = TWITTER_CHAR_LIMIT) -> str:
    """Truncate text to at most ``limit`` characters with an ellipsis."""
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    window = text[: limit - 1].rstrip()
    space_idx = window.rfind(" ")
    if space_idx > limit // 2:
        window = window[:space_idx].rstrip()
    return f"{window}…"


class BufferClient:
    """Minimal async client for Buffer's GraphQL API."""

    def __init__(
        self,
        api_key: str,
        channel_ids: Optional[Sequence[str]] = None,
        mode: str = "addToQueue",
        api_url: str = BUFFER_GRAPHQL_URL,
    ) -> None:
        self.api_key = api_key.strip()
        self.channel_ids: List[str] = [c.strip() for c in (channel_ids or []) if c.strip()]
        self.mode = (mode or "addToQueue").strip()
        self.api_url = api_url
        self._session: Optional[aiohttp.ClientSession] = None
        self._discovered_channels: Optional[List[str]] = None
        self._channel_info: Dict[str, Dict[str, str]] = {}

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "g4f-discord-buffer/1.0",
                },
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _graphql(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        session = self._get_session()
        payload: Dict[str, Any] = {"query": query}
        if variables is not None:
            payload["variables"] = variables

        async with session.post(self.api_url, json=payload) as resp:
            data = await resp.json(content_type=None)
            if resp.status >= 400:
                raise RuntimeError(f"Buffer API HTTP {resp.status}: {data}")
            if isinstance(data, dict) and data.get("errors"):
                raise RuntimeError(f"Buffer GraphQL errors: {data['errors']}")
            return data.get("data", {}) if isinstance(data, dict) else {}

    async def get_target_channel_ids(self) -> List[str]:
        """Return configured channel IDs or discover active channels on the account."""
        if self.channel_ids:
            return list(self.channel_ids)
        if self._discovered_channels is not None:
            return list(self._discovered_channels)

        org_data = await self._graphql(_GET_ORGANIZATIONS_QUERY)
        orgs = (org_data.get("account") or {}).get("organizations") or []
        discovered: List[str] = []

        for org in orgs:
            org_id = org.get("id")
            if not org_id:
                continue
            ch_data = await self._graphql(
                _GET_CHANNELS_QUERY, variables={"organizationId": org_id}
            )
            for ch in ch_data.get("channels") or []:
                ch_id = ch.get("id")
                if ch_id and not ch.get("isQueuePaused"):
                    ch_id_str = str(ch_id)
                    discovered.append(ch_id_str)
                    self._channel_info[ch_id_str] = {
                        "name": str(ch.get("name") or ""),
                        "service": str(ch.get("service") or "").lower(),
                        "type": str(ch.get("type") or "").lower(),
                    }

        self._discovered_channels = discovered
        log.info("Discovered %d active Buffer channel(s)", len(discovered))
        return list(discovered)

    def _build_post_input(
        self,
        channel_id: str,
        clean_text: str,
        assets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        info = self._channel_info.get(channel_id) or {}
        service = info.get("service", "")
        ch_type = info.get("type", "")

        scheduling_type = "automatic"
        if (service == "facebook" and ch_type == "group") or (
            service == "instagram" and ch_type == "profile"
        ):
            scheduling_type = "notification"

        post_input: Dict[str, Any] = {
            "text": clean_text,
            "channelId": channel_id,
            "schedulingType": scheduling_type,
            "mode": self.mode,
        }

        if service == "instagram":
            post_input["metadata"] = {
                "instagram": {
                    "type": "post",
                    "shouldShareToFeed": True,
                }
            }
            if assets:
                post_input["assets"] = assets
        elif service in ("twitter", "x") and len(clean_text) > TWITTER_CHAR_LIMIT:
            chunks = _split_text(clean_text, TWITTER_CHAR_LIMIT)
            if len(chunks) > 1:
                thread_items: List[Dict[str, Any]] = [{"text": c} for c in chunks]
                if assets:
                    thread_items[0]["assets"] = assets
                post_input["text"] = chunks[0]
                post_input["metadata"] = {"twitter": {"thread": thread_items}}
            else:
                post_input["text"] = chunks[0]
                if assets:
                    post_input["assets"] = assets
        elif assets:
            post_input["assets"] = assets

        return post_input

    async def _send_post_with_fallback(
        self,
        post_input: Dict[str, Any],
        clean_text: str,
        assets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        current_input = dict(post_input)
        if "metadata" in post_input:
            current_input["metadata"] = dict(post_input["metadata"])

        for _ in range(3):
            data = await self._graphql(
                _CREATE_POST_MUTATION, variables={"input": current_input}
            )
            result = data.get("createPost") or {}
            if result.get("post"):
                return result["post"]

            err_msg = result.get("message")
            if not err_msg:
                raise RuntimeError(f"unexpected response {result}")

            err_lower = err_msg.lower()
            retried = False

            if (
                "notification scheduling" in err_lower
                and current_input.get("schedulingType") != "notification"
            ):
                current_input["schedulingType"] = "notification"
                retried = True
            elif (
                "instagram posts require a type" in err_lower
                and "instagram" not in (current_input.get("metadata") or {})
            ):
                metadata = dict(current_input.get("metadata") or {})
                metadata["instagram"] = {"type": "post", "shouldShareToFeed": True}
                current_input["metadata"] = metadata
                retried = True
            elif "280 characters" in err_lower:
                metadata = dict(current_input.get("metadata") or {})
                if "twitter" not in metadata:
                    chunks = _split_text(clean_text, TWITTER_CHAR_LIMIT)
                    if len(chunks) > 1:
                        thread_items: List[Dict[str, Any]] = [{"text": c} for c in chunks]
                        if assets:
                            thread_items[0]["assets"] = assets
                        current_input.pop("assets", None)
                        current_input["text"] = chunks[0]
                        metadata["twitter"] = {"thread": thread_items}
                        current_input["metadata"] = metadata
                        retried = True
                    else:
                        target_len = min(TWITTER_CHAR_LIMIT, max(1, len(current_input["text"]) - 20))
                        new_text = _truncate_text(clean_text, target_len)
                        if new_text != current_input["text"]:
                            current_input["text"] = new_text
                            retried = True
                else:
                    metadata.pop("twitter", None)
                    if metadata:
                        current_input["metadata"] = metadata
                    else:
                        current_input.pop("metadata", None)
                    current_input["text"] = _truncate_text(clean_text, TWITTER_CHAR_LIMIT)
                    if assets:
                        current_input["assets"] = assets
                    retried = True
            elif "twitter" in (current_input.get("metadata") or {}):
                # Fallback to truncated single tweet if thread creation is rejected
                metadata = dict(current_input.get("metadata") or {})
                metadata.pop("twitter", None)
                if metadata:
                    current_input["metadata"] = metadata
                else:
                    current_input.pop("metadata", None)
                current_input["text"] = _truncate_text(clean_text, TWITTER_CHAR_LIMIT)
                if assets:
                    current_input["assets"] = assets
                retried = True

            if not retried:
                raise RuntimeError(err_msg)

        raise RuntimeError(err_msg)

    async def create_post(
        self,
        text: str,
        image_urls: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Create a post (optionally with images) on all target Buffer channels.

        Returns a list of created post objects. Raises RuntimeError if no
        channels are available or all channel posts fail.
        """
        target_channels = await self.get_target_channel_ids()
        if not target_channels:
            raise RuntimeError("No active Buffer channels found for this account")

        clean_text = (text or "").strip() or "📢 Update"
        assets = [
            {"image": {"url": url}}
            for url in (image_urls or [])
            if isinstance(url, str) and url.startswith(("http://", "https://"))
        ]

        results: List[Dict[str, Any]] = []
        errors: List[str] = []

        for channel_id in target_channels:
            post_input = self._build_post_input(channel_id, clean_text, assets)

            try:
                post = await self._send_post_with_fallback(
                    post_input, clean_text, assets
                )
                results.append(post)
            except Exception as exc:
                errors.append(f"{channel_id}: {exc}")

        if not results and errors:
            raise RuntimeError("; ".join(errors))
        if errors:
            log.warning("Partial Buffer post errors: %s", "; ".join(errors))

        return results
