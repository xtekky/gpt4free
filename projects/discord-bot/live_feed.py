"""
Live feed cog for the g4f Discord bot.

Polls the g4f API request log and the g4f.dev members service and posts
interesting events to a designated Discord channel:

- 🖼️  Image generation thumbnails
- 🔧  Tool calls (web search, scraping, image gen, etc.)
- 📝  File edits (apply_patch, file_write, file_delete)
- ⚡  Heavy token usage completions
- 🚨  Server errors (5xx)
- 👋  New g4f.dev users
- 📊  Periodic activity summaries

Configuration is read from environment variables (see bot.py):
    G4F_LIVE_FEED_CHANNEL       — Discord channel ID (required to enable)
    G4F_API_BASE                — g4f API base URL (default: http://localhost:8080)
    G4F_PUBLIC_BASE             — public base URL for Discord-accessible image
                                  links (defaults to G4F_API_BASE)
    G4F_MEMBERS_BASE            — g4f.dev base URL (default: https://g4f.dev)
    G4F_FEED_POLL_INTERVAL      — seconds between polls (default: 15)
    G4F_HEAVY_TOKEN_THRESHOLD   — token count to flag as "heavy" (default: 10000)
    G4F_FEED_SUMMARY_INTERVAL   — seconds between summaries (default: 3600)
    G4F_FEED_MAX_POSTS_PER_CYCLE— max embeds per poll cycle (default: 5)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp
import discord
from discord.ext import commands, tasks

log = logging.getLogger("g4f-discord.feed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_PATHS = {
    "/v1/images/generate",
    "/v1/media/generate",
    "/v1/images/generations",
}

FILE_EDIT_TOOLS = {"apply_patch", "file_write", "file_delete"}

COLORS = {
    "image": 0x9D50BB,
    "tools": 0x4776E6,
    "file_edit": 0xFFC107,
    "heavy": 0xFF8C00,
    "new_user": 0x28A745,
    "error": 0xDC3545,
    "summary": 0x6E48AA,
}


# ---------------------------------------------------------------------------
# Path classification helpers
# ---------------------------------------------------------------------------

def _is_image_path(path: str) -> bool:
    return path in IMAGE_PATHS or path.endswith("/images/generations")


def _is_chat_path(path: str) -> bool:
    return path == "/v1/chat/completions" or path.endswith("/chat/completions")


def _is_responses_path(path: str) -> bool:
    return path == "/v1/responses" or path.endswith("/responses")


def _is_messages_path(path: str) -> bool:
    return path == "/v1/messages" or path.endswith("/messages")


def _is_chat_like_path(path: str) -> bool:
    return _is_chat_path(path) or _is_responses_path(path) or _is_messages_path(path)


# ---------------------------------------------------------------------------
# Body parsing helpers
# ---------------------------------------------------------------------------

def _parse_sse_response(body: Any) -> List[dict]:
    """Parse an SSE response body (string) into a list of JSON data objects."""
    if not isinstance(body, str):
        return []
    results: List[dict] = []
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                continue
            try:
                results.append(json.loads(data_str))
            except (json.JSONDecodeError, ValueError):
                pass
    return results


def _normalize_usage(usage: dict) -> dict:
    """Normalize OpenAI/Anthropic usage dicts into a common shape."""
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total = usage.get("total_tokens") or (prompt + completion)
    return {
        "total_tokens": total,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }


def _extract_usage(body: Any) -> Optional[dict]:
    """Extract normalized usage stats from a response body (dict or SSE string)."""
    if body is None:
        return None
    if isinstance(body, str):
        for chunk in reversed(_parse_sse_response(body)):
            if chunk.get("usage"):
                return _normalize_usage(chunk["usage"])
        return None
    if isinstance(body, dict):
        usage = body.get("usage")
        if usage:
            return _normalize_usage(usage)
    return None


def _extract_tool_calls(body: Any) -> List[dict]:
    """Extract tool calls from a response body (dict or SSE string)."""
    if body is None:
        return []

    if isinstance(body, str):
        # SSE streaming — accumulate tool_calls across chunks
        tool_calls_map: Dict[int, dict] = {}
        for chunk in _parse_sse_response(body):
            choices = chunk.get("choices", [])
            for choice in choices:
                delta = choice.get("delta", {}) or choice.get("message", {})
                tcs = delta.get("tool_calls", [])
                for tc in tcs:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {"name": "", "arguments": ""}
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_calls_map[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tool_calls_map[idx]["arguments"] += fn["arguments"]
        return list(tool_calls_map.values())

    if isinstance(body, dict):
        choices = body.get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            tcs = message.get("tool_calls", [])
            if tcs:
                return [
                    {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    }
                    for tc in tcs
                ]
    return []


def _extract_image_urls(body: Any) -> List[str]:
    """Extract image URLs from an image generation response body."""
    if not isinstance(body, dict):
        return []
    data = body.get("data", [])
    if not isinstance(data, list):
        return []
    urls: List[str] = []
    for item in data:
        if isinstance(item, dict):
            url = item.get("url")
            if url and isinstance(url, str) and url.startswith("http"):
                urls.append(url)
    return urls


def _has_b64_images(body: Any) -> bool:
    """Check if an image generation response contains base64 images."""
    if not isinstance(body, dict):
        return False
    data = body.get("data", [])
    if not isinstance(data, list):
        return False
    return any(
        isinstance(item, dict) and item.get("b64_json")
        for item in data
    )


def _extract_prompt(body: Any) -> str:
    """Extract the prompt from an image generation request body."""
    if isinstance(body, dict):
        return str(body.get("prompt", ""))
    return ""


def _extract_first_user_message(body: Any) -> str:
    """Extract the first user message from a chat completion request body."""
    if not isinstance(body, dict):
        return ""
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text", ""))
    return ""


def _extract_model(body: Any) -> str:
    """Extract the model from a request body."""
    if isinstance(body, dict):
        return str(body.get("model", ""))
    return ""


def _extract_provider(body: Any) -> str:
    """Extract the provider from a response body."""
    if isinstance(body, dict):
        return str(body.get("provider", ""))
    return ""


# ---------------------------------------------------------------------------
# URL & formatting helpers
# ---------------------------------------------------------------------------

def _to_thumbnail_url(image_url: str, public_base: str) -> str:
    """Convert a g4f media URL to a thumbnail URL.

    For URLs pointing to ``/media/`` or ``/images/`` on the g4f API,
    rewrite them to ``/thumbnail/`` using *public_base* so Discord can
    fetch them.  External URLs are returned unchanged.
    """
    try:
        parsed = urlparse(image_url)
        path = parsed.path
        if path.startswith("/media/"):
            filename = path[len("/media/"):]
            return f"{public_base}/thumbnail/{filename}"
        if path.startswith("/images/"):
            filename = path[len("/images/"):]
            return f"{public_base}/thumbnail/{filename}"
    except Exception:
        pass
    return image_url


def _to_media_url(image_url: str, public_base: str) -> str:
    """Rewrite a g4f media URL to use *public_base* (for full-size links)."""
    try:
        parsed = urlparse(image_url)
        if parsed.path.startswith("/media/") or parsed.path.startswith("/images/"):
            return f"{public_base}{parsed.path}"
    except Exception:
        pass
    return image_url


def _truncate(text: str, limit: int = 1900) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _format_duration(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


# ---------------------------------------------------------------------------
# LiveFeed cog
# ---------------------------------------------------------------------------

class LiveFeed(commands.Cog):
    """Poll g4f API + g4f.dev members and post interesting events to a channel.

    All config is passed in from ``bot.py`` (read from env vars there).
    """

    def __init__(
        self,
        bot: commands.Bot,
        channel_id: int,
        api_base: str,
        api_key: Optional[str],
        public_base: str,
        members_base: Optional[str],
        poll_interval: int = 15,
        heavy_token_threshold: int = 10_000,
        summary_interval: int = 3600,
        max_posts_per_cycle: int = 5,
    ):
        self.bot = bot
        self.channel_id = channel_id
        self.api_base = api_base.rstrip("/")
        self.public_base = public_base.rstrip("/")
        self.api_key = api_key
        self.members_base = members_base.rstrip("/") if members_base else None
        self.heavy_token_threshold = heavy_token_threshold
        self.max_posts_per_cycle = max_posts_per_cycle
        self._summary_interval = summary_interval

        self._last_log_id: int = 0
        self._initialized: bool = False
        self._seen_user_keys: Set[str] = set()
        self._session: Optional[aiohttp.ClientSession] = None

        # Rolling stats for periodic summary
        self._stats: Dict[str, int] = {
            "requests": 0,
            "tokens": 0,
            "images": 0,
            "tool_calls": 0,
            "file_edits": 0,
            "errors": 0,
            "new_users": 0,
        }
        self._models: Counter = Counter()
        self._providers: Counter = Counter()
        self._last_summary: float = time.time()

        self.poll.start()
        log.info(
            "LiveFeed cog started — channel=%s api=%s members=%s interval=%ds",
            channel_id, self.api_base, self.members_base, poll_interval,
        )

    async def cog_unload(self) -> None:
        self.poll.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # HTTP session
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20),
                headers={
                    "User-Agent": "g4f-discord-livefeed/1.0",
                    **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
                },
            )
        return self._session

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    @tasks.loop(seconds=15)
    async def poll(self) -> None:
        """Main polling loop — runs every *poll_interval* seconds."""
        try:
            await self._poll_api_logs()
        except Exception:
            log.exception("API logs poll failed")

        if self.members_base:
            try:
                await self._poll_new_users()
            except Exception:
                log.exception("New users poll failed")

        if time.time() - self._last_summary >= self._summary_interval:
            await self._post_summary()
            self._last_summary = time.time()

    @poll.before_loop
    async def _wait_ready(self) -> None:
        await self.bot.wait_until_ready()

    # ------------------------------------------------------------------
    # API logs polling
    # ------------------------------------------------------------------

    async def _poll_api_logs(self) -> None:
        session = self._get_session()
        url = f"{self.api_base}/api/logs?limit=100"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return

        entries = data.get("entries", [])
        if not entries:
            return

        if not self._initialized:
            self._last_log_id = max(e.get("id", 0) for e in entries)
            self._initialized = True
            log.info("LiveFeed initialised at log id %d", self._last_log_id)
            return

        new_entries = [e for e in entries if e.get("id", 0) > self._last_log_id]
        if not new_entries:
            return

        new_entries.sort(key=lambda e: e.get("id", 0))

        posted = 0
        for entry in new_entries:
            self._track_stats(entry)
            if posted >= self.max_posts_per_cycle:
                continue
            if await self._handle_entry(entry):
                posted += 1

        self._last_log_id = max(e.get("id", 0) for e in new_entries)

    # ------------------------------------------------------------------
    # New users polling
    # ------------------------------------------------------------------

    async def _poll_new_users(self) -> None:
        if not self.members_base:
            return
        session = self._get_session()
        url = f"{self.members_base}/members/api/recent-users?limit=20"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return

        users = data.get("users", [])
        if not users:
            return

        if not self._initialized:
            for u in users:
                self._seen_user_keys.add(f"{u.get('provider')}:{u.get('username')}")
            return

        for user in users:
            key = f"{user.get('provider')}:{user.get('username')}"
            if key in self._seen_user_keys:
                continue
            self._seen_user_keys.add(key)
            self._stats["new_users"] += 1
            await self._post_new_user(user)

    # ------------------------------------------------------------------
    # Entry dispatch
    # ------------------------------------------------------------------

    async def _handle_entry(self, entry: dict) -> bool:
        """Dispatch a log entry to the appropriate handler. Returns True if posted."""
        path = entry.get("path", "")
        status = entry.get("status", 200)

        if status >= 500:
            return await self._post_error(entry)

        if _is_image_path(path):
            return await self._post_image(entry)

        if _is_chat_like_path(path):
            return await self._post_chat(entry)

        return False

    # ------------------------------------------------------------------
    # Stats tracking
    # ------------------------------------------------------------------

    def _track_stats(self, entry: dict) -> None:
        self._stats["requests"] += 1
        status = entry.get("status", 200)
        if status >= 500:
            self._stats["errors"] += 1

        path = entry.get("path", "")
        if _is_image_path(path):
            self._stats["images"] += 1

        req = entry.get("request_body")
        if isinstance(req, dict):
            model = req.get("model")
            if model:
                self._models[str(model)] += 1

        body = entry.get("response_body")
        usage = _extract_usage(body)
        if usage:
            self._stats["tokens"] += usage.get("total_tokens", 0)

        provider = _extract_provider(body)
        if provider:
            self._providers[provider] += 1

        tool_calls = _extract_tool_calls(body)
        if tool_calls:
            self._stats["tool_calls"] += len(tool_calls)
            for tc in tool_calls:
                if tc.get("name") in FILE_EDIT_TOOLS:
                    self._stats["file_edits"] += 1

    # ------------------------------------------------------------------
    # Event posters
    # ------------------------------------------------------------------

    async def _post_image(self, entry: dict) -> bool:
        """Post an image generation thumbnail."""
        body = entry.get("response_body")
        urls = _extract_image_urls(body)
        has_b64 = _has_b64_images(body)

        req = entry.get("request_body", {})
        prompt = _extract_prompt(req)
        model = _extract_model(req) or "unknown"
        duration = entry.get("duration_ms", 0)
        user = entry.get("user")

        embed = discord.Embed(
            title="🖼️ Image Generated",
            color=COLORS["image"],
        )
        embed.add_field(
            name="Prompt",
            value=_truncate(prompt, 256) or "_(empty)_",
            inline=False,
        )
        embed.add_field(name="Model", value=f"`{model}`", inline=True)
        embed.add_field(name="Duration", value=_format_duration(duration), inline=True)
        if user:
            embed.add_field(name="User", value=user, inline=True)

        if urls:
            thumb_url = _to_thumbnail_url(urls[0], self.public_base)
            full_url = _to_media_url(urls[0], self.public_base)
            embed.set_image(url=thumb_url)
            embed.add_field(
                name="Full image",
                value=f"[Open]({full_url})",
                inline=True,
            )
        elif has_b64:
            embed.add_field(
                name="Output",
                value="_(base64 image — no public URL)_",
                inline=False,
            )
        else:
            embed.add_field(
                name="Output",
                value="_(no image URL in response)_",
                inline=False,
            )

        await self._send(embed)
        return True

    async def _post_chat(self, entry: dict) -> bool:
        """Post a chat completion event (tool calls and/or heavy token usage)."""
        body = entry.get("response_body")
        tool_calls = _extract_tool_calls(body)
        usage = _extract_usage(body)

        has_tools = bool(tool_calls)
        has_file_edits = any(tc.get("name") in FILE_EDIT_TOOLS for tc in tool_calls)
        total_tokens = (usage or {}).get("total_tokens", 0) or 0
        is_heavy = total_tokens >= self.heavy_token_threshold

        if not has_tools and not is_heavy:
            return False

        if has_file_edits:
            title = "📝 File Edit via AI"
            color = COLORS["file_edit"]
        elif has_tools:
            title = "🔧 Tool Calls"
            color = COLORS["tools"]
        else:
            title = "⚡ Heavy Token Usage"
            color = COLORS["heavy"]

        embed = discord.Embed(title=title, color=color)

        if has_tools:
            tool_names = [tc.get("name", "?") for tc in tool_calls]
            embed.add_field(
                name="Tools",
                value=" ".join(f"`{n}`" for n in tool_names),
                inline=False,
            )

        if is_heavy:
            prompt_t = (usage or {}).get("prompt_tokens", 0)
            comp_t = (usage or {}).get("completion_tokens", 0)
            embed.add_field(
                name="Tokens",
                value=f"**{_format_tokens(total_tokens)}** (↑{_format_tokens(prompt_t)} ↓{_format_tokens(comp_t)})",
                inline=True,
            )

        req = entry.get("request_body", {})
        model = _extract_model(req) or "unknown"
        embed.add_field(name="Model", value=f"`{model}`", inline=True)

        user = entry.get("user")
        if user:
            embed.add_field(name="User", value=user, inline=True)

        first_msg = _extract_first_user_message(req)
        if first_msg:
            embed.add_field(
                name="Prompt",
                value=_truncate(first_msg, 200),
                inline=False,
            )

        embed.add_field(
            name="Duration",
            value=_format_duration(entry.get("duration_ms", 0)),
            inline=True,
        )

        await self._send(embed)
        return True

    async def _post_error(self, entry: dict) -> bool:
        """Post a server error alert."""
        embed = discord.Embed(
            title="🚨 Server Error",
            color=COLORS["error"],
        )
        embed.add_field(
            name="Path",
            value=f"`{entry.get('method', '?')} {entry.get('path', '?')}`",
            inline=False,
        )
        embed.add_field(name="Status", value=str(entry.get("status", "?")), inline=True)
        embed.add_field(
            name="Duration",
            value=_format_duration(entry.get("duration_ms", 0)),
            inline=True,
        )
        user = entry.get("user")
        if user:
            embed.add_field(name="User", value=user, inline=True)

        await self._send(embed)
        return True

    async def _post_new_user(self, user: dict) -> None:
        """Post a new g4f.dev user announcement."""
        username = user.get("username", "unknown")
        provider = user.get("provider", "unknown")
        avatar = user.get("avatar")
        tier = user.get("tier", "new")
        created_at = user.get("created_at", "")

        embed = discord.Embed(
            title="👋 New g4f.dev User",
            color=COLORS["new_user"],
        )
        embed.add_field(name="Username", value=f"**{username}**", inline=True)
        embed.add_field(name="Provider", value=provider, inline=True)
        embed.add_field(name="Tier", value=tier, inline=True)
        if created_at:
            embed.add_field(name="Joined", value=created_at, inline=False)
        if avatar and isinstance(avatar, str) and avatar.startswith("http"):
            embed.set_thumbnail(url=avatar)

        await self._send(embed)

    async def _post_summary(self) -> None:
        """Post a periodic activity summary."""
        s = self._stats
        if s["requests"] == 0 and s["new_users"] == 0:
            return  # nothing happened

        embed = discord.Embed(
            title="📊 Activity Summary",
            description=f"Last {self._summary_interval // 60} minutes",
            color=COLORS["summary"],
        )

        embed.add_field(name="Requests", value=str(s["requests"]), inline=True)
        embed.add_field(name="Tokens", value=_format_tokens(s["tokens"]), inline=True)
        embed.add_field(name="Images", value=str(s["images"]), inline=True)
        embed.add_field(name="Tool calls", value=str(s["tool_calls"]), inline=True)
        embed.add_field(name="File edits", value=str(s["file_edits"]), inline=True)
        embed.add_field(name="Errors", value=str(s["errors"]), inline=True)
        embed.add_field(name="New users", value=str(s["new_users"]), inline=True)

        top_models = self._models.most_common(3)
        if top_models:
            embed.add_field(
                name="Top models",
                value="\n".join(f"`{m}` ({n})" for m, n in top_models),
                inline=False,
            )

        top_providers = self._providers.most_common(3)
        if top_providers:
            embed.add_field(
                name="Top providers",
                value="\n".join(f"`{p}` ({n})" for p, n in top_providers),
                inline=False,
            )

        await self._send(embed)

        # Reset rolling stats
        for k in self._stats:
            self._stats[k] = 0
        self._models.clear()
        self._providers.clear()

    # ------------------------------------------------------------------
    # Send helper
    # ------------------------------------------------------------------

    async def _send(self, embed: discord.Embed) -> None:
        """Send an embed to the feed channel, logging on failure."""
        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            channel = await self.bot.fetch_channel(self.channel_id)
        if channel is None:
            log.warning("Feed channel %s not found", self.channel_id)
            return
        try:
            await channel.send(embed=embed)
        except discord.HTTPException:
            log.exception("Failed to send feed embed")
