from __future__ import annotations

import json
import base64
import os
from urllib.parse import urlencode, quote

from ..typing import AsyncResult, Messages, ImageType
from ..requests import StreamSession, raise_for_status
from ..providers.response import (
    ProviderInfo, JsonConversation, Sources,
    SuggestedFollowups, Usage, Reasoning
)
from ..image import to_bytes
from ..tools.media import merge_media
from ..providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .. import debug


BRAVE_URL = "https://search.brave.com"
BRAVE_ASK_URL = f"{BRAVE_URL}/ask"
DATA_ENDPOINT = f"{BRAVE_ASK_URL}/__data.json"
API_BASE = f"{BRAVE_URL}/api/tap/v1"
NEW_ENDPOINT = f"{API_BASE}/new"
STREAM_ENDPOINT = f"{API_BASE}/stream"

# Deep Research takes much longer due to multi-iteration web crawling
DEEP_RESEARCH_TIMEOUT = 600


class Conversation(JsonConversation):
    """Manages session state for Brave Search Ask."""
    conversation_id: str = None
    symmetric_key: str = None
    nonce: str = None
    sig: str = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.symmetric_key is None:
            self.symmetric_key = base64.urlsafe_b64encode(
                os.urandom(32)
            ).decode().rstrip("=")


class BraveSearch(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Free provider for Brave Search's Ask AI feature.
    
    Uses the Brave Search Ask endpoint (search.brave.com/ask) which provides
    AI-powered answers augmented with web search results. No authentication 
    required.
    
    Models:
    - brave: Standard Ask Brave (fast, single-pass answer)
    - brave-deep-research: Deep Research mode (multi-iteration web crawling,
      generates comprehensive research reports, takes 1-5 minutes)
    """

    label = "Brave Search"
    url = BRAVE_ASK_URL
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = False
    supports_message_history = False

    default_model = "brave"
    models = [default_model, "brave-deep-research"]

    @classmethod
    def _is_deep_research(cls, model: str) -> bool:
        """Check if the model is Deep Research mode."""
        return model == "brave-deep-research"

    @classmethod
    def _parse_sveltekit_tap_data(cls, raw_json: dict) -> dict:
        """
        Parse SvelteKit __data.json format to extract nonce and sig 
        from the tap (Ask) page node.
        
        SvelteKit data format uses indexed arrays where:
        - node_data[0] is a schema dict mapping field names to indices
        - remaining elements are values referenced by those indices
        - The "token" field contains a nested schema with q/nonce/sig indices
        """
        result = {}

        if raw_json.get("type") != "data" or "nodes" not in raw_json:
            return result

        for node in raw_json["nodes"]:
            if node.get("type") != "data" or "data" not in node:
                continue

            node_data = node["data"]
            if not isinstance(node_data, list) or len(node_data) < 2:
                continue

            schema = node_data[0]
            if not isinstance(schema, dict):
                continue

            # Find the tap node (page == "tap")
            page_idx = schema.get("page")
            if not (isinstance(page_idx, int) and 0 < page_idx < len(node_data)):
                continue
            if node_data[page_idx] != "tap":
                continue

            # Extract token which contains q, nonce, sig
            token_idx = schema.get("token")
            if isinstance(token_idx, int) and 0 < token_idx < len(node_data):
                token = node_data[token_idx]
                if isinstance(token, dict):
                    for key, val_idx in token.items():
                        if isinstance(val_idx, int) and 0 < val_idx < len(node_data):
                            result[key] = node_data[val_idx]
            break

        return result

    @classmethod
    def _process_research_event(cls, event: dict):
        """
        Process a Deep Research 'research' event and return 
        a Reasoning object for status display, or None.
        """
        inner = event.get("event", {})
        if not isinstance(inner, dict):
            return None

        research_event = inner.get("event", "")

        if research_event == "queries":
            queries = inner.get("queries", [])
            return Reasoning(
                status=f"Researching: {', '.join(queries[:3])}",
                label="Deep Research"
            )

        elif research_event == "analyzing":
            query = inner.get("query", "")
            urls = inner.get("urls", 0)
            return Reasoning(
                status=f"Analyzing {urls} sources for: {query}",
                label="Deep Research"
            )

        elif research_event == "thinking":
            query = inner.get("query", "")
            chunks = inner.get("chunks_analyzed", 0)
            urls = inner.get("urls_analyzed", 0)
            return Reasoning(
                status=f"Processing {chunks} chunks from {urls} sources: {query}",
                label="Deep Research"
            )

        elif research_event == "progress":
            iterations = inner.get("number_of_iterations", 0)
            queries = inner.get("number_of_queries", 0)
            urls = inner.get("number_of_urls_analyzed", 0)
            elapsed = inner.get("elasped_seconds", 0)
            return Reasoning(
                status=f"Iteration {iterations}: {queries} queries, "
                       f"{urls} URLs analyzed ({elapsed:.0f}s)",
                label="Deep Research"
            )

        elif research_event == "blindspots":
            spots = inner.get("blindspots", [])
            if spots:
                return Reasoning(
                    status=f"Identified gaps: {', '.join(spots[:3])}",
                    label="Deep Research"
                )

        elif research_event == "answer" and inner.get("final"):
            return Reasoning(
                status="Research complete, generating final answer",
                label="Deep Research"
            )

        return None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        images: ImageType | list[ImageType] = None,
        media: MediaListType = None,
        conversation: Conversation = None,
        prompt: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        is_deep = cls._is_deep_research(model)

        # Extract the last user message as the query
        if not prompt:
            for message in reversed(messages):
                if message["role"] == "user":
                    prompt = message["content"]
                    if isinstance(prompt, list):
                        prompt = "\n".join(
                            [item.get("text", "") for item in prompt if isinstance(item, dict)]
                        ) if prompt else ""
                    break

        if not prompt:
            raise ValueError("No user message found in messages")

        # Initialize conversation if needed
        if conversation is None:
            conversation = Conversation()

        headers = {
            "accept": "application/json",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
        }

        # Deep Research needs a much longer timeout
        timeout = DEEP_RESEARCH_TIMEOUT if is_deep else 120

        async with StreamSession(
            headers=headers,
            proxy=proxy,
            impersonate="chrome",
            timeout=timeout,
        ) as session:
            # ===== STEP 1: Get nonce + sig from __data.json =====
            referer = f"{BRAVE_ASK_URL}?q=&source=llmSuggest"
            if conversation.conversation_id:
                referer = (
                    f"{BRAVE_ASK_URL}?q={quote(prompt)}&source=llmSuggest"
                    f"&conversation={conversation.conversation_id}"
                )

            data_params = {
                "q": prompt,
                "x-sveltekit-invalidated": "11",
            }
            if is_deep:
                data_params["enable_research"] = "true"

            async with session.get(
                DATA_ENDPOINT,
                params=data_params,
                headers={"referer": referer}
            ) as response:
                await raise_for_status(response)
                data_json = await response.json()
                tap_data = cls._parse_sveltekit_tap_data(data_json)

                nonce = tap_data.get("nonce")
                sig = tap_data.get("sig")
                if not nonce or not sig:
                    raise RuntimeError(
                        "Failed to extract nonce/sig from __data.json. "
                        f"Got: {tap_data}"
                    )
                conversation.nonce = nonce
                conversation.sig = sig
                debug.log(f"BraveSearch: Got nonce={nonce[:8]}... sig={sig[:8]}...")

            # ===== STEP 2: Create new conversation =====
            new_params = {
                "language": "en",
                "country": "us",
                "ui_lang": "en-us",
                "safesearch": "moderate",
                "force_safesearch": "0",
                "units_of_measurement": "metric",
                "use_location": "1",
                "geoloc": "50.457x30.532",
                "premium_cookie_name": "__Secure-sku#brave-search-premium",
                "symmetric_key": conversation.symmetric_key,
                "source": "newThread" if is_deep else "llmSuggest",
                "enable_research": "true" if is_deep else "false",
                "q": prompt,
                "nonce": nonce,
                "sig": sig,
            }

            async with session.get(
                NEW_ENDPOINT,
                params=new_params,
                headers={
                    "referer": f"{BRAVE_ASK_URL}?q={quote(prompt)}&source=newThread"
                }
            ) as response:
                await raise_for_status(response)
                new_data = await response.json()
                conversation.conversation_id = new_data.get("id")
                if not conversation.conversation_id:
                    raise RuntimeError(
                        f"Failed to get conversation ID from /new: {new_data}"
                    )
                debug.log(
                    f"BraveSearch: Conversation ID: {conversation.conversation_id}"
                )

            yield conversation
            yield ProviderInfo(**cls.get_dict(), model=model)

            # ===== STEP 3: Stream the response =====
            stream_params = {
                "language": "en",
                "country": "us",
                "ui_lang": "en-us",
                "safesearch": "moderate",
                "force_safesearch": "0",
                "units_of_measurement": "metric",
                "use_location": "1",
                "geoloc": "50.457x30.532",
                "premium_cookie_name": "__Secure-sku#brave-search-premium",
                "id": conversation.conversation_id,
                "query": prompt,
                "symmetric_key": conversation.symmetric_key,
                "enable_inline_entities": "true",
            }

            stream_url = f"{STREAM_ENDPOINT}?{urlencode(stream_params)}"

            request_kwargs = {
                "headers": {
                    "accept": "text/event-stream",
                    "origin": "https://search.brave.com",
                    "referer": (
                        f"{BRAVE_ASK_URL}?q={quote(prompt)}&source=newThread"
                        f"&conversation={conversation.conversation_id}"
                    ),
                }
            }
            
            async with session.get(stream_url, **request_kwargs) as response:
                await raise_for_status(response)

                sources = []
                buffer = b""
                # Track if we've received the final research answer
                # to avoid yielding intermediate answers as text
                research_final_received = False

                async for chunk in response.iter_content():
                    if not chunk:
                        continue
                    buffer += chunk

                    # Process complete lines (newline-delimited JSON)
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue

                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type", "")

                        if event_type == "text_delta":
                            delta = event.get("delta", "")
                            if delta:
                                yield delta

                        elif event_type == "research_start":
                            yield Reasoning(
                                status="Starting deep research...",
                                label="Deep Research"
                            )

                        elif event_type == "research":
                            reasoning = cls._process_research_event(event)
                            if reasoning:
                                yield reasoning

                        elif event_type == "research_end":
                            yield Reasoning(
                                status="Research phase complete",
                                label="Deep Research"
                            )

                        elif event_type == "augment_with_inline_citation":
                            url = event.get("url")
                            title = event.get("title")
                            if url:
                                sources.append({
                                    "url": url,
                                    "title": title or url,
                                })

                        elif event_type == "followups":
                            followups = event.get("followups", [])
                            if followups:
                                yield SuggestedFollowups(followups)

                        elif event_type == "usage":
                            yield Usage(
                                input_tokens=event.get("prompt_tokens"),
                                output_tokens=event.get("completion_tokens"),
                            )

                # Process remaining buffer
                if buffer:
                    line = buffer.decode("utf-8", errors="replace").strip()
                    if line:
                        try:
                            event = json.loads(line)
                            event_type = event.get("type", "")
                            if event_type == "text_delta":
                                delta = event.get("delta", "")
                                if delta:
                                    yield delta
                        except json.JSONDecodeError:
                            pass

                # Yield sources at the end
                if sources:
                    # Deduplicate sources by URL
                    seen_urls = set()
                    unique_sources = []
                    for src in sources:
                        if src["url"] not in seen_urls:
                            seen_urls.add(src["url"])
                            unique_sources.append(src)
                    yield Sources(unique_sources)

                yield conversation
