from __future__ import annotations

import random
import uuid

from ..typing import AsyncResult, Messages, Cookies
from ..requests import StreamSession, raise_for_status, sse_stream
from ..cookies import get_cookies
from ..providers.response import ProviderInfo, JsonConversation, JsonRequest, JsonResponse, Reasoning, Sources, SuggestedFollowups, ImageResponse, PreviewResponse, YouTubeResponse
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .. import debug

class Perplexity(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Perplexity"
    url = "https://www.perplexity.ai"
    cookie_domain = ".perplexity.ai"
    working = True
    active_by_default = True

    default_model = "auto"
    models = [
        default_model,
        "turbo",
        "gpt41",
        "gpt5",
        "gpt5_thinking",
        "o3",
        "o3pro",
        "claude2",
        "claude37sonnetthinking",
        "claude40opus",
        "claude40opusthinking",
        "claude41opusthinking",
        "claude45sonnet",
        "claude45sonnetthinking",
        "experimental",
        "grok",
        "grok4",
        "gemini2flash",
        "pplx_pro",
        "pplx_pro_upgraded",
        "pplx_alpha",
        "pplx_beta",
        "comet_max_assistant",
        "o3_research",
        "o3pro_research",
        "claude40sonnet_research",
        "claude40sonnetthinking_research",
        "claude40opus_research",
        "claude40opusthinking_research",
        "o3_labs",
        "o3pro_labs",
        "claude40sonnetthinking_labs",
        "claude40opusthinking_labs",
        "o4mini",
        "o1",
        "gpt4o",
        "gpt45",
        "gpt4",
        "o3mini",
        "claude35haiku",
        "llama_x_large",
        "mistral",
        "claude3opus",
        "gemini",
        "pplx_reasoning",
        "r1"
    ]
    fallback_models = ["perplexity", "pplx_pro"]
    model_aliases = {
        "gpt-5": "gpt5",
        "gpt-5-thinking": "gpt5_thinking",
        "r1-1776": "r1",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        cookies: Cookies = None,
        conversation: JsonConversation = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = cls.default_model
        if cookies is None:
            cookies = get_cookies(cls.cookie_domain, False)
        if conversation is None:
            conversation = JsonConversation(
                frontend_uid=str(uuid.uuid4()),
                frontend_context_uuid=str(uuid.uuid4()),
                visitor_id=str(uuid.uuid4()),
                user_id=None,
            )
        request_id = str(uuid.uuid4())
        
        headers = {
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
            "x-perplexity-request-reason": "perplexity-query-state-provider",
            "x-request-id": request_id,
        }
    
        # Extract the last user message as the query
        query = ""
        for message in reversed(messages):
            if message["role"] == "user":
                query = message["content"]
                break

        async with StreamSession(headers=headers, cookies=cookies, proxy=proxy, impersonate="chrome") as session:
            if conversation.user_id is None:
                async with session.get(f"{cls.url}/api/auth/session") as response:
                    await raise_for_status(response)
                    user = await response.json()
                    conversation.user_id = user.get("user", {}).get("id")
                    debug.log(f"Perplexity user id: {conversation.user_id}")
            yield conversation
            if model == "auto" or model == "perplexity":
                model = "pplx_pro" if conversation.user_id else "turbo"
            yield ProviderInfo(**cls.get_dict(), model=model)
            if model in cls.model_aliases:
                model = cls.model_aliases[model]
            if conversation.user_id is None:
                data = {
                    "params": {
                        "attachments": [],
                        "language": "en-US",
                        "timezone": "America/New_York",
                        "search_focus": "internet",
                        "sources": ["web"],
                        "search_recency_filter": None,
                        "frontend_uuid": conversation.frontend_uid,
                        "mode": "concise",
                        "model_preference": model,
                        "is_related_query": False,
                        "is_sponsored": False,
                        "visitor_id": conversation.visitor_id,
                        "frontend_context_uuid": conversation.frontend_context_uuid,
                        "prompt_source": "user",
                        "query_source": "home",
                        "is_incognito": False,
                        "time_from_first_type": 0,
                        "local_search_enabled": False,
                        "use_schematized_api": True,
                        "send_back_text_in_streaming_api": False,
                        "supported_block_use_cases": [
                            "answer_modes",
                            "media_items",
                            "knowledge_cards",
                            "inline_entity_cards",
                            "place_widgets",
                            "finance_widgets",
                            "sports_widgets",
                            "shopping_widgets",
                            "jobs_widgets",
                            "search_result_widgets",
                            "clarification_responses",
                            "inline_images",
                            "inline_assets",
                            "inline_finance_widgets",
                            "placeholder_cards",
                            "diff_blocks",
                            "inline_knowledge_cards",
                            "entity_group_v2",
                            "refinement_filters",
                            "canvas_mode"
                        ],
                        "client_coordinates": None,
                        "mentions": [],
                        "dsl_query": query,
                        "skip_search_enabled": False,
                        "is_nav_suggestions_disabled": False,
                        "always_search_override": False,
                        "override_no_search": False,
                        "comet_max_assistant_enabled": False,
                        "version": "2.18"
                    },
                    "query_str": query
                }
            else:
                data = {
                    "params": {
                        "last_backend_uuid": None,
                        "read_write_token": None,
                        "attachments": [],
                        "language": "en-US",
                        "timezone": "America/New_York",
                        "search_focus": "internet",
                        "sources": [
                            "web"
                        ],
                        "frontend_uuid": conversation.frontend_uid,
                        "mode": "copilot",
                        "model_preference": model,
                        "is_related_query": False,
                        "is_sponsored": False,
                        "visitor_id": conversation.visitor_id,
                        "user_nextauth_id": conversation.user_id,
                        "prompt_source": "user",
                        "query_source": "followup",
                        "is_incognito": False,
                        "time_from_first_type": random.randint(0, 1000),
                        "local_search_enabled": False,
                        "use_schematized_api": True,
                        "send_back_text_in_streaming_api": False,
                        "supported_block_use_cases": [
                            "answer_modes",
                            "media_items",
                            "knowledge_cards",
                            "inline_entity_cards",
                            "place_widgets",
                            "finance_widgets",
                            "sports_widgets",
                            "shopping_widgets",
                            "jobs_widgets",
                            "search_result_widgets",
                            "clarification_responses",
                            "inline_images",
                            "inline_assets",
                            "inline_finance_widgets",
                            "placeholder_cards",
                            "diff_blocks",
                            "inline_knowledge_cards",
                            "entity_group_v2",
                            "refinement_filters",
                            "canvas_mode"
                        ],
                        "client_coordinates": None,
                        "mentions": [],
                        "skip_search_enabled": True,
                        "is_nav_suggestions_disabled": False,
                        "followup_source": "link",
                        "always_search_override": False,
                        "override_no_search": False,
                        "comet_max_assistant_enabled": False,
                        "version": "2.18"
                    },
                    "query_str": query
                }
            yield JsonRequest.from_dict(data)
            async with session.post(
                f"{cls.url}/rest/sse/perplexity_ask",
                json=data,
            ) as response:
                await raise_for_status(response)
                full_response = ""
                full_reasoning = ""
                async for json_data in sse_stream(response):
                    yield JsonResponse.from_dict(json_data)
                    for block in json_data.get("blocks", []):
                        if block.get("intended_usage") == "sources_answer_mode":
                            yield Sources(block.get("sources_mode_block", {}).get("web_results", []))
                            continue
                        if block.get("intended_usage") == "media_items":
                            yield PreviewResponse([
                                ImageResponse(item.get("url"), item.get("name"), {
                                    "height": item.get("image_height"),
                                    "width": item.get("image_width"),
                                    **item
                                }) if item.get("medium") == "image" else YouTubeResponse(item.get("url").split("=").pop())
                                for item in block.get("media_block", {}).get("media_items", [])
                            ])
                            continue
                        for patch in block.get("diff_block", {}).get("patches", []):
                            if patch.get("path") == "/progress":
                                continue
                            value = patch.get("value", "")
                            if isinstance(value, dict) and "chunks" in value:
                                value = "".join(value.get("chunks", []))
                            if patch.get("path").startswith("/goals"):
                                if isinstance(value, str):
                                    if value.startswith(full_reasoning):
                                        value = value[len(full_reasoning):]
                                    yield Reasoning(value)
                                    full_reasoning += value
                                else:
                                    yield Reasoning(status="")
                                continue
                            if block.get("diff_block").get("field") != "markdown_block":
                                continue
                            value = value.get("answer", "") if isinstance(value, dict) else value
                            if value and isinstance(value, str):
                                if value.startswith(full_response):
                                    value = value[len(full_response):]
                                elif full_response.endswith(value):
                                    value = ""
                                if value:
                                    full_response += value
                                    yield value
                    if "related_query_items" in json_data:
                        followups = []
                        for item in json_data["related_query_items"]:
                            followups.append(item.get("text", ""))
                        yield SuggestedFollowups(followups)
