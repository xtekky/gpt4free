from __future__ import annotations

import os
import uuid
import json
from typing import AsyncIterator

from ..typing import AsyncResult, Messages, Cookies
from ..requests import StreamSession, raise_for_status, sse_stream
from ..cookies import get_cookies, get_cookies_dir
from ..providers.response import (
    ProviderInfo, JsonConversation, JsonRequest, JsonResponse, 
    Reasoning, Sources, SuggestedFollowups, ImageResponse, 
    PreviewResponse, YouTubeResponse
)
from ..providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .. import debug

# Perplexity API endpoints
PERPLEXITY_URL = "https://www.perplexity.ai"
PERPLEXITY_DOMAIN = ".perplexity.ai"
AUTH_ENDPOINT = f"{PERPLEXITY_URL}/api/auth/session"
QUERY_ENDPOINT = f"{PERPLEXITY_URL}/rest/sse/perplexity_ask"


def get_har_files():
    """Get list of Perplexity HAR files from har_and_cookies directory."""
    if not os.access(get_cookies_dir(), os.R_OK):
        return []
    
    har_files = []
    for root, _, files in os.walk(get_cookies_dir()):
        for file in files:
            # Look for Perplexity HAR files
            if file.endswith(".har") and "perplexity" in file.lower():
                har_files.append(os.path.join(root, file))
    
    # Sort by modification time, newest first
    har_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return har_files


def read_perplexity_har():
    """
    Read Perplexity HAR file to extract cookies.
    Returns: cookies dict or None if not found
    """
    har_files = get_har_files()
    
    if not har_files:
        debug.log("PerplexityEmulated: No Perplexity HAR files found in har_and_cookies/")
        return None
    
    for har_path in har_files:
        debug.log(f"PerplexityEmulated: Reading HAR file: {har_path}")
        try:
            with open(har_path, 'r', encoding='utf-8') as f:
                har_data = json.load(f)
            
            # Look for Perplexity requests
            for entry in har_data.get('log', {}).get('entries', []):
                url = entry.get('request', {}).get('url', '')
                
                # We want requests to Perplexity API
                if 'perplexity.ai' in url.lower():
                    # Extract cookies
                    cookies = {}
                    for c in entry.get('request', {}).get('cookies', []):
                        cookies[c.get('name')] = c.get('value')
                    
                    if cookies:
                        debug.log(f"PerplexityEmulated: Found {len(cookies)} cookies in HAR")
                        return cookies
            
            debug.log(f"PerplexityEmulated: No Perplexity requests found in {har_path}")
        
        except json.JSONDecodeError as e:
            debug.log(f"PerplexityEmulated: Failed to parse HAR file {har_path}: {e}")
        except Exception as e:
            debug.error(f"PerplexityEmulated: Error reading HAR file {har_path}: {e}")
    
    debug.log("PerplexityEmulated: No valid Perplexity cookies found in any HAR file")
    return None


class PerplexityEmulated(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Perplexity provider using browser emulation with HAR file support.
    
    This provider extends the base Perplexity implementation with HAR file support
    for easier authentication management. It uses curl_cffi's Chrome impersonation
    for realistic browser-like requests.
    """
    
    label = "Perplexity (Emulated)"
    url = PERPLEXITY_URL
    cookie_domain = PERPLEXITY_DOMAIN
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
        proxy: str = None,
        conversation: JsonConversation = None,
        **kwargs
    ) -> AsyncResult:
        """
        Create async generator for Perplexity requests with HAR file support.
        
        Authentication priority:
        1. HAR file cookies (har_and_cookies/perplexity*.har)
        2. Cookie jar from get_cookies()
        """
        if not model:
            model = cls.default_model
        
        # Try to get cookies from HAR file first
        if cookies is None:
            cookies = read_perplexity_har()
            if cookies:
                debug.log(f"PerplexityEmulated: Using {len(cookies)} cookies from HAR file")
            else:
                # Fall back to cookie jar
                cookies = get_cookies(cls.cookie_domain, False)
                if cookies:
                    debug.log(f"PerplexityEmulated: Using {len(cookies)} cookies from cookie jar")
                else:
                    debug.log("PerplexityEmulated: No cookies found")
        
        # Initialize conversation if needed
        if conversation is None:
            conversation = JsonConversation(
                frontend_uid=str(uuid.uuid4()),
                frontend_context_uuid=str(uuid.uuid4()),
                visitor_id=str(uuid.uuid4()),
                user_id=None,
                thread_url_slug=None,  # For conversation continuity via Referer header
            )
        
        request_id = str(uuid.uuid4())
        
        # Set referer based on thread_url_slug for conversation continuity
        referer = f"{cls.url}/"
        if hasattr(conversation, 'thread_url_slug') and conversation.thread_url_slug:
            referer = f"{cls.url}/search/{conversation.thread_url_slug}"
            debug.log(f"PerplexityEmulated: Using conversation referer: {referer}")
        
        headers = {
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": referer,
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
            "x-perplexity-request-reason": "perplexity-query-state-provider",
            "x-request-id": request_id,
        }
        
        # Extract query from messages
        query = ""
        for message in reversed(messages):
            if message["role"] == "user":
                query = message["content"]
                break
        
        # Use StreamSession with Chrome impersonation
        async with StreamSession(headers=headers, cookies=cookies, proxy=proxy, impersonate="chrome") as session:
            # Get user info if needed
            if conversation.user_id is None:
                try:
                    async with session.get(f"{cls.url}/api/auth/session") as response:
                        await raise_for_status(response)
                        user = await response.json()
                        conversation.user_id = user.get("user", {}).get("id")
                        debug.log(f"PerplexityEmulated: User ID: {conversation.user_id}")
                except Exception as e:
                    debug.error(f"PerplexityEmulated: Failed to get user info: {e}")
            
            yield conversation
            
            # Determine model
            if model == "auto" or model == "perplexity":
                model = "pplx_pro" if conversation.user_id else "turbo"
            
            yield ProviderInfo(**cls.get_dict(), model=model)
            
            if model in cls.model_aliases:
                model = cls.model_aliases[model]
            
            # Build request data (same as original Perplexity)
            # Check if this is a followup request (has session tokens)
            is_followup = hasattr(conversation, 'last_backend_uuid') and conversation.last_backend_uuid
            
            # Generate new frontend_uuid for followup requests (browser does this)
            if is_followup:
                conversation.frontend_uid = str(uuid.uuid4())
            
            if not is_followup:
                data = {
                    "params": {
                        "attachments": [],
                        "language": "en-US",
                        "timezone": "America/Los_Angeles",
                        "search_focus": "internet",
                        "sources": ["web"],
                        "search_recency_filter": None,
                        "frontend_uuid": conversation.frontend_uid,
                        "mode": "concise",
                        "model_preference": model,
                        "is_related_query": False,
                        "is_sponsored": False,
                        "frontend_context_uuid": conversation.frontend_context_uuid,
                        "prompt_source": "user",
                        "query_source": "home",
                        "is_incognito": False,
                        "time_from_first_type": 15872,
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
                            "prediction_market_widgets",
                            "sports_widgets",
                            "flight_status_widgets",
                            "news_widgets",
                            "shopping_widgets",
                            "jobs_widgets",
                            "search_result_widgets",
                            "inline_images",
                            "inline_assets",
                            "placeholder_cards",
                            "diff_blocks",
                            "inline_knowledge_cards",
                            "entity_group_v2",
                            "refinement_filters",
                            "canvas_mode",
                            "maps_preview",
                            "answer_tabs",
                            "price_comparison_widgets",
                            "preserve_latex",
                            "generic_onboarding_widgets",
                            "in_context_suggestions",
                            "inline_claims"
                        ],
                        "client_coordinates": None,
                        "mentions": [],
                        "dsl_query": query,
                        "skip_search_enabled": True,
                        "is_nav_suggestions_disabled": False,
                        "source": "default",
                        "always_search_override": False,
                        "override_no_search": False,
                        "should_ask_for_mcp_tool_confirmation": True,
                        "browser_agent_allow_once_from_toggle": False,
                        "force_enable_browser_agent": False,
                        "supported_features": [
                            "browser_agent_permission_banner_v1.1"
                        ],
                        "version": "2.18"
                    },
                    "query_str": query
                }
            else:
                import random
                data = {
                    "params": {
                        "last_backend_uuid": getattr(conversation, 'last_backend_uuid', None),
                        "read_write_token": getattr(conversation, 'read_write_token', None),
                        "attachments": [],
                        "language": "en-US",
                        "timezone": "America/Los_Angeles",
                        "search_focus": "internet",
                        "sources": ["web"],
                        "search_recency_filter": None,
                        "frontend_uuid": conversation.frontend_uid,  # New UUID for followup
                        "mode": "concise",
                        "model_preference": model,
                        "is_related_query": False,
                        "is_sponsored": False,
                        "prompt_source": "user",
                        "query_source": "followup",
                        "followup_source": "link",  # Critical for conversation continuity
                        "is_incognito": False,
                        "time_from_first_type": 5106,  # Use HAR value for consistency
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
                            "prediction_market_widgets",
                            "sports_widgets",
                            "flight_status_widgets",
                            "news_widgets",
                            "shopping_widgets",
                            "jobs_widgets",
                            "search_result_widgets",
                            "inline_images",
                            "inline_assets",
                            "placeholder_cards",
                            "diff_blocks",
                            "inline_knowledge_cards",
                            "entity_group_v2",
                            "refinement_filters",
                            "canvas_mode",
                            "maps_preview",
                            "answer_tabs",
                            "price_comparison_widgets",
                            "preserve_latex",
                            "generic_onboarding_widgets",
                            "in_context_suggestions",
                            "inline_claims"
                        ],
                        "client_coordinates": None,
                        "mentions": [],
                        "dsl_query": query,
                        "skip_search_enabled": True,
                        "is_nav_suggestions_disabled": False,
                        "source": "default",
                        "always_search_override": False,
                        "override_no_search": False,
                        "should_ask_for_mcp_tool_confirmation": True,
                        "force_enable_browser_agent": False,
                        "supported_features": [
                            "browser_agent_permission_banner_v1.1"
                        ],
                        "version": "2.18"
                    },
                    "query_str": query
                }
            
            yield JsonRequest.from_dict(data)
            
            # Log full request data for debugging
            debug.log(f"PerplexityEmulated: Request data: {json.dumps(data, indent=2, default=str)[:1000]}")
            
            # Send request
            debug.log(f"PerplexityEmulated: Sending request to {QUERY_ENDPOINT}")
            
            async with session.post(QUERY_ENDPOINT, json=data) as response:
                # Process SSE stream
                debug.log(f"PerplexityEmulated: Processing response...")
                await raise_for_status(response)
                
                full_response = ""
                full_reasoning = ""
                
                async for json_data in sse_stream(response):
                    yield JsonResponse.from_dict(json_data)
                    
                    # Capture session tokens for conversation continuity
                    # Note: The 'backend_uuid' field in responses is the backend UUID we need for followups
                    if 'backend_uuid' in json_data:
                        conversation.last_backend_uuid = json_data['backend_uuid']
                    
                    # Only capture read_write_token if we don't have one yet (like a session cookie)
                    if 'read_write_token' in json_data and not hasattr(conversation, 'read_write_token'):
                        conversation.read_write_token = json_data['read_write_token']
                    
                    # Capture thread_url_slug for conversation continuity via Referer header
                    if 'thread_url_slug' in json_data and (not hasattr(conversation, 'thread_url_slug') or not conversation.thread_url_slug):
                        conversation.thread_url_slug = json_data.get('thread_url_slug')
                    
                    for block in json_data.get("blocks", []):
                        # Handle sources
                        if block.get("intended_usage") == "sources_answer_mode":
                            yield Sources(block.get("sources_mode_block", {}).get("web_results", []))
                            continue
                        
                        # Handle media items
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
                        
                        # Handle response text
                        for patch in block.get("diff_block", {}).get("patches", []):
                            if patch.get("path") == "/progress":
                                continue
                            
                            value = patch.get("value", "")
                            
                            # Handle reasoning
                            if isinstance(value, dict) and "chunks" in value:
                                value = "".join(value.get("chunks", []))
                            
                            if patch.get("path").startswith("/goals"):
                                if isinstance(value, str):
                                    if value.startswith(full_reasoning):
                                        value = value[len(full_reasoning):]
                                    if value:
                                        yield Reasoning(value)
                                        full_reasoning += value
                                else:
                                    yield Reasoning(status="")
                                continue
                            
                            # Handle regular response
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
                    
                    # Handle follow-ups
                    if "related_query_items" in json_data:
                        followups = []
                        for item in json_data["related_query_items"]:
                            followups.append(item.get("text", ""))
                        yield SuggestedFollowups(followups)
                
                debug.log("PerplexityEmulated: Request completed successfully")
                debug.log(f"PerplexityEmulated: last_backend_uuid={getattr(conversation, 'last_backend_uuid', None)}, read_write_token={getattr(conversation, 'read_write_token', None)}")
