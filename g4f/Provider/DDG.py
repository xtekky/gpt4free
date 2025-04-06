from __future__ import annotations

import time
from aiohttp import ClientSession, ClientTimeout
import json
import asyncio
import random
import base64
import hashlib
from yarl import URL

from ..typing import AsyncResult, Messages, Cookies
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt, get_last_user_message
from ..providers.response import FinishReason, JsonConversation
from ..errors import ModelNotSupportedError, ResponseStatusError, RateLimitError, TimeoutError, ConversationLimitError

try:
    from bs4 import BeautifulSoup
    has_bs4 = True
except ImportError:
    has_bs4 = False


class DuckDuckGoSearchException(Exception):
    """Base exception class for duckduckgo_search."""

class DuckDuckGoChallengeError(ResponseStatusError):
    """Raised when DuckDuckGo presents a challenge that needs to be solved."""

class Conversation(JsonConversation):
    vqd: str = None
    vqd_hash_1: str = None
    message_history: Messages = []
    cookies: dict = {}
    fe_version: str = None

    def __init__(self, model: str):
        self.model = model

class DDG(AsyncGeneratorProvider, ProviderModelMixin):
    label = "DuckDuckGo AI Chat"
    url = "https://duckduckgo.com/aichat"
    api_endpoint = "https://duckduckgo.com/duckchat/v1/chat"
    status_url = "https://duckduckgo.com/duckchat/v1/status"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "gpt-4o-mini"
    
    # Model mapping from user-friendly names to API model names
    _chat_models = {
        "gpt-4": default_model,
        "gpt-4o-mini": default_model,
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "o3-mini": "o3-mini",
        "mixtral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    }
    
    # Available models (user-friendly names)
    models = list(_chat_models.keys())

    last_request_time = 0
    max_retries = 3
    base_delay = 2
    
    # Class variable to store the x-fe-version across instances
    _chat_xfe = ""
    
    @staticmethod
    def sha256_base64(text: str) -> str:
        """Return the base64 encoding of the SHA256 digest of the text."""
        sha256_hash = hashlib.sha256(text.encode("utf-8")).digest()
        return base64.b64encode(sha256_hash).decode()

    @staticmethod
    def parse_dom_fingerprint(js_text: str) -> str:
        if not has_bs4:
            # Fallback if BeautifulSoup is not available
            return "1000"
        
        try:
            html_snippet = js_text.split("e.innerHTML = '")[1].split("';")[0]
            offset_value = js_text.split("return String(")[1].split(" ")[0]
            soup = BeautifulSoup(html_snippet, "html.parser")
            corrected_inner_html = soup.body.decode_contents()
            inner_html_length = len(corrected_inner_html)
            fingerprint = int(offset_value) + inner_html_length
            return str(fingerprint)
        except Exception:
            # Return a fallback value if parsing fails
            return "1000"

    @staticmethod
    def parse_server_hashes(js_text: str) -> list:
        try:
            return js_text.split('server_hashes: ["', maxsplit=1)[1].split('"]', maxsplit=1)[0].split('","')
        except Exception:
            # Return a fallback value if parsing fails
            return ["1", "2"]

    @classmethod
    def build_x_vqd_hash_1(cls, vqd_hash_1: str, headers: dict) -> str:
        """Build the x-vqd-hash-1 header value."""
        try:
            # If we received a valid base64 string, try to decode it
            if vqd_hash_1 and len(vqd_hash_1) > 20:
                try:
                    # Try to decode and parse as JSON first
                    decoded_json = json.loads(base64.b64decode(vqd_hash_1).decode())
                    # If it's already a complete structure with meta, return it as is
                    if isinstance(decoded_json, dict) and "meta" in decoded_json:
                        return vqd_hash_1
                    
                    # Otherwise, extract what we can from it
                    if isinstance(decoded_json, dict) and "server_hashes" in decoded_json:
                        server_hashes = decoded_json.get("server_hashes", ["1", "2"])
                    else:
                        # Fall back to parsing from string
                        decoded = base64.b64decode(vqd_hash_1).decode()
                        server_hashes = cls.parse_server_hashes(decoded)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # If it's not valid JSON, try to parse it as a string
                    decoded = base64.b64decode(vqd_hash_1).decode()
                    server_hashes = cls.parse_server_hashes(decoded)
            else:
                # Default server hashes if we can't extract them
                server_hashes = ["1", "2"]
            
            # Generate fingerprints
            dom_fingerprint = "1000"  # Default value
            ua_fingerprint = headers.get("User-Agent", "") + headers.get("sec-ch-ua", "")
            ua_hash = cls.sha256_base64(ua_fingerprint)
            dom_hash = cls.sha256_base64(dom_fingerprint)
            
            # Create a challenge ID (random hex string)
            challenge_id = ''.join(random.choice('0123456789abcdef') for _ in range(40)) + 'h8jbt'
            
            # Build the complete structure including meta
            final_result = {
                "server_hashes": server_hashes,
                "client_hashes": [ua_hash, dom_hash],
                "signals": {},
                "meta": {
                    "v": "1",
                    "challenge_id": challenge_id,
                    "origin": "https://duckduckgo.com",
                    "stack": "Error\nat ke (https://duckduckgo.com/dist/wpm.chat.js:1:29526)\nat async dispatchServiceInitialVQD (https://duckduckgo.com/dist/wpm.chat.js:1:45076)"
                }
            }
            
            base64_final_result = base64.b64encode(json.dumps(final_result).encode()).decode()
            return base64_final_result
        except Exception as e:
            # If anything fails, return an empty string
            return ""

    @classmethod 
    def validate_model(cls, model: str) -> str:
        """Validates and returns the correct model name for the API"""
        if not model:
            return cls.default_model
            
        # Check aliases first
        if model in cls.model_aliases:
            model = cls.model_aliases[model]
            
        # Check if it's a valid model name
        if model not in cls.models:
            raise ModelNotSupportedError(f"Model {model} not supported. Available models: {cls.models}")
            
        return model

    @classmethod
    async def sleep(cls, multiplier=1.0):
        """Implements rate limiting between requests"""
        now = time.time()
        if cls.last_request_time > 0:
            delay = max(0.0, 1.5 - (now - cls.last_request_time)) * multiplier
            if delay > 0:
                await asyncio.sleep(delay)
        cls.last_request_time = time.time()

    @classmethod
    async def get_default_cookies(cls, session: ClientSession) -> dict:
        """Obtains default cookies needed for API requests"""
        try:
            await cls.sleep()
            # Make initial request to get cookies
            async with session.get(cls.url) as response:
                # Set the required cookies
                cookies = {}
                cookies_dict = {'dcs': '1', 'dcm': '3'}
                
                # Add any cookies from the response
                for cookie in response.cookies.values():
                    cookies[cookie.key] = cookie.value
                
                # Ensure our required cookies are set
                for name, value in cookies_dict.items():
                    cookies[name] = value
                    url_obj = URL(cls.url)
                    session.cookie_jar.update_cookies({name: value}, url_obj)
                
                # Make a second request to the status endpoint to get any additional cookies
                headers = {
                    "accept": "text/event-stream",
                    "accept-language": "en",
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
                    "origin": "https://duckduckgo.com",
                    "referer": "https://duckduckgo.com/",
                }
                
                await cls.sleep()
                async with session.get(cls.status_url, headers=headers) as status_response:
                    # Add any cookies from the status response
                    for cookie in status_response.cookies.values():
                        cookies[cookie.key] = cookie.value
                        url_obj = URL(cls.url)
                        session.cookie_jar.update_cookies({cookie.key: cookie.value}, url_obj)
                
                return cookies
        except Exception as e:
            # Return at least the required cookies on error
            cookies = {'dcs': '1', 'dcm': '3'}
            url_obj = URL(cls.url)
            for name, value in cookies.items():
                session.cookie_jar.update_cookies({name: value}, url_obj)
            return cookies
    
    @classmethod
    async def fetch_fe_version(cls, session: ClientSession) -> str:
        """Fetches the fe-version from the initial page load."""
        if cls._chat_xfe:
            return cls._chat_xfe
            
        try:
            url = "https://duckduckgo.com/?q=DuckDuckGo+AI+Chat&ia=chat&duckai=1"
            await cls.sleep()
            async with session.get(url) as response:
                await raise_for_status(response)
                content = await response.text()
                
                # Extract x-fe-version components
                try:
                    # Try to extract the version components
                    xfe1 = content.split('__DDG_BE_VERSION__="', 1)[1].split('"', 1)[0]
                    xfe2 = content.split('__DDG_FE_CHAT_HASH__="', 1)[1].split('"', 1)[0]
                    
                    # Format it like "serp_YYYYMMDD_HHMMSS_ET-hash"
                    from datetime import datetime
                    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cls._chat_xfe = f"serp_{current_date}_ET-{xfe2}"
                    
                    return cls._chat_xfe
                except Exception:
                    # Fallback to a default format if extraction fails
                    from datetime import datetime
                    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cls._chat_xfe = f"serp_{current_date}_ET-78c2e87e3d286691cc21"
                    return cls._chat_xfe
        except Exception:
            # Fallback to a default format if request fails
            from datetime import datetime
            current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            cls._chat_xfe = f"serp_{current_date}_ET-78c2e87e3d286691cc21"
            return cls._chat_xfe

    @classmethod
    async def fetch_vqd_and_hash(cls, session: ClientSession, retry_count: int = 0) -> tuple[str, str]:
        """Fetches the required VQD token and hash for the chat session with retries."""
        headers = {
            "accept": "text/event-stream",
            "accept-language": "en",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "origin": "https://duckduckgo.com",
            "referer": "https://duckduckgo.com/",
            "x-vqd-accept": "1",
        }

        # Make sure we have cookies first
        if len(session.cookie_jar) == 0:
            await cls.get_default_cookies(session)

        try:
            await cls.sleep(multiplier=1.0 + retry_count * 0.5)
            async with session.get(cls.status_url, headers=headers) as response:
                await raise_for_status(response)
                
                vqd = response.headers.get("x-vqd-4", "")
                vqd_hash_1 = response.headers.get("x-vqd-hash-1", "")
                
                if vqd:
                    # Return the fetched vqd and vqd_hash_1
                    return vqd, vqd_hash_1
                
                response_text = await response.text()
                raise RuntimeError(f"Failed to fetch VQD token and hash: {response.status} {response_text}")
                
        except Exception as e:
            if retry_count < cls.max_retries:
                wait_time = cls.base_delay * (2 ** retry_count) * (1 + random.random())
                await asyncio.sleep(wait_time)
                return await cls.fetch_vqd_and_hash(session, retry_count + 1)
            else:
                raise RuntimeError(f"Failed to fetch VQD token and hash after {cls.max_retries} attempts: {str(e)}")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 60,
        cookies: Cookies = None,
        conversation: Conversation = None,
        return_conversation: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.validate_model(model)
        retry_count = 0

        while retry_count <= cls.max_retries:
            try:
                session_timeout = ClientTimeout(total=timeout)
                async with ClientSession(timeout=session_timeout, cookies=cookies) as session:
                    # Step 1: Ensure we have the fe_version
                    if not cls._chat_xfe:
                        cls._chat_xfe = await cls.fetch_fe_version(session)
                    
                    # Step 2: Initialize or update conversation
                    if conversation is None:
                        # Get initial cookies if not provided
                        if not cookies:
                            await cls.get_default_cookies(session)
                        
                        # Create a new conversation
                        conversation = Conversation(model)
                        conversation.fe_version = cls._chat_xfe
                        
                        # Step 3: Get VQD tokens
                        vqd, vqd_hash_1 = await cls.fetch_vqd_and_hash(session)
                        conversation.vqd = vqd
                        conversation.vqd_hash_1 = vqd_hash_1
                        conversation.message_history = [{"role": "user", "content": format_prompt(messages)}]
                    else:
                        # Update existing conversation with new message
                        last_message = get_last_user_message(messages.copy())
                        conversation.message_history.append({"role": "user", "content": last_message})
                    
                    # Step 4: Prepare headers with proper x-vqd-hash-1
                    headers = {
                        "accept": "text/event-stream",
                        "accept-language": "en",
                        "cache-control": "no-cache",
                        "content-type": "application/json",
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
                        "origin": "https://duckduckgo.com",
                        "referer": "https://duckduckgo.com/",
                        "pragma": "no-cache",
                        "priority": "u=1, i",
                        "sec-ch-ua": '"Not:A-Brand";v="24", "Chromium";v="134"',
                        "sec-ch-ua-mobile": "?0",
                        "sec-ch-ua-platform": '"Linux"',
                        "sec-fetch-dest": "empty",
                        "sec-fetch-mode": "cors",
                        "sec-fetch-site": "same-origin",
                        "x-fe-version": conversation.fe_version or cls._chat_xfe,
                        "x-vqd-4": conversation.vqd,
                    }
                    
                    # For the first request, send an empty x-vqd-hash-1 header
                    # This matches the behavior in the duckduckgo_search module
                    headers["x-vqd-hash-1"] = ""

                    # Step 5: Prepare the request data
                    # Convert the user-friendly model name to the API model name
                    api_model = cls._chat_models.get(model, model)
                    
                    data = {
                        "model": api_model,
                        "messages": conversation.message_history,
                    }

                    # Step 6: Send the request
                    await cls.sleep(multiplier=1.0 + retry_count * 0.5)
                    async with session.post(cls.api_endpoint, json=data, headers=headers, proxy=proxy) as response:
                        # Handle 429 and 418 errors specifically
                        if response.status == 429:
                            response_text = await response.text()
                            
                            if retry_count < cls.max_retries:
                                retry_count += 1
                                wait_time = cls.base_delay * (2 ** retry_count) * (1 + random.random())
                                await asyncio.sleep(wait_time)
                                
                                # Get fresh tokens and cookies
                                cookies = await cls.get_default_cookies(session)
                                continue
                            else:
                                raise RateLimitError(f"Rate limited after {cls.max_retries} retries")
                        elif response.status == 418:
                            # Check if it's a challenge error
                            try:
                                response_text = await response.text()
                                try:
                                    response_json = json.loads(response_text)
                                    
                                    # Extract challenge data if available
                                    challenge_data = None
                                    if response_json.get("type") == "ERR_CHALLENGE" and "cd" in response_json:
                                        challenge_data = response_json["cd"]
                                    
                                    if retry_count < cls.max_retries:
                                        retry_count += 1
                                        wait_time = cls.base_delay * (2 ** retry_count) * (1 + random.random())
                                        await asyncio.sleep(wait_time)
                                        
                                        # Reset tokens and try again with fresh session
                                        conversation = None
                                        cls._chat_xfe = ""
                                        
                                        # Get fresh cookies
                                        cookies = await cls.get_default_cookies(session)
                                        
                                        # If we have challenge data, try to use it
                                        if challenge_data and isinstance(challenge_data, dict):
                                            # Extract any useful information from challenge data
                                            # This could be used to build a better response in the future
                                            pass
                                            
                                        continue
                                    else:
                                        raise DuckDuckGoChallengeError(f"Challenge error after {cls.max_retries} retries")
                                except json.JSONDecodeError:
                                    # If we can't parse the JSON, assume it's a challenge error anyway
                                    if retry_count < cls.max_retries:
                                        retry_count += 1
                                        wait_time = cls.base_delay * (2 ** retry_count) * (1 + random.random())
                                        await asyncio.sleep(wait_time)
                                        
                                        # Reset tokens and try again with fresh session
                                        conversation = None
                                        cls._chat_xfe = ""
                                        cookies = await cls.get_default_cookies(session)
                                        continue
                                    else:
                                        raise DuckDuckGoChallengeError(f"Challenge error after {cls.max_retries} retries")
                            except Exception as e:
                                # If any other error occurs during handling, still try to recover
                                if retry_count < cls.max_retries:
                                    retry_count += 1
                                    wait_time = cls.base_delay * (2 ** retry_count) * (1 + random.random())
                                    await asyncio.sleep(wait_time)
                                    
                                    # Reset tokens and try again with fresh session
                                    conversation = None
                                    cls._chat_xfe = ""
                                    cookies = await cls.get_default_cookies(session)
                                    continue
                                else:
                                    raise DuckDuckGoChallengeError(f"Challenge error after {cls.max_retries} retries: {str(e)}")
                        
                        # For other status codes, use the standard error handler
                        await raise_for_status(response)
                        reason = None
                        full_message = ""

                        # Step 7: Process the streaming response
                        async for line in response.content:
                            line = line.decode("utf-8").strip()

                            if line.startswith("data:"):
                                try:
                                    message = json.loads(line[5:].strip())
                                except json.JSONDecodeError:
                                    continue

                                if "action" in message and message["action"] == "error":
                                    error_type = message.get("type", "")
                                    if message.get("status") == 429:
                                        if error_type == "ERR_CONVERSATION_LIMIT":
                                            raise ConversationLimitError(error_type)
                                        raise RateLimitError(error_type)
                                    elif message.get("status") == 418 and error_type == "ERR_CHALLENGE":
                                        # Handle challenge error by refreshing tokens and retrying
                                        if retry_count < cls.max_retries:
                                            # Don't raise here, let the outer exception handler retry
                                            raise DuckDuckGoChallengeError(f"Challenge detected: {error_type}")
                                    raise DuckDuckGoSearchException(error_type)

                                if "message" in message:
                                    if message["message"]:
                                        yield message["message"]
                                        full_message += message["message"]
                                        reason = "length"
                                    else:
                                        reason = "stop"

                        # Step 8: Update conversation with response information
                        # Always update the VQD tokens from the response headers
                        conversation.vqd = response.headers.get("x-vqd-4", conversation.vqd)
                        conversation.vqd_hash_1 = response.headers.get("x-vqd-hash-1", conversation.vqd_hash_1)
                        
                        # Update cookies
                        conversation.cookies = {
                            n: c.value 
                            for n, c in session.cookie_jar.filter_cookies(URL(cls.url)).items()
                        }
                        
                        # If requested, return the updated conversation
                        if return_conversation:
                            conversation.message_history.append({"role": "assistant", "content": full_message})
                            yield conversation

                        if reason is not None:
                            yield FinishReason(reason)
                        
                        # If we got here, the request was successful
                        break

            except (RateLimitError, ResponseStatusError, DuckDuckGoChallengeError) as e:
                if ("429" in str(e) or isinstance(e, DuckDuckGoChallengeError)) and retry_count < cls.max_retries:
                    retry_count += 1
                    wait_time = cls.base_delay * (2 ** retry_count) * (1 + random.random())
                    await asyncio.sleep(wait_time)
                    
                    # For challenge errors, refresh tokens and cookies
                    if isinstance(e, DuckDuckGoChallengeError):
                        # Reset conversation to force new token acquisition
                        conversation = None
                        # Clear class cache to force refresh
                        cls._chat_xfe = ""
                else:
                    raise
            except asyncio.TimeoutError as e:
                raise TimeoutError(f"Request timed out: {str(e)}")
            except Exception as e:
                raise
