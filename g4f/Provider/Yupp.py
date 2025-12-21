import asyncio
import hashlib
import json
import os
import re
import time
import uuid

import aiohttp

from .helper import get_last_user_message
from .yupp.models import YuppModelManager, ModelProcessor
from ..cookies import get_cookies
from ..debug import log
from ..errors import RateLimitError, ProviderException, MissingAuthError
from ..image import is_accepted_format, to_bytes
from ..providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..providers.response import Reasoning, PlainTextResponse, PreviewResponse, JsonConversation, ImageResponse, \
    ProviderInfo, FinishReason, JsonResponse
from ..requests.aiohttp import StreamSession
from ..tools.auth import AuthManager
from ..tools.media import merge_media
from ..typing import AsyncResult, Messages, Optional, Dict, Any, List

# Global variables to manage Yupp accounts
YUPP_ACCOUNT = Dict[str, Any]
YUPP_ACCOUNTS: List[YUPP_ACCOUNT] = []
account_rotation_lock = asyncio.Lock()

# Global variables to manage Yupp Image Cache
ImagesCache: Dict[str, dict] = {}


class YuppAccount:
    """Yupp account representation"""

    def __init__(self, token: str, is_valid: bool = True, error_count: int = 0, last_used: float = 0):
        self.token = token
        self.is_valid = is_valid
        self.error_count = error_count
        self.last_used = last_used


def load_yupp_accounts(tokens_str: str):
    """Load Yupp accounts from token string"""
    global YUPP_ACCOUNTS
    if not tokens_str:
        return
    tokens = [token.strip() for token in tokens_str.split(',') if token.strip()]
    YUPP_ACCOUNTS = [
        {
            "token": token,
            "is_valid": True,
            "error_count": 0,
            "last_used": 0.0
        }
        for token in tokens
    ]


def create_headers() -> Dict[str, str]:
    """Create headers for requests"""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
        "Accept": "text/x-component, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }


async def get_best_yupp_account() -> Optional[YUPP_ACCOUNT]:
    """Get the best available Yupp account using smart selection algorithm"""
    max_error_count = int(os.getenv("MAX_ERROR_COUNT", "3"))
    error_cooldown = int(os.getenv("ERROR_COOLDOWN", "300"))

    async with account_rotation_lock:
        now = time.time()
        valid_accounts = [
            acc
            for acc in YUPP_ACCOUNTS
            if acc["is_valid"]
               and (
                       acc["error_count"] < max_error_count
                       or now - acc["last_used"] > error_cooldown
               )
        ]

        if not valid_accounts:
            return None

        # Reset error count for accounts in cooldown
        for acc in valid_accounts:
            if (
                    acc["error_count"] >= max_error_count
                    and now - acc["last_used"] > error_cooldown
            ):
                acc["error_count"] = 0

        # Sort by last used and error count
        valid_accounts.sort(key=lambda x: (x["last_used"], x["error_count"]))
        account = valid_accounts[0]
        account["last_used"] = now
        return account


async def claim_yupp_reward(session: aiohttp.ClientSession, account: YUPP_ACCOUNT, reward_id: str):
    """Claim Yupp reward asynchronously"""
    try:
        log_debug(f"Claiming reward {reward_id}...")
        url = "https://yupp.ai/api/trpc/reward.claim?batch=1"
        payload = {"0": {"json": {"rewardId": reward_id}}}
        headers = {
            "Content-Type": "application/json",
            "Cookie": f"__Secure-yupp.session-token={account['token']}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",

        }
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            balance = data[0]["result"]["data"]["json"]["currentCreditBalance"]
            log_debug(f"Reward claimed successfully. New balance: {balance}")
            return balance
    except Exception as e:
        log_debug(f"Failed to claim reward {reward_id}. Error: {e}")
        return None


async def make_chat_private(session: aiohttp.ClientSession, account: YUPP_ACCOUNT, chat_id: str) -> bool:
    """Set a Yupp chat's sharing status to PRIVATE"""
    try:
        log_debug(f"Setting chat {chat_id} to PRIVATE...")
        url = "https://yupp.ai/api/trpc/chat.updateSharingSettings?batch=1"
        payload = {
            "0": {
                "json": {
                    "chatId": chat_id,
                    "status": "PRIVATE"
                }
            }
        }
        headers = {
            "Content-Type": "application/json",
            "Cookie": f"__Secure-yupp.session-token={account['token']}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",

        }

        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            if (
                    isinstance(data, list) and len(data) > 0
                    and "json" in data[0].get("result", {}).get("data", {})
            ):
                log_debug(f"Chat {chat_id} is now PRIVATE ✅")
                return True

            log_debug(f"Unexpected response while setting chat private: {data}")
            return False

    except Exception as e:
        log_debug(f"Failed to make chat {chat_id} private: {e}")
        return False


def log_debug(message: str):
    """Debug logging"""
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        print(f"[DEBUG] {message}")
    else:
        log(f"[Yupp] {message}")


def format_messages_for_yupp(messages: Messages) -> str:
    """Format multi-turn conversation for Yupp single-turn format"""
    if not messages:
        return ""

    if len(messages) == 1 and isinstance(messages[0].get("content"), str):
        return messages[0].get("content", "").strip()

    formatted = []

    # Handle system messages
    system_messages = [msg for msg in messages if msg.get("role") in ["developer", "system"]]
    if system_messages:
        for sys_msg in system_messages:
            content = sys_msg.get("content", "")
            formatted.append(content)

    # Handle user and assistant messages
    user_assistant_msgs = [msg for msg in messages if msg.get("role") in ["user", "assistant"]]
    for msg in user_assistant_msgs:
        role = "Human" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        for part in content if isinstance(content, list) else [{"text": content}]:
            if part.get("text", "").strip():
                formatted.append(f"\n\n{role}: {part.get('text', '')}")

    # Ensure it ends with Assistant: for the model to continue
    if not formatted or not formatted[-1].strip().startswith("Assistant:"):
        formatted.append("\n\nAssistant:")

    result = "".join(formatted)
    if result.startswith("\n\n"):
        result = result[2:]

    return result


class Yupp(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Yupp.ai Provider for g4f
    Uses multiple account rotation and smart error handling
    """

    url = "https://yupp.ai"
    login_url = "https://discord.gg/qXA4Wf4Fsm"
    working = True
    active_by_default = True
    supports_stream = True
    image_cache = True

    @classmethod
    def get_models(cls, api_key: str = None, **kwargs) -> List[str]:
        if not cls.models:
            if not api_key:
                api_key = AuthManager.load_api_key(cls)
            if not api_key:
                api_key = get_cookies("yupp.ai", False).get("__Secure-yupp.session-token")
            if not api_key:
                raise MissingAuthError("No Yupp accounts configured. Set YUPP_API_KEY environment variable.")
            manager = YuppModelManager(api_key=api_key)
            models = manager.client.fetch_models()
            if models:
                cls.models_tags = {model.get("name"): manager.processor.generate_tags(model) for model in models}
                cls.models = [model.get("name") for model in models]
                cls.image_models = [model.get("name") for model in models if model.get("isImageGeneration")]
                cls.vision_models = [model.get("name") for model in models if
                                     "image/*" in model.get("supportedAttachmentMimeTypes", [])]
        return cls.models

    @classmethod
    async def prepare_files(cls, media, session: aiohttp.ClientSession, account: YUPP_ACCOUNT) -> list:
        files = []
        if not media:
            return files
        for file, name in media:
            data = to_bytes(file)
            hasher = hashlib.md5()
            hasher.update(data)
            image_hash = hasher.hexdigest()
            file = ImagesCache.get(image_hash)
            if cls.image_cache and file:
                log_debug("Using cached image")
                files.append(file)
                continue
            presigned_resp = await session.post(
                "https://yupp.ai/api/trpc/chat.createPresignedURLForUpload?batch=1",
                json={
                    "0": {"json": {"fileName": name, "fileSize": len(data), "contentType": is_accepted_format(data)}}},
                headers={"Content-Type": "application/json",
                         "Cookie": f"__Secure-yupp.session-token={account['token']}",
                         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",

                         }
            )
            presigned_resp.raise_for_status()
            upload_info = (await presigned_resp.json())[0]["result"]["data"]["json"]
            upload_url = upload_info["signedUrl"]

            await session.put(
                upload_url,
                data=data,
                headers={
                    "Content-Type": is_accepted_format(data),
                    "Content-Length": str(len(data))
                }
            )

            attachment_resp = await session.post(
                "https://yupp.ai/api/trpc/chat.createAttachmentForUploadedFile?batch=1",
                json={"0": {"json": {"fileName": name, "contentType": is_accepted_format(data),
                                     "fileId": upload_info["fileId"]}}},
                cookies={"__Secure-yupp.session-token": account["token"]}
            )
            attachment_resp.raise_for_status()
            attachment = (await attachment_resp.json())[0]["result"]["data"]["json"]
            file = {
                "fileName": attachment["file_name"],
                "contentType": attachment["content_type"],
                "attachmentId": attachment["attachment_id"],
                "chatMessageId": ""
            }
            ImagesCache[image_hash] = file
            files.append(file)
        return files

    @classmethod
    async def user_info(cls, account: YUPP_ACCOUNT, kwargs: dict):
        history: dict = {}
        user_info = {}

        def pars_children(data):
            data = data["children"]
            if len(data) < 4:
                return
            if data[1] in ["div", "defs", "style", "script"]:
                return
            pars_data(data[3])

        def pars_data(data):
            if not isinstance(data, (list, dict)):
                return
            if isinstance(data, dict):
                json_data = data.get("json") or {}
            elif data[0] == "$":
                if data[1] in ["div", "defs", "style", "script"]:
                    return
                json_data = data[3]
            else:
                return

            if 'session' in json_data:
                user_info.update(json_data['session']['user'])
            elif "state" in json_data:
                for query in json_data["state"]["queries"]:
                    if query["state"]["dataUpdateCount"] == 0:
                        continue
                    if "getCredits" in query["queryHash"]:
                        credits = query["state"]["data"]["json"]
                        user_info["credits"] = credits
                    elif "getSidebarChatsV2" in query["queryHash"]:
                        for page in query["state"]["data"]["json"]["pages"]:
                            for item in page["items"]:
                                history[item["id"]] = item

            elif 'categories' in json_data:
                ...
            elif 'children' in json_data:
                pars_children(json_data)
            elif isinstance(json_data, list):
                if "supportedAttachmentMimeTypes" in json_data[0]:
                    models = json_data
                    cls.models_tags = {model.get("name"): ModelProcessor.generate_tags(model) for
                                       model in models}
                    cls.models = [model.get("name") for model in models]
                    cls.image_models = [model.get("name") for model in models if
                                        model.get("isImageGeneration")]
                    cls.vision_models = [model.get("name") for model in models if
                                         "image/*" in model.get("supportedAttachmentMimeTypes", [])]

        try:
            async with StreamSession() as session:
                headers = {
                    "content-type": "text/plain;charset=UTF-8",
                    "cookie": f"__Secure-yupp.session-token={account['token']}",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
                }
                async with session.get("https://yupp.ai", headers=headers, ) as response:
                    response.raise_for_status()
                    response.content._high_water = 10 * 1024 * 1024  # 10MB per line
                    line_pattern = re.compile("^([0-9a-fA-F]+):(.*)")
                    async for line in response.content:
                        line = line.decode()
                        # Pattern to match self.__next_f.push([...])
                        pattern = r'self\.__next_f\.push\((\[[\s\S]*?\])\)(?=<\/script>)'
                        matches = re.findall(pattern, line)
                        for match in matches:
                            # Parse the JSON array
                            data = json.loads(match)
                            for chunk in data[1].split("\n"):
                                match = line_pattern.match(chunk)
                                if not match:
                                    continue
                                chunk_id, chunk_data = match.groups()

                                if chunk_data.startswith("I["):
                                    data = json.loads(chunk_data[1:])
                                    if data[2] == "HomePagePromptForm":
                                        for js in data[1][::-1]:
                                            js_url = f"{cls.url}{js}"
                                            async with session.get(js_url, headers=headers, ) as js_response:
                                                js_text = await js_response.text()
                                                if "startNewChat" in js_text:
                                                    # changeStyle, continueChat, retryResponse, showMoreResponses, startNewChat
                                                    start_id = re.findall(r'\("([a-f0-9]{40,})".*?"(\w+)"\)', js_text)
                                                    for v, k in start_id:
                                                        kwargs[k] = v
                                                    break
                                elif chunk_data.startswith(("[", "{")):
                                    try:
                                        data = json.loads(chunk_data)
                                        pars_data(data)
                                    except json.decoder.JSONDecodeError:
                                        ...
                                    except Exception as e:
                                        log_debug(f"user_info error: {str(e)}")

        except Exception as e:
            log_debug(f"user_info error: {str(e)}")
        if user_info:
            log_debug(
                f"user:{user_info.get('name')} credits:{user_info.get('credits')} onboardingStatus:{user_info.get('onboardingStatus')}")
        return user_info

    @classmethod
    async def create_async_generator(
            cls,
            model: str,
            messages: Messages,
            proxy: str = None,
            **kwargs,
    ) -> AsyncResult:
        """
        Create async completion using Yupp.ai API with account rotation
        """
        # Initialize Yupp accounts
        api_key = kwargs.get("api_key")
        if not api_key:
            api_key = get_cookies("yupp.ai", False).get("__Secure-yupp.session-token")
        if api_key:
            load_yupp_accounts(api_key)
            log_debug(f"Yupp provider initialized with {len(YUPP_ACCOUNTS)} accounts")
        else:
            raise MissingAuthError("No Yupp accounts configured. Set YUPP_API_KEY environment variable.")

        # Format messages
        conversation = kwargs.get("conversation")
        url_uuid = conversation.url_uuid if conversation else None
        is_new_conversation = url_uuid is None

        prompt = kwargs.get("prompt")
        if prompt is None:
            if is_new_conversation:
                prompt = format_messages_for_yupp(messages)
            else:
                prompt = get_last_user_message(messages, bool(prompt))

        log_debug(
            f"Use url_uuid: {url_uuid}, Formatted prompt length: {len(prompt)}, Is new conversation: {is_new_conversation}")

        # Try all accounts with rotation
        max_attempts = len(YUPP_ACCOUNTS)
        for attempt in range(max_attempts):
            account = await get_best_yupp_account()
            if not account:
                raise ProviderException("No valid Yupp accounts available")
            # user_info, models. prev conversation, credits
            user_info: dict = await cls.user_info(account, kwargs)
            yield PlainTextResponse(str(user_info))
            try:
                async with StreamSession() as session:
                    turn_id = str(uuid.uuid4())
                    # Handle media attachments
                    media = kwargs.get("media")
                    if media:
                        media_ = list(merge_media(media, messages))
                        files = await cls.prepare_files(media_, session=session, account=account)
                    else:
                        files = []
                    mode = "image" if model in cls.image_models else "text"

                    # Build payload and URL - FIXED: Use consistent url_uuid handling
                    if is_new_conversation:
                        url_uuid = str(uuid.uuid4())
                        payload = [
                            url_uuid,
                            turn_id,
                            prompt,
                            "$undefined",
                            "$undefined",
                            files,
                            "$undefined",
                            [{"modelName": model, "promptModifierId": "$undefined"}] if model else "none",
                            mode,
                            True,
                            "$undefined",
                        ]
                        url = f"https://yupp.ai/chat/{url_uuid}?stream=true"
                        # Yield the conversation info first
                        yield JsonConversation(url_uuid=url_uuid)
                        next_action = kwargs.get("startNewChat") or kwargs.get("next_action", "7f7de0a21bc8dc3cee8ba8b6de632ff16f769649dd")
                    else:
                        # Continuing existing conversation
                        payload = [
                            url_uuid,
                            turn_id,
                            prompt,
                            False,
                            [],
                            [{"modelName": model, "promptModifierId": "$undefined"}] if model else [],
                            mode,
                            files
                        ]
                        url = f"https://yupp.ai/chat/{url_uuid}?stream=true"
                        next_action = kwargs.get("continueChat") or  kwargs.get("next_action", "7f9ec99a63cbb61f69ef18c0927689629bda07f1bf")
                    headers = {
                        "accept": "text/x-component",
                        "content-type": "text/plain;charset=UTF-8",
                        "next-action": next_action,
                        "cookie": f"__Secure-yupp.session-token={account['token']}",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
                    }

                    log_debug(f"Sending request to: {url}")
                    log_debug(f"Payload structure: {type(payload)}, length: {len(str(payload))}")
                    timeout = kwargs.get("timeout") or 5 * 60
                    # Send request
                    async with session.post(url, json=payload, headers=headers, proxy=proxy,
                                            timeout=timeout) as response:
                        response.raise_for_status()
                        if response.status == 303:
                            ...
                        # Make chat private in background
                        asyncio.create_task(make_chat_private(session, account, url_uuid))
                        # ٍSolve ValueError: Chunk too big
                        response.content._high_water = 10 * 1024 * 1024  # 10MB per line
                        # Process stream
                        async for chunk in cls._process_stream_response(response.content, account, session, prompt,
                                                                        model):
                            yield chunk
                    return

            except RateLimitError:
                log_debug(f"Account ...{account['token'][-4:]} hit rate limit, rotating")
                async with account_rotation_lock:
                    account["error_count"] += 1
                continue
            except ProviderException as e:
                log_debug(f"Account ...{account['token'][-4:]} failed: {str(e)}")
                async with account_rotation_lock:
                    if "auth" in str(e).lower() or "401" in str(e) or "403" in str(e):
                        account["is_valid"] = False
                    else:
                        account["error_count"] += 1
                continue
            except aiohttp.ClientResponseError as e:
                log_debug(f"Account ...{account['token'][-4:]} failed: {str(e)}")
                # No Available Yupp credits
                if e.status == 500 and 'Internal Server Error' in e.message:
                    account["is_valid"] = False
                # Need User-Agent
                # elif e.status == 429 and 'Too Many Requests' in e.message:
                #     account["is_valid"] = False
                else:
                    async with account_rotation_lock:
                        account["error_count"] += 1
                    raise ProviderException(f"Yupp request failed: {str(e)}") from e
            except Exception as e:
                log_debug(f"Unexpected error with account ...{account['token'][-4:]}: {str(e)}")
                async with account_rotation_lock:
                    account["error_count"] += 1
                raise ProviderException(f"Yupp request failed: {str(e)}") from e

        raise ProviderException("All Yupp accounts failed after rotation attempts")

    @classmethod
    async def _process_stream_response(
            cls,
            response_content,
            account: YUPP_ACCOUNT,
            session: aiohttp.ClientSession,
            prompt: str,
            model_id: str
    ) -> AsyncResult:
        """Process Yupp stream response asynchronously"""

        line_pattern = re.compile(b"^([0-9a-fA-F]+):(.*)")
        target_stream_id = None
        reward_info = None
        # Stream segmentation buffers
        is_thinking = False
        thinking_content = ""  # model's "thinking" channel (if activated later)
        normal_content = ""
        quick_content = ""  # quick-response short message
        variant_text = ""  # variant model output (comparison stream)
        stream = {
            "target": [],
            "variant": [],
            "quick": [],
            "thinking": [],
            "extra": []
        }
        # Holds leftStream / rightStream definitions to determine target/variant
        select_stream = [None, None]
        # State for capturing a multi-line <think> + <yapp> block (fa-style)
        capturing_ref_id: Optional[str] = None
        capturing_lines: List[bytes] = []

        # Storage for special referenced blocks like $fa
        think_blocks: Dict[str, str] = {}
        image_blocks: Dict[str, str] = {}

        def extract_ref_id(ref):
            """Extract ID from reference string, e.g., from '$@123' extract '123'"""
            return ref[2:] if ref and isinstance(ref, str) and ref.startswith("$@") else None

        def extract_ref_name(ref: str) -> Optional[str]:
            """Extract simple ref name from '$fa' → 'fa'"""
            if not isinstance(ref, str):
                return None
            if ref.startswith("$@"):
                return ref[2:]
            if ref.startswith("$") and len(ref) > 1:
                return ref[1:]
            return None

        def is_valid_content(content: str) -> bool:
            """Check if content is valid"""
            if not content or content in [None, "", "$undefined"]:
                return False
            return True

        async def process_content_chunk(content: str, chunk_id: str, line_count: int, *, for_target: bool = False):
            """
            Process a single content chunk from a stream.

            - If for_target=True → chunk belongs to the target model output.
            """
            nonlocal normal_content

            if not is_valid_content(content):
                return

            # Handle image-gen chunks
            if '<yapp class="image-gen">' in content:
                img_block = content.split('<yapp class="image-gen">').pop().split('</yapp>')[0]
                url = "https://yupp.ai/api/trpc/chat.getSignedImage"
                async with session.get(
                        url,
                        params={
                            "batch": "1",
                            "input": json.dumps(
                                {"0": {"json": {"imageId": json.loads(img_block).get("image_id")}}}
                            )
                        }
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    img = ImageResponse(
                        data[0]["result"]["data"]["json"]["signed_url"],
                        prompt
                    )
                    yield img
                return
            # Optional: thinking-mode support (disabled by default)
            if is_thinking:
                yield Reasoning(content)
            else:
                if for_target:
                    normal_content += content
                yield content

        def finalize_capture_block(ref_id: str, lines: List[bytes]):
            """Parse captured <think> + <yapp> block for a given ref ID."""
            text = b"".join(lines).decode("utf-8", errors="ignore")

            # Extract <think>...</think>
            think_start = text.find("<think>")
            think_end = text.find("</think>")
            if think_start != -1 and think_end != -1 and think_end > think_start:
                inner = text[think_start + len("<think>"):think_end].strip()
                if inner:
                    think_blocks[ref_id] = inner

            # Extract <yapp class="image-gen">...</yapp>
            yapp_start = text.find('<yapp class="image-gen">')
            if yapp_start != -1:
                yapp_end = text.find("</yapp>", yapp_start)
                if yapp_end != -1:
                    yapp_block = text[yapp_start:yapp_end + len("</yapp>")]
                    image_blocks[ref_id] = yapp_block

        try:
            line_count = 0
            quick_response_id = None
            variant_stream_id = None
            is_started: bool = False
            variant_image: Optional[ImageResponse] = None
            # "a" use as default then extract from "1"
            reward_id = "a"
            routing_id = "e"
            turn_id = None
            persisted_turn_id = None
            left_message_id = None
            right_message_id = None
            nudge_new_chat_id = None
            nudge_new_chat = False
            async for line in response_content:
                line_count += 1
                # If we are currently capturing a think/image block for some ref ID
                if capturing_ref_id is not None:
                    capturing_lines.append(line)

                    # Check if this line closes the <yapp> block; after that, block is complete
                    if b"</yapp>" in line:  # or b':{"curr"' in line:
                        # We may have trailing "2:{...}" after </yapp> on the same line
                        # Get id using re
                        idx = line.find(b"</yapp>")
                        suffix = line[idx + len(b"</yapp>"):]

                        # Finalize captured block for this ref ID
                        finalize_capture_block(capturing_ref_id, capturing_lines)
                        capturing_ref_id = None
                        capturing_lines = []

                        # If there is trailing content (e.g. '2:{"curr":"$fa"...}')
                        if suffix.strip():
                            # Process suffix as a new "line" in the same iteration
                            line = suffix
                        else:
                            # Nothing more on this line
                            continue
                    else:
                        # Still inside captured block; skip normal processing
                        continue

                # Detect start of a <think> block assigned to a ref like 'fa:...<think>'
                if b"<think>" in line:
                    m = line_pattern.match(line)
                    if m:
                        capturing_ref_id = m.group(1).decode()
                        capturing_lines = [line]
                        # Skip normal parsing; the rest of the block will be captured until </yapp>
                        continue

                match = line_pattern.match(line)
                if not match:
                    continue

                chunk_id, chunk_data = match.groups()
                chunk_id = chunk_id.decode()

                if nudge_new_chat_id and chunk_id == nudge_new_chat_id:
                    nudge_new_chat = chunk_data.decode()
                    continue

                try:
                    data = json.loads(chunk_data) if chunk_data != b"{}" else {}
                except json.JSONDecodeError:
                    continue
                # Process reward info
                if chunk_id == reward_id and isinstance(data, dict) and "unclaimedRewardInfo" in data:
                    reward_info = data
                    log_debug(f"Found reward info")

                # Process initial setup
                elif chunk_id == "1":
                    yield PlainTextResponse(line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        left_stream = data.get("leftStream", {})
                        right_stream = data.get("rightStream", {})
                        if data.get("quickResponse", {}) != "$undefined":
                            quick_response_id = extract_ref_id(
                                data.get("quickResponse", {}).get("stream", {}).get("next"))

                        if data.get("turnId", {}) != "$undefined":
                            turn_id = extract_ref_id(data.get("turnId", {}).get("next"))
                        if data.get("persistedTurn", {}) != "$undefined":
                            persisted_turn_id = extract_ref_id(data.get("persistedTurn", {}).get("next"))
                        if data.get("leftMessageId", {}) != "$undefined":
                            left_message_id = extract_ref_id(data.get("leftMessageId", {}).get("next"))
                        if data.get("rightMessageId", {}) != "$undefined":
                            right_message_id = extract_ref_id(data.get("rightMessageId", {}).get("next"))

                        reward_id = extract_ref_id(data.get("pendingRewardActionResult", "")) or reward_id
                        routing_id = extract_ref_id(data.get("routingResultPromise", "")) or routing_id
                        nudge_new_chat_id = extract_ref_id(data.get("nudgeNewChatPromise", "")) or nudge_new_chat_id
                        select_stream = [left_stream, right_stream]
                # Routing / model selection block
                elif chunk_id == routing_id:
                    yield PlainTextResponse(line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        provider_info = cls.get_dict()
                        provider_info['model'] = model_id
                        # Determine target & variant stream IDs
                        for i, selection in enumerate(data.get("modelSelections", [])):
                            if selection.get("selectionSource") == "USER_SELECTED":
                                target_stream_id = extract_ref_id(select_stream[i].get("next"))
                                provider_info["modelLabel"] = selection.get("shortLabel")
                                provider_info["modelUrl"] = selection.get("externalUrl")
                                log_debug(f"Found target stream ID: {target_stream_id}")
                            else:
                                variant_stream_id = extract_ref_id(select_stream[i].get("next"))
                                provider_info["variantLabel"] = selection.get("shortLabel")
                                provider_info["variantUrl"] = selection.get("externalUrl")
                                log_debug(f"Found variant stream ID: {variant_stream_id}")
                        yield ProviderInfo.from_dict(provider_info)

                # Process target stream content
                elif target_stream_id and chunk_id == target_stream_id:
                    yield PlainTextResponse(line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        target_stream_id = extract_ref_id(data.get("next"))
                        content = data.get("curr", "")
                        if content:
                            # Handle special "$fa" / "$<id>" reference
                            ref_name = extract_ref_name(content)
                            if ref_name and (ref_name in think_blocks or ref_name in image_blocks):
                                # Thinking block
                                if ref_name in think_blocks:
                                    t_text = think_blocks[ref_name]
                                    if t_text:
                                        reasoning = Reasoning(t_text)
                                        # thinking_content += t_text
                                        stream["thinking"].append(reasoning)
                                        # yield reasoning

                                # Image-gen block
                                if ref_name in image_blocks:
                                    img_block_text = image_blocks[ref_name]
                                    async for chunk in process_content_chunk(
                                            img_block_text,
                                            ref_name,
                                            line_count,
                                            for_target=True
                                    ):
                                        stream["target"].append(chunk)
                                        is_started = True
                                        yield chunk
                            else:
                                # Normal textual chunk
                                async for chunk in process_content_chunk(
                                        content,
                                        chunk_id,
                                        line_count,
                                        for_target=True
                                ):
                                    stream["target"].append(chunk)
                                    is_started = True
                                    yield chunk
                # Variant stream (comparison)
                elif variant_stream_id and chunk_id == variant_stream_id:
                    yield PlainTextResponse("[Variant] " + line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        variant_stream_id = extract_ref_id(data.get("next"))
                        content = data.get("curr", "")
                        if content:
                            async for chunk in process_content_chunk(
                                    content,
                                    chunk_id,
                                    line_count,
                                    for_target=False
                            ):
                                stream["variant"].append(chunk)
                                if isinstance(chunk, ImageResponse):
                                    yield PreviewResponse(str(chunk))
                                else:
                                    variant_text += str(chunk)
                                    if not is_started:
                                        yield PreviewResponse(variant_text)
                # Quick response (short preview)
                elif quick_response_id and chunk_id == quick_response_id:
                    yield PlainTextResponse("[Quick] " + line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        content = data.get("curr", "")
                        if content:
                            async for chunk in process_content_chunk(
                                    content,
                                    chunk_id,
                                    line_count,
                                    for_target=False
                            ):
                                stream["quick"].append(chunk)
                            quick_content += content
                            yield PreviewResponse(content)

                elif chunk_id in [turn_id, persisted_turn_id]:
                    ...
                elif chunk_id == right_message_id:
                    ...
                elif chunk_id == left_message_id:
                    ...
                # Miscellaneous extra content
                elif isinstance(data, dict) and "curr" in data:
                    content = data.get("curr", "")
                    if content:
                        async for chunk in process_content_chunk(
                                content,
                                chunk_id,
                                line_count,
                                for_target=False
                        ):
                            stream["extra"].append(chunk)
                            if isinstance(chunk, str) and "<streaming stopped unexpectedly" in chunk:
                                yield FinishReason(chunk)

                        yield PlainTextResponse("[Extra] " + line.decode(errors="ignore"))

            if variant_image is not None:
                yield variant_image
            elif variant_text:
                yield PreviewResponse(variant_text)
            yield JsonResponse(**stream)
            log_debug(f"Finished processing {line_count} lines")
        except:
            raise

        finally:
            # Claim reward in background
            if reward_info and "unclaimedRewardInfo" in reward_info:
                reward_id = reward_info["unclaimedRewardInfo"].get("rewardId")
                if reward_id:
                    await claim_yupp_reward(session, account, reward_id)
