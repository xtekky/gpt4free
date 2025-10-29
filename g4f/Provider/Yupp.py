import json
import time
import uuid
import re
import os
import asyncio
import aiohttp

from ..typing import AsyncResult, Messages, Optional, Dict, Any, List
from ..providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..providers.response import Reasoning, PlainTextResponse, PreviewResponse, JsonConversation, ImageResponse, ProviderInfo
from ..errors import RateLimitError, ProviderException, MissingAuthError
from ..cookies import get_cookies
from ..tools.auth import AuthManager
from ..tools.media import merge_media
from ..image import is_accepted_format, to_bytes
from .yupp.models import YuppModelManager
from .helper import get_last_user_message
from ..debug import log

# Global variables to manage Yupp accounts
YUPP_ACCOUNTS: List[Dict[str, Any]] = []
account_rotation_lock = asyncio.Lock()

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

async def get_best_yupp_account() -> Optional[Dict[str, Any]]:
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

async def claim_yupp_reward(session: aiohttp.ClientSession, account: Dict[str, Any], reward_id: str):
    """Claim Yupp reward asynchronously"""
    try:
        log_debug(f"Claiming reward {reward_id}...")
        url = "https://yupp.ai/api/trpc/reward.claim?batch=1"
        payload = {"0": {"json": {"rewardId": reward_id}}}
        headers = {
            "Content-Type": "application/json",
            "Cookie": f"__Secure-yupp.session-token={account['token']}",
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

async def make_chat_private(session: aiohttp.ClientSession, account: Dict[str, Any], chat_id: str) -> bool:
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
                cls.vision_models = [model.get("name") for model in models if "image/*" in model.get("supportedAttachmentMimeTypes", [])]
        return cls.models

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
                prompt = get_last_user_message(messages, prompt)
        
        log_debug(f"Use url_uuid: {url_uuid}, Formatted prompt length: {len(prompt)}, Is new conversation: {is_new_conversation}")

        # Try all accounts with rotation
        max_attempts = len(YUPP_ACCOUNTS)
        for attempt in range(max_attempts):
            account = await get_best_yupp_account()
            if not account:
                raise ProviderException("No valid Yupp accounts available")

            try:
                async with aiohttp.ClientSession() as session:
                    turn_id = str(uuid.uuid4())
                    files = []

                    # Handle media attachments
                    media = kwargs.get("media")
                    if media:
                        for file, name in list(merge_media(media, messages)):
                            data = to_bytes(file)
                            presigned_resp = await session.post(
                                "https://yupp.ai/api/trpc/chat.createPresignedURLForUpload?batch=1",
                                json={"0": {"json": {"fileName": name, "fileSize": len(data), "contentType": is_accepted_format(data)}}},
                                headers={"Content-Type": "application/json", "Cookie": f"__Secure-yupp.session-token={account['token']}"}
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
                                json={"0": {"json": {"fileName": name, "contentType": is_accepted_format(data), "fileId": upload_info["fileId"]}}},
                                cookies={"__Secure-yupp.session-token": account["token"]}
                            )
                            attachment_resp.raise_for_status()
                            attachment = (await attachment_resp.json())[0]["result"]["data"]["json"]
                            files.append({
                                "fileName": attachment["file_name"],
                                "contentType": attachment["content_type"],
                                "attachmentId": attachment["attachment_id"],
                                "chatMessageId": ""
                            })
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
                        next_action = kwargs.get("next_action", "7f7de0a21bc8dc3cee8ba8b6de632ff16f769649dd")
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
                        next_action = kwargs.get("next_action", "7f9ec99a63cbb61f69ef18c0927689629bda07f1bf")

                    headers = {
                        "accept": "text/x-component",
                        "content-type": "text/plain;charset=UTF-8",
                        "next-action": next_action,
                        "cookie": f"__Secure-yupp.session-token={account['token']}",
                    }

                    log_debug(f"Sending request to: {url}")
                    log_debug(f"Payload structure: {type(payload)}, length: {len(str(payload))}")

                    # Send request
                    async with session.post(url, json=payload, headers=headers, proxy=proxy) as response:
                        response.raise_for_status()

                        # Make chat private in background
                        asyncio.create_task(make_chat_private(session, account, url_uuid))

                        # Process stream
                        async for chunk in cls._process_stream_response(response.content, account, session, prompt, model):
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
        account: Dict[str, Any],
        session: aiohttp.ClientSession,
        prompt: str,
        model_id: str
    ) -> AsyncResult:
        """Process Yupp stream response asynchronously"""

        line_pattern = re.compile(b"^([0-9a-fA-F]+):(.*)")
        target_stream_id = None
        reward_info = None
        is_thinking = False
        thinking_content = ""
        normal_content = ""
        select_stream = [None, None]
        
        def extract_ref_id(ref):
            """Extract ID from reference string, e.g., from '$@123' extract '123'"""
            return ref[2:] if ref and isinstance(ref, str) and ref.startswith("$@") else None
        def is_valid_content(content: str) -> bool:
            """Check if content is valid"""
            if not content or content in [None, "", "$undefined"]:
                return False
            return True

        async def process_content_chunk(content: str, chunk_id: str, line_count: int):
            """Process single content chunk"""
            nonlocal is_thinking, thinking_content, normal_content, session
            
            if not is_valid_content(content):
                return
            
            if '<yapp class="image-gen">' in content:
                content = content.split('<yapp class="image-gen">').pop().split('</yapp>')[0]
                url = "https://yupp.ai/api/trpc/chat.getSignedImage"
                async with session.get(url, params={"batch": "1", "input": json.dumps({"0": {"json": {"imageId": json.loads(content).get("image_id")}}})}) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    yield ImageResponse(data[0]["result"]["data"]["json"]["signed_url"], prompt)
                return
            
            # log_debug(f"Processing chunk #{line_count} with content: '{content[:50]}...'")
            
            if is_thinking:
                yield Reasoning(content)
            else:
                normal_content += content
                yield content
        
        try:
            line_count = 0
            quick_response_id = None
            variant_stream_id = None
            is_started: bool = False
            variant_image: Optional[ImageResponse] = None
            variant_text = ""
            
            async for line in response_content:
                line_count += 1
                
                match = line_pattern.match(line)
                if not match:
                    continue
                
                chunk_id, chunk_data = match.groups()
                chunk_id = chunk_id.decode()
                
                try:
                    data = json.loads(chunk_data) if chunk_data != b"{}" else {}
                except json.JSONDecodeError:
                    continue
                
                # Process reward info
                if chunk_id == "a":
                    reward_info = data
                    log_debug(f"Found reward info")
                
                # Process initial setup
                elif chunk_id == "1":
                    yield PlainTextResponse(line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        left_stream = data.get("leftStream", {})
                        right_stream = data.get("rightStream", {})
                        quick_response_id = extract_ref_id(data.get("quickResponse", {}).get("stream", {}).get("next"))
                        select_stream = [left_stream, right_stream]
                
                elif chunk_id == "e":
                    yield PlainTextResponse(line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        provider_info = cls.get_dict()
                        provider_info['model'] = model_id
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
                            async for chunk in process_content_chunk(content, chunk_id, line_count):
                                is_started = True
                                yield chunk
                
                elif variant_stream_id and chunk_id == variant_stream_id:
                    yield PlainTextResponse("[Variant] " + line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        variant_stream_id = extract_ref_id(data.get("next"))
                        content = data.get("curr", "")
                        if content:
                            async for chunk in process_content_chunk(content, chunk_id, line_count):
                                if isinstance(chunk, ImageResponse):
                                    yield PreviewResponse(str(chunk))
                                else:
                                    variant_text += str(chunk)
                                    if not is_started:
                                        yield PreviewResponse(variant_text)
                
                elif quick_response_id and chunk_id == quick_response_id:
                    yield PlainTextResponse("[Quick] " + line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        content = data.get("curr", "")
                        if content:
                            yield PreviewResponse(content)

                elif isinstance(data, dict) and "curr" in data:
                    content = data.get("curr", "")
                    if content:
                        yield PlainTextResponse("[Extra] " + line.decode(errors="ignore"))
            
            if variant_image is not None:
                yield variant_image
            elif variant_text:
                yield PreviewResponse(variant_text)

            log_debug(f"Finished processing {line_count} lines")
            
        except:
            raise
        
        finally:
            # Claim reward in background
            if reward_info and "unclaimedRewardInfo" in reward_info:
                reward_id = reward_info["unclaimedRewardInfo"].get("rewardId")
                if reward_id:
                    await claim_yupp_reward(session, account, reward_id)