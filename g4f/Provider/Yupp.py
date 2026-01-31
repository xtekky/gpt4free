import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List

try:
    import cloudscraper
    from cloudscraper import CloudScraper

    has_cloudscraper = True
except ImportError:
    from typing import Type as CloudScraper

    has_cloudscraper = False

from .helper import get_last_user_message
from .yupp.models import YuppModelManager
from .yupp.token_extractor import get_token_extractor
from ..cookies import get_cookies
from ..debug import log
from ..errors import (
    RateLimitError,
    ProviderException,
    MissingAuthError,
    MissingRequirementsError,
)
from ..image import is_accepted_format, to_bytes
from ..providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..providers.response import (
    Reasoning,
    PlainTextResponse,
    PreviewResponse,
    JsonConversation,
    ImageResponse,
    ProviderInfo,
    FinishReason,
    JsonResponse,
    VariantResponse,
)
from ..tools.auth import AuthManager
from ..tools.media import merge_media
from ..typing import AsyncResult, Messages

YUPP_ACCOUNTS: List[Dict[str, Any]] = []
account_rotation_lock = asyncio.Lock()
ImagesCache: Dict[str, dict] = {}
_accounts_loaded = False
_executor = ThreadPoolExecutor(max_workers=32)
MAX_CACHE_SIZE = 1000


def create_scraper():
    scraper = cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "windows",
            "desktop": True,
            "mobile": False,
        },
        delay=10,
        interpreter="nodejs",
    )
    scraper.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
            "Accept": "text/x-component, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Ch-Ua": '"Microsoft Edge";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
        }
    )
    return scraper


def load_yupp_accounts(tokens_str: str):
    global YUPP_ACCOUNTS, _accounts_loaded
    if _accounts_loaded:
        return
    if not tokens_str:
        return
    tokens = [token.strip() for token in tokens_str.split(",") if token.strip()]
    YUPP_ACCOUNTS = [
        {"token": token, "is_valid": True, "error_count": 0, "last_used": 0.0}
        for token in tokens
    ]
    _accounts_loaded = True


async def get_best_yupp_account() -> Optional[Dict[str, Any]]:
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

        for acc in valid_accounts:
            if (
                acc["error_count"] >= max_error_count
                and now - acc["last_used"] > error_cooldown
            ):
                acc["error_count"] = 0

        valid_accounts.sort(key=lambda x: (x["last_used"], x["error_count"]))
        account = valid_accounts[0]
        account["last_used"] = now
        return account


def sync_claim_yupp_reward(
    scraper: CloudScraper, account: Dict[str, Any], eval_id: str
):
    try:
        log_debug(f"Claiming reward {eval_id}...")
        url = "https://yupp.ai/api/trpc/reward.claim?batch=1"
        payload = {"0": {"json": {"evalId": eval_id}}}
        scraper.cookies.set("__Secure-yupp.session-token", account["token"])
        response = scraper.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        balance = data[0]["result"]["data"]["json"]["currentCreditBalance"]
        log_debug(f"Reward claimed successfully. New balance: {balance}")
        return balance
    except Exception as e:
        log_debug(f"Failed to claim reward {eval_id}. Error: {e}")
        return None


async def claim_yupp_reward(
    scraper: CloudScraper, account: Dict[str, Any], eval_id: str
):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, sync_claim_yupp_reward, scraper, account, eval_id
    )


def sync_record_model_feedback(
    scraper: CloudScraper,
    account: Dict[str, Any],
    turn_id: str,
    left_message_id: str,
    right_message_id: str,
) -> Optional[str]:
    try:
        log_debug(f"Recording model feedback for turn {turn_id}...")
        url = "https://yupp.ai/api/trpc/evals.recordModelFeedback?batch=1"
        payload = {
            "0": {
                "json": {
                    "turnId": turn_id,
                    "evalType": "SELECTION",
                    "messageEvals": [
                        {
                            "messageId": right_message_id,
                            "rating": "GOOD",
                            "reasons": ["Fast"],
                        },
                        {"messageId": left_message_id, "rating": "BAD", "reasons": []},
                    ],
                    "comment": "",
                    "requireReveal": False,
                }
            }
        }
        scraper.cookies.set("__Secure-yupp.session-token", account["token"])
        response = scraper.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        for result in data:
            json_data = result.get("result", {}).get("data", {}).get("json", {})
            eval_id = json_data.get("evalId")
            final_reward = json_data.get("finalRewardAmount")
            log_debug(f"Feedback recorded - evalId: {eval_id}, reward: {final_reward}")

            if final_reward:
                return eval_id
        return None
    except Exception as e:
        log_debug(f"Failed to record model feedback. Error: {e}")
        return None


async def record_model_feedback(
    scraper: CloudScraper,
    account: Dict[str, Any],
    turn_id: str,
    left_message_id: str,
    right_message_id: str,
) -> Optional[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        sync_record_model_feedback,
        scraper,
        account,
        turn_id,
        left_message_id,
        right_message_id,
    )


def sync_delete_chat(
    scraper: CloudScraper, account: Dict[str, Any], chat_id: str
) -> bool:
    try:
        log_debug(f"Deleting chat {chat_id}...")
        url = "https://yupp.ai/api/trpc/chat.deleteChat?batch=1"
        payload = {"0": {"json": {"chatId": chat_id}}}
        scraper.cookies.set("__Secure-yupp.session-token", account["token"])
        response = scraper.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if (
            isinstance(data, list)
            and len(data) > 0
            and data[0].get("result", {}).get("data", {}).get("json") is None
        ):
            log_debug(f"Chat {chat_id} deleted successfully")
            return True
        log_debug(f"Unexpected response while deleting chat: {data}")
        return False
    except Exception as e:
        log_debug(f"Failed to delete chat {chat_id}: {e}")
        return False


async def delete_chat(
    scraper: CloudScraper, account: Dict[str, Any], chat_id: str
) -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, sync_delete_chat, scraper, account, chat_id
    )


def sync_make_chat_private(
    scraper: CloudScraper, account: Dict[str, Any], chat_id: str
) -> bool:
    try:
        log_debug(f"Setting chat {chat_id} to PRIVATE...")
        url = "https://yupp.ai/api/trpc/chat.updateSharingSettings?batch=1"
        payload = {"0": {"json": {"chatId": chat_id, "status": "PRIVATE"}}}
        scraper.cookies.set("__Secure-yupp.session-token", account["token"])
        response = scraper.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if (
            isinstance(data, list)
            and len(data) > 0
            and "json" in data[0].get("result", {}).get("data", {})
        ):
            log_debug(f"Chat {chat_id} is now PRIVATE")
            return True
        log_debug(f"Unexpected response while setting chat private: {data}")
        return False
    except Exception as e:
        log_debug(f"Failed to make chat {chat_id} private: {e}")
        return False


async def make_chat_private(
    scraper: CloudScraper, account: Dict[str, Any], chat_id: str
) -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, sync_make_chat_private, scraper, account, chat_id
    )


def log_debug(message: str):
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        print(f"[DEBUG] {message}")
    else:
        log(f"[Yupp] {message}")


def format_messages_for_yupp(messages: Messages) -> str:
    if not messages:
        return ""

    if len(messages) == 1 and isinstance(messages[0].get("content"), str):
        return messages[0].get("content", "").strip()

    formatted = []

    system_messages = [
        msg for msg in messages if msg.get("role") in ["developer", "system"]
    ]
    if system_messages:
        for sys_msg in system_messages:
            content = sys_msg.get("content", "")
            formatted.append(content)

    user_assistant_msgs = [
        msg for msg in messages if msg.get("role") in ["user", "assistant"]
    ]
    for msg in user_assistant_msgs:
        role = "Human" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        for part in content if isinstance(content, list) else [{"text": content}]:
            if part.get("text", "").strip():
                formatted.append(f"\n\n{role}: {part.get('text', '')}")

    if not formatted or not formatted[-1].strip().startswith("Assistant:"):
        formatted.append("\n\nAssistant:")

    result = "".join(formatted)
    if result.startswith("\n\n"):
        result = result[2:]

    return result


def evict_cache_if_needed():
    global ImagesCache
    if len(ImagesCache) > MAX_CACHE_SIZE:
        keys_to_remove = list(ImagesCache.keys())[
            : len(ImagesCache) - MAX_CACHE_SIZE + 100
        ]
        for key in keys_to_remove:
            del ImagesCache[key]


class Yupp(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://yupp.ai"
    login_url = "https://discord.gg/qXA4Wf4Fsm"
    working = has_cloudscraper
    active_by_default = True
    supports_stream = True
    image_cache = True

    @classmethod
    def get_models(cls, api_key: str = None, **kwargs) -> List[str]:
        if not cls.models:
            if not api_key:
                api_key = AuthManager.load_api_key(cls)
            if not api_key:
                api_key = get_cookies("yupp.ai", False).get(
                    "__Secure-yupp.session-token"
                )
            if not api_key:
                raise MissingAuthError(
                    "No Yupp accounts configured. Set YUPP_API_KEY environment variable."
                )
            manager = YuppModelManager(api_key=api_key, session=create_scraper())
            models = manager.client.fetch_models()
            if models:
                cls.models_tags = {
                    model.get("name"): manager.processor.generate_tags(model)
                    for model in models
                }
                cls.models = [model.get("name") for model in models]
                cls.image_models = [
                    model.get("name")
                    for model in models
                    if model.get("isImageGeneration")
                ]
                cls.vision_models = [
                    model.get("name")
                    for model in models
                    if "image/*" in model.get("supportedAttachmentMimeTypes", [])
                ]
        return cls.models

    @classmethod
    def sync_prepare_files(
        cls, media, scraper: CloudScraper, account: Dict[str, Any]
    ) -> list:
        files = []
        if not media:
            return files
        for file, name in media:
            data = to_bytes(file)
            hasher = hashlib.md5()
            hasher.update(data)
            image_hash = hasher.hexdigest()
            cached_file = ImagesCache.get(image_hash)
            if cls.image_cache and cached_file:
                log_debug("Using cached image")
                files.append(cached_file)
                continue

            scraper.cookies.set("__Secure-yupp.session-token", account["token"])
            presigned_resp = scraper.post(
                "https://yupp.ai/api/trpc/chat.createPresignedURLForUpload?batch=1",
                json={
                    "0": {
                        "json": {
                            "fileName": name,
                            "fileSize": len(data),
                            "contentType": is_accepted_format(data),
                        }
                    }
                },
                headers={"Content-Type": "application/json"},
            )
            presigned_resp.raise_for_status()
            upload_info = presigned_resp.json()[0]["result"]["data"]["json"]
            upload_url = upload_info["signedUrl"]

            scraper.put(
                upload_url,
                data=data,
                headers={
                    "Content-Type": is_accepted_format(data),
                    "Content-Length": str(len(data)),
                },
            )

            attachment_resp = scraper.post(
                "https://yupp.ai/api/trpc/chat.createAttachmentForUploadedFile?batch=1",
                json={
                    "0": {
                        "json": {
                            "fileName": name,
                            "contentType": is_accepted_format(data),
                            "fileId": upload_info["fileId"],
                        }
                    }
                },
                cookies={"__Secure-yupp.session-token": account["token"]},
            )
            attachment_resp.raise_for_status()
            attachment = attachment_resp.json()[0]["result"]["data"]["json"]
            file_info = {
                "fileName": attachment["file_name"],
                "contentType": attachment["content_type"],
                "attachmentId": attachment["attachment_id"],
                "chatMessageId": "",
            }
            evict_cache_if_needed()
            ImagesCache[image_hash] = file_info
            files.append(file_info)
        return files

    @classmethod
    async def prepare_files(
        cls, media, scraper: CloudScraper, account: Dict[str, Any]
    ) -> list:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, cls.sync_prepare_files, media, scraper, account
        )

    @classmethod
    def sync_get_signed_image(cls, scraper: CloudScraper, image_id: str) -> str:
        url = "https://yupp.ai/api/trpc/chat.getSignedImage"
        resp = scraper.get(
            url,
            params={
                "batch": "1",
                "input": json.dumps({"0": {"json": {"imageId": image_id}}}),
            },
        )
        resp.raise_for_status()
        data = resp.json()[0]["result"]["data"]["json"]
        return data.get("signed_url", data.get("signedURL"))

    @classmethod
    async def get_signed_image(cls, scraper: CloudScraper, image_id: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, cls.sync_get_signed_image, scraper, image_id
        )

    @classmethod
    def sync_stream_request(
        cls, scraper: CloudScraper, url: str, payload: list, headers: dict, timeout: int
    ):
        response = scraper.post(
            url, json=payload, headers=headers, stream=True, timeout=timeout
        )
        response.raise_for_status()
        return response

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        api_key: str = None,
        **kwargs,
    ) -> AsyncResult:
        if not has_cloudscraper:
            raise MissingRequirementsError(
                "cloudscraper library is required for Yupp provider | install it via 'pip install cloudscraper'"
            )
        if not api_key:
            api_key = AuthManager.load_api_key(cls)
        if not api_key:
            api_key = get_cookies("yupp.ai", False).get("__Secure-yupp.session-token")
        if api_key:
            load_yupp_accounts(api_key)
            log_debug(f"Yupp provider initialized with {len(YUPP_ACCOUNTS)} accounts")
        else:
            raise MissingAuthError(
                "No Yupp accounts configured. Set YUPP_API_KEY environment variable."
            )

        conversation = kwargs.get("conversation")
        url_uuid = conversation.url_uuid if conversation else None
        is_new_conversation = url_uuid is None

        prompt = kwargs.get("prompt")
        if prompt is None:
            if is_new_conversation:
                prompt = format_messages_for_yupp(messages)
            else:
                prompt = get_last_user_message(messages, prompt)

        log_debug(
            f"Use url_uuid: {url_uuid}, Formatted prompt length: {len(prompt)}, Is new conversation: {is_new_conversation}"
        )

        max_attempts = len(YUPP_ACCOUNTS)
        for attempt in range(max_attempts):
            account = await get_best_yupp_account()
            if not account:
                raise ProviderException("No valid Yupp accounts available")

            try:
                scraper = create_scraper()
                if proxy:
                    scraper.proxies = {"http": proxy, "https": proxy}

                # Initialize token extractor for automatic token swapping
                token_extractor = get_token_extractor(
                    jwt_token=account["token"], scraper=scraper
                )

                turn_id = str(uuid.uuid4())

                media = kwargs.get("media")
                if media:
                    media_ = list(merge_media(media, messages))
                    files = await cls.prepare_files(
                        media_, scraper=scraper, account=account
                    )
                else:
                    files = []

                mode = "image" if model in cls.image_models else "text"

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
                        [{"modelName": model, "promptModifierId": "$undefined"}]
                        if model
                        else "none",
                        mode,
                        True,
                        "$undefined",
                    ]
                    url = f"https://yupp.ai/chat/{url_uuid}?stream=true"
                    yield JsonConversation(url_uuid=url_uuid)
                    next_action = kwargs.get(
                        "next_action",
                        await token_extractor.get_token("new_conversation"),
                    )
                else:
                    payload = [
                        url_uuid,
                        turn_id,
                        prompt,
                        False,
                        [],
                        [{"modelName": model, "promptModifierId": "$undefined"}]
                        if model
                        else [],
                        mode,
                        files,
                    ]
                    url = f"https://yupp.ai/chat/{url_uuid}?stream=true"
                    next_action = kwargs.get(
                        "next_action",
                        await token_extractor.get_token("existing_conversation"),
                    )

                headers = {
                    "accept": "text/x-component",
                    "content-type": "text/plain;charset=UTF-8",
                    "next-action": next_action,
                    "cookie": f"__Secure-yupp.session-token={account['token']}",
                }

                log_debug(f"Sending request to: {url}")
                log_debug(
                    f"Payload structure: {type(payload)}, length: {len(str(payload))}"
                )

                _timeout = kwargs.get("timeout")
                if isinstance(_timeout, (int, float)):
                    timeout = int(_timeout)
                else:
                    timeout = 5 * 60

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    _executor,
                    cls.sync_stream_request,
                    scraper,
                    url,
                    payload,
                    headers,
                    timeout,
                )

                try:
                    async for chunk in cls._process_stream_response(
                        response, account, scraper, prompt, model
                    ):
                        yield chunk
                finally:
                    response.close()
                    if not kwargs.get("conversation"):
                        asyncio.create_task(delete_chat(scraper, account, url_uuid))
                return

            except RateLimitError:
                log_debug(
                    f"Account ...{account['token'][-4:]} hit rate limit, rotating"
                )
                async with account_rotation_lock:
                    account["error_count"] += 1
                continue

            except ProviderException as e:
                log_debug(f"Account ...{account['token'][-4:]} failed: {str(e)}")
                error_msg = str(e).lower()

                # Check if this is a token-related error
                if any(
                    x in error_msg
                    for x in [
                        "auth",
                        "401",
                        "403",
                        "404",
                        "invalid action",
                        "action",
                        "next-action",
                    ]
                ):
                    # Mark token as failed to trigger extraction
                    token_type = (
                        "new_conversation"
                        if is_new_conversation
                        else "existing_conversation"
                    )
                    await token_extractor.mark_token_failed(token_type, next_action)
                    log_debug(
                        f"Token failure detected, marked for extraction: {token_type}"
                    )

                async with account_rotation_lock:
                    if "auth" in error_msg or "401" in error_msg or "403" in error_msg:
                        account["is_valid"] = False
                    else:
                        account["error_count"] += 1
                continue

            except Exception as e:
                log_debug(
                    f"Unexpected error with account ...{account['token'][-4:]}: {str(e)}"
                )
                error_str = str(e).lower()

                # Check for token-related errors in generic exceptions too
                if any(x in error_str for x in ["404", "401", "403", "invalid action"]):
                    token_type = (
                        "new_conversation"
                        if is_new_conversation
                        else "existing_conversation"
                    )
                    await token_extractor.mark_token_failed(token_type, next_action)
                    log_debug(
                        f"Token failure detected in exception handler: {token_type}"
                    )

                if "500" in error_str or "internal server error" in error_str:
                    async with account_rotation_lock:
                        account["error_count"] += 1
                    continue
                async with account_rotation_lock:
                    account["error_count"] += 1
                raise ProviderException(f"Yupp request failed: {str(e)}") from e

        raise ProviderException("All Yupp accounts failed after rotation attempts")

    @classmethod
    async def _process_stream_response(
        cls,
        response,
        account: Dict[str, Any],
        scraper: CloudScraper,
        prompt: str,
        model_id: str,
    ) -> AsyncResult:
        line_pattern = re.compile(b"^([0-9a-fA-F]+):(.*)")
        target_stream_id = None
        reward_info = None
        is_thinking = False
        thinking_content = ""
        normal_content = ""
        quick_content = ""
        variant_text = ""
        stream = {"target": [], "variant": [], "quick": [], "thinking": [], "extra": []}
        select_stream = [None, None]
        capturing_ref_id: Optional[str] = None
        capturing_lines: List[bytes] = []
        think_blocks: Dict[str, str] = {}
        image_blocks: Dict[str, str] = {}

        def extract_ref_id(ref):
            return (
                ref[2:]
                if ref and isinstance(ref, str) and ref.startswith("$@")
                else None
            )

        def extract_ref_name(ref: str) -> Optional[str]:
            if not isinstance(ref, str):
                return None
            if ref.startswith("$@"):
                return ref[2:]
            if ref.startswith("$") and len(ref) > 1:
                return ref[1:]
            return None

        def is_valid_content(content: str) -> bool:
            if not content or content in [None, "", "$undefined"]:
                return False
            return True

        async def process_content_chunk(
            content: str, chunk_id: str, line_count: int, *, for_target: bool = False
        ):
            nonlocal normal_content

            if not is_valid_content(content):
                return

            if '<yapp class="image-gen">' in content:
                img_block = (
                    content.split('<yapp class="image-gen">').pop().split("</yapp>")[0]
                )
                image_id = json.loads(img_block).get("image_id")
                signed_url = await cls.get_signed_image(scraper, image_id)
                img = ImageResponse(signed_url, prompt)
                yield img
                return

            if is_thinking:
                yield Reasoning(content)
            else:
                if for_target:
                    normal_content += content
                yield content

        def finalize_capture_block(ref_id: str, lines: List[bytes]):
            text = b"".join(lines).decode("utf-8", errors="ignore")

            think_start = text.find("<think>")
            think_end = text.find("</think>")
            if think_start != -1 and think_end != -1 and think_end > think_start:
                inner = text[think_start + len("<think>") : think_end].strip()
                if inner:
                    think_blocks[ref_id] = inner

            yapp_start = text.find('<yapp class="image-gen">')
            if yapp_start != -1:
                yapp_end = text.find("</yapp>", yapp_start)
                if yapp_end != -1:
                    yapp_block = text[yapp_start : yapp_end + len("</yapp>")]
                    image_blocks[ref_id] = yapp_block

        try:
            line_count = 0
            quick_response_id = None
            variant_stream_id = None
            is_started: bool = False
            variant_image: Optional[ImageResponse] = None
            reward_id = "a"
            reward_kw = {}
            routing_id = "e"
            turn_id = None
            persisted_turn_id = None
            left_message_id = None
            right_message_id = None
            nudge_new_chat_id = None
            nudge_new_chat = False

            loop = asyncio.get_event_loop()

            def iter_lines():
                for line in response.iter_lines():
                    if line:
                        yield line

            lines_iterator = iter_lines()

            while True:
                try:
                    line = await loop.run_in_executor(
                        _executor, lambda: next(lines_iterator, None)
                    )
                    if line is None:
                        break
                except StopIteration:
                    break

                line_count += 1

                if isinstance(line, str):
                    line = line.encode()

                if capturing_ref_id is not None:
                    capturing_lines.append(line)

                    if b"</yapp>" in line:
                        idx = line.find(b"</yapp>")
                        suffix = line[idx + len(b"</yapp>") :]
                        finalize_capture_block(capturing_ref_id, capturing_lines)
                        capturing_ref_id = None
                        capturing_lines = []

                        if suffix.strip():
                            line = suffix
                        else:
                            continue
                    else:
                        continue

                match = line_pattern.match(line)
                if not match:
                    if b"<think>" in line:
                        m = line_pattern.match(line)
                        if m:
                            capturing_ref_id = m.group(1).decode()
                            capturing_lines = [line]
                            continue
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

                if (
                    chunk_id == reward_id
                    and isinstance(data, dict)
                    and "unclaimedRewardInfo" in data
                ):
                    reward_info = data
                    log_debug(f"Found reward info")

                elif chunk_id == "1":
                    yield PlainTextResponse(line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        left_stream = data.get("leftStream", {})
                        right_stream = data.get("rightStream", {})
                        if data.get("quickResponse", {}) != "$undefined":
                            quick_response_id = extract_ref_id(
                                data.get("quickResponse", {})
                                .get("stream", {})
                                .get("next")
                            )

                        if data.get("turnId", {}) != "$undefined":
                            turn_id = extract_ref_id(data.get("turnId", {}).get("next"))
                        if data.get("persistedTurn", {}) != "$undefined":
                            persisted_turn_id = extract_ref_id(
                                data.get("persistedTurn", {}).get("next")
                            )
                        if data.get("leftMessageId", {}) != "$undefined":
                            left_message_id = extract_ref_id(
                                data.get("leftMessageId", {}).get("next")
                            )
                        if data.get("rightMessageId", {}) != "$undefined":
                            right_message_id = extract_ref_id(
                                data.get("rightMessageId", {}).get("next")
                            )

                        reward_id = (
                            extract_ref_id(data.get("pendingRewardActionResult", ""))
                            or reward_id
                        )
                        routing_id = (
                            extract_ref_id(data.get("routingResultPromise", ""))
                            or routing_id
                        )
                        nudge_new_chat_id = (
                            extract_ref_id(data.get("nudgeNewChatPromise", ""))
                            or nudge_new_chat_id
                        )
                        select_stream = [left_stream, right_stream]

                elif chunk_id == routing_id:
                    yield PlainTextResponse(line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        provider_info = cls.get_dict()
                        provider_info["model"] = model_id
                        for i, selection in enumerate(data.get("modelSelections", [])):
                            if selection.get("selectionSource") == "USER_SELECTED":
                                target_stream_id = extract_ref_id(
                                    select_stream[i].get("next")
                                )
                                provider_info["modelLabel"] = selection.get(
                                    "shortLabel"
                                )
                                provider_info["modelUrl"] = selection.get("externalUrl")
                                log_debug(f"Found target stream ID: {target_stream_id}")
                            else:
                                variant_stream_id = extract_ref_id(
                                    select_stream[i].get("next")
                                )
                                provider_info["variantLabel"] = selection.get(
                                    "shortLabel"
                                )
                                provider_info["variantUrl"] = selection.get(
                                    "externalUrl"
                                )
                                log_debug(
                                    f"Found variant stream ID: {variant_stream_id}"
                                )
                        yield ProviderInfo.from_dict(provider_info)

                elif target_stream_id and chunk_id == target_stream_id:
                    yield PlainTextResponse(line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        target_stream_id = extract_ref_id(data.get("next"))
                        content = data.get("curr", "")
                        if content:
                            ref_name = extract_ref_name(content)
                            if ref_name and (
                                ref_name in think_blocks or ref_name in image_blocks
                            ):
                                if ref_name in think_blocks:
                                    t_text = think_blocks[ref_name]
                                    if t_text:
                                        reasoning = Reasoning(t_text)
                                        stream["thinking"].append(reasoning)

                                if ref_name in image_blocks:
                                    img_block_text = image_blocks[ref_name]
                                    async for chunk in process_content_chunk(
                                        img_block_text,
                                        ref_name,
                                        line_count,
                                        for_target=True,
                                    ):
                                        stream["target"].append(chunk)
                                        is_started = True
                                        yield chunk
                            else:
                                async for chunk in process_content_chunk(
                                    content, chunk_id, line_count, for_target=True
                                ):
                                    stream["target"].append(chunk)
                                    is_started = True
                                    yield chunk

                elif variant_stream_id and chunk_id == variant_stream_id:
                    yield PlainTextResponse("[Variant] " + line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        variant_stream_id = extract_ref_id(data.get("next"))
                        content = data.get("curr", "")
                        if content:
                            async for chunk in process_content_chunk(
                                content, chunk_id, line_count, for_target=False
                            ):
                                stream["variant"].append(chunk)
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
                            async for chunk in process_content_chunk(
                                content, chunk_id, line_count, for_target=False
                            ):
                                stream["quick"].append(chunk)
                            quick_content += content
                            yield PreviewResponse(content)

                elif chunk_id == turn_id:
                    reward_kw["turn_id"] = data.get("curr", "")

                elif chunk_id == persisted_turn_id:
                    pass

                elif chunk_id == right_message_id:
                    reward_kw["right_message_id"] = data.get("curr", "")

                elif chunk_id == left_message_id:
                    reward_kw["left_message_id"] = data.get("curr", "")

                elif isinstance(data, dict) and "curr" in data:
                    content = data.get("curr", "")
                    if content:
                        async for chunk in process_content_chunk(
                            content, chunk_id, line_count, for_target=False
                        ):
                            stream["extra"].append(chunk)
                            if (
                                isinstance(chunk, str)
                                and "<streaming stopped unexpectedly" in chunk
                            ):
                                yield FinishReason(chunk)

                        yield PlainTextResponse(
                            "[Extra] " + line.decode(errors="ignore")
                        )

            if variant_image is not None:
                yield variant_image
            elif variant_text:
                yield VariantResponse(variant_text)
            yield JsonResponse(**stream)
            log_debug(f"Finished processing {line_count} lines")

        finally:
            log_debug(f"Get Reward: {reward_kw}")
            if (
                reward_kw.get("turn_id")
                and reward_kw.get("left_message_id")
                and reward_kw.get("right_message_id")
            ):
                eval_id = await record_model_feedback(
                    scraper,
                    account,
                    reward_kw["turn_id"],
                    reward_kw["left_message_id"],
                    reward_kw["right_message_id"],
                )
                if eval_id:
                    await claim_yupp_reward(scraper, account, eval_id)
