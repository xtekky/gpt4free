import json
import time
import uuid
import re
import os
from typing import Iterable, Optional, Dict, Any, Generator, List
import threading
import requests

from ..providers.base_provider import AbstractProvider, ProviderModelMixin
from ..providers.response import Reasoning, PlainTextResponse, PreviewResponse, JsonConversation, ImageResponse, ProviderInfo
from ..errors import RateLimitError, ProviderException, MissingAuthError
from ..cookies import get_cookies
from ..tools.auth import AuthManager
from ..tools.media import merge_media
from ..image import is_accepted_format, to_bytes
from .yupp.models import YuppModelManager
from .helper import get_last_user_message
from ..debug import log

# Global variables to manage Yupp accounts (should be set by your main application)
YUPP_ACCOUNTS: List[Dict[str, Any]] = []
YUPP_MODELS: List[Dict[str, Any]] = []
account_rotation_lock = threading.Lock()

class YuppAccount:
    """Yupp account representation"""
    def __init__(self, token: str, is_valid: bool = True, error_count: int = 0, last_used: float = 0):
        self.token = token
        self.is_valid = is_valid
        self.error_count = error_count
        self.last_used = last_used

def load_yupp_accounts(tokens_str: str):
    """Load Yupp accounts from token string (compatible with your existing system)"""
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

def create_requests_session():
    """Create a requests session with proper headers"""
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
        "Accept": "text/x-component, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    })
    return session

def get_best_yupp_account() -> Optional[Dict[str, Any]]:
    """Get the best available Yupp account using a smart selection algorithm."""
    max_error_count = int(os.getenv("MAX_ERROR_COUNT", "3"))
    error_cooldown = int(os.getenv("ERROR_COOLDOWN", "300"))

    with account_rotation_lock:
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

        # Reset error count for accounts that have been in cooldown
        for acc in valid_accounts:
            if (
                acc["error_count"] >= max_error_count
                and now - acc["last_used"] > error_cooldown
            ):
                acc["error_count"] = 0

        # Sort by last used (oldest first) and error count (lowest first)
        valid_accounts.sort(key=lambda x: (x["last_used"], x["error_count"]))
        account = valid_accounts[0]
        account["last_used"] = now
        return account

def claim_yupp_reward(account: Dict[str, Any], reward_id: str):
    """Claim Yupp reward synchronously"""
    try:
        import requests
        log_debug(f"Claiming reward {reward_id}...")
        url = "https://yupp.ai/api/trpc/reward.claim?batch=1"
        payload = {"0": {"json": {"rewardId": reward_id}}}
        headers = {
            "Content-Type": "application/json",
            "Cookie": f"__Secure-yupp.session-token={account['token']}",
        }
        session = create_requests_session()
        response = session.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        balance = data[0]["result"]["data"]["json"]["currentCreditBalance"]
        log_debug(f"Reward claimed successfully. New balance: {balance}")
        return balance
    except Exception as e:
        log_debug(f"Failed to claim reward {reward_id}. Error: {e}")
        return None

def make_chat_private(account: Dict[str, Any], chat_id: str) -> bool:
    """
    Set a Yupp chat's sharing status to PRIVATE.
    Returns True if successful, False otherwise.
    """
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

        session = create_requests_session()
        response = session.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        # Expected: [{"result":{"data":{"json":{}}}}]
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
    """Debug logging (can be replaced with your logging system)"""
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        print(f"[DEBUG] {message}")
    else:
        log(f"[Yupp] {message}")

def format_messages_for_yupp(messages: List[Dict[str, str]]) -> str:
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
    # Remove leading \n\n if present
    if result.startswith("\n\n"):
        result = result[2:]

    return result

class Yupp(AbstractProvider, ProviderModelMixin):
    """
    Yupp.ai Provider for g4f
    Uses multiple account rotation and smart error handling
    """

    url = "https://yupp.ai"
    login_url = "https://discord.gg/qXA4Wf4Fsm"
    working = True
    active_by_default = True

    @classmethod
    def get_models(cls, api_key: str = None, **kwargs) -> List[Dict[str, Any]]:
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
        return cls.models

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: List[Dict[str, str]] = None,
        stream: bool = False,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
        conversation: JsonConversation = None,
        **kwargs,
    ) -> Generator[str, Any, None]:
        """
        Create completion using Yupp.ai API with account rotation
        """
        # Initialize Yupp accounts and models
        if not api_key:
            api_key = get_cookies("yupp.ai", False).get("__Secure-yupp.session-token")
        if api_key:
            load_yupp_accounts(api_key)
            log_debug(f"Yupp provider initialized with {len(YUPP_ACCOUNTS)} accounts")
        else:
            raise MissingAuthError("No Yupp accounts configured. Set YUPP_API_KEY environment variable.")

        if messages is None:
            messages = []

        # Format messages - use the new format_messages_for_yupp function
        url_uuid = None
        if conversation is not None:
            url_uuid = conversation.url_uuid
            
        # Determine the prompt based on conversation context
        is_new_conversation = url_uuid is None
        if prompt is None:
            if is_new_conversation:
                # New conversation - format all messages
                prompt = format_messages_for_yupp(messages)
            else:
                # Continuing conversation - use only the last user message
                prompt = get_last_user_message(messages, prompt)
                
        log_debug(f"Use url_uuid: {url_uuid}, Formatted prompt length: {len(prompt)}, Is new conversation: {is_new_conversation}")

        # Try all accounts with rotation
        max_attempts = len(YUPP_ACCOUNTS)
        for attempt in range(max_attempts):
            account = get_best_yupp_account()
            if not account:
                raise ProviderException("No valid Yupp accounts available")

            try:
                # Prepare the request
                session = create_requests_session()
                turn_id = str(uuid.uuid4())
                files = []

                # Handle media attachments if any
                media = kwargs.get("media", None)
                if media:
                    for file, name in list(merge_media(media, messages)):
                        data = to_bytes(file)
                        presigned_resp = session.post(
                            "https://yupp.ai/api/trpc/chat.createPresignedURLForUpload?batch=1",
                            json={"0": {"json": {"fileName": name, "fileSize": len(data), "contentType": is_accepted_format(data)}}},
                            headers={"Content-Type": "application/json", "Cookie": f"__Secure-yupp.session-token={account['token']}"}
                        )
                        presigned_resp.raise_for_status()
                        upload_info = presigned_resp.json()[0]["result"]["data"]["json"]
                        upload_url = upload_info["signedUrl"]
                        session.put(upload_url, data=data, headers={"Content-Type": is_accepted_format(data), "Content-Length": str(len(data))})
                        attachment_resp = session.post(
                            "https://yupp.ai/api/trpc/chat.createAttachmentForUploadedFile?batch=1",
                            json={"0": {"json": {"fileName": name, "contentType": is_accepted_format(data), "fileId": upload_info["fileId"]}}},
                            cookies={"__Secure-yupp.session-token": account["token"]}
                        )
                        attachment_resp.raise_for_status()
                        attachment = attachment_resp.json()[0]["result"]["data"]["json"]
                        files.append({
                            "fileName": attachment["file_name"],
                            "contentType": attachment["content_type"],
                            "attachmentId": attachment["attachment_id"],
                            "chatMessageId": ""
                        })

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
                        "text",
                        True,
                        "$undefined",
                    ]
                    url = f"https://yupp.ai/chat/{url_uuid}?stream=true"
                    # Yield the conversation info first
                    yield JsonConversation(url_uuid=url_uuid)
                    next_action = kwargs.get("next_action", "7f2a2308b5fc462a2c26df714cb2cccd02a9c10fbb")
                else:
                    # Continuing existing conversation
                    payload = [
                        url_uuid,
                        turn_id,
                        prompt,
                        False,
                        [],
                        [{"modelName": model, "promptModifierId": "$undefined"}] if model else [],
                        "text",
                        files
                    ]
                    url = f"https://yupp.ai/chat/{url_uuid}?stream=true"
                    next_action = kwargs.get("next_action", "7f1e9eec4ab22c8bfc73a50c026db603cd8380f87d")

                headers = {
                    "accept": "text/x-component",
                    "content-type": "text/plain;charset=UTF-8",
                    "next-action": next_action,
                    "cookie": f"__Secure-yupp.session-token={account['token']}",
                }

                log_debug(f"Sending request to: {url}")
                log_debug(f"Payload structure: {type(payload)}, length: {len(str(payload))}")

                # Send request
                response = session.post(url, data=json.dumps(payload), headers=headers, stream=True, timeout=60)
                response.raise_for_status()

                # Attempt to make chat private
                try:
                    make_chat_private(account, url_uuid)
                except Exception as e:
                    log_debug(f"Failed to set chat private for {url_uuid}: {e}")

                # Yield streaming responses
                yield from cls._process_stream_response(response.iter_lines(), account, session, prompt, model)

                return  # Exit after successful completion

            except RateLimitError:
                log_debug(f"Account ...{account['token'][-4:]} hit rate limit, rotating")
                with account_rotation_lock:
                    account["error_count"] += 1
                continue
            except ProviderException as e:
                log_debug(f"Account ...{account['token'][-4:]} failed: {str(e)}")
                with account_rotation_lock:
                    if "auth" in str(e).lower() or "401" in str(e) or "403" in str(e):
                        account["is_valid"] = False
                    else:
                        account["error_count"] += 1
                continue
            except Exception as e:
                log_debug(f"Unexpected error with account ...{account['token'][-4:]}: {str(e)}")
                with account_rotation_lock:
                    account["error_count"] += 1
                raise ProviderException(f"Yupp request failed: {str(e)}") from e

        raise ProviderException("All Yupp accounts failed after rotation attempts")

    @classmethod
    def _process_stream_response(
        cls,
        response_lines: Iterable[bytes],
        account: Dict[str, Any],
        session: requests.Session,
        prompt: str,
        model_id: str
    ) -> Generator[str, Any, None]:
        """Process Yupp stream response and convert to OpenAI format"""

        line_pattern = re.compile(b"^([0-9a-fA-F]+):(.*)")
        chunks = {}
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
            """Check if content is valid, avoid over-filtering"""
            if not content or content in [None, "", "$undefined"]:
                return False
            
            return True

        def process_content_chunk(content: str, chunk_id: str, line_count: int):
            """Process single content chunk"""
            nonlocal is_thinking, thinking_content, normal_content, session
            
            if not is_valid_content(content):
                return
            
            if '<yapp class="image-gen">' in content:
                content = content.split('<yapp class="image-gen">').pop().split('</yapp>')[0]
                url = f"https://yupp.ai/api/trpc/chat.getSignedImage"
                response = session.get(url, params={"batch": "1", "input": json.dumps({"0": {"json": {"imageId": json.loads(content).get("image_id")}}})})
                response.raise_for_status()
                yield ImageResponse(response.json()[0]["result"]["data"]["json"]["signed_url"], prompt)
                return
            
            # log_debug(f"Processing chunk #{line_count} with content: '{content[:50]}...'")
            
            if is_thinking:
                yield Reasoning(content)
            else:
                normal_content += content
                yield content
        
        try:
            # log_debug("Starting to process Yupp stream response...")
            line_count = 0
            quick_response_id = None
            variant_stream_id = None
            is_started: bool = False
            variant_image: Optional[ImageResponse] = None
            variant_text = ""
            
            for line in response_lines:

                line_count += 1
                
                match = line_pattern.match(line)
                if not match:
                    log_debug(f"Line {line_count}: No pattern match")
                    continue
                
                chunk_id, chunk_data = match.groups()
                chunk_id = chunk_id.decode()
                
                try:
                    data = json.loads(chunk_data) if chunk_data != b"{}" else {}
                    chunks[chunk_id] = data
                except json.JSONDecodeError:
                    log_debug(f"Failed to parse JSON for chunk {chunk_id}")
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
                            for chunk in process_content_chunk(content, chunk_id, line_count):
                                is_started = True
                                yield chunk
                
                elif variant_stream_id and chunk_id == variant_stream_id:
                    yield PlainTextResponse("[Variant] " + line.decode(errors="ignore"))
                    if isinstance(data, dict):
                        variant_stream_id = extract_ref_id(data.get("next"))
                        content = data.get("curr", "")
                        if content:
                            for chunk in process_content_chunk(content, chunk_id, line_count):
                                if isinstance(chunk, ImageResponse):
                                    variant_image = chunk
                                    yield PreviewResponse(str(variant_image))
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

                # Fallback: process any chunk with "curr"
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
                    try:
                        claim_yupp_reward(account, reward_id)
                    except Exception as e:
                        log_debug(f"Failed to claim reward: {e}")
            
            # log_debug(f"Stream completed. Content length: {len(normal_content)}")
    
# Initialize the provider
def init_yupp_provider():
    """Initialize Yupp provider with environment configuration"""
    tokens = os.getenv("YUPP_TOKENS", "")
    if tokens:
        load_yupp_accounts(tokens)
    
    log_debug(f"Yupp provider initialized: {len(YUPP_ACCOUNTS)} accounts, {len(YUPP_MODELS)} models")
    return Yupp

# Example usage and testing
if __name__ == "__main__":
    # Set up environment for testing
    os.environ["DEBUG_MODE"] = "true"
    
    # Initialize provider
    provider = init_yupp_provider()
    
    # Test stream completion
    try:
        print("\nTesting stream completion...")
        for chunk in provider.create_completion(
            model="claude-sonnet-4-5-20250929<>thinking",
            messages=[{"role": "user", "content": "What is Python?"}],
            stream=True
        ):
            if isinstance(chunk, str) and chunk.strip():
                print(chunk, end="")
    except Exception as e:
        print(f"\nStream test failed: {e}")