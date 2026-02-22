from __future__ import annotations

import os
import json
import time
import uuid
import base64
from datetime import datetime
from typing import AsyncIterator
from pathlib import Path

from g4f.typing import AsyncResult, Messages, Cookies
from g4f.requests import StreamSession, raise_for_status, sse_stream, FormData
from g4f.cookies import get_cookies, get_headers, get_cookies_dir
from g4f.providers.response import (
    JsonConversation, JsonRequest, JsonResponse, 
    Reasoning, FinishReason
)
from g4f.providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from g4f.providers.helper import get_last_user_message
from g4f import debug
from g4f.errors import MissingAuthError
from g4f.image import to_bytes

# Inline PoW (Proof of Work) implementation for DeepSeek
# Based on reference implementation in gpt4free/projects/deepseek4free/dsk/pow.py

try:
    import wasmtime
    import numpy
    has_wasmtime_and_numpy = True
except ImportError:
    has_wasmtime_and_numpy = False

try:
    from curl_cffi import CurlHttpVersion
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

WASM_PATH = os.path.join(os.path.dirname(__file__), "deepseek", "pow_solver.wasm")

class DeepSeekHash:
    """Custom SHA3 hash solver using WebAssembly"""
    
    def __init__(self):
        self.instance = None
        self.memory = None
        self.store = None
        
    def init(self, wasm_path: str):
        if not has_wasmtime_and_numpy:
            raise ImportError("wasmtime and numpy are required for PoW solving")
        
        if not Path(wasm_path).exists():
            raise FileNotFoundError(f"WASM file not found: {wasm_path}")
        
        engine = wasmtime.Engine()
        
        with open(wasm_path, 'rb') as f:
            wasm_bytes = f.read()
            
        module = wasmtime.Module(engine, wasm_bytes)
        
        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        
        self.instance = linker.instantiate(self.store, module)
        self.memory = self.instance.exports(self.store)["memory"]
        
        return self
    
    def _write_to_memory(self, text: str) -> tuple[int, int]:
        encoded = text.encode('utf-8')
        length = len(encoded)
        ptr = self.instance.exports(self.store)["__wbindgen_export_0"](self.store, length, 1)
        
        memory_view = self.memory.data_ptr(self.store)
        for i, byte in enumerate(encoded):
            memory_view[ptr + i] = byte
            
        return ptr, length
    
    def calculate_hash(self, algorithm: str, challenge: str, salt: str, 
                      difficulty: int, expire_at: int) -> int:
        
        prefix = f"{salt}_{expire_at}_"
        retptr = self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](self.store, -16)
        
        try:
            challenge_ptr, challenge_len = self._write_to_memory(challenge)
            prefix_ptr, prefix_len = self._write_to_memory(prefix)
            
            self.instance.exports(self.store)["wasm_solve"](
                self.store,
                retptr,
                challenge_ptr,
                challenge_len,
                prefix_ptr,
                prefix_len,
                float(difficulty)
            )
            
            memory_view = self.memory.data_ptr(self.store)
            status = int.from_bytes(bytes(memory_view[retptr:retptr + 4]), byteorder='little', signed=True)
            
            if status == 0:
                return None
            
            value_bytes = bytes(memory_view[retptr + 8:retptr + 16])
            value = numpy.frombuffer(value_bytes, dtype=numpy.float64)[0]
            
            return int(value)
            
        finally:
            self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](self.store, 16)


class DeepSeekPOW:
    """Proof of Work solver for DeepSeek challenges"""
    
    def __init__(self):
        self.hasher = DeepSeekHash().init(WASM_PATH)
    
    def solve_challenge(self, config: dict) -> str:
        """Solves a proof-of-work challenge and returns the encoded response"""
        answer = self.hasher.calculate_hash(
            config['algorithm'],
            config['challenge'],
            config['salt'],
            config['difficulty'],
            config['expire_at']
        )
        
        result = {
            'algorithm': config['algorithm'],
            'challenge': config['challenge'],
            'salt': config['salt'],
            'answer': answer,
            'signature': config['signature'],
            'target_path': config.get('target_path', '')
        }

        return base64.b64encode(json.dumps(result).encode()).decode()

# DeepSeek API endpoints
DEEPSEEK_URL = "https://chat.deepseek.com"
DEEPSEEK_DOMAIN = "chat.deepseek.com"
CHAT_SESSION_CREATE_ENDPOINT = f"{DEEPSEEK_URL}/api/v0/chat_session/create"
CHAT_SESSION_DELETE_ENDPOINT = f"{DEEPSEEK_URL}/api/v0/chat_session/delete"
CHAT_COMPLETION_ENDPOINT = f"{DEEPSEEK_URL}/api/v0/chat/completion"
POW_CHALLENGE_ENDPOINT = f"{DEEPSEEK_URL}/api/v0/chat/create_pow_challenge"
FILE_UPLOAD_ENDPOINT = f"{DEEPSEEK_URL}/v0/file/upload_file"

def generate_client_stream_id() -> str:
    """
    Generate DeepSeek client_stream_id in format: YYYYMMDD-<hex_string>
    Based on HAR file analysis of DeepSeek web client.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    # Generate a random hex string (16 chars like in HAR)
    hex_part = uuid.uuid4().hex[:16]
    return f"{date_str}-{hex_part}"

class DeepSeekAPI(AsyncGeneratorProvider, ProviderModelMixin):
    """
    DeepSeek provider using browser emulation with HAR file support.
    
    This provider extends DeepSeek implementation with HAR file support
    for easier authentication management. It uses curl_cffi's Chrome impersonation
    for realistic browser-like requests.
    """
    
    label = "DeepSeek (HAR Auth)"
    url = DEEPSEEK_URL
    cookie_domain = DEEPSEEK_DOMAIN
    working = has_wasmtime_and_numpy
    active_by_default = True
    needs_auth = True
    supports_file_upload = True
    
    default_model = "deepseek-v3"
    models = ["deepseek-v3", "deepseek-r1"]
    model_aliases = {"deepseek-chat": "deepseek-v3"}
    
    @classmethod
    async def upload_file(
        cls,
        session: StreamSession,
        file: bytes,
        filename: str = None
    ) -> dict:
        """
        Upload a file to DeepSeek.
        
        Returns dict with file info including file_id
        """
        if not filename:
            filename = "document.pdf"
        
        debug.log(f"DeepSeekAuth: Starting file upload: {filename} ({len(file)} bytes)")
        debug.log(f"DeepSeekAuth: Upload endpoint: {FILE_UPLOAD_ENDPOINT}")
        
        # Create multipart form data
        data = FormData()
        data.add_field("file", file, filename=filename, content_type="application/pdf")
        
        async with session.post(
            FILE_UPLOAD_ENDPOINT,
            data=data,
            headers={"accept": "application/json"}
        ) as response:
            debug.log(f"DeepSeekAuth: File upload response status: {response.status}")
            await raise_for_status(response)
            result = await response.json()
            debug.log(f"DeepSeekAuth: File upload response: {result}")
            
        if "data" in result:
            file_id = result["data"].get("id")
            debug.log(f"DeepSeekAuth: File uploaded successfully, file_id: {file_id}")
            return {
                "file_id": file_id,
                "filename": filename,
                "size": len(file)
            }
        else:
            debug.error(f"DeepSeekAuth: Failed to upload file: {result}")
            raise Exception(f"Failed to upload file: {result}")
    
    @classmethod
    async def delete_chat_session(
        cls,
        session: StreamSession,
        chat_session_id: str,
        headers: dict
    ):
        """
        Delete a chat session from DeepSeek.
        
        Tries multiple approaches (DELETE/POST with JSON body/query params) until one succeeds.
        
        Args:
            session: StreamSession instance
            chat_session_id: The session ID to delete
            headers: Request headers including authorization
        """
        import json as json_module
        
        # Try different deletion approaches - POST with JSON body first (as seen in HAR)
        deletion_methods = [
            {
                "name": "POST with JSON body",
                "method": "post",
                "url": CHAT_SESSION_DELETE_ENDPOINT,
                "use_json_body": True,
            },
            {
                "name": "POST with query params",
                "method": "post",
                "url": f"{CHAT_SESSION_DELETE_ENDPOINT}?chat_session_id={chat_session_id}",
                "use_json_body": False,
            },
            {
                "name": "DELETE with JSON body",
                "method": "delete",
                "url": CHAT_SESSION_DELETE_ENDPOINT,
                "use_json_body": True,
            },
            {
                "name": "DELETE with query params",
                "method": "delete",
                "url": f"{CHAT_SESSION_DELETE_ENDPOINT}?chat_session_id={chat_session_id}",
                "use_json_body": False,
            },
        ]
        
        for method_info in deletion_methods:
            try:
                debug.log(f"DeepSeekAuth: Attempting deletion - {method_info['name']}")
                debug.log(f"DeepSeekAuth:   URL: {method_info['url']}")
                
                # Prepare request parameters
                request_params = {}
                if method_info['use_json_body']:
                    request_params['json'] = {"chat_session_id": chat_session_id}
                    debug.log(f"DeepSeekAuth:   JSON body: {{'chat_session_id': '{chat_session_id}'}}")
                else:
                    debug.log(f"DeepSeekAuth:   Query params: chat_session_id={chat_session_id}")
                
                # Make the request - pass headers to each request
                if method_info['method'] == 'delete':
                    async with session.delete(method_info['url'], headers=headers, **request_params) as response:
                        debug.log(f"DeepSeekAuth:   Response status: {response.status}")
                        debug.log(f"DeepSeekAuth:   Response headers: {dict(response.headers)}")
                        await raise_for_status(response)
                        result = await response.json()
                        debug.log(f"DeepSeekAuth:   Response body: {result}")
                        debug.log(f"DeepSeekAuth: Chat session deleted successfully using {method_info['name']}")
                        return  # Success - exit early
                else:  # POST
                    async with session.post(method_info['url'], headers=headers, **request_params) as response:
                        debug.log(f"DeepSeekAuth:   Response status: {response.status}")
                        debug.log(f"DeepSeekAuth:   Response headers: {dict(response.headers)}")
                        await raise_for_status(response)
                        result = await response.json()
                        debug.log(f"DeepSeekAuth:   Response body: {result}")
                        debug.log(f"DeepSeekAuth: Chat session deleted successfully using {method_info['name']}")
                        return  # Success - exit early
            
            except Exception as e:
                debug.error(f"DeepSeekAuth: Failed to delete using {method_info['name']}: {e}")
                # Continue to next method
        
        # All methods failed
        debug.error(f"DeepSeekAuth: All deletion methods failed for session {chat_session_id}")
        # Don't raise - deletion is not critical
    
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        cookies: Cookies = None,
        headers: dict = None,
        proxy: str = None,
        conversation: JsonConversation = None,
        web_search: bool = False,
        media: list = None,
        delete_session: bool = False,
        **kwargs
    ) -> AsyncResult:
        """
        Create async generator for DeepSeek requests with HAR file support.
        
        Authentication priority:
        1. HAR file cookies and auth token (har_and_cookies/deepseek*.har)
        2. Cookie jar from get_cookies()
        
        Note: DeepSeek requires proof-of-work challenge which may require
        additional handling. This implementation provides basic HAR-based auth.
        
        Args:
            model: Model name to use
            messages: Message history
            cookies: Optional cookies
            proxy: Optional proxy
            conversation: JsonConversation object for continuing sessions
            web_search: Enable web search
            media: List of (file_bytes, filename) tuples for file upload
        """
        if not model:
            model = cls.default_model
        
        # Try to get auth from HAR file first
        if cookies is None:
            cookies = get_cookies(cls.cookie_domain, False)
            headers = get_headers(cls.cookie_domain)
            if cookies:
                debug.log(f"DeepSeekAuth: Using {len(cookies)} cookies and {len(headers)} headers from cookie jar")
            else:
                raise MissingAuthError(
                    "DeepSeekAuth: No authentication found. "
                    "Please add a DeepSeek HAR file to har_and_cookies/ directory "
                    "with an authorization token."
                )
        
        # Initialize conversation if needed
        if conversation is None:
            conversation = JsonConversation(
                parent_message_id=None
            )
        
        # Get auth token from HAR data or conversation
        authorization = None
        if headers:
            authorization = headers.get("authorization")
        elif hasattr(conversation, 'authorization'):
            authorization = conversation.authorization
        
        if not authorization:
            raise MissingAuthError(
                "DeepSeekAuth: Authorization token required. "
                "Please ensure HAR file contains authorization header."
            )
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
            "x-app-version": "20241129.1",
            "x-client-locale": "en_US",
            "x-client-platform": "web",
            "x-client-timezone-offset": "-28800",
            "x-client-version": "1.7.0",
            "authorization": authorization,
        }
        
        # Extract query from messages
        prompt = get_last_user_message(messages)
        
        # Determine thinking mode
        thinking_enabled = bool(model) and "deepseek-r1" in model
        
        yield JsonRequest.from_dict({
            "prompt": prompt,
            "thinking_enabled": thinking_enabled,
            "search_enabled": web_search,
        })
        
        # Get proof-of-work challenge (required by DeepSeek)
        debug.log(f"DeepSeekAuth: Requesting PoW challenge from {POW_CHALLENGE_ENDPOINT}")
        async with StreamSession(
            headers=headers, 
            cookies=cookies, 
            proxy=proxy, 
            impersonate="chrome"
        ) as session:
            async with session.post(
                POW_CHALLENGE_ENDPOINT,
                json={"target_path": "/api/v0/chat/completion"}
            ) as response:
                await raise_for_status(response)
                pow_data = await response.json()
                debug.log("DeepSeekAuth: PoW challenge received")
                
                # Extract challenge data
                if 'data' in pow_data and 'biz_data' in pow_data['data']:
                    challenge = pow_data['data']['biz_data']['challenge']
                    debug.log(f"DeepSeekAuth: Challenge: algorithm={challenge.get('algorithm')}, difficulty={challenge.get('difficulty')}")
                    
                    # Use inline PoW solver to solve the challenge
                    pow_solver = DeepSeekPOW()
                    pow_response_str = pow_solver.solve_challenge(challenge)
                    debug.log(f"DeepSeekAuth: PoW challenge solved successfully")
                    headers["x-ds-pow-response"] = pow_response_str
        
        # Always create a new chat session for the first request
        if not hasattr(conversation, 'chat_session_id') or not conversation.chat_session_id:
            debug.log(f"DeepSeekAuth: Creating new chat session...")
            async with StreamSession(
                headers=headers, 
                cookies=cookies, 
                proxy=proxy, 
                impersonate="chrome"
            ) as session:
                async with session.post(CHAT_SESSION_CREATE_ENDPOINT) as response:
                    await raise_for_status(response)
                    session_data = await response.json()
                    # ID is nested in data.biz_data.id
                    if ('data' in session_data and 
                        'biz_data' in session_data['data'] and 
                        'id' in session_data['data']['biz_data']):
                        chat_session_id = session_data['data']['biz_data']['id']
                        conversation.chat_session_id = chat_session_id
                        debug.log(f"DeepSeekAuth: Chat session created: {chat_session_id}")
                    else:
                        debug.error(f"DeepSeekAuth: Unexpected session response: {session_data}")
                        raise Exception(f"Failed to parse session response: {session_data}")
        else:
            debug.log(f"DeepSeekAuth: Reusing existing chat session: {conversation.chat_session_id}")
        
        # Yield conversation object so caller can reuse it for subsequent messages
        yield conversation
        
        # Upload file if provided - use HTTP/1.1 to avoid HTTP/2 stream errors
        ref_file_ids = []
        if media is not None and len(media) > 0:
            # Take first file from media list
            file_bytes, filename = media[0]
            async with StreamSession(
                headers=headers, 
                cookies=cookies, 
                proxy=proxy, 
                impersonate="chrome",
                http_version=CurlHttpVersion.V1_1 if has_curl_cffi else None  # Force HTTP/1.1 to avoid HTTP/2 stream errors
            ) as session:
                upload_result = await cls.upload_file(session, file_bytes, filename)
                ref_file_ids.append(upload_result["file_id"])
                debug.log(f"DeepSeekAuth: Using file_id: {upload_result['file_id']}")
        
        # Build request data
        json_data = {
            "chat_session_id": getattr(conversation, 'chat_session_id', str(uuid.uuid4())),
            "prompt": prompt,
            "ref_file_ids": ref_file_ids,
            "thinking_enabled": thinking_enabled,
            "search_enabled": web_search,
            "client_stream_id": generate_client_stream_id(),
        }
        
        # Add parent_message_id if continuing conversation
        if hasattr(conversation, 'parent_message_id') and conversation.parent_message_id:
            json_data["parent_message_id"] = conversation.parent_message_id
        
        # debug.log(f"DeepSeekAuth: Sending request to {CHAT_COMPLETION_ENDPOINT}")
        
        async with StreamSession(
            headers=headers, 
            cookies=cookies, 
            proxy=proxy, 
            impersonate="chrome"
        ) as session:
            async with session.post(CHAT_COMPLETION_ENDPOINT, json=json_data) as response:
                # debug.log(f"DeepSeekAuth: Processing response... status={response.status}, content-type={response.headers.get('content-type', 'unknown')}")
                await raise_for_status(response)
                
                # Check if response is actually SSE or regular JSON
                content_type = response.headers.get('content-type', '')
                if 'text/event-stream' not in content_type.lower():
                    raise RuntimeError(f"Expected SSE response but got content-type: {content_type}")
                
                is_thinking = False
                async for stream_data in sse_stream(response):
                    # Handle different stream data formats
                    if isinstance(stream_data, dict):
                        # Handle first chunk with message IDs (for conversation continuity)
                        if 'response_message_id' in stream_data:
                            conversation.parent_message_id = stream_data['response_message_id']
                            # debug.log(f"DeepSeekAuth: Set parent_message_id to {conversation.parent_message_id}")
                        
                        # Handle initial response with fragments (most common case)
                        # Format: {'v': {'response': {'fragments': [{'content': '42', ...}]}}}
                        if 'v' in stream_data and isinstance(stream_data['v'], dict):
                            response_obj = stream_data['v'].get('response', {})
                            fragments = response_obj.get('fragments', [])
                            for fragment in fragments:
                                if isinstance(fragment, dict) and 'content' in fragment:
                                    if fragment.get('type') == 'THINK':
                                        is_thinking = True
                                    content = fragment['content']
                                    if isinstance(content, str) and content:
                                        yield Reasoning(content) if is_thinking else content
                                        # debug.log(f"DeepSeekAuth: Initial fragment content: '{content}'")
                        
                        # Handle APPEND operations that create new fragments with initial content
                        elif ('p' in stream_data and stream_data['p'] == 'response/fragments' and 
                            'o' in stream_data and stream_data['o'] == 'APPEND' and 
                            'v' in stream_data and isinstance(stream_data['v'], list)):
                            
                            # Extract content from the new fragment
                            for fragment in stream_data['v']:
                                if isinstance(fragment, dict) and 'content' in fragment and isinstance(fragment['content'], str):
                                    is_thinking = False 
                                    yield fragment['content']
                                    # debug.log(f"DeepSeekAuth: APPEND fragment content: '{fragment['content']}'")
                        
                        # Handle path-based updates (like 'response/fragments/-1/content')
                        elif 'p' in stream_data and 'v' in stream_data:
                            path = stream_data['p']
                            value = stream_data['v']
                            
                            # Handle content updates
                            if path.endswith('/content') and isinstance(value, str):
                                yield Reasoning(value) if is_thinking else value
                                # debug.log(f"DeepSeekAuth: Content update: '{value}'")
                            
                            # Handle status updates
                            elif path == 'response/status' and value == 'FINISHED':
                                # debug.log("DeepSeekAuth: Stream finished")
                                break
                        
                        # Handle batch updates
                        elif 'o' in stream_data and stream_data['o'] == 'BATCH' and 'v' in stream_data:
                            for batch_item in stream_data['v']:
                                if isinstance(batch_item, dict) and 'p' in batch_item and 'v' in batch_item:
                                    if batch_item['p'] == 'response/status' and batch_item['v'] == 'FINISHED':
                                        # debug.log("DeepSeekAuth: Stream finished (batch)")
                                        break
                        
                        # Handle shorthand content updates
                        elif 'v' in stream_data and isinstance(stream_data['v'], str):
                            yield Reasoning(stream_data['v']) if is_thinking else stream_data['v']
                            # debug.log(f"DeepSeekAuth: Shorthand content: '{stream_data['v']}'")

                
                # Ensure we yield the conversation object at the end
                yield conversation

                # Delete chat session only if explicitly requested (when conversation is fully done)
                if delete_session and hasattr(conversation, 'chat_session_id') and conversation.chat_session_id:
                    async with StreamSession(
                        headers=headers,
                        cookies=cookies,
                        proxy=proxy,
                        impersonate="chrome"
                    ) as delete_session_obj:
                        await cls.delete_chat_session(
                            delete_session_obj,
                            conversation.chat_session_id,
                            headers
                        )