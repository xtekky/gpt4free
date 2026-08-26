from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Literal

from g4f import debug
from g4f.cookies import get_cookies, get_headers
from g4f.errors import MissingAuthError
from g4f.image import to_bytes, detect_file_type
from g4f.providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from g4f.providers.helper import get_last_user_message
from g4f.providers.response import (
    FinishReason,
    JsonConversation,
    JsonRequest,
    Reasoning,
)
from g4f.requests import StreamSession, raise_for_status, FormData
from g4f.typing import AsyncResult, Messages, Cookies

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

        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()

        module = wasmtime.Module(engine, wasm_bytes)

        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()

        self.instance = linker.instantiate(self.store, module)
        self.memory = self.instance.exports(self.store)["memory"]

        return self

    def _write_to_memory(self, text: str) -> tuple[int, int]:
        encoded = text.encode("utf-8")
        length = len(encoded)
        ptr = self.instance.exports(self.store)["__wbindgen_export_0"](
            self.store, length, 1
        )

        memory_view = self.memory.data_ptr(self.store)
        for i, byte in enumerate(encoded):
            memory_view[ptr + i] = byte

        return ptr, length

    def calculate_hash(
            self, algorithm: str, challenge: str, salt: str, difficulty: int, expire_at: int
    ) -> int:
        prefix = f"{salt}_{expire_at}_"
        retptr = self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](
            self.store, -16
        )

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
                float(difficulty),
            )

            memory_view = self.memory.data_ptr(self.store)
            status = int.from_bytes(
                bytes(memory_view[retptr: retptr + 4]), byteorder="little", signed=True
            )

            if status == 0:
                return None

            value_bytes = bytes(memory_view[retptr + 8: retptr + 16])
            value = numpy.frombuffer(value_bytes, dtype=numpy.float64)[0]

            return int(value)

        finally:
            self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](
                self.store, 16
            )


class DeepSeekPOW:
    """Proof of Work solver for DeepSeek challenges"""

    def __init__(self):
        self.hasher = DeepSeekHash().init(WASM_PATH)

    def solve_challenge(self, config: dict) -> str:
        """Solves a proof-of-work challenge and returns the encoded response"""
        answer = self.hasher.calculate_hash(
            config["algorithm"],
            config["challenge"],
            config["salt"],
            config["difficulty"],
            config["expire_at"],
        )

        result = {
            "algorithm": config["algorithm"],
            "challenge": config["challenge"],
            "salt": config["salt"],
            "answer": answer,
            "signature": config["signature"],
            "target_path": config.get("target_path", ""),
        }

        return base64.b64encode(json.dumps(result).encode()).decode()


# DeepSeek API endpoints
DEEPSEEK_URL = "https://chat.deepseek.com"
DEEPSEEK_DOMAIN = "chat.deepseek.com"
CHAT_SESSION_CREATE_ENDPOINT = f"{DEEPSEEK_URL}/api/v0/chat_session/create"
CHAT_SESSION_CONTINUE_ENDPOINT = f"{DEEPSEEK_URL}/api/v0/chat/continue"
CHAT_SESSION_RESUME_STREAM_ENDPOINT = f"{DEEPSEEK_URL}/api/v0/chat/resume_stream"
CHAT_SESSION_DELETE_ENDPOINT = f"{DEEPSEEK_URL}/api/v0/chat_session/delete"
CHAT_COMPLETION_ENDPOINT = f"{DEEPSEEK_URL}/api/v0/chat/completion"
POW_CHALLENGE_ENDPOINT = f"{DEEPSEEK_URL}/api/v0/chat/create_pow_challenge"
CHAT_COMPLETION_PATH = "/api/v0/chat/completion"
FILE_UPLOAD_PATH = "/api/v0/file/upload_file"
FILE_UPLOAD_ENDPOINT = f"{DEEPSEEK_URL}{FILE_UPLOAD_PATH}"
FILE_FETCH_ENDPOINT = f"{DEEPSEEK_URL}/api/v0/file/fetch_files"

DEEPSEEK_MESSAGE_STATUSES = {
    "FINISHED",
    "CONTENT_FILTER",
    "CONTEXT_LENGTH_EXCEEDED",
    "INCOMPLETE",
    "WIP",
    "TIMEOUT",
}
DEEPSEEK_FINISH_REASONS = {
    "FINISHED": "stop",
    "CONTENT_FILTER": "content_filter",
    "CONTEXT_LENGTH_EXCEEDED": "length",
    "INCOMPLETE": "incomplete",
    "WIP": "wip",
    "TIMEOUT": "timeout",
}

DEEPSEEK_RESPONSE_FRAGMENT_TYPES = {"RESPONSE", "TEMPLATE_RESPONSE"}
DEEPSEEK_REASONING_FRAGMENT_TYPES = {
    "THINK",
    "THINKING",
    "REASONING",
    "SEARCH",
    "TOOL_SEARCH",
    "TOOL_OPEN",
    "TOOL_FIND",
}
DEEPSEEK_METADATA_FRAGMENT_TYPES = {
    "REQUEST",
    "FILE",
    "TIP",
    "READ_LINK",
}
DEEPSEEK_STREAM_OPERATIONS = {"SET", "BATCH", "APPEND"}

CHAT_HEADER_DEFAULTS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "application/json",
    "origin": DEEPSEEK_URL,
    "referer": f"{DEEPSEEK_URL}/",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
    ),
    "x-app-version": "20241129.1",
    "x-client-bundle-id": "com.deepseek.chat",
    "x-client-locale": "en_US",
    "x-client-platform": "web",
    "x-client-timezone-offset": "-28800",
    "x-client-version": "2.4.0",
}

CHAT_HEADER_PASSTHROUGH = {
    "accept-language",
    "dnt",
    "priority",
    "referer",
    "sec-ch-ua",
    "sec-ch-ua-arch",
    "sec-ch-ua-bitness",
    "sec-ch-ua-full-version",
    "sec-ch-ua-full-version-list",
    "sec-ch-ua-mobile",
    "sec-ch-ua-model",
    "sec-ch-ua-platform",
    "sec-ch-ua-platform-version",
    "sec-fetch-dest",
    "sec-fetch-mode",
    "sec-fetch-site",
    "user-agent",
    "x-app-version",
    "x-client-bundle-id",
    "x-client-locale",
    "x-client-platform",
    "x-client-timezone-offset",
    "x-client-version",
    "x-hif-leim",
}


def _normalized_headers(headers: Optional[dict]) -> dict:
    return {
        str(key).lower(): value
        for key, value in (headers or {}).items()
        if value is not None
    }


def _extract_chat_session_id(session_data: Any) -> Optional[str]:
    """Read the current and legacy chat-session response shapes."""
    data = session_data.get("data") if isinstance(session_data, dict) else None
    biz_data = data.get("biz_data") if isinstance(data, dict) else None
    if not isinstance(biz_data, dict):
        return None

    chat_session = biz_data.get("chat_session")
    if isinstance(chat_session, dict) and chat_session.get("id"):
        return chat_session["id"]
    return biz_data.get("id")


def _build_chat_headers(source_headers: Optional[dict], authorization: str) -> dict:
    """Build JSON request headers while retaining current browser-bound values."""
    normalized = _normalized_headers(source_headers)
    headers = dict(CHAT_HEADER_DEFAULTS)
    headers.update(
        {
            name: normalized[name]
            for name in CHAT_HEADER_PASSTHROUGH
            if name in normalized
        }
    )
    headers["authorization"] = authorization
    # A PoW answer is valid only for the challenge/target that produced it.
    headers.pop("x-ds-pow-response", None)
    return headers


async def iter_deepseek_sse(response) -> AsyncIterator[tuple[str, Any]]:
    """Parse DeepSeek SSE frames without losing their ``event`` field."""
    event_type = "message"
    data_lines: list[str] = []

    def decode_event() -> Optional[tuple[str, Any]]:
        if not data_lines:
            return None
        raw_data = "\n".join(data_lines)
        if raw_data.strip() == "[DONE]":
            return None
        try:
            return event_type, json.loads(raw_data)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid DeepSeek SSE JSON data: {raw_data!r}") from error

    async for raw_line in response.iter_lines():
        line = (
            raw_line.decode("utf-8")
            if isinstance(raw_line, (bytes, bytearray))
            else str(raw_line)
        ).rstrip("\r")

        if not line:
            event = decode_event()
            if event is not None:
                yield event
            event_type = "message"
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        field_name, separator, field_value = line.partition(":")
        if not separator:
            continue
        if field_value.startswith(" "):
            field_value = field_value[1:]
        if field_name == "event":
            event_type = field_value or "message"
        elif field_name == "data":
            data_lines.append(field_value)

    event = decode_event()
    if event is not None:
        yield event


@dataclass
class _DeepSeekStreamState:
    message_id: Any = None
    status: Optional[str] = None
    closed: bool = False
    active_kind: Optional[str] = "response"
    emitted: dict[str, str] = field(
        default_factory=lambda: {"reasoning": "", "response": ""}
    )

    def append(self, kind: str, content: str) -> str:
        self.emitted[kind] += content
        return content

    def snapshot_delta(self, kind: str, content: str) -> str:
        """Return only text not already emitted by an earlier stream attempt."""
        previous = self.emitted[kind]
        if content.startswith(previous):
            delta = content[len(previous):]
            self.emitted[kind] = content
            return delta
        if previous.startswith(content):
            return ""

        max_overlap = min(len(previous), len(content))
        for overlap in range(max_overlap, 0, -1):
            if previous.endswith(content[:overlap]):
                delta = content[overlap:]
                self.emitted[kind] += delta
                return delta

        self.emitted[kind] += content
        return content


def _fragment_kind(fragment: dict) -> Optional[str]:
    fragment_type = str(fragment.get("type", "")).upper()
    if not fragment_type:
        # Preserve compatibility with older snapshots that omitted the type.
        return "response"
    if fragment_type in DEEPSEEK_RESPONSE_FRAGMENT_TYPES:
        return "response"
    if fragment_type in DEEPSEEK_REASONING_FRAGMENT_TYPES:
        return "reasoning"
    if fragment_type in DEEPSEEK_METADATA_FRAGMENT_TYPES:
        return None
    return None


def _stream_output(kind: str, content: str):
    if not content:
        return None
    return Reasoning(content) if kind == "reasoning" else content


def _record_response_message_id(
        state: _DeepSeekStreamState,
        conversation: JsonConversation,
        message_id: Any,
) -> None:
    if message_id is not None:
        state.message_id = message_id
        conversation.parent_message_id = message_id


def _record_stream_status(
        state: _DeepSeekStreamState,
        path: str,
        value: Any,
        *,
        operation: Optional[str] = None,
        source: Literal["patch", "snapshot"] = "snapshot",
) -> bool:
    if path not in {"response/status", "quasi_status", "response/quasi_status"}:
        return False
    status = str(value).upper()
    if status in DEEPSEEK_MESSAGE_STATUSES:
        state.status = status
        if status == "INCOMPLETE":
            interpretation = "requires_continue"
        else:
            interpretation = None

        operation_label = operation if operation is not None else "none"
        message = (
            "DeepSeekAuth: Stream status: "
            f"status={status} source={source} operation={operation_label}"
        )
        if interpretation is not None:
            message += f" interpretation={interpretation}"
        debug.log(message)
    return True


def _stream_log_value(value: Any) -> str:
    """Format a small, non-content stream field for diagnostic logging."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        sanitized = "".join(
            character
            if character.isalnum() or character in {"_", "-", "."}
            else "_"
            for character in value[:64]
        )
        return sanitized or "empty"
    return type(value).__name__


def _process_fragments(
        fragments: list,
        state: _DeepSeekStreamState,
        *,
        snapshot: bool,
) -> list:
    chunks = []
    content_by_kind = {"reasoning": "", "response": ""}
    kind_order = []

    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        kind = _fragment_kind(fragment)
        state.active_kind = kind
        content = fragment.get("content")
        if kind is None or not isinstance(content, str):
            continue

        if snapshot:
            if kind not in kind_order:
                kind_order.append(kind)
            content_by_kind[kind] += content
            continue

        output = _stream_output(kind, state.append(kind, content))
        if output is not None:
            chunks.append(output)

    if snapshot:
        for kind in kind_order:
            output = _stream_output(
                kind,
                state.snapshot_delta(kind, content_by_kind[kind]),
            )
            if output is not None:
                chunks.append(output)

    return chunks


def _process_stream_payload(
        payload: Any,
        state: _DeepSeekStreamState,
        conversation: JsonConversation,
) -> list:
    """Apply one DeepSeek message event and return newly visible output chunks."""
    if not isinstance(payload, dict):
        return []

    chunks = []
    _record_response_message_id(
        state, conversation, payload.get("response_message_id")
    )

    operation = payload.get("o")
    value = payload.get("v")

    if operation == "BATCH" and isinstance(value, list):
        for batch_item in value:
            chunks.extend(
                _process_stream_payload(batch_item, state, conversation)
            )
        return chunks

    if isinstance(value, dict) and isinstance(value.get("response"), dict):
        response_obj = value["response"]
        _record_response_message_id(
            state, conversation, response_obj.get("message_id")
        )
        response_status = response_obj.get("status")
        if response_status is not None:
            _record_stream_status(
                state,
                "response/status",
                response_status,
                source="snapshot",
            )

        fragments = response_obj.get("fragments", [])
        if isinstance(fragments, list):
            chunks.extend(_process_fragments(fragments, state, snapshot=True))
        return chunks

    path = payload.get("p")
    if (
            path == "response/fragments"
            and operation in {"SET", "APPEND"}
            and isinstance(value, list)
    ):
        chunks.extend(
            _process_fragments(value, state, snapshot=operation == "SET")
        )
        return chunks

    if isinstance(path, str) and "v" in payload:
        if _record_stream_status(
                state,
                path,
                value,
                operation=operation,
                source="patch",
        ):
            return chunks
        if path.endswith("/content") and isinstance(value, str):
            kind = state.active_kind
            if kind is None:
                return chunks
            content = (
                state.snapshot_delta(kind, value)
                if operation == "SET"
                else state.append(kind, value)
            )
            output = _stream_output(kind, content)
            if output is not None:
                chunks.append(output)
        return chunks

    if isinstance(value, str):
        kind = state.active_kind
        if kind is None:
            return chunks
        output = _stream_output(kind, state.append(kind, value))
        if output is not None:
            chunks.append(output)

    return chunks


def _build_upload_session_headers(headers: dict) -> dict:
    """Copy shared headers without values that would corrupt a multipart upload."""
    excluded_headers = {"content-type", "x-ds-pow-response"}
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in excluded_headers
    }


def generate_client_stream_id() -> str:
    """
    Generate DeepSeek client_stream_id in format: YYYYMMDD-<hex_string>
    Based on HAR file analysis of DeepSeek web client.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    # Generate a random hex string (16 chars like in HAR)
    hex_part = uuid.uuid4().hex[:16]
    return f"{date_str}-{hex_part}"


class DeepSeek(AsyncGeneratorProvider, ProviderModelMixin):
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
    async def create_pow_response(
            cls, session: StreamSession, target_path: str
    ) -> str:
        """Request and solve a PoW challenge for one exact API target path."""
        debug.log(
            f"DeepSeekAuth: Requesting PoW challenge for {target_path} "
            f"from {POW_CHALLENGE_ENDPOINT}"
        )
        async with session.post(
                POW_CHALLENGE_ENDPOINT,
                json={"target_path": target_path},
                headers={"content-type": "application/json"},
        ) as response:
            await raise_for_status(response)
            pow_data = await response.json()

        try:
            challenge = pow_data["data"]["biz_data"]["challenge"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"DeepSeek returned an invalid PoW challenge for {target_path}"
            ) from error

        if challenge.get("target_path") != target_path:
            raise RuntimeError(
                "DeepSeek returned a PoW challenge for an unexpected target path: "
                f"{challenge.get('target_path')!r}"
            )

        debug.log(
            "DeepSeekAuth: Challenge: "
            f"algorithm={challenge.get('algorithm')}, "
            f"difficulty={challenge.get('difficulty')}"
        )
        pow_response = DeepSeekPOW().solve_challenge(challenge)
        debug.log(f"DeepSeekAuth: PoW challenge solved for {target_path}")
        return pow_response

    @classmethod
    async def upload_file(
            cls,
            session: StreamSession,
            file: bytes,
            filename: str = None,
            thinking_enabled: bool = False,
            model_type: str = "default",
    ) -> dict:
        """
        Upload a file to DeepSeek.

        Returns dict with file info including file_id
        """
        data_bytes = to_bytes(file)
        extension, file_type = detect_file_type(data_bytes)
        filename = filename or f"file-{len(data_bytes)}{extension}"

        debug.log(f"DeepSeekAuth: Starting file upload: {filename} ({len(data_bytes)} bytes)")
        debug.log(f"DeepSeekAuth: Upload endpoint: {FILE_UPLOAD_ENDPOINT}")

        pow_response = await cls.create_pow_response(session, FILE_UPLOAD_PATH)

        # Create multipart form data
        data = FormData()
        data.add_field("file", data_bytes, filename=filename, content_type=file_type)

        upload_headers = {
            "accept": "*/*",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-ds-pow-response": pow_response,
            "x-file-size": str(len(data_bytes)),
            "x-model-type": model_type,
            "x-thinking-enabled": "1" if thinking_enabled else "0",
        }
        async with session.post(
                FILE_UPLOAD_ENDPOINT, data=data, headers=upload_headers
        ) as response:
            debug.log(f"DeepSeekAuth: File upload response status: {response.status}")
            await raise_for_status(response)
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type.lower():
                raise RuntimeError(
                    "DeepSeek file upload returned a non-JSON response "
                    f"(content-type: {content_type or 'unknown'}) from "
                    f"{FILE_UPLOAD_ENDPOINT}"
                )
            result = await response.json()

        response_data = result.get("data") if isinstance(result, dict) else None
        biz_data = response_data.get("biz_data") if isinstance(response_data, dict) else None
        file_id = biz_data.get("id") if isinstance(biz_data, dict) else None
        if not file_id and isinstance(response_data, dict):
            # Keep compatibility with the older, unnested response shape.
            file_id = response_data.get("id")

        if result.get("code") not in (None, 0) or not file_id:
            message = result.get("msg") or "missing data.biz_data.id"
            raise RuntimeError(f"DeepSeek file upload failed: {message}")

        debug.log(f"DeepSeekAuth: File uploaded successfully, file_id: {file_id}")
        return {
            "file_id": file_id,
            "filename": filename,
            "size": len(data_bytes),
        }

    @classmethod
    async def wait_for_file_parsed(
            cls,
            session: StreamSession,
            file_id: str,
            timeout: float = 120,
            poll_interval: float = 1,
    ) -> None:
        """Wait until DeepSeek finishes extracting the uploaded file."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            async with session.get(
                    FILE_FETCH_ENDPOINT,
                    params={"file_ids": file_id},
                    headers={"accept": "application/json"},
            ) as response:
                await raise_for_status(response)
                result = await response.json()

            response_data = result.get("data") if isinstance(result, dict) else None
            biz_data = response_data.get("biz_data") if isinstance(response_data, dict) else None
            files = biz_data.get("files") if isinstance(biz_data, dict) else None
            file_info = files[0] if isinstance(files, list) and files else {}
            status = str(file_info.get("status", "")).upper()

            if status == "SUCCESS":
                debug.log(f"DeepSeekAuth: File parsing completed, file_id: {file_id}")
                return
            if status in {"FAILED", "ERROR"}:
                error_code = file_info.get("error_code") or "unknown"
                raise RuntimeError(
                    f"DeepSeek file parsing failed for {file_id}: {error_code}"
                )
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"DeepSeek file parsing timed out after {timeout:g}s for {file_id}"
                )

            await asyncio.sleep(poll_interval)

    @classmethod
    async def delete_chat_session(
            cls, session: StreamSession, chat_session_id: str, headers: dict
    ):
        """
        Delete a chat session from DeepSeek.

        Tries multiple approaches (DELETE/POST with JSON body/query params) until one succeeds.

        Args:
            session: StreamSession instance
            chat_session_id: The session ID to delete
            headers: Request headers including authorization
        """

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
                if method_info["use_json_body"]:
                    request_params["json"] = {"chat_session_id": chat_session_id}
                    debug.log(
                        f"DeepSeekAuth:   JSON body: {{'chat_session_id': '{chat_session_id}'}}"
                    )
                else:
                    debug.log(
                        f"DeepSeekAuth:   Query params: chat_session_id={chat_session_id}"
                    )

                # Make the request - pass headers to each request
                if method_info["method"] == "delete":
                    async with session.delete(
                            method_info["url"], headers=headers, **request_params
                    ) as response:
                        debug.log(f"DeepSeekAuth:   Response status: {response.status}")
                        debug.log(
                            f"DeepSeekAuth:   Response headers: {dict(response.headers)}"
                        )
                        await raise_for_status(response)
                        result = await response.json()
                        debug.log(f"DeepSeekAuth:   Response body: {result}")
                        debug.log(
                            f"DeepSeekAuth: Chat session deleted successfully using {method_info['name']}"
                        )
                        return  # Success - exit early
                else:  # POST
                    async with session.post(
                            method_info["url"], headers=headers, **request_params
                    ) as response:
                        debug.log(f"DeepSeekAuth:   Response status: {response.status}")
                        debug.log(
                            f"DeepSeekAuth:   Response headers: {dict(response.headers)}"
                        )
                        await raise_for_status(response)
                        result = await response.json()
                        debug.log(f"DeepSeekAuth:   Response body: {result}")
                        debug.log(
                            f"DeepSeekAuth: Chat session deleted successfully using {method_info['name']}"
                        )
                        return  # Success - exit early

            except Exception as e:
                debug.error(
                    f"DeepSeekAuth: Failed to delete using {method_info['name']}: {e}"
                )
                # Continue to next method

        # All methods failed
        debug.error(
            f"DeepSeekAuth: All deletion methods failed for session {chat_session_id}"
        )
        # Don't raise - deletion is not critical

    @classmethod
    async def get_quota(cls, **kwargs):
        cookies = get_cookies(cls.cookie_domain, False)
        headers = get_headers(cls.cookie_domain)
        if cookies and headers.get("authorization"):
            return {"success": True}
        raise MissingAuthError("DeepSeekAuth: No authentication found.")

    @classmethod
    async def iter_chat_stream(
            cls,
            session: StreamSession,
            conversation: JsonConversation,
            initial_payload: dict,
            initial_headers: Optional[dict] = None,
            auto_continue: bool = True,
            max_continue_attempts: Optional[int] = 20,
            max_resume_attempts: Optional[int] = 5,
    ) -> AsyncResult:
        """Consume one logical answer across completion, resume, and continue calls."""
        for name, limit in (
                ("max_continue_attempts", max_continue_attempts),
                ("max_resume_attempts", max_resume_attempts),
        ):
            if limit is not None and limit < 0:
                raise ValueError(f"{name} must be non-negative or None")

        chat_session_id = initial_payload.get("chat_session_id")
        if not chat_session_id:
            raise ValueError("DeepSeek chat_session_id is required for streaming")

        endpoint = CHAT_COMPLETION_ENDPOINT
        payload = initial_payload
        request_headers = initial_headers
        state = _DeepSeekStreamState()
        continue_attempts = 0
        resume_attempts = 0

        while True:
            state.closed = False
            stream_error = None
            close_payload = {}
            request_kwargs = {"json": payload}
            if request_headers:
                request_kwargs["headers"] = request_headers

            async with session.post(endpoint, **request_kwargs) as response:
                await raise_for_status(response)
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type.lower():
                    raise RuntimeError(
                        "Expected SSE response but got content-type: "
                        f"{content_type or 'unknown'}"
                    )

                events = iter_deepseek_sse(response).__aiter__()
                while True:
                    try:
                        event_type, stream_data = await events.__anext__()
                    except StopAsyncIteration:
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        stream_error = error
                        break

                    if event_type == "close":
                        state.closed = True
                        if isinstance(stream_data, dict):
                            close_payload = stream_data
                        break
                    if event_type not in {"message", "ready"}:
                        continue
                    for chunk in _process_stream_payload(
                            stream_data, state, conversation
                    ):
                        yield chunk

            if state.closed:
                resume_attempts = 0
                should_continue = (
                        state.status == "INCOMPLETE"
                        and auto_continue
                )
                close_details = (
                    "auto_resume="
                    f"{_stream_log_value(close_payload.get('auto_resume'))} "
                    "click_behavior="
                    f"{_stream_log_value(close_payload.get('click_behavior'))} "
                    f"response_chars={len(state.emitted['response'])} "
                    f"reasoning_chars={len(state.emitted['reasoning'])} "
                    "message_id_present="
                    f"{_stream_log_value(state.message_id is not None)}"
                )
                if not should_continue:
                    finish_reason = DEEPSEEK_FINISH_REASONS.get(state.status)
                    if state.status == "INCOMPLETE":
                        stop_reason = "auto_continue_disabled"
                    elif state.status is None:
                        stop_reason = "no_status"
                    elif finish_reason is not None:
                        stop_reason = "terminal_status"
                    else:
                        stop_reason = "unknown_status"
                    debug.log(
                        "DeepSeekAuth: Stream closed: "
                        f"status={_stream_log_value(state.status)} action=stop "
                        f"reason={stop_reason} "
                        f"finish_reason={_stream_log_value(finish_reason)} "
                        f"{close_details}"
                    )
                    if finish_reason is not None:
                        yield FinishReason(finish_reason)
                    return
                if state.message_id is None:
                    debug.log(
                        "DeepSeekAuth: Stream closed: "
                        "status=INCOMPLETE action=error "
                        "reason=missing_message_id finish_reason=none "
                        f"{close_details}"
                    )
                    raise RuntimeError(
                        "DeepSeek closed an incomplete stream without a message_id"
                    )
                if (
                        max_continue_attempts is not None
                        and continue_attempts >= max_continue_attempts
                ):
                    debug.log(
                        "DeepSeekAuth: Stream closed: "
                        "status=INCOMPLETE action=error "
                        "reason=max_continue_attempts finish_reason=none "
                        f"{close_details}"
                    )
                    raise RuntimeError(
                        "DeepSeek response remained INCOMPLETE after "
                        f"{continue_attempts} continue attempt(s)"
                    )

                debug.log(
                    "DeepSeekAuth: Stream closed: "
                    "status=INCOMPLETE action=continue "
                    "reason=incomplete_status finish_reason=none "
                    f"{close_details}"
                )
                continue_attempts += 1
                debug.log(
                    "DeepSeekAuth: Continuing incomplete response: "
                    f"action=continue attempt={continue_attempts}"
                )
                endpoint = CHAT_SESSION_CONTINUE_ENDPOINT
                payload = {
                    "chat_session_id": chat_session_id,
                    "message_id": state.message_id,
                    "fallback_to_resume": True,
                }
                request_headers = None
                state.status = None
                continue

            debug.log(
                "DeepSeekAuth: Stream ended without close: "
                f"status={_stream_log_value(state.status)} "
                "message_id_present="
                f"{_stream_log_value(state.message_id is not None)} "
                "error="
                f"{type(stream_error).__name__ if stream_error is not None else 'none'} "
                f"response_chars={len(state.emitted['response'])} "
                f"reasoning_chars={len(state.emitted['reasoning'])}"
            )
            if state.message_id is None:
                debug.log(
                    "DeepSeekAuth: Interrupted stream action: "
                    "action=error reason=missing_message_id"
                )
                message = "DeepSeek stream ended without close or message_id"
                if stream_error is not None:
                    raise RuntimeError(message) from stream_error
                raise RuntimeError(message)
            if (
                    max_resume_attempts is not None
                    and resume_attempts >= max_resume_attempts
            ):
                debug.log(
                    "DeepSeekAuth: Interrupted stream action: "
                    "action=error reason=max_resume_attempts"
                )
                message = (
                    "DeepSeek stream did not close normally after "
                    f"{resume_attempts} resume attempt(s)"
                )
                if stream_error is not None:
                    raise RuntimeError(message) from stream_error
                raise RuntimeError(message)

            resume_attempts += 1
            debug.log(
                "DeepSeekAuth: Resuming interrupted response stream: "
                f"action=resume_stream attempt={resume_attempts}"
            )
            endpoint = CHAT_SESSION_RESUME_STREAM_ENDPOINT
            payload = {
                "chat_session_id": chat_session_id,
                "message_id": state.message_id,
            }
            request_headers = None

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
            reasoning_effort: Optional[
                Literal["none", "low", "medium", "high", "x-high"]
            ] = None,
            delete_session: bool = False,
            auto_continue: bool = True,
            max_continue_attempts: Optional[int] = 20,
            max_resume_attempts: Optional[int] = 5,
            **kwargs,
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
            auto_continue: Continue responses that close with INCOMPLETE status
            max_continue_attempts: Safety cap for consecutive continue requests
            max_resume_attempts: Safety cap for resume requests before a close event
        """
        if not model:
            model = cls.default_model

        source_headers = dict(headers or {})
        # Try to get auth from HAR file first
        if cookies is None:
            cookies = get_cookies(cls.cookie_domain, False)
            source_headers = get_headers(cls.cookie_domain) or {}
            normalized_source_headers = _normalized_headers(source_headers)
            if cookies and normalized_source_headers.get("authorization"):
                debug.log(
                    "DeepSeekAuth: Using "
                    f"{len(cookies)} cookies and {len(source_headers)} headers "
                    "from cookie jar"
                )
            # else:
            #     raise MissingAuthError(
            #         "DeepSeekAuth: No authentication found. "
            #         "Please add a DeepSeek HAR file to har_and_cookies/ directory "
            #         "with an authorization token."
            #     )

        # Initialize conversation if needed
        if conversation is None:
            conversation = JsonConversation(parent_message_id=None)

        token = kwargs.get("token", "") or kwargs.get("api_key", "")
        authorization = (token if token.lower().startswith("bearer ") else f"Bearer {token}") if token else ""
        # Get auth token from HAR data or conversation
        if not authorization:
            authorization = _normalized_headers(source_headers).get("authorization")
            if not authorization and hasattr(conversation, "authorization"):
                authorization = conversation.authorization

        if not authorization:
            raise MissingAuthError(
                "DeepSeekAuth: Authorization token required. "
                "Please ensure HAR file contains authorization header."
            )

        headers = _build_chat_headers(source_headers, authorization)

        # Extract query from messages
        prompt = get_last_user_message(messages)

        # Determine thinking mode
        if reasoning_effort is not None and reasoning_effort != "none":
            thinking_enabled = True
        else:
            thinking_enabled = bool(model) and "deepseek-r1" in model
        model_type = kwargs.get("model_type", "default")  # "default", "expert", "vision"

        yield JsonRequest.from_dict(
            {
                "prompt": prompt,
                "thinking_enabled": thinking_enabled,
                "search_enabled": web_search,
            }
        )

        # Always create a new chat session for the first request
        if (
                not hasattr(conversation, "chat_session_id")
                or not conversation.chat_session_id
        ):
            debug.log(f"DeepSeekAuth: Creating new chat session...")
            async with StreamSession(
                    headers=headers, cookies=cookies, proxy=proxy, impersonate="chrome"
            ) as session:
                async with session.post(CHAT_SESSION_CREATE_ENDPOINT) as response:
                    await raise_for_status(response)
                    session_data = await response.json()
                    chat_session_id = _extract_chat_session_id(session_data)
                    if chat_session_id:
                        conversation.chat_session_id = chat_session_id
                        debug.log(
                            f"DeepSeekAuth: Chat session created: {chat_session_id}"
                        )
                    else:
                        debug.error(
                            f"DeepSeekAuth: Unexpected session response: {session_data}"
                        )
                        raise Exception(
                            f"Failed to parse session response: {session_data}"
                        )
        else:
            debug.log(
                f"DeepSeekAuth: Reusing existing chat session: {conversation.chat_session_id}"
            )

        # Yield conversation object so caller can reuse it for subsequent messages
        yield conversation

        # Upload file if provided - use HTTP/1.1 to avoid HTTP/2 stream errors
        ref_file_ids = []
        if media is not None and len(media) > 0:
            # Take first file from media list
            file_bytes, filename = media[0]
            upload_session_headers = _build_upload_session_headers(headers)
            async with StreamSession(
                    headers=upload_session_headers,
                    cookies=cookies,
                    proxy=proxy,
                    impersonate="chrome",
                    http_version=CurlHttpVersion.V1_1
                    if has_curl_cffi
                    else None,  # Force HTTP/1.1 to avoid HTTP/2 stream errors
            ) as session:
                upload_result = await cls.upload_file(
                    session,
                    file_bytes,
                    filename,
                    thinking_enabled=thinking_enabled,
                    model_type=model_type,
                )
                await cls.wait_for_file_parsed(session, upload_result["file_id"])
                ref_file_ids.append(upload_result["file_id"])
                debug.log(f"DeepSeekAuth: Using file_id: {upload_result['file_id']}")

        # Build request data

        json_data = {
            "action": None,
            "chat_session_id": getattr(
                conversation, "chat_session_id", str(uuid.uuid4())
            ),
            "model_type": model_type,
            # "parent_message_id":None,
            # "preempt":False,
            "prompt": prompt,
            "ref_file_ids": ref_file_ids,
            "thinking_enabled": thinking_enabled,
            "search_enabled": web_search,
            "client_stream_id": generate_client_stream_id(),
        }

        # Add parent_message_id if continuing conversation
        if (
                hasattr(conversation, "parent_message_id")
                and conversation.parent_message_id
        ):
            json_data["parent_message_id"] = conversation.parent_message_id

        async with StreamSession(
                headers=headers, cookies=cookies, proxy=proxy, impersonate="chrome"
        ) as session:
            chat_pow_response = await cls.create_pow_response(
                session, CHAT_COMPLETION_PATH
            )
            async for chunk in cls.iter_chat_stream(
                    session,
                    conversation,
                    json_data,
                    {"x-ds-pow-response": chat_pow_response},
                    auto_continue=auto_continue,
                    max_continue_attempts=max_continue_attempts,
                    max_resume_attempts=max_resume_attempts,
            ):
                yield chunk

        # Yield the updated message ID only after the logical response has closed.
        yield conversation

        if (
                delete_session
                and hasattr(conversation, "chat_session_id")
                and conversation.chat_session_id
        ):
            async with StreamSession(
                    headers=headers,
                    cookies=cookies,
                    proxy=proxy,
                    impersonate="chrome",
            ) as delete_session_obj:
                await cls.delete_chat_session(
                    delete_session_obj, conversation.chat_session_id, headers
                )
