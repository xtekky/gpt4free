from __future__ import annotations

import asyncio
import mimetypes
from datetime import datetime
from typing import Any, Optional, Literal

from g4f import debug
from g4f.cookies import get_cookies, get_headers
from g4f.errors import MissingAuthError, ResponseError
from g4f.image import to_bytes, detect_file_type
from g4f.providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from g4f.providers.helper import get_last_user_message
from g4f.providers.response import (
    FinishReason,
    JsonConversation,
    JsonRequest,
)
from g4f.requests import StreamSession, raise_for_status, FormData
from g4f.typing import AsyncResult, Messages, Cookies
from .deepseek.pow import (
    DEEPSEEK_POW_ALGORITHM,
    WASM_PATH,
    DeepSeekHash,
    DeepSeekPOW,
    has_wasmtime_and_numpy,
)
from .deepseek.stream import (
    DEEPSEEK_FINISH_REASONS,
    DEEPSEEK_MESSAGE_STATUSES,
    DEEPSEEK_METADATA_FRAGMENT_TYPES,
    DEEPSEEK_REASONING_FRAGMENT_TYPES,
    DEEPSEEK_RESPONSE_FRAGMENT_TYPES,
    _DeepSeekStreamState,
    _fragment_kind,
    _process_fragments,
    _process_full_message,
    _process_stream_payload,
    _record_response_message_id,
    _record_stream_status,
    _stream_log_value,
    _stream_output,
    iter_deepseek_sse,
)

try:
    from curl_cffi import CurlHttpVersion

    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

def _solve_pow_challenge(challenge: dict) -> str:
    """Create the WASM solver and solve entirely outside the event loop."""
    return DeepSeekPOW().solve_challenge(challenge)


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
RESUME_MESSAGE_GOT_FULL_MESSAGE_CODE = 22
DEEPSEEK_FILE_FAILURE_STATUSES = {
    "FAILED",
    "ERROR",
    "CONTENT_FILTER",
    "CONTENT_TOO_LONG",
    "CANCELLED",
    "CONTENT_EMPTY",
    "_CUSTOM_SYSTEM_ERROR_FAIL",
    "_CUSTOM_FROM_SHARE",
}

CHAT_HEADER_DEFAULTS = {
    "accept": "*/*",
    "cache-control": "no-cache",
    "content-type": "application/json",
    "origin": DEEPSEEK_URL,
    "referer": f"{DEEPSEEK_URL}/a/chat/",
    "x-client-bundle-id": "com.deepseek.chat",
    "x-client-locale": "en_US",
    "x-client-platform": "web",
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
    "x-hif-dliq",
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


def _unwrap_biz_response(
        payload: Any,
        context: str,
        *,
        allowed_codes: tuple[int, ...] = (),
) -> tuple[Any, Any, Optional[str]]:
    """Validate DeepSeek's current business envelope and return its payload."""
    if not isinstance(payload, dict):
        raise RuntimeError(f"DeepSeek {context} returned an invalid JSON response")

    response_data = payload.get("data")
    if isinstance(response_data, dict):
        code = response_data.get("biz_code", payload.get("code"))
        message = response_data.get("biz_msg") or payload.get("msg")
        biz_data = response_data.get("biz_data")
    else:
        code = payload.get("code")
        message = payload.get("msg")
        biz_data = None

    allowed_code_strings = {str(value) for value in allowed_codes}
    if code not in (None, 0, "0") and str(code) not in allowed_code_strings:
        detail = message or "unknown business error"
        raise RuntimeError(f"DeepSeek {context} failed ({code}): {detail}")
    return code, biz_data, message


def _build_chat_headers(source_headers: Optional[dict], authorization: str) -> dict:
    """Build JSON request headers while retaining current browser-bound values."""
    normalized = _normalized_headers(source_headers)
    headers = dict(CHAT_HEADER_DEFAULTS)
    timezone_offset = datetime.now().astimezone().utcoffset()
    headers["x-client-timezone-offset"] = str(
        -int(timezone_offset.total_seconds()) if timezone_offset else 0
    )
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


def _build_upload_session_headers(headers: dict) -> dict:
    """Copy shared headers without values that would corrupt a multipart upload."""
    excluded_headers = {"content-type", "x-ds-pow-response"}
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in excluded_headers
    }


def _resolve_upload_metadata(data: bytes, filename: Optional[str]) -> tuple[str, str]:
    """Resolve a stable filename and MIME type without rejecting text/code files."""
    if filename:
        guessed_type, _encoding = mimetypes.guess_type(filename, strict=False)
        if guessed_type:
            return filename, guessed_type
        try:
            _extension, detected_type = detect_file_type(data)
        except ValueError:
            detected_type = "application/octet-stream"
        return filename, detected_type

    extension, detected_type = detect_file_type(data)
    return f"file-{len(data)}{extension}", detected_type


def _build_completion_payload(
        conversation: JsonConversation,
        *,
        prompt: str,
        model_type: str,
        ref_file_ids: list[str],
        thinking_enabled: bool,
        search_enabled: bool,
) -> dict:
    """Build the current DeepSeek web completion request contract."""
    chat_session_id = getattr(conversation, "chat_session_id", None)
    if not chat_session_id:
        raise ValueError("DeepSeek chat_session_id is required for completion")

    return {
        "action": None,
        "chat_session_id": chat_session_id,
        "parent_message_id": getattr(conversation, "parent_message_id", None),
        "model_type": model_type,
        "prompt": prompt,
        "ref_file_ids": ref_file_ids,
        "thinking_enabled": thinking_enabled,
        "search_enabled": search_enabled,
        "preempt": False,
    }


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

        _code, biz_data, _message = _unwrap_biz_response(
            pow_data, "PoW challenge"
        )
        try:
            challenge = biz_data["challenge"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"DeepSeek returned an invalid PoW challenge for {target_path}"
            ) from error

        if challenge.get("target_path") != target_path:
            raise RuntimeError(
                "DeepSeek returned a PoW challenge for an unexpected target path: "
                f"{challenge.get('target_path')!r}"
            )
        if challenge.get("algorithm") != DEEPSEEK_POW_ALGORITHM:
            raise RuntimeError(
                "DeepSeek returned an unsupported PoW algorithm: "
                f"{challenge.get('algorithm')!r}"
            )

        debug.log(
            "DeepSeekAuth: Challenge: "
            f"algorithm={challenge.get('algorithm')}, "
            f"difficulty={challenge.get('difficulty')}"
        )
        loop = asyncio.get_running_loop()
        pow_response = await loop.run_in_executor(
            None, _solve_pow_challenge, challenge
        )
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
        filename, file_type = _resolve_upload_metadata(data_bytes, filename)

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

        _code, biz_data, _message = _unwrap_biz_response(result, "file upload")
        response_data = result.get("data") if isinstance(result, dict) else None
        file_id = biz_data.get("id") if isinstance(biz_data, dict) else None
        if not file_id and isinstance(response_data, dict):
            # Keep compatibility with the older, unnested response shape.
            file_id = response_data.get("id")

        if not file_id:
            raise RuntimeError(
                "DeepSeek file upload failed: missing data.biz_data.id"
            )

        debug.log(f"DeepSeekAuth: File uploaded successfully, file_id: {file_id}")
        return {
            "file_id": file_id,
            "filename": filename,
            "size": len(data_bytes),
        }

    @classmethod
    async def upload_files(
            cls,
            session: StreamSession,
            media: list,
            *,
            thinking_enabled: bool = False,
            model_type: str = "default",
    ) -> list[str]:
        """Upload and parse every media item, preserving caller order."""
        file_ids = []
        for file_bytes, filename in media:
            upload_result = await cls.upload_file(
                session,
                file_bytes,
                filename,
                thinking_enabled=thinking_enabled,
                model_type=model_type,
            )
            file_id = upload_result["file_id"]
            await cls.wait_for_file_parsed(session, file_id)
            file_ids.append(file_id)
            debug.log(f"DeepSeekAuth: Using file_id: {file_id}")
        return file_ids

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

            _code, biz_data, _message = _unwrap_biz_response(
                result, "file status"
            )
            files = biz_data.get("files") if isinstance(biz_data, dict) else None
            file_info = files[0] if isinstance(files, list) and files else {}
            status = str(file_info.get("status", "")).upper()

            if status == "SUCCESS":
                debug.log(f"DeepSeekAuth: File parsing completed, file_id: {file_id}")
                return
            if status in DEEPSEEK_FILE_FAILURE_STATUSES:
                error_code = file_info.get("error_code") or status
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
    ) -> bool:
        """Delete one session with the observed POST contract."""
        try:
            async with session.post(
                    CHAT_SESSION_DELETE_ENDPOINT,
                    headers=headers,
                    json={"chat_session_id": chat_session_id},
            ) as response:
                await raise_for_status(response)
                result = await response.json()
            _unwrap_biz_response(result, "chat session deletion")
        except Exception as error:
            debug.error(f"DeepSeekAuth: Chat session deletion failed: {error}")
            return False

        debug.log("DeepSeekAuth: Chat session deleted successfully")
        return True

    @classmethod
    async def get_quota(cls, **kwargs):
        cookies = get_cookies(cls.cookie_domain, False)
        headers = _normalized_headers(get_headers(cls.cookie_domain) or {})
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
        empty_response_resume_attempted = False

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
                    result = await response.json()
                    code, biz_data, _message = _unwrap_biz_response(
                        result,
                        "chat stream",
                        allowed_codes=(RESUME_MESSAGE_GOT_FULL_MESSAGE_CODE,),
                    )
                    if str(code) == str(RESUME_MESSAGE_GOT_FULL_MESSAGE_CODE):
                        for chunk in _process_full_message(
                                biz_data, state, conversation
                        ):
                            yield chunk
                        if not state.emitted["response"]:
                            debug.log(
                                "DeepSeekAuth: Stream closed: "
                                f"status={_stream_log_value(state.status)} "
                                "action=error "
                                "reason=empty_response_after_resume "
                                "finish_reason=none"
                            )
                            raise ResponseError(
                                "DeepSeek finished without a response"
                            )
                        finish_reason = DEEPSEEK_FINISH_REASONS.get(state.status)
                        if finish_reason is not None:
                            yield FinishReason(finish_reason)
                        return
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
                if (
                        state.status == "FINISHED"
                        and not state.emitted["response"]
                ):
                    can_resume_empty_response = (
                        state.message_id is not None
                        and not empty_response_resume_attempted
                        and (
                            max_resume_attempts is None
                            or resume_attempts < max_resume_attempts
                        )
                    )
                    if can_resume_empty_response:
                        empty_response_resume_attempted = True
                        resume_attempts += 1
                        debug.log(
                            "DeepSeekAuth: Stream closed: "
                            "status=FINISHED action=resume_stream "
                            "reason=empty_response finish_reason=none "
                            f"attempt={resume_attempts} "
                            f"{close_details}"
                        )
                        endpoint = CHAT_SESSION_RESUME_STREAM_ENDPOINT
                        payload = {
                            "chat_session_id": chat_session_id,
                            "message_id": state.message_id,
                        }
                        request_headers = None
                        continue

                    debug.log(
                        "DeepSeekAuth: Stream closed: "
                        "status=FINISHED action=error "
                        "reason=empty_response finish_reason=none "
                        f"{close_details}"
                    )
                    raise ResponseError(
                        "DeepSeek finished without a response"
                    )
                if (
                        empty_response_resume_attempted
                        and not state.emitted["response"]
                        and not should_continue
                ):
                    debug.log(
                        "DeepSeekAuth: Stream closed: "
                        f"status={_stream_log_value(state.status)} "
                        "action=error reason=empty_response_after_resume "
                        "finish_reason=none "
                        f"{close_details}"
                    )
                    raise ResponseError(
                        "DeepSeek finished without a response"
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
            discovered_headers = get_headers(cls.cookie_domain) or {}
            # Explicit caller headers override browser/HAR values, including when
            # their casing differs (normalization happens below).
            source_headers = {**discovered_headers, **source_headers}
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
        if reasoning_effort is not None:
            thinking_enabled = reasoning_effort != "none"
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
                    _unwrap_biz_response(
                        session_data, "chat session creation"
                    )
                    chat_session_id = _extract_chat_session_id(session_data)
                    if chat_session_id:
                        conversation.chat_session_id = chat_session_id
                        debug.log(
                            f"DeepSeekAuth: Chat session created: {chat_session_id}"
                        )
                    else:
                        debug.error(
                            "DeepSeekAuth: Session response did not include an id"
                        )
                        raise RuntimeError(
                            "DeepSeek chat session creation failed: missing session id"
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
                ref_file_ids = await cls.upload_files(
                    session,
                    media,
                    thinking_enabled=thinking_enabled,
                    model_type=model_type,
                )

        # Build request data

        json_data = _build_completion_payload(
            conversation,
            prompt=prompt,
            model_type=model_type,
            ref_file_ids=ref_file_ids,
            thinking_enabled=thinking_enabled,
            search_enabled=web_search,
        )

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
