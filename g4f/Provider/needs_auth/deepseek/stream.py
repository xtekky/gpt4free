from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional

from g4f import debug
from g4f.providers.response import JsonConversation, Reasoning


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
            raise ValueError(
                f"Invalid DeepSeek SSE JSON data: {raw_data!r}"
            ) from error

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
    fragment_kinds: dict[str, Optional[str]] = field(default_factory=dict)
    next_fragment_index: int = 0
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
        return "response"
    if fragment_type in DEEPSEEK_RESPONSE_FRAGMENT_TYPES:
        return "response"
    if fragment_type in DEEPSEEK_REASONING_FRAGMENT_TYPES:
        return "reasoning"
    return None


def _stream_output(kind: str, content: str):
    if not content:
        return None
    return Reasoning(content) if kind == "reasoning" else content


def _record_fragment_kind(
        state: _DeepSeekStreamState,
        fragment_index: str,
        kind: Optional[str],
        *,
        append_fragment: bool = False,
) -> None:
    state.active_kind = kind
    debug.log(
        "DeepSeekAuth: Stream fragment: "
        f"index={_stream_log_value(fragment_index)} "
        f"kind={_stream_log_value(kind)} "
        f"append={_stream_log_value(append_fragment)}"
    )

    if fragment_index == "-1":
        state.fragment_kinds["-1"] = kind
        if append_fragment:
            state.fragment_kinds[str(state.next_fragment_index)] = kind
            state.next_fragment_index += 1
        elif state.next_fragment_index:
            state.fragment_kinds[str(state.next_fragment_index - 1)] = kind
        return

    state.fragment_kinds[fragment_index] = kind
    try:
        numeric_index = int(fragment_index)
    except ValueError:
        return
    if numeric_index >= state.next_fragment_index:
        state.next_fragment_index = numeric_index + 1
    if numeric_index == state.next_fragment_index - 1:
        state.fragment_kinds["-1"] = kind


def _fragment_index_from_path(path: str) -> Optional[str]:
    path_parts = path.split("/")
    if (
            len(path_parts) >= 3
            and path_parts[0] == "response"
            and path_parts[1] == "fragments"
    ):
        return path_parts[2]
    return None


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
        interpretation = "requires_continue" if status == "INCOMPLETE" else None
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

    if snapshot:
        state.fragment_kinds.clear()
        state.next_fragment_index = 0

    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        kind = _fragment_kind(fragment)
        fragment_index = str(state.next_fragment_index)
        _record_fragment_kind(state, fragment_index, kind)
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
            chunks.extend(_process_stream_payload(batch_item, state, conversation))
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

    fragment_index = (
        _fragment_index_from_path(path)
        if isinstance(path, str)
        else None
    )
    if (
            fragment_index is not None
            and path.count("/") == 2
            and operation in {"SET", "APPEND"}
            and isinstance(value, dict)
    ):
        kind = _fragment_kind(value)
        _record_fragment_kind(
            state,
            fragment_index,
            kind,
            append_fragment=operation == "APPEND",
        )
        content = value.get("content")
        if kind is not None and isinstance(content, str):
            delta = (
                state.snapshot_delta(kind, content)
                if operation == "SET"
                else state.append(kind, content)
            )
            output = _stream_output(kind, delta)
            if output is not None:
                chunks.append(output)
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
        if (
                fragment_index is not None
                and path.endswith("/type")
                and isinstance(value, str)
        ):
            _record_fragment_kind(
                state,
                fragment_index,
                _fragment_kind({"type": value}),
            )
            return chunks
        if path.endswith("/content") and isinstance(value, str):
            kind = (
                state.fragment_kinds[fragment_index]
                if fragment_index in state.fragment_kinds
                else state.active_kind
            )
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


def _process_full_message(
        biz_data: Any,
        state: _DeepSeekStreamState,
        conversation: JsonConversation,
) -> list:
    """Convert a non-SSE full-message response into normal provider chunks."""
    if not isinstance(biz_data, dict):
        raise RuntimeError("DeepSeek resume returned an invalid full message")

    response = biz_data.get("response")
    if not isinstance(response, dict):
        response = biz_data.get("response_message") or biz_data.get("message")
    if not isinstance(response, dict) and "fragments" in biz_data:
        response = biz_data
    if not isinstance(response, dict):
        raise RuntimeError("DeepSeek resume returned an invalid full message")

    return _process_stream_payload(
        {"v": {"response": response}},
        state,
        conversation,
    )
