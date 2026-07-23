from __future__ import annotations

import json
import re

from ...errors import RateLimitError, ResponseError


BARD_ERROR_PATTERN = re.compile(r"BardErrorInfo\s*\[(\d+)\]")
LENGTH_MARKER_PATTERN = re.compile(r"(\d+)\n")
ACCOUNT_STATUS_AVAILABLE = 1000
ACCOUNT_STATUS_UNAUTHENTICATED = 1016
MODEL_HEADER_KEY = "x-goog-ext-525001261-jspb"
MODEL_HEADER_AUXILIARY = {
    "x-goog-ext-73010989-jspb": "[0]",
    "x-goog-ext-73010990-jspb": "[0]",
}
MODEL_FAMILIES = {
    "gemini-3.5-flash": "flash",
    "gemini-3.5-flash-thinking": "thinking",
    "gemini-3.1-pro": "pro",
}
ANONYMOUS_MODELS = {"gemini-3.5-flash", "gemini-auto"}
KNOWN_MODEL_IDS = {
    "fbb127bbb056c959": "flash",
    "5bf011840784117a": "thinking",
    "9d8ca3786ebdfbea": "pro",
}
GEMINI_ERROR_MESSAGES = {
    1013: "Gemini encountered a temporary generation error",
    1037: "Gemini usage limit exceeded for the requested model",
    1050: "The requested Gemini model is inconsistent with the conversation",
    1052: "The requested Gemini model header is invalid or unavailable",
    1060: "Gemini temporarily blocked this IP address",
}


def iter_wrb_payloads(value):
    if isinstance(value, list):
        if len(value) >= 3 and value[0] == "wrb.fr" and isinstance(value[2], str):
            yield value[2]
            return
        for item in value:
            yield from iter_wrb_payloads(item)


def get_nested_value(value, path, default=None):
    current = value
    for key in path:
        if isinstance(key, int):
            if not isinstance(current, list) or not -len(current) <= key < len(current):
                return default
        elif not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def _utf16_char_count(value: str, start: int, units: int) -> tuple[int, int]:
    count = found = 0
    while found < units and start + count < len(value):
        size = 2 if ord(value[start + count]) > 0xFFFF else 1
        if found + size > units:
            break
        found += size
        count += 1
    return count, found


def parse_google_frames(content: str) -> tuple[list, str]:
    """Parse Google's length-prefixed frames, including UTF-16 lengths."""
    frames = []
    position = 0
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        if position >= len(content):
            break
        match = LENGTH_MARKER_PATTERN.match(content, position)
        if match is None:
            break
        length = int(match.group(1))
        start = match.start() + len(match.group(1))
        char_count, units_found = _utf16_char_count(content, start, length)
        if units_found < length:
            break
        end = start + char_count
        chunk = content[start:end].strip()
        position = end
        if not chunk:
            continue
        try:
            parsed = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(parsed, list):
            frames.extend(parsed)
        else:
            frames.append(parsed)
    return frames, content[position:]


def _iter_google_json(content: str):
    content = content.lstrip()
    if content.startswith(")]}'"):
        content = content[4:].lstrip()
    frames, _ = parse_google_frames(content)
    if frames:
        yield from frames
        return
    try:
        parsed = json.loads(content)
    except ValueError:
        parsed = None
    if parsed is not None:
        if isinstance(parsed, list):
            yield from parsed
        else:
            yield parsed
        return
    for line in content.splitlines():
        line = line.strip()
        if not line or line.isdigit():
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, list):
            yield from parsed
        else:
            yield parsed


def _compute_model_capacity(tier_flags: list, capability_flags: list) -> tuple[int, int]:
    if 21 in tier_flags:
        return 1, 13
    if 22 in tier_flags:
        return 2, 13
    if 115 in capability_flags:
        return 4, 12
    if 16 in tier_flags or 106 in capability_flags:
        return 3, 12
    if 8 in tier_flags or (106 not in capability_flags and 19 in capability_flags):
        return 2, 12
    return 1, 12


def _model_family(model_id: str, display_name: str, description: str) -> str | None:
    if model_id in KNOWN_MODEL_IDS:
        return KNOWN_MODEL_IDS[model_id]
    label = f"{display_name} {description}".lower()
    if "thinking" in label:
        return "thinking"
    if "pro" in label:
        return "pro"
    if "flash" in label:
        return "flash"
    return None


def build_model_headers(model_id: str, capacity: int, capacity_field: int) -> dict[str, str]:
    header = [1, None, None, None, model_id, None, None, 0, [4], None, None]
    if capacity_field == 13:
        header.extend([None, capacity])
    else:
        header.append(capacity)
    return {
        MODEL_HEADER_KEY: json.dumps(header, separators=(",", ":")),
        **MODEL_HEADER_AUXILIARY,
    }


def parse_account_models(content: str) -> tuple[int | None, dict[str, dict]]:
    status_code = None
    registry = {}
    for frame in _iter_google_json(content):
        for payload in iter_wrb_payloads(frame):
            try:
                body = json.loads(payload)
            except (TypeError, ValueError):
                continue
            current_status = get_nested_value(body, [14])
            if isinstance(current_status, int):
                status_code = current_status
            models_list = get_nested_value(body, [15], [])
            if not isinstance(models_list, list):
                continue
            tier_flags = get_nested_value(body, [16], [])
            capability_flags = get_nested_value(body, [17], [])
            tier_flags = tier_flags if isinstance(tier_flags, list) else []
            capability_flags = capability_flags if isinstance(capability_flags, list) else []
            capacity, capacity_field = _compute_model_capacity(
                tier_flags, capability_flags
            )
            for model_data in models_list:
                if not isinstance(model_data, list):
                    continue
                model_id = get_nested_value(model_data, [0], "")
                display_name = get_nested_value(model_data, [1], "")
                description = get_nested_value(model_data, [2], "")
                if not isinstance(model_id, str) or not model_id:
                    continue
                family = _model_family(model_id, display_name, description)
                # The authenticated response commonly omits field 14. Gemini's
                # web client treats an absent status as the normal/available
                # state; explicit non-1000 values describe restricted states.
                available = status_code in (None, ACCOUNT_STATUS_AVAILABLE)
                if status_code == ACCOUNT_STATUS_UNAUTHENTICATED:
                    available = family == "flash"
                registry[model_id] = {
                    "model_id": model_id,
                    "family": family,
                    "display_name": display_name,
                    "description": description,
                    "capacity": capacity,
                    "capacity_field": capacity_field,
                    "available": available,
                    "headers": build_model_headers(
                        model_id, capacity, capacity_field
                    ),
                }
    if registry and status_code is None:
        status_code = ACCOUNT_STATUS_AVAILABLE
    return status_code, registry


def extract_reasoning(response_part: list) -> str | None:
    candidates = get_nested_value(response_part, [4], [])
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        reasoning = get_nested_value(candidate, [37, 0, 0])
        if isinstance(reasoning, str) and reasoning:
            return reasoning
    return None


def extract_gemini_error_code(value) -> int | None:
    if isinstance(value, str):
        match = BARD_ERROR_PATTERN.search(value)
        return int(match.group(1)) if match else None
    if not isinstance(value, list):
        return None
    if value and value[0] == "wrb.fr":
        code = get_nested_value(value, [5, 2, 0, 1, 0])
        if isinstance(code, int):
            return code
    for item in value:
        code = extract_gemini_error_code(item)
        if code is not None:
            return code
    return None


def raise_gemini_error(code: int, model: str) -> None:
    message = GEMINI_ERROR_MESSAGES.get(
        code, f"Gemini rejected the request with error code {code}"
    )
    if code in (1037, 1060):
        raise RateLimitError(f"{message}: {model}")
    raise ResponseError(f"{message}: {model}")
