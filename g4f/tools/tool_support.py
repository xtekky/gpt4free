from __future__ import annotations

import json
import re
import inspect
import typing
from typing import Any, Callable, Dict, List, Optional, Union


def _py_type_to_json_type(py_type: Any) -> tuple[str, Optional[dict]]:
    """Map Python type annotation to JSON schema type string and optional schema detail."""
    if py_type is inspect.Parameter.empty or py_type is None:
        return "string", None

    origin = typing.get_origin(py_type)
    args = typing.get_args(py_type)

    if origin is Union:
        # Handle Optional[T] / Union[T, None]
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _py_type_to_json_type(non_none[0])
        return "string", None

    target = origin or py_type

    if target in (int,):
        return "integer", None
    elif target in (float,):
        return "number", None
    elif target in (bool,):
        return "boolean", None
    elif target in (str,):
        return "string", None
    elif target in (list, tuple, set, List):
        if args:
            item_type, item_schema = _py_type_to_json_type(args[0])
            items = {"type": item_type}
            if item_schema:
                items.update(item_schema)
            return "array", {"items": items}
        return "array", {"items": {"type": "string"}}
    elif target in (dict, Dict):
        return "object", None

    return "string", None


def function_to_tool_def(func: Callable) -> dict:
    """Convert a Python function into an OpenAI-compatible tool definition dictionary."""
    name = getattr(func, "__name__", str(func))
    doc = getattr(func, "__doc__", "") or ""
    description = inspect.cleandoc(doc) if doc else f"Function {name}"

    try:
        sig = inspect.signature(func)
        hints = typing.get_type_hints(func)
    except Exception:
        sig = None
        hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []

    if sig:
        for p_name, param in sig.parameters.items():
            if p_name in ("self", "cls"):
                continue

            py_type = hints.get(p_name, param.annotation)
            json_type, detail = _py_type_to_json_type(py_type)
            prop_def: dict[str, Any] = {"type": json_type}
            if detail:
                prop_def.update(detail)

            properties[p_name] = prop_def

            if param.default is inspect.Parameter.empty:
                required.append(p_name)

    parameters = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def normalize_tool_defs(tools: Any) -> list[dict]:
    """Normalize various tool definition formats into standard OpenAI tool dictionaries.

    Supported inputs:
    - Standard OpenAI dict: ``{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}``
    - Flat dict: ``{"name": ..., "description": ..., "parameters" / "input_schema": ...}``
    - Anthropic format: ``{"name": ..., "description": ..., "input_schema": ...}``
    - Python callable (function / method / lambda)
    - Python object with ``.to_dict()``, ``.dict()``, or attributes ``name``, ``description``, ``parameters``
    """
    if not tools:
        return []

    tool_list = tools if isinstance(tools, (list, tuple)) else [tools]
    normalized: list[dict] = []

    for t in tool_list:
        if callable(t) and not isinstance(t, dict):
            normalized.append(function_to_tool_def(t))
            continue

        if isinstance(t, dict):
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                fn = dict(t["function"])
                name = fn.get("name")
                if not name or not isinstance(name, str):
                    continue
                desc = fn.get("description") or ""
                params = fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}}
                normalized.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": params,
                    }
                })
            elif "name" in t and isinstance(t["name"], str):
                name = t["name"]
                desc = t.get("description") or ""
                params = t.get("parameters") or t.get("input_schema") or {"type": "object", "properties": {}}
                normalized.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": params,
                    }
                })
            continue

        # Object with method / attributes
        dict_val = None
        if hasattr(t, "to_dict") and callable(t.to_dict):
            try:
                dict_val = t.to_dict()
            except Exception:
                pass
        elif hasattr(t, "dict") and callable(t.dict):
            try:
                dict_val = t.dict()
            except Exception:
                pass

        if isinstance(dict_val, dict):
            normalized.extend(normalize_tool_defs([dict_val]))
            continue

        if hasattr(t, "name") and isinstance(getattr(t, "name"), str):
            name = getattr(t, "name")
            desc = getattr(t, "description", "") or ""
            params = (
                getattr(t, "parameters", None)
                or getattr(t, "input_schema", None)
                or {"type": "object", "properties": {}}
            )
            normalized.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params,
                }
            })

    return normalized


def normalize_tool_calls(tool_calls: Any) -> list[dict]:
    """Normalize tool call representations into standard OpenAI tool call dictionaries.

    Returns a list of dicts:
    ``[{"index": i, "id": call_id, "type": "function", "function": {"name": name, "arguments": json_str}}]``
    """
    if not tool_calls:
        return []

    if hasattr(tool_calls, "get_list") and callable(tool_calls.get_list):
        tool_calls = tool_calls.get_list()

    calls_list = tool_calls if isinstance(tool_calls, (list, tuple)) else [tool_calls]
    normalized: list[dict] = []

    for idx, tc in enumerate(calls_list):
        if not isinstance(tc, dict):
            continue

        call_id = str(tc.get("id") or f"call_{idx}")
        name = None
        args: Any = {}

        # 1. Standard OpenAI format
        if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
            fn = tc["function"]
            name = fn.get("name")
            args = fn.get("arguments", {})
        # 2. Anthropic tool_use block
        elif tc.get("type") == "tool_use":
            name = tc.get("name")
            args = tc.get("input", {})
            call_id = tc.get("id") or call_id
        # 3. Gemini functionCall
        elif "functionCall" in tc and isinstance(tc["functionCall"], dict):
            fc = tc["functionCall"]
            name = fc.get("name")
            args = fc.get("args", {})
        # 4. Flat dict / ReAct dict
        else:
            name = tc.get("name") or tc.get("tool") or tc.get("action")
            args = (
                tc.get("arguments")
                if "arguments" in tc
                else tc.get("args")
                if "args" in tc
                else tc.get("parameters")
                if "parameters" in tc
                else tc.get("input")
                if "input" in tc
                else tc.get("action_input", {})
            )

        if not name or not isinstance(name, str):
            continue

        if isinstance(args, str):
            args_str = args
        else:
            try:
                args_str = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=True)
            except Exception:
                args_str = "{}"

        normalized.append({
            "index": idx,
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": args_str,
            }
        })

    return normalized


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping a payload."""
    if not text:
        return text
    text = text.strip()
    m = re.match(r"^```(?:json|JSON)?\s*\n?([\s\S]*?)\n?```\s*$", text)
    if m:
        return m.group(1).strip()
    m = re.match(r"^```(?:json|JSON)?\s*\n?([\s\S]*)$", text)
    if m:
        return m.group(1).strip()
    return text


def _extract_balanced_json(s: str, start: int) -> tuple[str, int]:
    """Return the balanced JSON object/array starting at s[start]."""
    if start >= len(s):
        return "", start
    open_ch = s[start]
    if open_ch not in "{[":
        end = s.find("\n", start)
        if end == -1:
            end = len(s)
        return s[start:end].strip(), end
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escape = False
    i = start
    while i < len(s):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1], i + 1
        i += 1
    return s[start:].strip(), len(s)


def parse_tool_calls_from_text(content: str, tool_names: Optional[list[str]] = None) -> list[dict]:
    """Parse tool calls from arbitrary LLM response text.

    Supports:
    - JSON object/array (code fenced or raw) with key `tool_calls`, `tool_call`, `function_call`, `function`, `name`, `tool`, `action`.
    - XML tags `<tool_call>...</tool_call>`, `<function_call>...</function_call>`, `<tool>...</tool>`.
    - Stringified tool calls `[Tool call: name] (id=id)\\nArguments: {json}`.
    - ReAct pattern `Action: name\\nAction Input: {json}`.
    """
    if not content:
        return []

    calls: list[dict] = []
    text = content.strip()

    # 1. Check stringified tool call header format [Tool call: NAME]
    header_re = re.compile(
        r"\[\s*Tool\s*call\s*:\s*([^\]]+?)\s*\](?:\s*\(id=([^\)]*)\))?\s*\n\s*Arguments\s*:\s*",
        re.IGNORECASE,
    )
    for m in header_re.finditer(text):
        name = (m.group(1) or "").strip()
        call_id = (m.group(2) or "").strip()
        args_str, _ = _extract_balanced_json(text, m.end())
        if name:
            calls.append({"name": name, "id": call_id, "arguments": args_str})
    if calls:
        return filter_tool_calls_by_names(calls, tool_names)

    # 2. Check XML tags <tool_call>...</tool_call> or <function_call>...</function_call> or <tool>...</tool>
    xml_re = re.compile(
        r"<(?:tool_call|function_call|tool)>(.*?)</(?:tool_call|function_call|tool)>",
        re.DOTALL | re.IGNORECASE,
    )
    for m in xml_re.finditer(text):
        inner = m.group(1).strip()
        # Inner might be JSON
        if inner.startswith("{") or inner.startswith("["):
            try:
                obj = json.loads(inner)
                parsed = _extract_calls_from_json_obj(obj)
                calls.extend(parsed)
                continue
            except Exception:
                pass
        # Inner might be XML sub-tags <name>...</name><arguments>...</arguments>
        name_m = re.search(r"<name>(.*?)</name>", inner, re.DOTALL | re.IGNORECASE)
        args_m = re.search(r"<(?:arguments|args|parameters|input)>(.*?)</(?:arguments|args|parameters|input)>", inner, re.DOTALL | re.IGNORECASE)
        if name_m:
            c_name = name_m.group(1).strip()
            c_args = args_m.group(1).strip() if args_m else "{}"
            calls.append({"name": c_name, "arguments": c_args})
    if calls:
        return filter_tool_calls_by_names(calls, tool_names)

    # 3. Check ReAct pattern: Action: name \n Action Input: {json}
    react_re = re.compile(
        r"Action\s*:\s*([^\n]+)\s*\n\s*Action\s*Input\s*:\s*",
        re.IGNORECASE,
    )
    for m in react_re.finditer(text):
        name = m.group(1).strip()
        args_str, _ = _extract_balanced_json(text, m.end())
        if name:
            calls.append({"name": name, "arguments": args_str})
    if calls:
        return filter_tool_calls_by_names(calls, tool_names)

    # 4. JSON parsing (strip code fences first)
    raw_text = _strip_code_fences(text)
    if "</tool_response>" in raw_text:
        raw_text = raw_text.split("</tool_response>", 1)[-1].strip()

    # Attempt direct json load or json substring match
    parsed_json = None
    if raw_text.startswith("{") and not raw_text.endswith("}"):
        raw_text += "}"
    try:
        parsed_json = json.loads(raw_text)
    except Exception:
        # Find json object or array substring
        m_obj = re.search(r"\{[\s\S]*\}", raw_text)
        if m_obj:
            try:
                parsed_json = json.loads(m_obj.group(0))
            except Exception:
                try:
                    fixed = re.sub(r",\s*([}\]])", r"\1", m_obj.group(0))
                    parsed_json = json.loads(fixed)
                except Exception:
                    pass
        if parsed_json is None:
            m_arr = re.search(r"\[[\s\S]*\]", raw_text)
            if m_arr:
                try:
                    parsed_json = json.loads(m_arr.group(0))
                except Exception:
                    pass

    if parsed_json is not None:
        calls = _extract_calls_from_json_obj(parsed_json)
        if calls:
            return filter_tool_calls_by_names(calls, tool_names)

    return []


def _extract_calls_from_json_obj(obj: Any) -> list[dict]:
    """Helper to extract tool call dicts from a parsed JSON object or array."""
    calls: list[dict] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("tool_calls"), list):
            for item in obj["tool_calls"]:
                calls.extend(_extract_calls_from_json_obj(item))
        elif isinstance(obj.get("tool_call"), dict):
            calls.extend(_extract_calls_from_json_obj(obj["tool_call"]))
        elif isinstance(obj.get("function_call"), dict):
            calls.extend(_extract_calls_from_json_obj(obj["function_call"]))
        elif isinstance(obj.get("function"), dict):
            fn = obj["function"]
            name = fn.get("name")
            args = fn.get("arguments") or fn.get("args") or fn.get("parameters") or {}
            if name:
                calls.append({"name": name, "id": obj.get("id", ""), "arguments": args})
        elif "name" in obj or "tool" in obj or "action" in obj:
            name = obj.get("name") or obj.get("tool") or obj.get("action")
            args = (
                obj.get("arguments")
                if "arguments" in obj
                else obj.get("args")
                if "args" in obj
                else obj.get("parameters")
                if "parameters" in obj
                else obj.get("input")
                if "input" in obj
                else obj.get("action_input", {})
            )
            if name and isinstance(name, str):
                calls.append({"name": name, "id": obj.get("id", ""), "arguments": args})
    elif isinstance(list, list) or isinstance(obj, list):
        for item in obj:
            calls.extend(_extract_calls_from_json_obj(item))
    return calls


def filter_tool_calls_by_names(calls: list[dict], tool_names: Optional[list[str]] = None) -> list[dict]:
    """Filter candidate tool calls against allowed tool_names if provided."""
    if not tool_names:
        return calls
    filtered = []
    for c in calls:
        name = c.get("name")
        if isinstance(name, str) and name in tool_names:
            filtered.append(c)
    return filtered
