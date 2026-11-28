"""
Antigravity Provider for gpt4free (v2)

Provides access to Google's Antigravity API (Code Assist) supporting:
- Gemini 2.5 & Gemini 3 (Pro/Flash) with thinkingBudget and thinkingLevel
- Gemini 3.1 / 3.5 / 3.6 / 3.7 / 3.8 Flash & Pro variants
- Claude (Sonnet 4.5 / 4.6, Opus 4.5 / 4.6) via Antigravity proxy
- Image generation models (gemini-3.1-flash-image)

Uses OAuth2 authentication with Antigravity-specific credentials.
Supports endpoint fallback chain for reliability.
Includes interactive OAuth login flow with PKCE support.
"""

import os
import sys
import json
import base64
import time
import secrets
import hashlib
import asyncio
import uuid
import re
import webbrowser
import threading
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Tuple
from urllib.parse import urlencode, parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from ...typing import AsyncResult, Messages, MediaListType
from ...errors import MissingAuthError, RateLimitError
from ...requests.raise_for_status import raise_for_status
from ...image.copy_images import save_response_media
from ...image import to_bytes, is_data_an_media
from ...providers.response import Usage, ImageResponse, ToolCalls, Reasoning
from ...providers.asyncio import get_running_loop
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin
from ..helper import get_connector, get_system_prompt, format_media_prompt
from ... import debug


# JSON Schema keywords not supported by the Gemini API
_UNSUPPORTED_SCHEMA_KEYS = {
    "patternProperties",
    "propertyNames",
    "$schema",
    "$id",
    "$ref",
    "$defs",
    "definitions",
    "if",
    "then",
    "else",
    "not",
    "allOf",
    "anyOf",
    "oneOf",
    "default",
    "examples",
    "readOnly",
    "writeOnly",
    "contentEncoding",
    "contentMediaType",
    "dependentRequired",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
    "minContains",
    "maxContains",
    "additionalProperties",
    "enumDescriptions",
    "$comment",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "const",
}


def _split_schema_type(value: Any) -> Tuple[Any, bool]:
    """Reduce a JSON Schema type (possibly a union) to one Gemini type."""
    if not isinstance(value, list):
        return value, False
    nullable = False
    types = []
    for item in value:
        if not isinstance(item, str):
            continue
        if item.lower() == "null":
            nullable = True
        elif item not in types:
            types.append(item)
    return (types[0] if types else None), nullable


def _sanitize_schema(schema: dict) -> dict:
    """
    Recursively make a JSON Schema acceptable to the Gemini API.

    Besides removing keywords Gemini rejects, this repairs schemas that the
    strict Antigravity/protobuf validation refuses:
    - arrays without ``items``
    - ``required`` entries that do not exist in ``properties``
    - union types such as ``["string", "null"]``
    """
    if not isinstance(schema, dict):
        return schema
    result = {}
    nullable = False
    for k, v in schema.items():
        if k in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if k == "type":
            v, is_nullable = _split_schema_type(v)
            nullable = nullable or is_nullable
            if not isinstance(v, str):
                continue
        if k == "properties":
            if isinstance(v, dict):
                result[k] = {
                    name: _sanitize_schema(sub) if isinstance(sub, dict) else {}
                    for name, sub in v.items()
                }
            continue
        if k == "items":
            if isinstance(v, list):
                v = v[0] if v else {}
            result[k] = _sanitize_schema(v) if isinstance(v, dict) else {
                "type": "string"
            }
            continue
        if k == "enum":
            # Gemini only accepts string enum values; other primitives are
            # rejected even when the declared type is numeric or boolean.
            if isinstance(v, list):
                result[k] = [
                    item if isinstance(item, str) else json.dumps(item)
                    for item in v
                ]
            continue
        if isinstance(v, dict):
            result[k] = _sanitize_schema(v)
        elif isinstance(v, list):
            result[k] = [_sanitize_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            result[k] = v

    if nullable:
        result["nullable"] = True
    if (
        "type" not in result
        and (
            "properties" in result
            or "required" in result
            or "additionalProperties" in schema
        )
    ):
        result["type"] = "object"
    # Gemini requires array schemas to declare their item type.
    if (
        isinstance(result.get("type"), str)
        and result["type"].lower() == "array"
        and "items" not in result
    ):
        result["items"] = {"type": "string"}
    required = result.get("required")
    if "required" in result:
        properties = result.get("properties")
        if isinstance(required, list) and isinstance(properties, dict):
            required = [
                name
                for name in required
                if isinstance(name, str) and name in properties
            ]
        else:
            required = []
        if required:
            result["required"] = required
        else:
            result.pop("required", None)
    return result


def get_antigravity_oauth_creds_path() -> Path:
    """Get the default path for Antigravity OAuth credentials."""
    return Path.home() / ".antigravity" / "oauth_creds.json"


def get_antigravity_accounts_dir() -> Path:
    """Get the directory containing credentials for additional accounts."""
    return Path.home() / ".antigravity" / "accounts"


def get_antigravity_account_credentials_path(name: Optional[str] = None) -> Path:
    """Get a stable credentials path for a named Antigravity account."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "")).strip("._")
    return get_antigravity_accounts_dir() / f"{safe_name or 'default'}.json"


class AntigravityQuotaError(RateLimitError):
    """Raised when an Antigravity account has exhausted its quota."""

    def __init__(self, message: str = "Antigravity quota reached", retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


# --- Constants ---
CREDENTIALS_DIR = ".antigravity"
CREDENTIALS_FILE = "oauth_creds.json"

# Base URLs
ANTIGRAVITY_BASE_URL_DAILY = "https://daily-cloudcode-pa.googleapis.com"
ANTIGRAVITY_BASE_URL_PROD = "https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_API_VERSION = "v1internal"

BASE_URLS = [
    f"{ANTIGRAVITY_BASE_URL_DAILY}/{ANTIGRAVITY_API_VERSION}",
    f"{ANTIGRAVITY_BASE_URL_PROD}/{ANTIGRAVITY_API_VERSION}",
]

PRODUCTION_URL = f"{ANTIGRAVITY_BASE_URL_PROD}/{ANTIGRAVITY_API_VERSION}"

OAUTH_CLIENT_ID = (
    "1071006060591-" + "tmhssin2h21lcre235vtolojh4g403ep" + ".apps.googleusercontent.com"
)
OAUTH_CLIENT_SECRET = "GOCSPX-" + "K58FWR486LdLJ1m" + "LB8sXC4z6qDAf"
DEFAULT_USER_AGENT = "antigravity/2.8.1 darwin/arm64"
REFRESH_SKEW = 3000  # 3000 seconds (50 minutes) advance refresh

ANTIGRAVITY_SYSTEM_PROMPT = (
    "You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team "
    "working on Advanced Agentic Coding.You are pair programming with a USER to solve their coding task. "
    "The task may require creating a new codebase, modifying or debugging an existing codebase, or simply "
    "answering a question.**Absolute paths only****Proactiveness**"
)

DEFAULT_THINKING_MIN = 1024
DEFAULT_THINKING_MAX = 100000
ANTIGRAVITY_EMPTY_TEXT_PLACEHOLDER = "."

ANTIGRAVITY_STREAM_FIRST_BYTE_TIMEOUT_MS = 180000
ANTIGRAVITY_STREAM_IDLE_TIMEOUT_MS = 300000

ANTIGRAVITY_RAW_FALLBACK_MAX_LINES = 20000
ANTIGRAVITY_ERROR_BODY_MAX_BYTES = 1024 * 1024

AUTO_MODEL = "auto"
DEFAULT_QUOTA_COOLDOWN = 3600
QUOTA_ERROR_MARKERS = (
    "resource_exhausted",
    "quota",
    "rate limit",
    "rate_limit",
    "too many requests",
)

ANTIGRAVITY_MODELS = [
    "claude-opus-4-6-thinking",
    "claude-sonnet-4-6",
    "gemini-3-flash",
    "gemini-3-pro-high",
    "gemini-3-pro-low",
    "gemini-3.1-flash-image",
    "gemini-pro-agent",
    "gemini-3.1-pro-high",
    "gemini-3.1-pro-low",
    "gpt-oss-120b-medium",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-low",
    "gemini-3.5-flash-high",
    "gemini-3.6-flash",
    "gemini-3.6-flash-low",
    "gemini-3.6-flash-high",
    "gemini-3.7-flash",
    "gemini-3.7-flash-low",
    "gemini-3.7-flash-high",
    "gemini-3.8-flash",
    "gemini-3.8-flash-low",
    "gemini-3.8-flash-high",
]

ANTIGRAVITY_CLIENT_TO_UPSTREAM_MODEL = {
    "gemini-3.1-pro-high": "gemini-pro-agent",
    "gemini-3.1-pro-preview": "gemini-pro-agent",
    "gemini-3.5-flash-high": "gemini-3.5-flash-low",
    "gemini-3.6-flash": "gemini-3.6-flash-low",
    "gemini-3.7-flash": "gemini-3.7-flash-low",
    "gemini-3.8-flash": "gemini-3.8-flash-low",
}

ANTIGRAVITY_UPSTREAM_TO_CLIENT_MODELS = {
    "gemini-pro-agent": ["gemini-3.1-pro-high", "gemini-3.1-pro-preview"],
    "gemini-3.6-flash-low": ["gemini-3.6-flash", "gemini-3.6-flash-low"],
    "gemini-3.7-flash-low": ["gemini-3.7-flash", "gemini-3.7-flash-low"],
    "gemini-3.8-flash-low": ["gemini-3.8-flash", "gemini-3.8-flash-low"],
}

ANTIGRAVITY_CLIENT_MODEL_THINKING_LEVEL = {
    "gemini-pro-agent": "high",
    "gemini-3.1-pro-high": "high",
    "gemini-3.1-pro-preview": "high",
    "gemini-3-pro-high": "high",
    "gemini-3-pro-preview": "high",
    "gemini-3.5-flash-high": "high",
    "gemini-3.6-flash-high": "high",
    "gemini-3.7-flash-high": "high",
    "gemini-3.8-flash-high": "high",
    "gemini-3.1-pro-low": "low",
    "gemini-3-pro-low": "low",
    "gemini-3.5-flash-low": "low",
    "gemini-3.6-flash-low": "low",
    "gemini-3.7-flash-low": "low",
    "gemini-3.8-flash-low": "low",
}

ANTIGRAVITY_MODEL_METADATA = {
    "claude-opus-4-6-thinking": {
        "maxOutputTokens": 64000,
        "thinking": {"min": 1024, "max": 64000, "zeroAllowed": True, "dynamicAllowed": True},
    },
    "claude-sonnet-4-6": {
        "maxOutputTokens": 64000,
        "thinking": {"min": 1024, "max": 64000, "zeroAllowed": True, "dynamicAllowed": True},
    },
    "gemini-3-flash": {
        "maxOutputTokens": 65536,
        "thinking": {
            "min": 128,
            "max": 32768,
            "dynamicAllowed": True,
            "levels": ["minimal", "low", "medium", "high"],
        },
    },
    "gemini-3-pro-high": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 128, "max": 32768, "dynamicAllowed": True, "levels": ["low", "high"]},
    },
    "gemini-3-pro-low": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 128, "max": 32768, "dynamicAllowed": True, "levels": ["low", "high"]},
    },
    "gemini-3.1-flash-image": {
        "thinking": {"min": 128, "max": 32768, "dynamicAllowed": True, "levels": ["minimal", "high"]},
    },
    "gemini-pro-agent": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 1, "max": 65535, "dynamicAllowed": True, "levels": ["low", "medium", "high"]},
    },
    "gemini-3.1-pro-high": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 1, "max": 65535, "dynamicAllowed": True, "levels": ["low", "medium", "high"]},
    },
    "gemini-3.1-pro-low": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 1, "max": 65535, "dynamicAllowed": True, "levels": ["low", "medium", "high"]},
    },
    "gpt-oss-120b-medium": {
        "maxOutputTokens": 32768,
    },
    "gemini-3.1-flash-lite": {
        "maxOutputTokens": 65535,
        "thinking": {
            "min": 1,
            "max": 65535,
            "zeroAllowed": True,
            "dynamicAllowed": True,
            "levels": ["minimal", "low", "medium", "high"],
        },
    },
    "gemini-3.5-flash-low": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 1, "max": 65535, "dynamicAllowed": True, "levels": ["low", "medium", "high"]},
    },
    "gemini-3.6-flash-low": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 1, "max": 65535, "dynamicAllowed": True, "levels": ["low", "medium", "high"]},
    },
    "gemini-3.6-flash-high": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 1, "max": 65535, "dynamicAllowed": True, "levels": ["low", "medium", "high"]},
    },
    "gemini-3.7-flash-low": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 1, "max": 65535, "dynamicAllowed": True, "levels": ["low", "medium", "high"]},
    },
    "gemini-3.7-flash-high": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 1, "max": 65535, "dynamicAllowed": True, "levels": ["low", "medium", "high"]},
    },
    "gemini-3.8-flash-low": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 1, "max": 65535, "dynamicAllowed": True, "levels": ["low", "medium", "high"]},
    },
    "gemini-3.8-flash-high": {
        "maxOutputTokens": 65535,
        "thinking": {"min": 1, "max": 65535, "dynamicAllowed": True, "levels": ["low", "medium", "high"]},
    },
}

ANTIGRAVITY_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": '{"ideType":"ANTIGRAVITY","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}

ANTIGRAVITY_AUTH_HEADERS = {
    "User-Agent": "google-api-nodejs-client/10.3.0",
    "X-Goog-Api-Client": "gl-node/22.18.0",
    "Client-Metadata": '{"ideType":"ANTIGRAVITY","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}

ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauthcallback"
ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]
CALLBACK_PORT = 51121
OAUTH_CALLBACK_PORT = CALLBACK_PORT
OAUTH_CALLBACK_PATH = "/oauthcallback"

TOOL_ID_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def parse_antigravity_quota_error(
    status: Optional[int], body: Any
) -> Optional[AntigravityQuotaError]:
    """Convert a quota/rate-limit response into an AntigravityQuotaError."""
    if isinstance(body, str):
        text = body
    elif body is not None:
        try:
            text = json.dumps(body)
        except (TypeError, ValueError):
            text = str(body)
    else:
        text = ""
    lowered = text.lower()
    quota_reported = status == 429 or any(
        marker in lowered for marker in QUOTA_ERROR_MARKERS
    )
    if not quota_reported:
        return None

    retry_after = None
    try:
        parsed = json.loads(text) if text.strip().startswith(("{", "[")) else None
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        details = (parsed.get("error") or {}).get("details", [])
        if isinstance(details, list):
            for detail in details:
                value = detail.get("retryDelay") if isinstance(detail, dict) else None
                if value is None:
                    continue
                try:
                    retry_after = float(str(value).rstrip("sS"))
                except ValueError:
                    continue
                break
    if retry_after is not None:
        retry_after = max(1.0, retry_after)
    summary = text.strip()[:500] or "quota reached"
    return AntigravityQuotaError(
        f"Antigravity quota reached (status {status}): {summary}",
        retry_after=retry_after,
    )


def get_antigravity_model_family(model_name: str) -> str:
    """Classify a model so quota exhaustion can be tracked per model family."""
    name = strip_model_suffix(normalize_antigravity_model_id(model_name)).lower()
    if "claude" in name:
        return "claude"
    if name.startswith("gemini-"):
        return "gemini"
    return "other"


def is_midsize_gemini_model(model_name: str) -> bool:
    """Mid-size Gemini models are Flash models without lite/pro/image suffixes."""
    name = strip_model_suffix(normalize_antigravity_model_id(model_name))
    if not name.startswith("gemini-") or "flash" not in name:
        return False
    return not any(
        marker in name for marker in ("lite", "pro", "image", "agent")
    )


def get_auto_model_candidates(available_models: Optional[List[str]] = None) -> List[str]:
    """
    Resolve the "auto" model: first mid-size Gemini model, then a Claude model.
    Dynamic models from the API are preferred over the built-in fallback list.
    """
    models: List[str] = []
    for candidate in [*(available_models or []), *ANTIGRAVITY_MODELS]:
        candidate = normalize_antigravity_model_id(candidate)
        if candidate and candidate not in models:
            models.append(candidate)

    gemini = next((m for m in models if is_midsize_gemini_model(m)), None)
    claude = None
    for preferred in (
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-opus-4-6-thinking",
        "claude-opus-4-5",
    ):
        if preferred in models:
            claude = preferred
            break
    if claude is None:
        claude = next((m for m in models if "claude" in m.lower()), None)

    return [model for model in (gemini, claude) if model]


def load_antigravity_accounts(env: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Load all configured Antigravity credentials, newest sources last."""
    env = env if env is not None else os.environ
    accounts: List[Dict[str, Any]] = []
    seen = set()

    def add_account(creds: Any, name: Optional[str] = None, path: Optional[Path] = None):
        if not isinstance(creds, dict):
            return
        if not (creds.get("refresh_token") or creds.get("access_token")):
            return
        identity = str(creds.get("refresh_token") or creds.get("email") or path)
        if identity in seen:
            return
        seen.add(identity)
        accounts.append({
            "credentials": creds,
            "name": name or creds.get("email") or (path.stem if path else f"account-{len(accounts) + 1}"),
            "path": path,
        })

    raw_service_account = env.get("ANTIGRAVITY_SERVICE_ACCOUNT")
    if raw_service_account:
        try:
            parsed = json.loads(raw_service_account)
            if isinstance(parsed, dict) and isinstance(parsed.get("accounts"), list):
                entries = parsed["accounts"]
            elif isinstance(parsed, list):
                entries = parsed
            else:
                entries = [parsed]
            for entry in entries:
                add_account(entry)
        except (json.JSONDecodeError, TypeError) as e:
            debug.log(f"Failed to parse ANTIGRAVITY_SERVICE_ACCOUNT: {e}")

    accounts_dir = Path(env.get("ANTIGRAVITY_ACCOUNTS_DIR") or get_antigravity_accounts_dir())
    if accounts_dir.is_dir():
        for path in sorted(accounts_dir.glob("*.json")):
            try:
                with path.open("r") as f:
                    creds = json.load(f)
            except Exception as e:
                debug.log(f"Failed to read Antigravity account {path}: {e}")
                continue
            add_account(creds, name=path.stem, path=path)

    for path in (AntigravityAuthManager.get_cache_file(), get_antigravity_oauth_creds_path()):
        if not path.exists():
            continue
        try:
            with path.open("r") as f:
                creds = json.load(f)
        except Exception as e:
            debug.log(f"Failed to read Antigravity credentials {path}: {e}")
            continue
        add_account(creds, name=path.stem, path=path)

    return accounts


def normalize_antigravity_model_id(model_name: str) -> str:
    if not model_name or not isinstance(model_name, str):
        return ""
    normalized = model_name.strip()
    if normalized.startswith("models/"):
        normalized = normalized[len("models/") :]
    return normalized


def strip_model_suffix(model_name: str) -> str:
    normalized = normalize_antigravity_model_id(model_name)
    match = re.match(r"^(.+?)\([^()]+\)$", normalized)
    return match.group(1).strip() if match else normalized


def resolve_antigravity_upstream_model(model_name: str) -> str:
    base_model = strip_model_suffix(model_name)
    if not base_model:
        return ""
    if base_model.startswith("gemini-claude-"):
        return base_model.replace("gemini-claude-", "claude-")
    return ANTIGRAVITY_CLIENT_TO_UPSTREAM_MODEL.get(base_model, base_model)


def expand_antigravity_client_models(upstream_model: str) -> List[str]:
    base_model = strip_model_suffix(upstream_model)
    if not base_model:
        return []
    out = []

    def push(m):
        if m and m not in out:
            out.append(m)

    if base_model.startswith("claude-"):
        push(f"gemini-{base_model}")
        return out

    exposed_alias = False
    for alias in ANTIGRAVITY_UPSTREAM_TO_CLIENT_MODELS.get(base_model, []):
        if alias in ANTIGRAVITY_MODELS:
            push(alias)
            exposed_alias = True

    if base_model in ANTIGRAVITY_MODELS or (
        not exposed_alias and base_model in ANTIGRAVITY_MODEL_METADATA
    ):
        push(base_model)

    return out


def get_antigravity_model_metadata(model_name: str) -> Optional[dict]:
    upstream_model = resolve_antigravity_upstream_model(model_name)
    return ANTIGRAVITY_MODEL_METADATA.get(
        upstream_model
    ) or ANTIGRAVITY_MODEL_METADATA.get(strip_model_suffix(model_name))


def is_known_antigravity_model(model_name: str) -> bool:
    base_model = strip_model_suffix(model_name)
    if not base_model:
        return False
    return (
        base_model in ANTIGRAVITY_MODELS
        or get_antigravity_model_metadata(base_model) is not None
    )


def antigravity_model_uses_thinking_levels(model_name: str) -> bool:
    metadata = get_antigravity_model_metadata(model_name)
    levels = metadata.get("thinking", {}).get("levels") if metadata else None
    return isinstance(levels, list) and len(levels) > 0


def antigravity_model_requires_stream_for_non_stream(model_name: str) -> bool:
    name = str(model_name or "").lower()
    return (
        "claude" in name
        or "gemini-3-pro" in name
        or "gemini-3.1-flash-image" in name
    )


def normalize_antigravity_text_part(part: dict) -> None:
    if not isinstance(part, dict) or "text" not in part:
        return
    if not isinstance(part["text"], str):
        part["text"] = "" if part["text"] is None else str(part["text"])
    if len(part["text"].strip()) == 0:
        part["text"] = ANTIGRAVITY_EMPTY_TEXT_PLACEHOLDER


def normalize_antigravity_text_parts(parts: list) -> None:
    if isinstance(parts, list):
        for p in parts:
            normalize_antigravity_text_part(p)


def get_antigravity_client_model_thinking_level(model_name: str) -> str:
    base_model = strip_model_suffix(model_name)
    return ANTIGRAVITY_CLIENT_MODEL_THINKING_LEVEL.get(base_model, "")


def apply_antigravity_thinking_level_config(
    thinking_config: dict, level: str
) -> dict:
    thinking_config["thinkingLevel"] = level
    thinking_config["includeThoughts"] = True
    thinking_config.pop("thinkingBudget", None)
    thinking_config.pop("thinking_budget", None)
    return thinking_config


def apply_antigravity_client_model_thinking_level(
    payload: dict, client_model_name: str
) -> dict:
    level = get_antigravity_client_model_thinking_level(client_model_name)
    if not level or not payload.get("request"):
        return payload
    gen_cfg = payload["request"].setdefault("generationConfig", {})
    th_cfg = gen_cfg.setdefault("thinkingConfig", {})
    apply_antigravity_thinking_level_config(th_cfg, level)
    return payload


def apply_antigravity_client_model_thinking_level_to_request(
    request_body: dict, client_model_name: str
) -> dict:
    level = get_antigravity_client_model_thinking_level(client_model_name)
    if not level or not request_body:
        return request_body
    gen_cfg = request_body.setdefault("generationConfig", {})
    th_cfg = gen_cfg.setdefault("thinkingConfig", {})
    apply_antigravity_thinking_level_config(th_cfg, level)
    return request_body


def is_claude(model_name: str) -> bool:
    return bool(model_name and "claude" in model_name.lower())


def is_image_model(model_name: str) -> bool:
    return bool(model_name and "image" in model_name.lower())


def model_supports_thinking(model_name: str) -> bool:
    if not model_name:
        return False
    metadata = get_antigravity_model_metadata(model_name)
    if metadata and "thinking" in metadata:
        return True
    name = model_name.lower()
    return (
        "gemini-3" in name
        or name.startswith("gemini-2.5-")
        or "-thinking" in name
    )


def generate_request_id() -> str:
    return f"agent-{uuid.uuid4()}"


def generate_image_gen_request_id() -> str:
    return f"image_gen/{int(time.time()*1000)}/{uuid.uuid4()}/12"


def generate_session_id() -> str:
    n = secrets.randbelow(9000)
    return f"-{n}"


def generate_stable_session_id(payload: dict) -> str:
    try:
        contents = payload.get("request", {}).get("contents")
        if isinstance(contents, list):
            for content in contents:
                if (
                    isinstance(content, dict)
                    and content.get("role") == "user"
                    and isinstance(content.get("parts"), list)
                ):
                    text = (
                        content["parts"][0].get("text")
                        if content["parts"]
                        else None
                    )
                    if text:
                        digest = hashlib.sha256(text.encode("utf-8")).digest()
                        n = (
                            int.from_bytes(digest[:8], byteorder="big")
                            & 0x7FFFFFFFFFFFFFFF
                        )
                        return f"-{n}"
    except Exception:
        pass
    return generate_session_id()


def generate_project_id() -> str:
    adjectives = ["useful", "bright", "swift", "calm", "bold"]
    nouns = ["fuze", "wave", "spark", "flow", "core"]
    adj = secrets.choice(adjectives)
    noun = secrets.choice(nouns)
    random_part = secrets.token_hex(3)[:5]
    return f"{adj}-{noun}-{random_part}"


def normalize_thinking_budget(model_name: str, budget: int) -> int:
    if budget == -1:
        return -1
    thinking = get_antigravity_model_metadata(model_name)
    thinking_cfg = thinking.get("thinking", {}) if thinking else {}
    min_b = thinking_cfg.get("min", DEFAULT_THINKING_MIN)
    max_b = thinking_cfg.get("max", DEFAULT_THINKING_MAX)
    if budget < min_b:
        return min_b
    if budget > max_b:
        return max_b
    return budget


def normalize_antigravity_thinking(
    model_name: str, payload: dict, is_claude_model: bool
) -> dict:
    if not model_supports_thinking(model_name):
        if payload.get("request", {}).get("generationConfig", {}).get(
            "thinkingConfig"
        ):
            payload["request"]["generationConfig"].pop("thinkingConfig", None)
        return payload

    thinking_config = (
        payload.get("request", {})
        .get("generationConfig", {})
        .get("thinkingConfig")
    )
    if not thinking_config:
        return payload

    thinking_level = thinking_config.get("thinkingLevel")
    budget = thinking_config.get("thinkingBudget")
    thinking_requested = (thinking_level is not None) or (
        budget is not None and budget != 0
    )

    if thinking_requested and "includeThoughts" not in thinking_config:
        thinking_config["includeThoughts"] = True

    if budget is None:
        return payload

    normalized_budget = normalize_thinking_budget(model_name, budget)

    gen_cfg = payload["request"].get("generationConfig", {})
    max_tokens = gen_cfg.get("maxOutputTokens") or gen_cfg.get(
        "max_output_tokens"
    )
    if max_tokens and max_tokens > 0 and normalized_budget >= max_tokens:
        normalized_budget = max(0, max_tokens - 1)

    if is_claude_model:
        min_budget = DEFAULT_THINKING_MIN
        if 0 <= normalized_budget < min_budget and normalized_budget != -1:
            payload["request"]["generationConfig"].pop("thinkingConfig", None)
            return payload

    payload["request"]["generationConfig"]["thinkingConfig"][
        "thinkingBudget"
    ] = normalized_budget
    return payload


def generate_synthetic_tool_id() -> str:
    bytes_data = secrets.token_bytes(26)
    s = "".join(TOOL_ID_ALPHABET[b % 62] for b in bytes_data)
    return f"toolu_vrtx_{s}"


def ensure_tool_call_ids(contents: list) -> None:
    if not isinstance(contents, list):
        return
    pending_by_name = {}
    for content in contents:
        if not isinstance(content, dict) or not isinstance(
            content.get("parts"), list
        ):
            continue
        for part in content["parts"]:
            if not isinstance(part, dict):
                continue
            if "functionCall" in part:
                fc = part["functionCall"]
                if not fc.get("id"):
                    fc["id"] = generate_synthetic_tool_id()
                fc_name = fc.get("name")
                if fc_name:
                    pending_by_name.setdefault(fc_name, []).append(fc["id"])
            elif "functionResponse" in part:
                fr = part["functionResponse"]
                fr_name = fr.get("name")
                if fr_name:
                    q = pending_by_name.get(fr_name, [])
                    pending_id = q.pop(0) if q else None
                    if not fr.get("id") and pending_id:
                        fr["id"] = pending_id


def normalize_antigravity_tool_config(
    request_obj: dict, is_claude_model: bool
) -> None:
    tool_cfg = request_obj.get("toolConfig")
    if not isinstance(tool_cfg, dict):
        return
    fc_cfg = tool_cfg.get("functionCallingConfig")
    if not isinstance(fc_cfg, dict):
        return
    mode = str(fc_cfg.get("mode", "")).upper()
    if mode:
        fc_cfg["mode"] = mode
        if mode == "ANY" and "allowedFunctionNames" not in fc_cfg:
            names = []
            tools = request_obj.get("tools", [])
            if isinstance(tools, list):
                for t in tools:
                    if isinstance(t, dict) and isinstance(
                        t.get("functionDeclarations"), list
                    ):
                        for fd in t["functionDeclarations"]:
                            if isinstance(fd, dict) and fd.get("name"):
                                names.append(fd["name"])
            if names:
                fc_cfg["allowedFunctionNames"] = names


def gemini_to_antigravity(
    model_name: str, payload: dict, project_id: str
) -> dict:
    template = json.loads(json.dumps(payload))
    ensure_tool_call_ids(template.get("request", {}).get("contents"))

    is_claude_model = is_claude(model_name)
    is_img_model = is_image_model(model_name)

    template["model"] = model_name
    template["userAgent"] = "antigravity"
    template["requestType"] = "image_gen" if is_img_model else "agent"

    if project_id:
        template["project"] = project_id
    else:
        template.pop("project", None)

    if is_img_model:
        template["requestId"] = generate_image_gen_request_id()
    else:
        template["requestId"] = generate_request_id()
        if "request" not in template:
            template["request"] = {}
        template["request"]["sessionId"] = generate_stable_session_id(template)

    if "request" not in template:
        template["request"] = {}

    template["request"].pop("safetySettings", None)

    if "tool_config" in template and "toolConfig" not in template:
        template["toolConfig"] = template.pop("tool_config")
    else:
        template.pop("tool_config", None)

    if "toolConfig" in template:
        if "toolConfig" not in template["request"]:
            template["request"]["toolConfig"] = template["toolConfig"]
        template.pop("toolConfig", None)

    normalize_antigravity_tool_config(template["request"], is_claude_model)

    gen_cfg = template["request"].get("generationConfig", {})
    max_tokens = gen_cfg.get("maxOutputTokens")
    metadata = get_antigravity_model_metadata(model_name)
    model_max_tokens = metadata.get("maxOutputTokens") if metadata else None

    if (
        isinstance(max_tokens, int)
        and model_max_tokens
        and max_tokens > model_max_tokens
    ):
        template["request"]["generationConfig"][
            "maxOutputTokens"
        ] = model_max_tokens

    if not is_claude_model and "maxOutputTokens" in gen_cfg:
        gen_cfg.pop("maxOutputTokens", None)

    tools = template["request"].get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and isinstance(
                tool.get("functionDeclarations"), list
            ):
                for func_decl in tool["functionDeclarations"]:
                    if "parametersJsonSchema" in func_decl:
                        func_decl["parameters"] = _sanitize_schema(
                            func_decl["parametersJsonSchema"]
                        )
                        func_decl.pop("parametersJsonSchema", None)
                    elif "parameters" in func_decl:
                        func_decl["parameters"] = _sanitize_schema(
                            func_decl["parameters"]
                        )

    if gen_cfg.get("responseJsonSchema"):
        gen_cfg["responseJsonSchema"] = _sanitize_schema(
            gen_cfg["responseJsonSchema"]
        )
    if gen_cfg.get("responseSchema"):
        gen_cfg["responseSchema"] = _sanitize_schema(gen_cfg["responseSchema"])

    if not antigravity_model_uses_thinking_levels(model_name):
        th_cfg = gen_cfg.get("thinkingConfig")
        if isinstance(th_cfg, dict) and "thinkingLevel" in th_cfg:
            th_cfg.pop("thinkingLevel", None)
            th_cfg["thinkingBudget"] = -1

    if is_img_model:
        gen_cfg = template["request"].setdefault("generationConfig", {})
        img_cfg = gen_cfg.setdefault("imageConfig", {})
        img_cfg["imageSize"] = "4K"
        th_cfg = gen_cfg.setdefault("thinkingConfig", {})
        th_cfg["includeThoughts"] = False

    template = normalize_antigravity_thinking(
        model_name, template, is_claude_model
    )
    return template


def function_call_matches_response(
    function_call: dict, function_response: dict
) -> bool:
    call_id = function_call.get("id")
    response_id = function_response.get("id")
    if call_id and response_id:
        return call_id == response_id
    call_name = function_call.get("name")
    response_name = function_response.get("name")
    if call_name and response_name:
        return call_name == response_name
    return not any([call_id, response_id, call_name, response_name])


def function_call_to_text(function_call: dict) -> dict:
    name = function_call.get("name") or "function"
    args = function_call.get("args")
    if isinstance(args, str):
        args_text = args
    else:
        try:
            args_text = json.dumps(args, ensure_ascii=False) if args else "{}"
        except (TypeError, ValueError):
            args_text = str(args)
    return {"text": f"Called function {name} with arguments {args_text}"}


def function_response_to_text(function_response_part: dict) -> dict:
    function_response = function_response_part.get("functionResponse")
    if not isinstance(function_response, dict):
        try:
            return {"text": json.dumps(function_response_part, ensure_ascii=False)}
        except (TypeError, ValueError):
            return {"text": str(function_response_part)}
    name = function_response.get("name") or "function"
    response = function_response.get("response")
    if isinstance(response, dict) and "result" in response:
        response = response["result"]
    if not isinstance(response, str):
        try:
            response = json.dumps(response, ensure_ascii=False)
        except (TypeError, ValueError):
            response = str(response)
    return {"text": f"Function {name} returned: {response}"}


def normalize_antigravity_contents(contents: list) -> list:
    """
    Repair OpenAI-style histories so they satisfy Gemini turn ordering.

    Gemini requires a functionCall turn to come immediately after a user turn
    or a function response turn, and it rejects function responses that do not
    answer a pending call. Clients may send consecutive assistant messages or a
    truncated history that starts with a tool call (for example after context
    compaction), which makes Gemini answer with a 400:

        Please ensure that function call turn comes immediately after a user
        turn or after a function response turn.

    Merge consecutive turns, add a leading user turn when the history starts
    with a model turn, and downgrade function calls/responses that cannot be
    paired to plain text so the request stays valid. Histories that would end
    with a model turn get a trailing user turn because Antigravity rejects
    requests ending with a model turn.
    """
    if not isinstance(contents, list):
        return contents

    merged: List[Dict[str, Any]] = []
    for content in contents:
        if not isinstance(content, dict):
            continue
        raw_role = content.get("role")
        role = "model" if raw_role in ("model", "assistant") else "user"
        raw_parts = content.get("parts")
        if isinstance(raw_parts, str):
            raw_parts = [{"text": raw_parts}]
        if not isinstance(raw_parts, list):
            continue
        parts = []
        for part in raw_parts:
            if not isinstance(part, dict):
                continue
            # Function calls only make sense in model turns and function
            # responses only in user turns. Downgrade misplaced parts to text
            # instead of sending a payload that Gemini will reject.
            if role == "user" and isinstance(part.get("functionCall"), dict):
                part = function_call_to_text(part["functionCall"])
            elif role == "model" and isinstance(
                part.get("functionResponse"), dict
            ):
                part = function_response_to_text(part)
            # Empty text parts are invalid and would otherwise leave an empty
            # turn behind (a common source of ordering errors).
            if (
                isinstance(part.get("text"), str)
                and not part["text"].strip()
                and len(part) == 1
            ):
                continue
            parts.append(part)
        if not parts:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["parts"].extend(parts)
        else:
            merged.append({"role": role, "parts": parts})

    if merged and merged[0]["role"] == "model":
        merged.insert(
            0,
            {
                "role": "user",
                "parts": [{"text": ANTIGRAVITY_EMPTY_TEXT_PLACEHOLDER}],
            },
        )

    pending_calls: List[Tuple[dict, dict]] = []
    repaired: List[Dict[str, Any]] = []

    def downgrade_pending_calls() -> None:
        # A call can only be answered by the user turn that directly follows it.
        # Anything left over can no longer be paired and must become plain text,
        # otherwise Gemini rejects the whole request.
        for model_content, function_call in pending_calls:
            for index, part in enumerate(model_content.get("parts", [])):
                if part.get("functionCall") is function_call:
                    model_content["parts"][index] = function_call_to_text(
                        function_call
                    )
                    break
        pending_calls.clear()

    for content in merged:
        if content["role"] == "model":
            repaired.append(content)
            for part in content["parts"]:
                function_call = part.get("functionCall")
                if isinstance(function_call, dict):
                    pending_calls.append((content, function_call))
            continue

        parts = []
        for part in content["parts"]:
            function_response = part.get("functionResponse")
            if not isinstance(function_response, dict):
                parts.append(part)
                continue
            match = next(
                (
                    index
                    for index, (_, call) in enumerate(pending_calls)
                    if function_call_matches_response(call, function_response)
                ),
                None,
            )
            if match is None:
                parts.append(function_response_to_text(part))
            else:
                _, call = pending_calls.pop(match)
                # Clients (and g4f's own tool loop) may put a tool_call_id in
                # the name field; Gemini matches by name, so trust the call.
                call_name = call.get("name")
                if isinstance(call_name, str) and call_name:
                    function_response["name"] = call_name
                call_id = call.get("id")
                if call_id and not function_response.get("id"):
                    function_response["id"] = call_id
                parts.append(part)
        if parts:
            repaired.append({"role": "user", "parts": parts})
        downgrade_pending_calls()

    downgrade_pending_calls()

    # Antigravity rejects requests whose last turn is a model turn, for
    # example when a client re-sends a finished assistant message after an
    # interrupted or aborted response. Close the history with a minimal user
    # turn so the request is accepted.
    if not repaired or repaired[-1]["role"] == "model":
        repaired.append(
            {
                "role": "user",
                "parts": [{"text": ANTIGRAVITY_EMPTY_TEXT_PLACEHOLDER}],
            }
        )

    return repaired


def ensure_roles_in_contents(request_body: dict, model_name: str) -> dict:
    request_body.pop("model", None)
    if "system_instruction" in request_body:
        request_body["systemInstruction"] = request_body.pop("system_instruction")

    original_system_prompt = request_body.get("systemInstruction")
    original_system_prompt_text = ""

    if original_system_prompt:
        if isinstance(original_system_prompt, str):
            original_system_prompt_text = original_system_prompt
        elif isinstance(original_system_prompt, dict):
            parts = original_system_prompt.get("parts")
            if isinstance(parts, list):
                text_parts = []
                for part in parts:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and isinstance(
                        part.get("text"), str
                    ):
                        text_parts.append(part["text"])
                original_system_prompt_text = "\n".join(
                    t for t in text_parts if t
                )
            elif isinstance(original_system_prompt.get("text"), str):
                original_system_prompt_text = original_system_prompt["text"]

    name = (model_name or "").lower()
    is_gemini3 = "gemini-3" in name
    use_antigravity = is_gemini3 or "claude" in name

    if use_antigravity:
        parts = [
            {"text": ANTIGRAVITY_SYSTEM_PROMPT},
            {
                "text": f"Please ignore following [ignore]{ANTIGRAVITY_SYSTEM_PROMPT}[/ignore]"
            },
        ]
        if original_system_prompt_text:
            parts.append({"text": original_system_prompt_text})
        request_body["systemInstruction"] = {"role": "user", "parts": parts}
    elif original_system_prompt_text:
        request_body["systemInstruction"] = {
            "role": "user",
            "parts": [{"text": original_system_prompt_text}],
        }
    else:
        request_body.pop("systemInstruction", None)

    contents = request_body.get("contents")
    if isinstance(contents, list):
        contents = normalize_antigravity_contents(contents)
        request_body["contents"] = contents
        for content in contents:
            if isinstance(content, dict):
                if not content.get("role"):
                    content["role"] = "user"
                if use_antigravity:
                    normalize_antigravity_text_parts(content.get("parts"))

    return request_body


def filter_sse_usage_metadata(line: str) -> str:
    if not line or not isinstance(line, str):
        return line
    if not line.startswith("data: "):
        return line
    try:
        data = json.loads(line[6:])
        finish_reason = None
        resp = data.get("response")
        if isinstance(resp, dict):
            candidates = resp.get("candidates")
            if isinstance(candidates, list) and candidates:
                finish_reason = candidates[0].get("finishReason")
        elif isinstance(data.get("candidates"), list) and data["candidates"]:
            finish_reason = data["candidates"][0].get("finishReason")

        if not finish_reason:
            if isinstance(data.get("response"), dict):
                data["response"].pop("usageMetadata", None)
            data.pop("usageMetadata", None)
            return "data: " + json.dumps(data)
    except Exception:
        pass
    return line


def convert_stream_to_non_stream(stream_text: str) -> dict:
    lines = stream_text.split("\n")
    response_template = ""
    trace_id = ""
    finish_reason = ""
    model_version = ""
    response_id = ""
    role = ""
    usage_raw = None
    parts = []

    pending_kind = ""
    pending_text = ""
    pending_thought_sig = ""

    def flush_pending():
        nonlocal pending_kind, pending_text, pending_thought_sig
        if not pending_kind:
            return
        text = pending_text
        if pending_kind == "text":
            if text.strip():
                parts.append({"text": text})
        elif pending_kind == "thought":
            if text.strip() or pending_thought_sig:
                part = {"thought": True, "text": text}
                if pending_thought_sig:
                    part["thoughtSignature"] = pending_thought_sig
                parts.append(part)
        pending_kind = ""
        pending_text = ""
        pending_thought_sig = ""

    def normalize_part(part):
        m = dict(part)
        sig = part.get("thoughtSignature") or part.get("thought_signature")
        if sig:
            m["thoughtSignature"] = sig
            m.pop("thought_signature", None)
        if "inline_data" in m:
            m["inlineData"] = m.pop("inline_data")
        return m

    for line in lines:
        trimmed = line.strip()
        if not trimmed:
            continue
        try:
            data = json.loads(trimmed)
        except Exception:
            continue

        response_node = data.get("response")
        if not response_node:
            if "candidates" in data:
                response_node = data
            else:
                continue
        response_template = json.dumps(response_node)

        if data.get("traceId"):
            trace_id = data["traceId"]

        candidates = response_node.get("candidates", [])
        if candidates and isinstance(candidates[0], dict):
            c0 = candidates[0]
            if c0.get("content", {}).get("role"):
                role = c0["content"]["role"]
            if c0.get("finishReason"):
                finish_reason = c0["finishReason"]

        if response_node.get("modelVersion"):
            model_version = response_node["modelVersion"]
        if response_node.get("responseId"):
            response_id = response_node["responseId"]
        if response_node.get("usageMetadata"):
            usage_raw = response_node["usageMetadata"]
        elif data.get("usageMetadata"):
            usage_raw = data["usageMetadata"]

        parts_array = (
            candidates[0].get("content", {}).get("parts")
            if candidates and isinstance(candidates[0], dict)
            else None
        )
        if isinstance(parts_array, list):
            for part in parts_array:
                if not isinstance(part, dict):
                    continue
                has_fc = "functionCall" in part
                has_inline = "inlineData" in part or "inline_data" in part
                sig = (
                    part.get("thoughtSignature")
                    or part.get("thought_signature")
                    or ""
                )
                text = part.get("text", "")
                thought = part.get("thought", False)

                if has_fc or has_inline:
                    flush_pending()
                    parts.append(normalize_part(part))
                    continue

                if thought or "text" in part:
                    kind = "thought" if thought else "text"
                    if pending_kind and pending_kind != kind:
                        flush_pending()
                    pending_kind = kind
                    pending_text += text
                    if kind == "thought" and sig:
                        pending_thought_sig = sig
                    continue

                flush_pending()
                parts.append(normalize_part(part))

    flush_pending()

    if not response_template:
        response_template = '{"candidates":[{"content":{"role":"model","parts":[]}}]}'

    result = json.loads(response_template)
    if "candidates" not in result or not result["candidates"]:
        result["candidates"] = [{"content": {"role": "model", "parts": []}}]
    c0 = result["candidates"][0]
    if "content" not in c0 or not isinstance(c0["content"], dict):
        c0["content"] = {"role": "model", "parts": []}
    c0["content"]["parts"] = parts

    if role:
        c0["content"]["role"] = role
    if finish_reason:
        c0["finishReason"] = finish_reason
    if model_version:
        result["modelVersion"] = model_version
    if response_id:
        result["responseId"] = response_id
    if usage_raw:
        result["usageMetadata"] = usage_raw
    elif "usageMetadata" not in result:
        result["usageMetadata"] = {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0,
        }

    return {
        "response": result,
        "traceId": trace_id or "",
    }


def to_gemini_api_response(antigravity_response: dict) -> Optional[dict]:
    if not antigravity_response:
        return None
    compliant_response = {
        "candidates": antigravity_response.get("candidates", [])
    }
    if "usageMetadata" in antigravity_response:
        compliant_response["usageMetadata"] = antigravity_response[
            "usageMetadata"
        ]
    if "promptFeedback" in antigravity_response:
        compliant_response["promptFeedback"] = antigravity_response[
            "promptFeedback"
        ]
    if "automaticFunctionCallingHistory" in antigravity_response:
        compliant_response["automaticFunctionCallingHistory"] = (
            antigravity_response["automaticFunctionCallingHistory"]
        )
    return compliant_response


def build_antigravity_payload(
    model: str,
    request_body: dict,
    project_id: str,
    available_models: Optional[list] = None,
) -> Tuple[dict, str, str]:
    selected_model = normalize_antigravity_model_id(model)
    avail = available_models or ANTIGRAVITY_MODELS
    if selected_model not in avail and not is_known_antigravity_model(
        selected_model
    ):
        selected_model = "gemini-3-flash"
        request_body["model"] = selected_model

    actual_model_name = resolve_antigravity_upstream_model(selected_model)

    apply_antigravity_client_model_thinking_level_to_request(
        request_body, selected_model
    )
    processed_request_body = ensure_roles_in_contents(
        json.loads(json.dumps(request_body)), selected_model
    )
    payload = apply_antigravity_client_model_thinking_level(
        gemini_to_antigravity(
            actual_model_name, {"request": processed_request_body}, project_id
        ),
        selected_model,
    )
    request_body["model"] = actual_model_name
    return payload, selected_model, actual_model_name


# --- PKCE / OAuth Callback Server ---
class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback."""

    callback_result: Optional[Dict[str, str]] = None
    callback_error: Optional[str] = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path != OAUTH_CALLBACK_PATH:
            self.send_error(404, "Not Found")
            return

        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        error = params.get("error", [None])[0]

        if error:
            OAuthCallbackHandler.callback_error = error
            self._send_error_response(error)
        elif code and state:
            OAuthCallbackHandler.callback_result = {
                "code": code,
                "state": state,
            }
            self._send_success_response()
        else:
            OAuthCallbackHandler.callback_error = (
                "Missing code or state parameter"
            )
            self._send_error_response("Missing parameters")

    def _send_success_response(self):
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Authentication Successful</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               display: flex; justify-content: center; align-items: center; height: 100vh; 
               margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .container { background: white; padding: 3rem; border-radius: 1rem; 
                     box-shadow: 0 20px 60px rgba(0,0,0,0.3); text-align: center; max-width: 400px; }
        h1 { color: #10B981; margin-bottom: 1rem; }
        p { color: #6B7280; line-height: 1.6; }
        .icon { font-size: 4rem; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">✅</div>
        <h1>Authentication Successful!</h1>
        <p>You have successfully authenticated with Google.<br>You can close this window and return to your terminal.</p>
    </div>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html.encode())))
        self.end_headers()
        self.wfile.write(html.encode())

    def _send_error_response(self, error: str):
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Authentication Failed</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               display: flex; justify-content: center; align-items: center; height: 100vh; 
               margin: 0; background: #FEE2E2; }}
        .container {{ background: white; padding: 3rem; border-radius: 1rem; 
                     box-shadow: 0 10px 40px rgba(0,0,0,0.1); text-align: center; }}
        h1 {{ color: #EF4444; }}
        p {{ color: #6B7280; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>❌ Authentication Failed</h1>
        <p>Error: {error}</p>
        <p>Please try again.</p>
    </div>
</body>
</html>"""
        self.send_response(400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html.encode())))
        self.end_headers()
        self.wfile.write(html.encode())


class OAuthCallbackServer:
    """Local HTTP server to capture OAuth callback."""

    def __init__(self, port: int = CALLBACK_PORT, timeout: float = 300.0):
        self.port = port
        self.timeout = timeout
        self.server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = False

    def start(self) -> bool:
        try:
            OAuthCallbackHandler.callback_result = None
            OAuthCallbackHandler.callback_error = None
            self._stop_flag = False

            self.server = HTTPServer(("localhost", self.port), OAuthCallbackHandler)
            self.server.timeout = 0.5

            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()
            return True
        except OSError as e:
            debug.log(f"Failed to start OAuth callback server: {e}")
            return False

    def _serve(self):
        start_time = time.time()
        while not self._stop_flag and self.server:
            if time.time() - start_time > self.timeout:
                break
            if (
                OAuthCallbackHandler.callback_result
                or OAuthCallbackHandler.callback_error
            ):
                time.sleep(0.3)
                break
            try:
                self.server.handle_request()
            except Exception:
                break

    def wait_for_callback(self) -> Optional[Dict[str, str]]:
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if (
                OAuthCallbackHandler.callback_result
                or OAuthCallbackHandler.callback_error
            ):
                break
            time.sleep(0.1)

        self._stop_flag = True
        if self._thread:
            self._thread.join(timeout=2.0)

        if OAuthCallbackHandler.callback_error:
            raise RuntimeError(
                f"OAuth error: {OAuthCallbackHandler.callback_error}"
            )

        return OAuthCallbackHandler.callback_result

    def stop(self):
        self._stop_flag = True
        if self.server:
            try:
                self.server.server_close()
            except Exception:
                pass
            self.server = None


def generate_pkce_pair() -> Tuple[str, str]:
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def encode_oauth_state(verifier: str, project_id: str = "") -> str:
    payload = {"verifier": verifier, "projectId": project_id}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


def decode_oauth_state(state: str) -> Dict[str, str]:
    padded = state + "=" * (4 - len(state) % 4) if len(state) % 4 else state
    normalized = padded.replace("-", "+").replace("_", "/")
    try:
        decoded = base64.b64decode(normalized).decode("utf-8")
        parsed = json.loads(decoded)
        return {
            "verifier": parsed.get("verifier", ""),
            "projectId": parsed.get("projectId", ""),
        }
    except Exception:
        return {"verifier": "", "projectId": ""}


class AntigravityAuthManager(AuthFileMixin):
    """
    Handles OAuth2 authentication for Google's Antigravity API.
    Uses Antigravity-specific OAuth credentials and supports endpoint fallback.
    """

    parent = "Antigravity"
    OAUTH_REFRESH_URL = "https://oauth2.googleapis.com/token"
    OAUTH_CLIENT_ID = os.environ.get("ANTIGRAVITY_CLIENT_ID", OAUTH_CLIENT_ID)
    OAUTH_CLIENT_SECRET = os.environ.get(
        "ANTIGRAVITY_CLIENT_SECRET", OAUTH_CLIENT_SECRET
    )
    TOKEN_BUFFER_TIME = REFRESH_SKEW
    KV_TOKEN_KEY = "antigravity_oauth_token_cache"

    def __init__(
        self,
        env: Dict[str, Any],
        credentials: Optional[Dict[str, Any]] = None,
        credentials_path: Optional[Path] = None,
        name: Optional[str] = None,
    ):
        self.env = env
        self._credentials = credentials
        self.credentials_path = credentials_path
        self.name = name or (credentials or {}).get("email") or (
            credentials_path.stem if credentials_path else None
        )
        self._access_token: Optional[str] = None
        self._expiry: Optional[float] = None
        self._token_cache = {}
        self._working_base_url: Optional[str] = None
        self._project_id: Optional[str] = None
        self.quota_until: Dict[str, float] = {}

    async def initialize_auth(self) -> None:
        cached = await self._get_cached_token()
        now = time.time()
        if cached:
            expires_at = cached["expiry_date"] / 1000
            if expires_at - now > self.TOKEN_BUFFER_TIME:
                self._access_token = cached["access_token"]
                self._expiry = expires_at
                return

        creds = self._credentials
        path = self.credentials_path
        if creds is None:
            path = path or AntigravityAuthManager.get_cache_file()
            if not path.exists():
                path = get_antigravity_oauth_creds_path()

            if path.exists():
                try:
                    with path.open("r") as f:
                        creds = json.load(f)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to read OAuth credentials from {path}: {e}"
                    )
            else:
                if "ANTIGRAVITY_SERVICE_ACCOUNT" not in self.env:
                    raise RuntimeError(
                        "ANTIGRAVITY_SERVICE_ACCOUNT environment variable not set. "
                        f"Please set it or create credentials at {get_antigravity_oauth_creds_path()}"
                    )
                creds = json.loads(self.env["ANTIGRAVITY_SERVICE_ACCOUNT"])

        self.credentials_path = path
        self.name = self.name or creds.get("email") or (path.stem if path else None)

        if creds.get("project_id"):
            self._project_id = creds["project_id"]

        refresh_token = creds.get("refresh_token")
        access_token = creds.get("access_token")
        expiry_date = creds.get("expiry_date")

        if access_token and expiry_date:
            expires_at = expiry_date / 1000
            if expires_at - now > self.TOKEN_BUFFER_TIME:
                self._access_token = access_token
                self._expiry = expires_at
                await self._cache_token(access_token, expiry_date)
                return

        if not refresh_token:
            raise RuntimeError("No refresh token found in credentials.")

        await self._refresh_and_cache_token(refresh_token)

    async def _refresh_and_cache_token(self, refresh_token: str) -> None:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "client_id": self.OAUTH_CLIENT_ID,
            "client_secret": self.OAUTH_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.OAUTH_REFRESH_URL, data=data, headers=headers
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    quota_error = parse_antigravity_quota_error(resp.status, text)
                    if quota_error is not None:
                        raise quota_error
                    raise RuntimeError(f"Token refresh failed: {text}")
                resp_data = await resp.json()
                access_token = resp_data.get("access_token")
                expires_in = resp_data.get("expires_in", 3600)

                if not access_token:
                    raise RuntimeError("No access_token in refresh response.")

                self._access_token = access_token
                self._expiry = time.time() + expires_in
                expiry_date_ms = int(self._expiry * 1000)
                await self._cache_token(access_token, expiry_date_ms)

    async def _cache_token(self, access_token: str, expiry_date: int) -> None:
        token_data = {
            "access_token": access_token,
            "expiry_date": expiry_date,
            "cached_at": int(time.time() * 1000),
        }
        self._token_cache[self.KV_TOKEN_KEY] = token_data

    async def _get_cached_token(self) -> Optional[Dict[str, Any]]:
        cached = self._token_cache.get(self.KV_TOKEN_KEY)
        if cached:
            expires_at = cached["expiry_date"] / 1000
            if expires_at - time.time() > self.TOKEN_BUFFER_TIME:
                return cached
        return None

    async def clear_token_cache(self) -> None:
        self._access_token = None
        self._expiry = None
        self._token_cache.pop(self.KV_TOKEN_KEY, None)

    def get_access_token(self) -> Optional[str]:
        if (
            self._access_token is not None
            and self._expiry is not None
            and self._expiry - time.time() > self.TOKEN_BUFFER_TIME
        ):
            return self._access_token
        return None

    def get_project_id(self) -> Optional[str]:
        return self._project_id

    def get_working_base_url(self) -> str:
        """Get the cached working base URL or default to first in list."""
        return self._working_base_url or BASE_URLS[0]

    def is_quota_exhausted(self, model_family: str) -> bool:
        """Whether this account is cooling down for a model family."""
        until = self.quota_until.get(model_family)
        return until is not None and until > time.time()

    def mark_quota_reached(
        self, model_family: str, retry_after: Optional[float] = None
    ) -> None:
        """Pause this account for a model family after a quota error."""
        cooldown = retry_after if retry_after is not None else DEFAULT_QUOTA_COOLDOWN
        self.quota_until[model_family] = time.time() + max(1.0, cooldown)

    async def call_endpoint(
        self,
        method: str,
        body: Dict[str, Any],
        is_retry: bool = False,
        use_auth_headers: bool = False,
    ) -> Any:
        if not self.get_access_token():
            await self.initialize_auth()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_access_token()}",
            **(
                ANTIGRAVITY_AUTH_HEADERS
                if use_auth_headers
                else ANTIGRAVITY_HEADERS
            ),
        }

        urls_to_try = []
        if self._working_base_url:
            urls_to_try.append(self._working_base_url)
        urls_to_try.extend([url for url in BASE_URLS if url != self._working_base_url])

        last_error = None
        for base_url in urls_to_try:
            url = f"{base_url}:{method}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, headers=headers, json=body, timeout=30
                    ) as resp:
                        if resp.status == 401 and not is_retry:
                            await self.clear_token_cache()
                            await self.initialize_auth()
                            return await self.call_endpoint(
                                method,
                                body,
                                is_retry=True,
                                use_auth_headers=use_auth_headers,
                            )
                        elif resp.ok:
                            self._working_base_url = base_url
                            return await resp.json()
                        try:
                            data = await resp.json()
                            if isinstance(data, list):
                                data = next(iter(data), {})
                            message = data.get("error", {}).get("message", "")
                            quota_error = parse_antigravity_quota_error(resp.status, message)
                            if quota_error is not None:
                                raise quota_error
                            last_error = f"Error {resp.status}: {message}"
                        except Exception as e:
                            last_error = str(e)
            except Exception as e:
                last_error = str(e)
                debug.log(f"Antigravity endpoint {base_url} failed: {e}")
                continue

        raise RuntimeError(
            f"All Antigravity endpoints failed. Last error: {last_error}"
        )

    @classmethod
    def build_authorization_url(cls, project_id: str = "") -> Tuple[str, str, str]:
        verifier, challenge = generate_pkce_pair()
        state = encode_oauth_state(verifier, project_id)

        params = {
            "client_id": cls.OAUTH_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
            "scope": " ".join(ANTIGRAVITY_SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }

        url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
        return url, verifier, state

    @classmethod
    async def exchange_code_for_tokens(
        cls,
        code: str,
        state: str,
    ) -> Dict[str, Any]:
        decoded_state = decode_oauth_state(state)
        verifier = decoded_state.get("verifier", "")
        project_id = decoded_state.get("projectId", "")

        if not verifier:
            raise RuntimeError("Missing PKCE verifier in state parameter")

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            token_data = {
                "client_id": cls.OAUTH_CLIENT_ID,
                "client_secret": cls.OAUTH_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
                "code_verifier": verifier,
            }

            async with session.post(
                "https://oauth2.googleapis.com/token",
                data=token_data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "google-api-nodejs-client/10.3.0",
                },
            ) as resp:
                if not resp.ok:
                    error_text = await resp.text()
                    raise RuntimeError(f"Token exchange failed: {error_text}")

                token_response = await resp.json()

            access_token = token_response.get("access_token")
            refresh_token = token_response.get("refresh_token")
            expires_in = token_response.get("expires_in", 3600)

            if not access_token or not refresh_token:
                raise RuntimeError("Missing tokens in response")

            email = None
            async with session.get(
                "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
                headers={"Authorization": f"Bearer {access_token}"},
            ) as resp:
                if resp.ok:
                    user_info = await resp.json()
                    email = user_info.get("email")

            effective_project_id = project_id
            if not effective_project_id:
                effective_project_id = await cls._fetch_project_id(
                    session, access_token
                )

        expires_at = int((start_time + expires_in) * 1000)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expires_at,
            "email": email,
            "project_id": effective_project_id,
        }

    @classmethod
    async def _fetch_project_id(
        cls, session: aiohttp.ClientSession, access_token: str
    ) -> str:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            **ANTIGRAVITY_AUTH_HEADERS,
        }

        load_request = {
            "metadata": {
                "ideType": "ANTIGRAVITY",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        }

        timeout = aiohttp.ClientTimeout(total=10)
        for base_url in BASE_URLS:
            try:
                url = f"{base_url}:loadCodeAssist"
                async with session.post(
                    url, headers=headers, json=load_request, timeout=timeout
                ) as resp:
                    if resp.ok:
                        data = await resp.json()
                        project = data.get("cloudaicompanionProject")
                        if isinstance(project, dict):
                            project = project.get("id")
                        if project:
                            return project
            except asyncio.TimeoutError:
                debug.log(f"Project discovery timed out at {base_url}")
                continue
            except Exception as e:
                debug.log(f"Project discovery failed at {base_url}: {e}")
                continue

        attempts = int(os.environ.get("ANTIGRAVITY_ONBOARD_ATTEMPTS", "10"))
        delay_seconds = float(os.environ.get("ANTIGRAVITY_ONBOARD_DELAY_S", "5"))
        tier_id = os.environ.get("ANTIGRAVITY_TIER_ID", "free-tier")
        configured_project = os.environ.get("ANTIGRAVITY_PROJECT_ID", "")

        if tier_id:
            onboard_request_body = {
                "tier_id": tier_id,
                "metadata": {
                    "ide_type": "ANTIGRAVITY",
                    "ide_version": "2.8.1",
                    "ide_name": "antigravity",
                },
            }
            if configured_project:
                onboard_request_body["metadata"][
                    "cloudaicompanionProject"
                ] = configured_project

            for base_url in BASE_URLS:
                for attempt in range(attempts):
                    try:
                        url = f"{base_url}:onboardUser"
                        onboard_headers = {
                            "Authorization": f"Bearer {access_token}",
                            "Content-Type": "application/json",
                            **ANTIGRAVITY_HEADERS,
                        }
                        async with session.post(
                            url,
                            headers=onboard_headers,
                            json=onboard_request_body,
                            timeout=timeout,
                        ) as resp:
                            if not resp.ok:
                                if resp.status == 403:
                                    raise MissingAuthError(
                                        "Account not eligible for Antigravity Code Assist."
                                    )
                                break

                            payload = await resp.json()
                            response_obj = payload.get("response") or {}
                            managed = response_obj.get("cloudaicompanionProject")
                            managed_id = (
                                managed.get("id")
                                if isinstance(managed, dict)
                                else None
                            )

                            done = bool(payload.get("done", False))
                            if done and managed_id:
                                return managed_id
                            if done and configured_project:
                                return configured_project
                    except MissingAuthError:
                        raise
                    except Exception as e:
                        debug.log(
                            f"Failed to onboard managed project at {base_url}: {e}"
                        )
                        break

                    await asyncio.sleep(delay_seconds)

        return generate_project_id()

    @classmethod
    async def interactive_login(
        cls,
        project_id: str = "",
        no_browser: bool = False,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        auth_url, verifier, state = cls.build_authorization_url(project_id)

        print("\n" + "=" * 60)
        print("Antigravity OAuth Login")
        print("=" * 60)

        callback_server = OAuthCallbackServer(timeout=timeout)
        server_started = callback_server.start()

        if server_started and not no_browser:
            print(f"\nOpening browser for authentication...")
            print(f"If browser doesn't open, visit this URL:\n")
            print(f"{auth_url}\n")
            try:
                webbrowser.open(auth_url)
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
                print("Please open the URL above manually.\n")
        else:
            if not server_started:
                print(
                    f"\nCould not start local callback server on port {CALLBACK_PORT}."
                )
            print(f"\nPlease open this URL in your browser:\n")
            print(f"{auth_url}\n")

        if server_started:
            print("Waiting for authentication callback...")
            try:
                callback_result = callback_server.wait_for_callback()
                if not callback_result:
                    raise RuntimeError("OAuth callback timed out")

                code = callback_result.get("code")
                callback_state = callback_result.get("state")

                if not code:
                    raise RuntimeError("No authorization code received")

                print("\n✓ Authorization code received. Exchanging for tokens...")
                tokens = await cls.exchange_code_for_tokens(
                    code, callback_state or state
                )

                print(f"✓ Authentication successful!")
                if tokens.get("email"):
                    print(f"  Logged in as: {tokens['email']}")
                if tokens.get("project_id"):
                    print(f"  Project ID: {tokens['project_id']}")

                return tokens
            finally:
                callback_server.stop()
        else:
            print(
                "\nAfter completing authentication, you'll be redirected to a localhost URL."
            )
            print("Copy and paste the full redirect URL or just the code below:\n")
            user_input = input("Paste redirect URL or code: ").strip()

            if not user_input:
                raise RuntimeError("No input provided")

            if user_input.startswith("http"):
                parsed = urlparse(user_input)
                params = parse_qs(parsed.query)
                code = params.get("code", [None])[0]
                callback_state = params.get("state", [state])[0]
            else:
                code = user_input
                callback_state = state

            if not code:
                raise RuntimeError("Could not extract authorization code")

            print("\nExchanging code for tokens...")
            tokens = await cls.exchange_code_for_tokens(code, callback_state)

            print(f"✓ Authentication successful!")
            if tokens.get("email"):
                print(f"  Logged in as: {tokens['email']}")

            return tokens

    @classmethod
    async def login_and_save(
        cls,
        project_id: str = "",
        no_browser: bool = False,
        credentials_path: Optional[Path] = None,
        account: Optional[str] = None,
    ) -> "AntigravityAuthManager":
        tokens = await cls.interactive_login(
            project_id=project_id, no_browser=no_browser
        )

        creds = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "expiry_date": tokens["expiry_date"],
            "email": tokens.get("email"),
            "project_id": tokens.get("project_id"),
            "client_id": cls.OAUTH_CLIENT_ID,
            "client_secret": cls.OAUTH_CLIENT_SECRET,
        }

        if credentials_path is None:
            path = get_antigravity_account_credentials_path(account or tokens.get("email"))
        else:
            path = Path(credentials_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(creds, f, indent=2)

        try:
            path.chmod(0o600)
        except Exception:
            pass

        print(f"\n✓ Credentials saved to: {path}")
        print("=" * 60 + "\n")

        auth_manager = cls(
            env=os.environ,
            credentials=creds,
            credentials_path=path,
            name=account or tokens.get("email") or path.stem,
        )
        auth_manager._access_token = tokens["access_token"]
        auth_manager._expiry = tokens["expiry_date"] / 1000
        return auth_manager


class AntigravityAccountPool:
    """
    Pool of Antigravity OAuth accounts. Accounts that exhaust their quota for a
    model family are skipped until their cooldown expires.
    """

    def __init__(
        self,
        env: Optional[Dict[str, Any]] = None,
        managers: Optional[List[AntigravityAuthManager]] = None,
    ):
        self.env = env if env is not None else os.environ
        try:
            self.cooldown = max(
                1, int(self.env.get("ANTIGRAVITY_QUOTA_COOLDOWN", DEFAULT_QUOTA_COOLDOWN))
            )
        except (TypeError, ValueError):
            self.cooldown = DEFAULT_QUOTA_COOLDOWN
        self._managers = managers if managers is not None else self._create_managers()

    def _create_managers(self) -> List[AntigravityAuthManager]:
        accounts = load_antigravity_accounts(self.env)
        if not accounts:
            return [AntigravityAuthManager(env=self.env)]
        return [
            AntigravityAuthManager(
                env=self.env,
                credentials=account["credentials"],
                credentials_path=account["path"],
                name=account["name"],
            )
            for account in accounts
        ]

    @property
    def managers(self) -> List[AntigravityAuthManager]:
        return self._managers

    @property
    def default_manager(self) -> AntigravityAuthManager:
        return self._managers[0]

    def add_manager(self, manager: AntigravityAuthManager) -> None:
        if manager not in self._managers:
            self._managers.insert(0, manager)

    def available_accounts(
        self, model_family: Optional[str] = None
    ) -> List[AntigravityAuthManager]:
        if not model_family:
            return list(self._managers)
        return [
            manager
            for manager in self._managers
            if not manager.is_quota_exhausted(model_family)
        ]

    def mark_quota_reached(
        self,
        manager: AntigravityAuthManager,
        model_family: str,
        retry_after: Optional[float] = None,
    ) -> None:
        manager.mark_quota_reached(
            model_family, retry_after if retry_after is not None else self.cooldown
        )
        debug.log(
            f"Antigravity account {manager.name or 'default'} paused for "
            f"{model_family} models after quota error"
        )


class AntigravityProvider:
    """
    Internal provider class for Antigravity API communication.
    Handles payload formatting, project discovery, and streaming content generation.
    """

    url = "https://cloud.google.com/code-assist"

    def __init__(self, env: dict, auth_manager: AntigravityAuthManager):
        self.env = env
        self.auth_manager = auth_manager
        self._project_id: Optional[str] = None
        self.available_models: List[str] = []

    async def discover_project_id(self) -> str:
        if self.env.get("ANTIGRAVITY_PROJECT_ID"):
            return self.env["ANTIGRAVITY_PROJECT_ID"]
        if self._project_id:
            return self._project_id

        auth_project_id = self.auth_manager.get_project_id()
        if auth_project_id:
            self._project_id = auth_project_id
            return auth_project_id

        try:
            access_token = self.auth_manager.get_access_token()
            if not access_token:
                raise RuntimeError(
                    "No valid access token available for project discovery"
                )

            async with aiohttp.ClientSession() as session:
                project = await self.auth_manager._fetch_project_id(
                    session=session, access_token=access_token
                )
            if project:
                self._project_id = project
                return project
        except MissingAuthError:
            raise
        except Exception as e:
            debug.error(f"Failed to discover project ID: {e}")

        fallback_id = generate_project_id()
        self._project_id = fallback_id
        return fallback_id

    @staticmethod
    def _messages_to_gemini_format(
        messages: list, media: MediaListType
    ) -> List[Dict[str, Any]]:
        format_messages = []
        tool_names_by_id: Dict[str, str] = {}
        for msg in messages:
            msg_role = msg.get("role")
            role = "model" if msg_role == "assistant" else "user"

            if msg_role == "tool":
                tool_result = msg.get("content", "")
                if tool_result is None:
                    tool_result = ""
                elif not isinstance(tool_result, str):
                    tool_result = json.dumps(tool_result)
                tool_call_id = msg.get("tool_call_id") or ""
                tool_name = tool_names_by_id.get(tool_call_id)
                message_name = msg.get("name")
                if not tool_name and message_name and message_name != tool_call_id:
                    tool_name = message_name
                function_response = {
                    "name": tool_name or "unknown_function",
                    "response": {"result": tool_result},
                }
                if tool_call_id:
                    function_response["id"] = tool_call_id
                func_response_part = {"functionResponse": function_response}
                if (
                    format_messages
                    and format_messages[-1]["role"] == "user"
                    and any(
                        "functionResponse" in p for p in format_messages[-1]["parts"]
                    )
                ):
                    format_messages[-1]["parts"].append(func_response_part)
                else:
                    format_messages.append(
                        {"role": "user", "parts": [func_response_part]}
                    )
                continue

            elif msg_role == "assistant" and msg.get("tool_calls"):
                parts = []
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append({"text": content})
                for tool_call in msg["tool_calls"]:
                    if not isinstance(tool_call, dict) or tool_call.get(
                        "type"
                    ) != "function":
                        continue
                    function = tool_call.get("function") or {}
                    func_name = function.get("name")
                    if not func_name:
                        continue
                    if tool_call.get("id"):
                        tool_names_by_id[tool_call["id"]] = func_name
                    func_args = function.get("arguments")
                    if isinstance(func_args, str):
                        try:
                            func_args = json.loads(func_args) if func_args.strip() else {}
                        except (TypeError, ValueError):
                            func_args = {"input": func_args}
                    if not isinstance(func_args, dict):
                        func_args = {} if func_args is None else {"value": func_args}
                    func_call = {
                        "name": func_name,
                        "args": func_args,
                    }
                    if tool_call.get("id"):
                        func_call["id"] = tool_call["id"]
                    thought_sig = (
                        tool_call.get("extra_content", {})
                        .get("google", {})
                        .get(
                            "thought_signature", "skip_thought_signature_validator"
                        )
                    )
                    parts.append(
                        {"functionCall": func_call, "thoughtSignature": thought_sig}
                    )

            elif isinstance(msg["content"], str):
                parts = [{"text": msg["content"]}]

            elif isinstance(msg["content"], list):
                parts = []
                for content in msg["content"]:
                    ctype = content.get("type")
                    if ctype == "text":
                        parts.append({"text": content["text"]})
                    elif ctype == "image_url":
                        image_url = content.get("image_url", {}).get("url")
                        if not image_url:
                            continue
                        if image_url.startswith("data:"):
                            prefix, b64data = image_url.split(",", 1)
                            mime_type = prefix.split(":")[1].split(";")[0]
                            parts.append(
                                {"inlineData": {"mimeType": mime_type, "data": b64data}}
                            )
                        else:
                            parts.append(
                                {
                                    "fileData": {
                                        "mimeType": "image/jpeg",
                                        "fileUri": image_url,
                                    }
                                }
                            )
            elif msg.get("content") is not None:
                parts = [{"text": str(msg["content"])}]
            else:
                parts = []

            format_messages.append({"role": role, "parts": parts})

        if media:
            if not format_messages:
                format_messages.append({"role": "user", "parts": []})
            for media_data, filename in media:
                if isinstance(media_data, str):
                    if not filename:
                        filename = media_data
                    extension = filename.split(".")[-1].replace("jpg", "jpeg")
                    format_messages[-1]["parts"].append(
                        {
                            "fileData": {
                                "mimeType": f"image/{extension}",
                                "fileUri": media_data,
                            }
                        }
                    )
                else:
                    media_bytes = to_bytes(media_data)
                    format_messages[-1]["parts"].append(
                        {
                            "inlineData": {
                                "mimeType": is_data_an_media(media_bytes, filename),
                                "data": base64.b64encode(media_bytes).decode(),
                            }
                        }
                    )
        return format_messages

    async def stream_content(
        self,
        model: str,
        messages: Messages,
        *,
        proxy: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator:
        if model in Antigravity.model_aliases:
            model = Antigravity.model_aliases[model]

        await self.auth_manager.initialize_auth()
        project_id = await self.discover_project_id()

        contents = self._messages_to_gemini_format(
            [m for m in messages if m["role"] not in ["developer", "system"]],
            media=kwargs.get("media", None),
        )
        system_prompt = get_system_prompt(messages)
        request_data = {}
        if system_prompt:
            request_data["system_instruction"] = {"parts": [{"text": system_prompt}]}

        gemini_tools = None
        function_declarations = []
        if tools:
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    func = tool["function"]
                    name = func.get("name")
                    if not name or not isinstance(name, str):
                        continue
                    parameters = _sanitize_schema(func.get("parameters") or {})
                    if not isinstance(parameters, dict) or not parameters:
                        parameters = {"type": "object", "properties": {}}
                    function_declarations.append(
                        {
                            "name": name,
                            "description": func.get("description", ""),
                            "parameters": parameters,
                        }
                    )
            if function_declarations:
                gemini_tools = [{"functionDeclarations": function_declarations}]

        generation_config = {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
            "stop": stop,
            "presencePenalty": presence_penalty,
            "frequencyPenalty": frequency_penalty,
            "seed": seed,
        }

        if response_format is not None and response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"

        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": thinking_budget,
                "includeThoughts": True,
            }

        req_body = {
            "contents": contents,
            "generationConfig": generation_config,
            "tools": gemini_tools,
            **request_data,
        }

        if tool_choice and gemini_tools:
            mode = tool_choice.upper()
            function_calling_config = {"mode": mode}
            if mode == "ANY":
                function_calling_config["allowedFunctionNames"] = [
                    fd["name"] for fd in function_declarations
                ]
            req_body["toolConfig"] = {
                "functionCallingConfig": function_calling_config
            }

        def clean_none(d):
            if isinstance(d, dict):
                return {k: clean_none(v) for k, v in d.items() if v is not None}
            if isinstance(d, list):
                return [clean_none(x) for x in d if x is not None]
            return d

        req_body = clean_none(req_body)

        payload, selected_model, actual_model_name = build_antigravity_payload(
            model, req_body, project_id, self.available_models
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_manager.get_access_token()}",
            "Accept": "text/event-stream",
            **ANTIGRAVITY_HEADERS,
        }

        base_url = self.auth_manager.get_working_base_url()
        url = f"{base_url}:streamGenerateContent?alt=sse"

        async def parse_sse_stream(
            stream: aiohttp.StreamReader,
        ) -> AsyncGenerator[Dict[str, Any], None]:
            buffer = ""
            raw_lines = []
            yield_count = 0

            def check_for_quota(data: Dict[str, Any]) -> None:
                error = data.get("error")
                if error is None and isinstance(data.get("response"), dict):
                    error = data["response"].get("error")
                if error is not None:
                    status = error.get("code") if isinstance(error, dict) else None
                    quota_error = parse_antigravity_quota_error(status, error)
                    if quota_error is not None:
                        raise quota_error

            async for chunk_bytes in stream.iter_any():
                chunk = chunk_bytes.decode("utf-8", errors="replace")
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines.pop()

                for line in lines:
                    trimmed = line.strip()
                    if yield_count == 0 and trimmed:
                        if len(raw_lines) < ANTIGRAVITY_RAW_FALLBACK_MAX_LINES:
                            raw_lines.append(trimmed)

                    if trimmed.startswith("data: "):
                        processed_line = filter_sse_usage_metadata(trimmed)
                        json_str = processed_line[6:]
                        try:
                            data = json.loads(json_str)
                            check_for_quota(data)
                            yield_count += 1
                            yield data
                        except AntigravityQuotaError:
                            raise
                        except Exception as e:
                            debug.error(f"Error parsing SSE JSON: {e}")
                    elif trimmed == "" and raw_lines and yield_count > 0:
                        pass

            if buffer.strip():
                trimmed = buffer.strip()
                if trimmed.startswith("data: "):
                    processed_line = filter_sse_usage_metadata(trimmed)
                    json_str = processed_line[6:]
                    try:
                        data = json.loads(json_str)
                        check_for_quota(data)
                        yield data
                        yield_count += 1
                    except AntigravityQuotaError:
                        raise
                    except Exception as e:
                        debug.error(f"Error parsing final SSE JSON: {e}")

            if yield_count == 0 and raw_lines:
                raw_body = "\n".join(raw_lines)
                try:
                    parsed = json.loads(raw_body)
                    items = parsed if isinstance(parsed, list) else [parsed]
                    for item in items:
                        if isinstance(item, dict):
                            check_for_quota(item)
                            yield item
                except AntigravityQuotaError:
                    raise
                except Exception as e:
                    debug.error(f"Failed to parse raw fallback JSON: {e}")

        timeout = ClientTimeout(total=None)
        connector = get_connector(None, proxy)

        async with ClientSession(
            headers=headers, timeout=timeout, connector=connector
        ) as session:
            async with session.post(url, json=payload) as resp:
                if not resp.ok:
                    body_text = await resp.text()
                    quota_error = parse_antigravity_quota_error(resp.status, body_text)
                    if quota_error is not None:
                        raise quota_error
                    if resp.status == 503:
                        try:
                            body = json.loads(body_text)
                            retry_delay = int(
                                max(
                                    [
                                        float(d.get("retryDelay", 0))
                                        for d in body.get("error", {}).get(
                                            "details", []
                                        )
                                    ]
                                )
                            )
                        except Exception:
                            retry_delay = 30
                        debug.log(
                            f"Received 503 error, retrying after {retry_delay}s"
                        )
                        if retry_delay <= 120:
                            await asyncio.sleep(retry_delay)
                            resp = await session.post(url, json=payload)
                            if not resp.ok:
                                body_text = await resp.text()
                                quota_error = parse_antigravity_quota_error(
                                    resp.status, body_text
                                )
                                if quota_error is not None:
                                    raise quota_error
                    try:
                        data = await resp.json()
                        if isinstance(data, list):
                            data = next(iter(data), {})
                        message = data.get("error", {}).get("message", "") or body_text
                    except Exception as e:
                        message = body_text
                    await raise_for_status(resp, message=body_text)

                usage_metadata = {}
                openai_tool_calls = []
                tool_calls_index = 0
                async for json_data in parse_sse_stream(resp.content):
                    candidates = json_data.get("response", {}).get(
                        "candidates", []
                    ) or json_data.get("candidates", [])
                    usage_metadata = (
                        json_data.get("response", {}).get("usageMetadata")
                        or json_data.get("usageMetadata")
                        or usage_metadata
                    )

                    if not candidates:
                        continue

                    candidate = candidates[0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    tool_calls = []

                    for part in parts:
                        if part.get("thought") is True and "text" in part:
                            yield Reasoning(part["text"])

                        elif "functionCall" in part:
                            tool_calls.append(part)

                        elif "text" in part:
                            yield part["text"]

                        elif "inlineData" in part:
                            async for media_chunk in save_response_media(
                                part["inlineData"], format_media_prompt(messages)
                            ):
                                yield media_chunk

                        elif "fileData" in part:
                            file_data = part["fileData"]
                            yield ImageResponse(file_data.get("fileUri"))

                    if tool_calls:
                        for i, part in enumerate(tool_calls):
                            tc = part["functionCall"]
                            tool_call_obj = {
                                "index": tool_calls_index,
                                "id": tc.get(
                                    "id",
                                    f"call_{i}_{tc.get('name', 'unknown')}",
                                ),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name"),
                                    "arguments": json.dumps(tc.get("args", {})),
                                },
                            }
                            if "thoughtSignature" in part:
                                tool_call_obj["extra_content"] = {
                                    "google": {
                                        "thought_signature": part[
                                            "thoughtSignature"
                                        ]
                                    }
                                }
                            openai_tool_calls.append(tool_call_obj)
                            tool_calls_index += 1

                if openai_tool_calls:
                    yield ToolCalls(openai_tool_calls)

                if usage_metadata:
                    yield Usage(**usage_metadata)


class Antigravity(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Antigravity Provider for gpt4free (v2).

    Provides access to Google's Antigravity API (Code Assist) supporting:
    - Gemini 2.5 & Gemini 3 Pro/Flash models
    - Gemini 3.1 / 3.5 / 3.6 / 3.7 / 3.8 Flash & Pro variants
    - Claude Sonnet 4.5/4.6 & Opus 4.5/4.6 via Antigravity proxy
    - Image generation models (gemini-3.1-flash-image)
    - "auto" model: mid-size Gemini first, Claude when Gemini quota is reached

    Requires OAuth2 credentials. Multiple accounts can be stored in
    ~/.antigravity/accounts/ and are switched automatically when quota is
    reached. Set ANTIGRAVITY_SERVICE_ACCOUNT or create credentials at
    ~/.antigravity/oauth_creds.json for a single account.
    """

    label = "Google Antigravity"
    url = "https://antigravity.google"
    screenshot_url = "https://antigravity.google"
    login_url = "https://cloud.google.com/code-assist"

    default_model = "gemini-3-flash"
    fallback_models = [AUTO_MODEL, *ANTIGRAVITY_MODELS]

    model_aliases = {
        "claude-sonnet-4.5": "claude-sonnet-4-5",
        "claude-opus-4.5": "claude-opus-4-5",
        "claude-sonnet-4.6": "claude-sonnet-4-6",
        "claude-opus-4.6": "claude-opus-4-6-thinking",
    }

    working = True
    supports_message_history = True
    supports_system_message = True
    supports_native_tools = True
    needs_auth = True
    active_by_default = True

    auth_manager: AntigravityAuthManager = None
    account_pool: Optional[AntigravityAccountPool] = None

    @classmethod
    def _get_account_pool(cls) -> AntigravityAccountPool:
        if cls.account_pool is None:
            cls.account_pool = AntigravityAccountPool(env=os.environ)
        return cls.account_pool

    @classmethod
    def _get_auth_manager(cls) -> AntigravityAuthManager:
        manager = cls._get_account_pool().default_manager
        cls.auth_manager = manager
        return manager

    @classmethod
    def get_models(cls, **kwargs) -> List[str]:
        if not cls.models and cls.has_credentials():
            try:
                get_running_loop(check_nested=True)
                cls.models = asyncio.run(cls._fetch_models())
            except Exception as e:
                debug.log(f"Failed to fetch dynamic models: {e}")

        if cls.live == 0:
            if cls._get_auth_manager().get_access_token() is not None:
                cls.live += 1

        models = list(cls.models) if cls.models else list(cls.fallback_models)
        if AUTO_MODEL not in models:
            models.insert(0, AUTO_MODEL)
        return models

    @classmethod
    async def _fetch_models(cls) -> List[str]:
        auth_manager = cls._get_auth_manager()

        await auth_manager.initialize_auth()

        try:
            response = await auth_manager.call_endpoint(
                method="fetchAvailableModels",
                body={"project": auth_manager.get_project_id()},
            )
            models_dict = response.get("models", {})
            if isinstance(models_dict, dict):
                raw_models = [
                    key
                    for key, value in models_dict.items()
                    if not value.get("isInternal", False) and not key.startswith("tab_")
                ]
                expanded_models = []
                for m in raw_models:
                    for expanded in expand_antigravity_client_models(m):
                        if expanded not in expanded_models:
                            expanded_models.append(expanded)
                return expanded_models if expanded_models else ANTIGRAVITY_MODELS
            return ANTIGRAVITY_MODELS
        except Exception as e:
            debug.log(f"Failed to fetch models: {e}")
            return ANTIGRAVITY_MODELS

    @classmethod
    async def get_quota(cls, api_key: Optional[str] = None) -> dict:
        auth_manager = cls._get_auth_manager()
        await auth_manager.initialize_auth()

        access_token = auth_manager.get_access_token()
        project_id = auth_manager.get_project_id()
        if not access_token or not project_id:
            raise MissingAuthError("Cannot fetch usage without valid authentication")

        return await auth_manager.call_endpoint(
            method="fetchAvailableModels",
            body={"project": project_id},
        )

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        media: MediaListType = None,
        tools: Optional[list] = None,
        **kwargs,
    ) -> AsyncResult:
        pool = cls._get_account_pool()
        model = cls.model_aliases.get(model, model)
        if model == AUTO_MODEL:
            models = get_auto_model_candidates(cls.models) or get_auto_model_candidates(
                cls.fallback_models
            )
        else:
            models = [model]
        if not models:
            models = ["gemini-3-flash"]

        last_error: Optional[Exception] = None
        for model_candidate in models:
            model_family = get_antigravity_model_family(model_candidate)
            tried_accounts = set()
            while True:
                available = [
                    manager
                    for manager in pool.available_accounts(model_family)
                    if manager not in tried_accounts
                ]
                if not available:
                    break
                manager = available[0]
                tried_accounts.add(manager)
                provider = AntigravityProvider(env=os.environ, auth_manager=manager)
                provider.available_models = list(cls.models or [])
                emitted = False
                try:
                    async for chunk in provider.stream_content(
                        model=model_candidate,
                        messages=messages,
                        stream=stream,
                        media=media,
                        tools=tools,
                        **kwargs,
                    ):
                        emitted = True
                        yield chunk
                    return
                except AntigravityQuotaError as e:
                    last_error = e
                    pool.mark_quota_reached(manager, model_family, e.retry_after)
                    debug.log(
                        f"Antigravity quota reached for {model_candidate} on account "
                        f"{manager.name or 'default'}, trying another account or model"
                    )
                    if emitted:
                        raise
                except MissingAuthError as e:
                    last_error = e
                    debug.log(
                        f"Antigravity authentication failed on account "
                        f"{manager.name or 'default'}: {e}"
                    )
                    if emitted:
                        raise

        if last_error is not None:
            raise last_error
        raise RateLimitError(
            "Antigravity quota reached for all accounts and models"
        )

    @classmethod
    async def login(
        cls,
        project_id: str = "",
        no_browser: bool = False,
        credentials_path: Optional[Path] = None,
        account: Optional[str] = None,
    ) -> "AntigravityAuthManager":
        auth_manager = await AntigravityAuthManager.login_and_save(
            project_id=project_id,
            no_browser=no_browser,
            credentials_path=credentials_path,
            account=account,
        )
        cls.auth_manager = auth_manager
        cls.account_pool = None
        cls._get_account_pool().add_manager(auth_manager)
        return auth_manager

    @classmethod
    def has_credentials(cls) -> bool:
        return len(load_antigravity_accounts(os.environ)) > 0

    @classmethod
    def get_credentials_path(cls) -> Path:
        accounts = load_antigravity_accounts(os.environ)
        if accounts and accounts[0]["path"] is not None:
            return accounts[0]["path"]
        cache_path = AntigravityAuthManager.get_cache_file()
        if cache_path.exists():
            return cache_path
        default_path = get_antigravity_oauth_creds_path()
        if default_path.exists():
            return default_path
        return cache_path


async def main(args: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Antigravity OAuth Authentication for gpt4free (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s login                    # Interactive login with browser
  %(prog)s login --account work     # Login as a named account
  %(prog)s login --no-browser       # Manual login (paste URL)
  %(prog)s login --project-id ID    # Login with specific project
  %(prog)s status                   # List all accounts
  %(prog)s logout work              # Remove one account
  %(prog)s logout                   # Remove all saved credentials
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")
    login_parser = subparsers.add_parser("login", help="Authenticate with Google")
    login_parser.add_argument(
        "--project-id",
        "-p",
        default="",
        help="Google Cloud project ID (optional, auto-discovered if not set)",
    )
    login_parser.add_argument(
        "--account",
        "-a",
        default=None,
        help="Account name for stored credentials (default: Google email)",
    )
    login_parser.add_argument(
        "--no-browser",
        "-n",
        action="store_true",
        help="Don't auto-open browser, print URL instead",
    )

    subparsers.add_parser("status", help="List all configured accounts")
    logout_parser = subparsers.add_parser("logout", help="Remove saved credentials")
    logout_parser.add_argument(
        "account",
        nargs="?",
        default=None,
        help="Account name or email to remove (default: all accounts)",
    )

    args = parser.parse_args(args)

    if args.command == "login":
        try:
            await Antigravity.login(
                project_id=args.project_id,
                no_browser=args.no_browser,
                account=args.account,
            )
        except KeyboardInterrupt:
            print("\n\nLogin cancelled.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Login failed: {e}")
            sys.exit(1)

    elif args.command == "status":
        print("\nAntigravity Authentication Status")
        print("=" * 40)

        accounts = load_antigravity_accounts(os.environ)
        if accounts:
            print(f"✓ {len(accounts)} account(s) found:")
            for account in accounts:
                creds = account["credentials"]
                email = creds.get("email")
                print(
                    f"  - {account['name']}"
                    + (f" ({email})" if email and email != account["name"] else "")
                )
                print(f"    Source: {account['path'] or 'ANTIGRAVITY_SERVICE_ACCOUNT'}")
                if creds.get("project_id"):
                    print(f"    Project: {creds['project_id']}")
                expiry = creds.get("expiry_date")
                if expiry:
                    expiry_time = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(expiry / 1000)
                    )
                    state = "expires" if expiry / 1000 > time.time() else "expired"
                    print(f"    Token {state}: {expiry_time}")
        else:
            print("✗ No credentials found")
            print("\nRun 'antigravity login' to authenticate.")
        print()

    elif args.command == "logout":
        print("\nAntigravity Logout")
        print("=" * 40)
        removed = False

        if args.account:
            paths = [
                account["path"]
                for account in load_antigravity_accounts(os.environ)
                if account["name"] == args.account
                or account["credentials"].get("email") == args.account
            ]
            named_path = get_antigravity_account_credentials_path(args.account)
            if named_path.exists() and named_path not in paths:
                paths.append(named_path)
        else:
            paths = [
                account["path"]
                for account in load_antigravity_accounts(os.environ)
                if account["path"] is not None
            ]

        for path in dict.fromkeys(paths):
            try:
                path.unlink()
                print(f"✓ Removed: {path}")
                removed = True
            except OSError as e:
                print(f"Could not remove {path}: {e}")

        if removed:
            print("\n✓ Credentials removed successfully.")
        else:
            print("No credentials found to remove.")
        print()
    else:
        parser.print_help()


def cli_main(args: Optional[List[str]] = None):
    asyncio.run(main(args))


if __name__ == "__main__":
    cli_main()
