"""Free-model discovery and health checks for the pi-freeflow proxy.

Only models returned by the proxy and answering a real chat completion are
used.  There is no fallback to paid providers: if no free model works the
pipeline stops instead of inventing a working model.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import PipelineConfig, ProjectConfig
from .util import truncate, utc_now

DEFAULT_BASE_URL = "http://127.0.0.1:28180/v1"


class FreeflowError(RuntimeError):
    """Raised when the freeflow proxy or every free model is unavailable."""


@dataclass
class ModelResult:
    id: str
    ok: bool
    latency_s: float = 0.0
    detail: str = ""
    checked_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ok": self.ok,
            "latency_s": round(self.latency_s, 2),
            "detail": truncate(self.detail, 300),
            "checked_at": self.checked_at or utc_now(),
        }


def _token_candidates(project_path: Path, root: Path) -> List[Optional[str]]:
    candidates: List[Optional[str]] = []
    for key in ("FREEFLOW_TOKEN", "FREEFLOW_API_KEY", "G4F_FREEFLOW_TOKEN"):
        value = os.environ.get(key)
        if value:
            candidates.append(value.strip())
    for path in (
        project_path / ".pi" / "agent" / "token",
        root / ".pi" / "agent" / "token",
        Path.home() / ".pi" / "agent" / "token",
    ):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            candidates.append(value)

    # Try the token-less route last.  A local relay may allow it.
    candidates.append(None)

    unique: List[Optional[str]] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _base_candidates(config: PipelineConfig) -> List[str]:
    values = [
        os.environ.get("FREEFLOW_BASE_URL"),
        os.environ.get("FREEFLOW_BASE"),
        config.base_url,
        DEFAULT_BASE_URL,
    ]
    result: List[str] = []
    for value in values:
        if not value:
            continue
        value = value.rstrip("/")
        if value not in result:
            result.append(value)
    return result


def _http_json(
    url: str,
    token: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        detail = body.decode("utf-8", "replace")
        raise FreeflowError(f"HTTP {exc.code} from {url}: {truncate(detail, 300)}") from exc
    except urllib.error.URLError as exc:
        raise FreeflowError(f"cannot reach freeflow at {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise FreeflowError(f"timed out waiting for {url}") from exc
    try:
        parsed = json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise FreeflowError(f"invalid JSON from {url}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise FreeflowError(f"unexpected response from {url}")
    return parsed


@dataclass
class FreeflowClient:
    base_url: str
    token: Optional[str]
    models: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def connect(cls, config: PipelineConfig, project: ProjectConfig) -> "FreeflowClient":
        errors: List[str] = []
        for base_url in _base_candidates(config):
            for token in _token_candidates(project.path, config.root):
                try:
                    payload = _http_json(f"{base_url}/models", token, timeout=15)
                except FreeflowError as exc:
                    errors.append(str(exc))
                    continue
                models = payload.get("data")
                if models is None and isinstance(payload.get("models"), list):
                    models = payload["models"]
                if not isinstance(models, list):
                    errors.append(f"{base_url}: /models did not return a data list")
                    continue
                return cls(base_url=base_url, token=token, models=models)
        detail = "; ".join(dict.fromkeys(errors))[-900:]
        raise FreeflowError(
            "pi-freeflow is not reachable. Start it with "
            "`sandbox-cli start-freeflow-api` or launch Pi with the freeflow "
            "extension installed.\n" + (detail or "no response")
        )

    @property
    def model_ids(self) -> List[str]:
        result = []
        for model in self.models:
            model_id = model.get("id") if isinstance(model, dict) else None
            if model_id:
                result.append(str(model_id))
        return result

    def probe(self, model_id: str, timeout: int = 60) -> ModelResult:
        started = time.monotonic()
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with the single word OK. Do not explain.",
                }
            ],
            "max_tokens": 64,
            "stream": False,
        }
        try:
            response = _http_json(
                f"{self.base_url}/chat/completions",
                self.token,
                payload=payload,
                timeout=timeout,
            )
        except FreeflowError as exc:
            return ModelResult(
                id=model_id,
                ok=False,
                latency_s=time.monotonic() - started,
                detail=str(exc),
                checked_at=utc_now(),
            )

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ModelResult(
                id=model_id,
                ok=False,
                latency_s=time.monotonic() - started,
                detail="response has no choices: " + truncate(json.dumps(response), 200),
                checked_at=utc_now(),
            )
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = (message.get("content") or first.get("text") or "").strip()
        reasoning = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
        if not content and not reasoning:
            return ModelResult(
                id=model_id,
                ok=False,
                latency_s=time.monotonic() - started,
                detail="empty assistant response",
                checked_at=utc_now(),
            )
        return ModelResult(
            id=model_id,
            ok=True,
            latency_s=time.monotonic() - started,
            detail=content[:80] or "reasoning-only response",
            checked_at=utc_now(),
        )


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _matches(model_norm: str, candidates: Sequence[str]) -> bool:
    """Return True when any short alias or full id matches ``model_norm``."""
    for candidate in candidates:
        candidate_norm = _normalise(candidate)
        if not candidate_norm:
            continue
        if candidate_norm in model_norm or model_norm in candidate_norm:
            return True
    return False


def is_model_disabled(model_id: str, disabled: Sequence[str]) -> bool:
    """Return True when ``model_id`` is excluded by ``disabled_models``."""
    return _matches(_normalise(model_id), disabled)


def rank_models(
    model_ids: Sequence[str],
    preferred: Sequence[str],
    disabled: Sequence[str] = (),
) -> List[str]:
    """Order catalog models by the configured coding preference.

    Models matching ``disabled`` are removed entirely so they are never probed
    or handed to the agent, even when the freeflow catalog still lists them.
    """
    preferred_norm = [_normalise(item) for item in preferred]

    def score(model_id: str) -> Tuple[int, int, str]:
        model_norm = _normalise(model_id)
        rank = len(preferred_norm) + 1
        for index, preferred_id in enumerate(preferred_norm):
            if not preferred_id:
                continue
            if preferred_id in model_norm or model_norm in preferred_id:
                rank = index
                break
        # Counting a wider context is impossible from /models, so prefer ids
        # that look like coding models as a gentle tie-breaker.
        coding_bonus = 0 if any(token in model_norm for token in ("code", "deepseek", "qwen", "glm")) else 1
        return (rank, coding_bonus, model_id.lower())

    allowed = [
        model_id
        for model_id in dict.fromkeys(model_ids)
        if not is_model_disabled(model_id, disabled)
    ]
    return sorted(allowed, key=score)


def _load_cache(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def select_working_models(
    config: PipelineConfig,
    project: ProjectConfig,
    refresh: bool = False,
    limit: Optional[int] = None,
    min_working: Optional[int] = None,
    explicit_model: Optional[str] = None,
    explore: bool = False,
) -> Tuple[FreeflowClient, List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Return ``(client, working, failed, from_cache)``.

    Every returned model completed a real chat completion against the
    freeflow proxy.  Cached health is used when fresh unless ``refresh`` is set.
    By default probing stops once ``min_working`` models work; ``explore=True``
    probes up to ``limit`` working models (used by ``models --refresh``).
    """
    client = FreeflowClient.connect(config, project)
    limit = limit or config.probe_limit
    min_working = min_working or min(config.min_models, limit)
    target = limit if explore else min_working

    if explicit_model:
        if is_model_disabled(explicit_model, config.disabled_models):
            raise FreeflowError(
                f"requested model {explicit_model!r} is disabled in "
                "dev-pipeline.json ('disabled_models')"
            )
        result = client.probe(explicit_model)
        if not result.ok:
            raise FreeflowError(
                f"requested model {explicit_model!r} is not working: {result.detail}"
            )
        return client, [result.as_dict()], [], False

    cache = _load_cache(config.cache_path)
    if not refresh and cache and cache.get("base_url") == client.base_url:
        age = time.time() - float(cache.get("checked_at_epoch") or 0)
        cached_working = [
            item for item in cache.get("working", []) if isinstance(item, dict) and item.get("id")
        ]
        if age <= config.model_cache_ttl and len(cached_working) >= target:
            # Keep only models that are still in the live catalog and not
            # disabled by the operator.
            live = set(client.model_ids)
            cached_working = [
                item
                for item in cached_working
                if item["id"] in live
                and not is_model_disabled(item["id"], config.disabled_models)
            ]
            if len(cached_working) >= target:
                return client, cached_working[:limit], list(cache.get("failed", [])), True

    ranked = rank_models(
        client.model_ids, config.preferred_models, config.disabled_models
    )
    working: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    attempts = 0

    for model_id in ranked:
        if len(working) >= limit or attempts >= config.probe_limit:
            break
        attempts += 1
        result = client.probe(model_id)
        if result.ok:
            working.append(result.as_dict())
        else:
            failed.append(result.as_dict())
        if len(working) >= target:
            # Enough models for this operation; stop to preserve free quotas.
            break

    if not working:
        summary = "; ".join(
            f"{item['id']}: {item.get('detail', 'failed')}" for item in failed[:5]
        )
        raise FreeflowError(
            "all probed freeflow models failed. Check `bin/dev-pipeline models --refresh` "
            "and the freeflow log at ~/.pi/agent/pi-freeflow.log.\n" + summary
        )

    _write_cache(
        config.cache_path,
        {
            "base_url": client.base_url,
            "checked_at": utc_now(),
            "checked_at_epoch": time.time(),
            "working": working,
            "failed": failed,
        },
    )
    return client, working, failed, False
