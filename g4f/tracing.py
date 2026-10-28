from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Set G4F_LANGFUSE_TRACING=0 to keep the SDK installed but stop sending traces.
TRACING_ENV = "G4F_LANGFUSE_TRACING"
# Extra tags added to every trace, comma separated.
TAGS_ENV = "G4F_LANGFUSE_TAGS"

# Request fields that describe how the model was called rather than what it was
# asked. They are reported to Langfuse as `model_parameters`.
_MODEL_PARAMETER_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "stop_sequences",
    "reasoning_effort",
)


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _get_client():
    """Return a configured Langfuse client, or None when tracing is off.

    Tracing stays off unless the SDK is installed *and* both credentials are
    present, so importing g4f never requires Langfuse.
    """
    if not _env_flag(TRACING_ENV):
        return None
    if not (
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    ):
        return None
    try:
        from langfuse import get_client
    except ImportError:
        logger.debug(
            "Langfuse credentials are set but the 'langfuse' package is not installed"
        )
        return None
    try:
        return get_client()
    except Exception as e:
        logger.warning(f"Could not initialize Langfuse tracing: {e}")
        return None


def is_enabled() -> bool:
    """Whether g4f will send traces to Langfuse."""
    return _get_client() is not None


def model_parameters(config: Any) -> Optional[dict]:
    """Collect the sampling parameters of a request config for Langfuse."""
    params = {}
    for field in _MODEL_PARAMETER_FIELDS:
        value = getattr(config, field, None)
        if value is None or isinstance(value, dict):
            continue
        if field == "stop" and isinstance(value, str):
            value = [value]
        params[field] = value
    return params or None


def _extract_text(result: Any) -> Optional[str]:
    """Read the assistant text out of a completion or a stream chunk."""
    choices = getattr(result, "choices", None)
    if not choices:
        return None
    choice = choices[0]
    for holder in ("message", "delta"):
        content = getattr(getattr(choice, holder, None), "content", None)
        if content:
            return str(content)
    return None


def _extract_usage(result: Any) -> Optional[dict]:
    """Map a g4f usage model onto Langfuse's usage_details keys."""
    usage = getattr(result, "usage", None)
    if usage is None:
        return None
    details = {}
    for key, field in (
        ("input", "prompt_tokens"),
        ("output", "completion_tokens"),
        ("total", "total_tokens"),
    ):
        value = getattr(usage, field, None)
        if value:
            details[key] = value
    return details or None


def _trace_tags(tags: Optional[list[str]]) -> Optional[list[str]]:
    merged = [t.strip() for t in os.environ.get(TAGS_ENV, "").split(",") if t.strip()]
    for tag in tags or []:
        if tag and tag not in merged:
            merged.append(tag)
    return merged or None


class RequestTrace:
    """A Langfuse observation wrapping one g4f API request.

    The object is inert when tracing is disabled, so callers never have to
    branch on whether Langfuse is configured.
    """

    def __init__(self, observation: Any = None, trace_id: Optional[str] = None):
        self._observation = observation
        self.trace_id = trace_id
        self._output_parts: list[str] = []
        self._usage: Optional[dict] = None
        self._model: Optional[str] = None
        self._provider: Optional[str] = None
        self._finished = False

    @property
    def enabled(self) -> bool:
        return self._observation is not None

    def record_output(self, result: Any) -> None:
        """Accumulate the text, model and token usage of a completion or chunk."""
        if self._observation is None or result is None:
            return
        text = _extract_text(result)
        if text:
            self._output_parts.append(text)
        usage = _extract_usage(result)
        if usage:
            self._usage = usage
        model = getattr(result, "model", None)
        if model:
            self._model = model
        provider = getattr(result, "provider", None)
        if provider:
            self._provider = provider

    def record_error(self, error: BaseException) -> None:
        """Mark the observation as failed."""
        if self._observation is None:
            return
        try:
            self._observation.update(
                level="ERROR", status_message=str(error)[:1000] or type(error).__name__
            )
        except Exception as e:
            logger.debug(f"Could not record Langfuse error: {e}")

    def finish(self) -> None:
        """Write the collected output and close the observation. Idempotent."""
        if self._observation is None or self._finished:
            return
        self._finished = True
        update = {}
        if self._output_parts:
            update["output"] = "".join(self._output_parts)
        if self._usage:
            update["usage_details"] = self._usage
        if self._model:
            update["model"] = self._model
        if self._provider:
            update["metadata"] = {"provider": self._provider}
        try:
            if update:
                self._observation.update(**update)
            self._observation.end()
        except Exception as e:
            logger.debug(f"Could not finish Langfuse observation: {e}")


def start_request_trace(
    name: str,
    *,
    input: Any = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    model_parameters: Optional[dict] = None,
    metadata: Optional[dict] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> RequestTrace:
    """Open a Langfuse generation for one API request.

    Trace level attributes (user, session, tags, name) are propagated while the
    observation is created so they land on the trace root, and the observation
    is returned unclosed: the caller ends it once the response is complete.
    """
    client = _get_client()
    if client is None:
        return RequestTrace()
    trace_metadata = {k: v for k, v in (metadata or {}).items() if v is not None}
    if provider:
        trace_metadata["provider"] = provider
    try:
        from langfuse import propagate_attributes

        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            trace_name=name,
            tags=_trace_tags(tags),
            metadata=trace_metadata or None,
        ):
            observation = client.start_observation(
                as_type="generation",
                name=name,
                input=input,
                model=model,
                model_parameters=model_parameters,
            )
    except Exception as e:
        logger.warning(f"Could not start Langfuse trace: {e}")
        return RequestTrace()
    return RequestTrace(observation, trace_id=getattr(observation, "trace_id", None))


def flush() -> None:
    """Send any buffered traces to Langfuse."""
    client = _get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:
        logger.debug(f"Langfuse flush failed: {e}")


def shutdown() -> None:
    """Flush and release the Langfuse client, e.g. when the server stops."""
    client = _get_client()
    if client is None:
        return
    try:
        client.shutdown()
    except Exception as e:
        logger.debug(f"Langfuse shutdown failed: {e}")