"""Lightweight hooks for observing outgoing HTTP requests.

The API server registers a sink so provider requests show up in ``/logs``
alongside the incoming API requests. The module deliberately does not import
anything from :mod:`g4f.api`, which keeps the dependency direction acyclic.

Only request metadata is put in the in-memory API log. A textual request
body may be attached so the API can persist it in its optional HAR log, but
response bodies and streaming payloads are never captured here.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Mapping, Optional

RequestLogSink = Callable[[dict], None]

_sinks: list[RequestLogSink] = []
_sinks_lock = threading.Lock()


def add_request_sink(sink: RequestLogSink) -> None:
    """Register a callback that receives one dict per finished request."""
    with _sinks_lock:
        if sink not in _sinks:
            _sinks.append(sink)


def remove_request_sink(sink: RequestLogSink) -> None:
    """Remove a previously registered callback."""
    with _sinks_lock:
        if sink in _sinks:
            _sinks.remove(sink)


def has_request_sinks() -> bool:
    with _sinks_lock:
        return bool(_sinks)


def _headers_to_dict(headers) -> dict:
    if not headers:
        return {}
    try:
        return {str(key): str(value) for key, value in headers.items()}
    except Exception:
        return {}


def _emit(entry: dict) -> None:
    with _sinks_lock:
        sinks = list(_sinks)
    for sink in sinks:
        try:
            sink(entry)
        except Exception:
            # Logging must never break an outgoing request.
            pass


class RequestTracker:
    """Collects metadata for a single outgoing request."""

    def __init__(
        self,
        method: str,
        url: str,
        headers: Optional[Mapping] = None,
        body: Optional[str] = None,
    ):
        self.method = str(method).upper()
        self.url = str(url)
        self.headers = _headers_to_dict(headers)
        self.body = body
        self._started = time.monotonic()
        self._timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._finished = False

    def finish(
        self,
        status: Optional[int] = None,
        headers: Optional[Mapping] = None,
        url: Optional[str] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        if url:
            self.url = str(url)
        _emit(
            {
                "method": self.method,
                "url": self.url,
                "status": status,
                "duration_ms": round((time.monotonic() - self._started) * 1000),
                "timestamp": self._timestamp,
                "request_headers": self.headers,
                "request_body": self.body,
                "response_headers": _headers_to_dict(headers),
                "error": str(error) if error is not None else None,
            }
        )


def start_request(
    method: str,
    url: str,
    headers: Optional[Mapping] = None,
    body: Optional[str] = None,
) -> Optional[RequestTracker]:
    """Start tracking a request, or return ``None`` when nobody is listening."""
    if not has_request_sinks():
        return None
    return RequestTracker(method, url, headers, body)
