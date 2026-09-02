from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from g4f.providers.response import JsonResponse, ProviderInfo, Reasoning


def _json_default(value: Any) -> Any:
    get_dict = getattr(value, "get_dict", None)
    if callable(get_dict):
        return get_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            key: item
            for key, item in attributes.items()
            if not key.startswith("__")
        }
    return str(value)


class ChunkJsonlWriter:
    """Append response metadata chunks to a durable, readable JSONL journal."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id or str(uuid.uuid4())
        self.count = 0
        self._file: TextIO | None = None

    def __enter__(self) -> ChunkJsonlWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, chunk: Any) -> dict[str, Any]:
        if self._file is None:
            raise RuntimeError("ChunkJsonlWriter must be used as a context manager")

        get_dict = getattr(chunk, "get_dict", None)
        data = get_dict() if callable(get_dict) else vars(chunk)
        self.count += 1
        record = {
            "run_id": self.run_id,
            "sequence": self.count,
            "recorded_at": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "chunk_type": type(chunk).__name__,
            "data": data,
        }
        self._file.write(
            json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        )
        self._file.flush()
        return record


def store_or_collect_chunk(
    chunk: Any,
    writer: ChunkJsonlWriter,
    visible_chunks: list[str],
) -> None:
    if isinstance(chunk, Exception):
        raise chunk
    if isinstance(chunk, (ProviderInfo, JsonResponse, Reasoning)):
        writer.write(chunk)
        return
    visible_chunks.append(str(chunk))


def read_chunk_records(
    path: str | Path,
    *,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        return []

    records = []
    with log_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL record at {log_path}:{line_number}"
                ) from error
            if run_id is None or record.get("run_id") == run_id:
                records.append(record)
    return records
