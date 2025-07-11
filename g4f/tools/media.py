from __future__ import annotations

import os
import base64
from typing import Iterator, Union
from urllib.parse import urlparse
from pathlib import Path

from ..typing import Messages
from ..image import is_data_an_media, to_input_audio, is_valid_media, is_valid_audio, to_data_uri
from .files import get_bucket_dir, read_bucket

def render_media(bucket_id: str, name: str, url: str, as_path: bool = False, as_base64: bool = False, **kwargs) -> Union[str, Path]:
    if (as_base64 or as_path or url.startswith("/")):
        file = Path(get_bucket_dir(bucket_id, "thumbnail", name))
        if not file.exists():
            file = Path(get_bucket_dir(bucket_id, "media", name))
        if as_path:
            return file
        data = file.read_bytes()
        data_base64 = base64.b64encode(data).decode()
        if as_base64:
            return data_base64
        return f"data:{is_data_an_media(data, name)};base64,{data_base64}"
    return url

def render_part(part: dict) -> dict:
    if "type" in part:
        return part
    text = part.get("text")
    if text:
        return {
            "type": "text",
            "text": text
        }
    filename = part.get("name")
    if (filename is None):
        bucket_dir = Path(get_bucket_dir(part.get("bucket_id")))
        return {
            "type": "text",
            "text": "".join(read_bucket(bucket_dir))
        }
    if is_valid_audio(filename=filename):
        return {
            "type": "input_audio",
            "input_audio": {
                "data": render_media(**part, as_base64=True),
                "format": os.path.splitext(filename)[1][1:]
            }
        }
    if is_valid_media(filename=filename):
        return {
            "type": "image_url",
            "image_url": {"url": render_media(**part)}
        }

def merge_media(media: list, messages: list) -> Iterator:
    buffer = []
    # Read media from the last user message
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if "type" not in part and "name" in part and "text" not in part:
                        path = render_media(**part, as_path=True)
                        buffer.append((path, os.path.basename(path)))
                    elif part.get("type") == "image_url":
                        path: str = urlparse(part.get("image_url")).path
                        if path.startswith("/files/"):
                            path = get_bucket_dir(path.split(path, "/")[1:])
                            if os.path.exists(path):
                                buffer.append((Path(path), os.path.basename(path)))
                            else:
                                buffer.append((part.get("image_url"), None))
                        else:
                            buffer.append((part.get("image_url"), None))
        else:
            buffer = []
    yield from buffer
    if media is not None:
        yield from media

def render_messages(messages: Messages, media: list = None) -> Iterator:
    last_is_assistant = False
    for idx, message in enumerate(messages):
        # Remove duplicate assistant messages
        if message.get("role") == "assistant":
            if last_is_assistant:
                continue
            last_is_assistant = True
        else:
            last_is_assistant = False
        # Render content parts
        if isinstance(message.get("content"), list):
            parts = [render_part(part) for part in message["content"] if part]
            yield {
                **message,
                "content": [part for part in parts if part]
            }
        else:
            # Append media to the last message
            if media is not None and idx == len(messages) - 1:
                yield {
                    **message,
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": to_input_audio(media_data, filename)
                        }
                        if is_valid_audio(media_data, filename) else {
                            "type": "image_url",
                            "image_url": {"url": to_data_uri(media_data)}
                        }
                        for media_data, filename in media
                        if media_data and is_valid_media(media_data, filename)
                    ] + ([{"type": "text", "text": message["content"]}] if isinstance(message["content"], str) else message["content"])
                }
            else:
                yield message