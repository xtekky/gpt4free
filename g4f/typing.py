from __future__ import annotations

import os
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    AsyncIterator,
    Iterator,
    NewType,
    Tuple,
    Union,
    List,
    Dict,
    Type,
    IO,
    Optional,
    TYPE_CHECKING,
)
from typing_extensions import TypedDict

# Only import PIL for type-checkers; no runtime dependency required.
if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
else:
    class PILImage:  # minimal placeholder to avoid runtime import errors
        pass

# Response chunk type from providers
from .providers.response import ResponseType

# ---- Hashes & cookie aliases -------------------------------------------------

SHA256 = NewType("SHA256", str)
Cookies = Dict[str, str]

# ---- Streaming result types --------------------------------------------------

CreateResult = Iterator[Union[str, ResponseType]]
AsyncResult = AsyncIterator[Union[str, ResponseType]]

# ---- Message schema ----------------------------------------------------------
# Typical message structure:
#   {"role": "user" | "assistant" | "system" | "tool", "content": str | [ContentPart, ...]}
# where content parts can be text or (optionally) structured pieces like images.

class ContentPart(TypedDict, total=False):
    type: str           # e.g., "text", "image_url", etc.
    text: str           # present when type == "text"
    image_url: Dict[str, str]  # present when type == "image_url"
    input_audio: Dict[str, str]  # present when type == "input_audio"
    bucket_id: str
    name: str

class Message(TypedDict):
    role: str
    content: Union[str, List[ContentPart]]

Messages = List[Message]

# ---- Media inputs ------------------------------------------------------------

# Paths, raw bytes, file-like objects, or PIL Image objects are accepted.
ImageType = Union[str, bytes, IO[bytes], PILImage, os.PathLike]
MediaListType = List[Tuple[ImageType, Optional[str]]]

__all__ = [
    "Any",
    "AsyncGenerator",
    "Generator",
    "AsyncIterator",
    "Iterator",
    "Tuple",
    "Union",
    "List",
    "Dict",
    "Type",
    "IO",
    "Optional",
    "TypedDict",
    "SHA256",
    "CreateResult",
    "AsyncResult",
    "Messages",
    "Message",
    "ContentPart",
    "Cookies",
    "Image",
    "ImageType",
    "MediaListType",
    "ResponseType",
]