from __future__ import annotations

import os
import time
import asyncio
import hashlib
import base64
from datetime import datetime
from typing import AsyncIterator
from urllib.parse import quote
from aiohttp import ClientSession, ClientError, ClientTimeout
from urllib.parse import urlparse

from ..typing import Optional, Cookies, Union
from ..requests.aiohttp import get_connector
from ..image import MEDIA_TYPE_MAP, EXTENSIONS_MAP, is_safe_url
from ..tools.files import secure_filename
from ..providers.response import ImageResponse, AudioResponse, VideoResponse, quote_url
from . import is_accepted_format, extract_data_uri
from .. import debug

# Directory for storing generated media files
images_dir = "./generated_images"
media_dir  = "./generated_media"


def get_media_dir() -> str:
    """Get the directory for storing generated media files."""
    if os.access(images_dir, os.R_OK):
        return images_dir
    return media_dir


def get_media_extension(media: str) -> str:
    """Extract media file extension from URL or filename."""
    path = urlparse(media).path
    extension = os.path.splitext(path)[1]
    if not extension and media:
        extension = os.path.splitext(media)[1]
    if not extension or len(extension) > 5:
        return ""
    if extension[1:] not in EXTENSIONS_MAP:
        return ""   # FIX: was `raise ""` — that's a TypeError in Python 3
    return extension


def ensure_media_dir():
    """Create images/media directory if it doesn't exist."""
    if not os.access(images_dir, os.R_OK):
        os.makedirs(media_dir, exist_ok=True)


def get_source_url(image: str, default: str = None) -> str:
    """Extract original URL from image parameter if present."""
    if "url=" in image:
        decoded_url = quote_url(image.split("url=", 1)[1])
        if decoded_url.startswith(("http://", "https://")):
            return decoded_url
    return default


def update_filename(response, filename: str) -> str:
    date = response.headers.get("last-modified", response.headers.get("date"))
    timestamp = datetime.strptime(date, '%a, %d %b %Y %H:%M:%S %Z').timestamp()
    return str(int(timestamp)) + "_" + filename.split("_", maxsplit=1)[-1]


async def save_response_media(
    response,
    prompt: str,
    tags: list[str] = None,
    transcript: str = None,
    content_type: str = None
) -> AsyncIterator:
    """Save media from response to local file and return a response object."""
    if isinstance(response, dict):
        content_type = response.get("mimeType", content_type or "audio/mpeg")
        transcript   = response.get("transcript")
        response     = response.get("data")
    elif hasattr(response, "headers"):
        content_type = response.headers.get("content-type", content_type)
    elif not content_type:
        raise ValueError("Response must be a dict or have headers")

    if isinstance(response, str):
        response = base64.b64decode(response)

    extension = MEDIA_TYPE_MAP.get(content_type)
    if extension is None:
        raise ValueError(f"Unsupported media type: {content_type}")

    filename = get_filename(tags, prompt, f".{extension}", prompt)
    if hasattr(response, "headers"):
        filename = update_filename(response, filename)
    target_path = os.path.join(get_media_dir(), filename)
    ensure_media_dir()

    with open(target_path, 'wb') as f:
        if isinstance(response, bytes):
            f.write(response)
        else:
            if hasattr(response, "iter_content"):
                iter_response = response.iter_content()
            else:
                iter_response = response.content.iter_any()
            async for chunk in iter_response:
                f.write(chunk)

    media_url = f"/media/{filename}"

    # FIX: guard against responses that have no .method attribute
    source_url = None
    if hasattr(response, "url") and getattr(response, "method", None) == "GET":
        source_url = str(response.url)

    if content_type.startswith("audio/"):
        yield AudioResponse(media_url, transcript, source_url=source_url)
    elif content_type.startswith("video/"):
        yield VideoResponse(media_url, prompt, source_url=source_url)
    else:
        yield ImageResponse(media_url, prompt, source_url=source_url)


def get_filename(tags: list[str], alt: str, extension: str, image: str) -> str:
    tags_str = f"{'+'.join([str(tag) for tag in tags if tag])}+" if tags else ""
    return "".join((
        f"{int(time.time())}_",
        f"{secure_filename(tags_str + alt)}_" if alt else secure_filename(tags_str),
        hashlib.sha256(
            str(time.time()).encode() if image is None else image.encode()
        ).hexdigest()[:16],
        extension
    ))


async def copy_media(
    images: list[str],
    cookies: Optional[Cookies] = None,
    headers: Optional[dict] = None,
    proxy: Optional[str] = None,
    alt: str = None,
    tags: list[str] = None,
    add_url: Union[bool, str] = True,
    target: str = None,
    thumbnail: bool = False,
    ssl: bool = None,
    timeout: Optional[int] = None,
    return_target: bool = False
) -> list[str]:
    """
    Download and store images/videos locally with Unicode-safe filenames.
    Returns list of relative media URLs.
    """
    ensure_media_dir()
    dest_dir = get_media_dir()
    if thumbnail:
        dest_dir = os.path.join(dest_dir, "thumbnails")
        os.makedirs(dest_dir, exist_ok=True)
    if headers is not None or cookies is not None:
        add_url = False  # Do not embed source URL if auth headers/cookies were used

    async with ClientSession(
        connector=get_connector(proxy=proxy),
        cookies=cookies,
        headers=headers,
        timeout=ClientTimeout(total=timeout) if timeout else None,
    ) as session:

        async def copy_image(image: str, target: str = None):
            """Process individual image/video and return its local URL."""
            if image is None or image.startswith("/"):
                return image

            target_path    = target
            media_extension = ""

            if target_path is None:
                media_extension = get_media_extension(image)
                path = urlparse(image).path
                if path.startswith("/media/"):
                    filename = secure_filename(path[len("/media/"):])
                else:
                    filename = get_filename(tags, alt, media_extension, image)
                target_path = os.path.join(dest_dir, filename)

            try:
                if image.startswith("data:"):
                    with open(target_path, "wb") as f:
                        f.write(extract_data_uri(image))

                elif not os.path.exists(target_path) or os.lstat(target_path).st_size <= 0:
                    if not is_safe_url(image):
                        raise ValueError(f"Invalid or unsafe image url: {image}")
                    async with session.get(image, ssl=ssl) as response:
                        response.raise_for_status()
                        if target is None:
                            filename    = update_filename(response, filename)
                            target_path = os.path.join(dest_dir, filename)
                        media_type = response.headers.get(
                            "content-type", "application/octet-stream"
                        )
                        if media_type not in (
                            "application/octet-stream",
                            "binary/octet-stream",
                        ):
                            if media_type not in MEDIA_TYPE_MAP:
                                raise ValueError(f"Unsupported media type: {media_type}")
                            if target is None and not media_extension:
                                media_extension = f".{MEDIA_TYPE_MAP[media_type]}"
                                target_path     = f"{target_path}{media_extension}"
                        with open(target_path, "wb") as f:
                            async for chunk in response.content.iter_any():
                                f.write(chunk)

                # Auto-detect extension from file magic if still unknown
                if target is None and not media_extension:
                    with open(target_path, "rb") as f:
                        file_header = f.read(12)
                    try:
                        detected_type   = is_accepted_format(file_header)
                        media_extension = f".{detected_type.split('/')[-1]}"
                        media_extension = media_extension.replace("jpeg", "jpg")
                        new_path        = f"{target_path}{media_extension}"
                        os.rename(target_path, new_path)
                        target_path = new_path
                    except ValueError:
                        pass

                if thumbnail:
                    uri = "/thumbnail/" + os.path.basename(target_path)
                else:
                    # FIX: removed pointless `'' +` prefix
                    if add_url and not image.startswith("data:"):
                        url_suffix = (
                            add_url if isinstance(add_url, str)
                            else "url=" + quote(image)
                        )
                        uri = f"/media/{os.path.basename(target_path)}?{url_suffix}"
                    else:
                        uri = f"/media/{os.path.basename(target_path)}"

                if return_target:
                    return uri, target_path
                return uri

            except (ClientError, IOError, OSError, ValueError) as e:
                debug.error("Image copying failed:", e)
                if target_path and os.path.exists(target_path):
                    os.unlink(target_path)
                return image

        return await asyncio.gather(*[copy_image(image, target) for image in images])