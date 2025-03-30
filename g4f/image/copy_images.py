from __future__ import annotations

import os
import time
import asyncio
import hashlib
import re
from typing import AsyncIterator
from urllib.parse import quote, unquote
from aiohttp import ClientSession, ClientError
from urllib.parse import urlparse

from ..typing import Optional, Cookies
from ..requests.aiohttp import get_connector, StreamResponse
from ..image import MEDIA_TYPE_MAP, EXTENSIONS_MAP
from ..tools.files import secure_filename
from ..providers.response import ImageResponse, AudioResponse, VideoResponse
from ..Provider.template import BackendApi
from . import is_accepted_format, extract_data_uri
from .. import debug

# Directory for storing generated images
images_dir = "./generated_images"

def get_media_extension(media: str) -> str:
    """Extract media file extension from URL or filename"""
    path = urlparse(media).path
    extension = os.path.splitext(path)[1]
    if not extension:
        extension = os.path.splitext(media)[1]
    if not extension:
        return ""
    if extension[1:] not in EXTENSIONS_MAP:
        raise ValueError(f"Unsupported media extension: {extension} in: {media}")
    return extension

def ensure_images_dir():
    """Create images directory if it doesn't exist"""
    os.makedirs(images_dir, exist_ok=True)

def get_source_url(image: str, default: str = None) -> str:
    """Extract original URL from image parameter if present"""
    if "url=" in image:
        decoded_url = unquote(image.split("url=", 1)[1])
        if decoded_url.startswith(("http://", "https://")):
            return decoded_url
    return default

def is_valid_media_type(content_type: str) -> bool:
    return content_type in MEDIA_TYPE_MAP or content_type.startswith("audio/") or content_type.startswith("video/")

async def save_response_media(response: StreamResponse, prompt: str, tags: list[str]) -> AsyncIterator:
    """Save media from response to local file and return URL"""
    content_type = response.headers["content-type"]
    if is_valid_media_type(content_type):
        extension = MEDIA_TYPE_MAP[content_type] if content_type in MEDIA_TYPE_MAP else content_type[6:].replace("mpeg", "mp3")
        if extension not in EXTENSIONS_MAP:
            raise ValueError(f"Unsupported media type: {content_type}")
        filename = get_filename(tags, prompt, f".{extension}", prompt)
        target_path = os.path.join(images_dir, filename)
        with open(target_path, 'wb') as f:
            async for chunk in response.iter_content() if hasattr(response, "iter_content") else response.content.iter_any():
                f.write(chunk)
        media_url = f"/media/{filename}"
        if response.method == "GET":
            media_url = f"{media_url}?url={str(response.url)}"
        if content_type.startswith("audio/"):
            yield AudioResponse(media_url)
        elif content_type.startswith("video/"):
            yield VideoResponse(media_url, prompt)
        else:
            yield ImageResponse(media_url, prompt)

def get_filename(tags: list[str], alt: str, extension: str, image: str) -> str:
    return "".join((
        f"{int(time.time())}_",
        f"{secure_filename('+'.join([tag for tag in tags if tag]))}+" if tags else "",
        f"{secure_filename(alt)}_",
        hashlib.sha256(image.encode()).hexdigest()[:16],
        extension
    ))

async def copy_media(
    images: list[str],
    cookies: Optional[Cookies] = None,
    headers: Optional[dict] = None,
    proxy: Optional[str] = None,
    alt: str = None,
    tags: list[str] = None,
    add_url: bool = True,
    target: str = None,
    ssl: bool = None
) -> list[str]:
    """
    Download and store images locally with Unicode-safe filenames
    Returns list of relative image URLs
    """
    if add_url:
        add_url = not cookies
    ensure_images_dir()

    async with ClientSession(
        connector=get_connector(proxy=proxy),
        cookies=cookies,
        headers=headers,
    ) as session:
        async def copy_image(image: str, target: str = None) -> str:
            """Process individual image and return its local URL"""
            # Skip if image is already local
            if image.startswith("/"):
                return image
            target_path = target
            if target_path is None:
                # Build safe filename with full Unicode support
                filename = get_filename(tags, alt, get_media_extension(image), image)
                target_path = os.path.join(images_dir, filename)
            try:
                # Handle different image types
                if image.startswith("data:"):
                    with open(target_path, "wb") as f:
                        f.write(extract_data_uri(image))
                else:
                    # Apply BackendApi settings if needed
                    if BackendApi.working and image.startswith(BackendApi.url):
                        request_headers = BackendApi.headers if headers is None else headers
                        request_ssl = BackendApi.ssl
                    else:
                        request_headers = headers
                        request_ssl = ssl

                    async with session.get(image, ssl=request_ssl, headers=request_headers) as response:
                        response.raise_for_status()
                        media_type = response.headers.get("content-type", "application/octet-stream")
                        if media_type not in ("application/octet-stream", "binary/octet-stream"):
                            if not is_valid_media_type(media_type):
                                raise ValueError(f"Unsupported media type: {media_type}")
                        with open(target_path, "wb") as f:
                            async for chunk in response.content.iter_any():
                                f.write(chunk)

                # Verify file format
                if target is None and not os.path.splitext(target_path)[1]:
                    with open(target_path, "rb") as f:
                        file_header = f.read(12)
                    try:
                        detected_type = is_accepted_format(file_header)
                        if detected_type:
                            new_ext = f".{detected_type.split('/')[-1]}"
                            os.rename(target_path, f"{target_path}{new_ext}")
                            target_path = f"{target_path}{new_ext}"
                    except ValueError:
                        pass

                # Build URL with safe encoding
                url_filename = quote(os.path.basename(target_path))
                return f"/media/{url_filename}" + (('?url=' + quote(image)) if add_url and not image.startswith('data:') else '')

            except (ClientError, IOError, OSError, ValueError) as e:
                debug.error(f"Image copying failed: {type(e).__name__}: {e}")
                if target_path and os.path.exists(target_path):
                    os.unlink(target_path)
                return get_source_url(image, image)

        return await asyncio.gather(*[copy_image(img, target) for img in images])
