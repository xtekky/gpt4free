from __future__ import annotations

import os
import time
import uuid
import asyncio
import hashlib
import re
from urllib.parse import quote, unquote
from aiohttp import ClientSession, ClientError

from ..typing import Optional, Cookies
from ..requests.aiohttp import get_connector, StreamResponse
from ..image import EXTENSIONS_MAP, ALLOWED_EXTENSIONS
from ..tools.files import get_bucket_dir
from ..providers.response import ImageResponse, AudioResponse, VideoResponse
from ..Provider.template import BackendApi
from . import is_accepted_format, extract_data_uri
from .. import debug

# Directory for storing generated images
images_dir = "./generated_images"

def get_media_extension(media: str) -> str:
    """Extract media file extension from URL or filename"""
    match = re.search(r"\.(jpe?g|png|gif|svg|webp|webm|mp4|mp3|wav|flac|opus|ogg|mkv)(?:\?|$)", media, re.IGNORECASE)
    return f".{match.group(1).lower()}" if match else ""

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

def secure_filename(filename: str) -> str:
    if filename is None:
        return None
    # Keep letters, numbers, basic punctuation and all Unicode chars
    filename = re.sub(
        r'[^\w.,_-]+',
        '_', 
        unquote(filename).strip(), 
        flags=re.UNICODE
    )
    filename = filename[:100].strip(".,_-")
    return filename

async def save_response_media(response: StreamResponse, prompt: str):
    content_type = response.headers["content-type"]
    if content_type in EXTENSIONS_MAP or content_type.startswith("audio/") or content_type.startswith("video/"):
        extension = EXTENSIONS_MAP[content_type] if content_type in EXTENSIONS_MAP else content_type[6:].replace("mpeg", "mp3")
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported media type: {content_type}")
        bucket_id = str(uuid.uuid4())
        dirname = str(int(time.time()))
        bucket_dir = get_bucket_dir(bucket_id, dirname)
        media_dir = os.path.join(bucket_dir, "media")
        os.makedirs(media_dir, exist_ok=True)
        filename = secure_filename(f"{content_type[0:5] if prompt is None else prompt}.{extension}")
        newfile = os.path.join(media_dir, filename)
        with open(newfile, 'wb') as f:
            async for chunk in response.iter_content() if hasattr(response, "iter_content") else response.content.iter_any():
                f.write(chunk)
        media_url = f"/files/{dirname}/{bucket_id}/media/{filename}"
        if response.method == "GET":
            media_url = f"{media_url}?url={str(response.url)}"
        if content_type.startswith("audio/"):
            yield AudioResponse(media_url)
        elif content_type.startswith("video/"):
            yield VideoResponse(media_url, prompt)
        else:
            yield ImageResponse(media_url, prompt)

async def copy_media(
    images: list[str],
    cookies: Optional[Cookies] = None,
    headers: Optional[dict] = None,
    proxy: Optional[str] = None,
    alt: str = None,
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
                filename = secure_filename("".join((
                    f"{int(time.time())}_",
                    (f"{alt}_" if alt else ""),
                    f"{hashlib.sha256(image.encode()).hexdigest()[:16]}",
                    f"{get_media_extension(image)}"
                )))
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
                        with open(target_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(4096):
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
                return f"/images/{url_filename}" + (('?url=' + quote(image)) if add_url and not image.startswith('data:') else '')

            except (ClientError, IOError, OSError) as e:
                debug.error(f"Image copying failed: {type(e).__name__}: {e}")
                if target_path and os.path.exists(target_path):
                    os.unlink(target_path)
                return get_source_url(image, image)

        return await asyncio.gather(*[copy_image(img, target) for img in images])
