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
from ..requests.aiohttp import get_connector
from ..Provider.template import BackendApi
from . import is_accepted_format, extract_data_uri
from .. import debug

# Directory for storing generated images
images_dir = "./generated_images"

def get_image_extension(image: str) -> str:
    """Extract image extension from URL or filename, default to .jpg"""
    match = re.search(r"\.(jpe?g|png|webp)$", image, re.IGNORECASE)
    return f".{match.group(1).lower()}" if match else ".jpg"

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

async def copy_images(
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
            target_path = target
            if target_path is None:
                # Generate filename components
                file_hash = hashlib.sha256(image.encode()).hexdigest()[:16]
                timestamp = int(time.time())
                
                # Sanitize alt text for filename (Unicode-safe)
                if alt:
                    # Keep letters, numbers, basic punctuation and all Unicode chars
                    clean_alt = re.sub(
                        r'[^\w\s.-]',  # Allow all Unicode word chars
                        '_', 
                        unquote(alt).strip(), 
                        flags=re.UNICODE
                    )
                    clean_alt = re.sub(r'[\s_]+', '_', clean_alt)[:100]
                else:
                    clean_alt = "image"

                # Build safe filename with full Unicode support
                extension = get_image_extension(image)
                filename = (
                    f"{timestamp}_"
                    f"{clean_alt}_"
                    f"{file_hash}"
                    f"{extension}"
                )
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
                if not os.path.splitext(target_path)[1]:
                    with open(target_path, "rb") as f:
                        file_header = f.read(12)
                    detected_type = is_accepted_format(file_header)
                    if detected_type:
                        new_ext = f".{detected_type.split('/')[-1]}"
                        os.rename(target_path, f"{target_path}{new_ext}")
                        target_path = f"{target_path}{new_ext}"

                # Build URL with safe encoding
                url_filename = quote(os.path.basename(target_path))
                return f"/images/{url_filename}{'?url=' + quote(image) if add_url and not image.startswith('data:') else ''}"

            except (ClientError, IOError, OSError) as e:
                debug.error(f"Image copying failed: {type(e).__name__}: {e}")
                if target_path and os.path.exists(target_path):
                    os.unlink(target_path)
                return get_source_url(image, image)

        return await asyncio.gather(*[copy_image(img, target) for img in images])
