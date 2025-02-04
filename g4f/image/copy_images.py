from __future__ import annotations

import os
import time
import uuid
import asyncio
import hashlib
import re
from urllib.parse import quote_plus
from aiohttp import ClientSession, ClientError

from ..typing import Optional, Cookies
from ..requests.aiohttp import get_connector
from ..Provider.template import BackendApi
from . import is_accepted_format, extract_data_uri
from .. import debug

# Define the directory for generated images
images_dir = "./generated_images"

def get_image_extension(image: str) -> str:
    match = re.search(r"\.(?:jpe?g|png|webp)", image)
    if match:
        return match.group(0)
    return ".jpg"

# Function to ensure the images directory exists
def ensure_images_dir():
    os.makedirs(images_dir, exist_ok=True)

def get_source_url(image: str, default: str = None) -> str:
    source_url = image.split("url=", 1)
    if len(source_url) > 1:
        source_url = source_url[1]
        source_url = source_url.replace("%2F", "/").replace("%3A", ":").replace("%3F", "?").replace("%3D", "=")
        if source_url.startswith("https://"):
            return source_url
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
    if add_url:
        add_url = not cookies
    ensure_images_dir()
    async with ClientSession(
        connector=get_connector(proxy=proxy),
        cookies=cookies,
        headers=headers,
    ) as session:
        async def copy_image(image: str, target: str = None, headers: dict = headers, ssl: bool = ssl) -> str:
            if target is None or len(images) > 1:
                hash = hashlib.sha256(image.encode()).hexdigest()
                target = f"{quote_plus('+'.join(alt.split()[:10])[:100], '')}_{hash}" if alt else str(uuid.uuid4())
                target = f"{int(time.time())}_{target}{get_image_extension(image)}"
                target = os.path.join(images_dir, target)
            try:
                if image.startswith("data:"):
                    with open(target, "wb") as f:
                        f.write(extract_data_uri(image))
                else:
                    try:
                        if BackendApi.working and image.startswith(BackendApi.url) and headers is None:
                            headers = BackendApi.headers
                            ssl = BackendApi.ssl
                        async with session.get(image, ssl=ssl, headers=headers) as response:
                            response.raise_for_status()
                            with open(target, "wb") as f:
                                async for chunk in response.content.iter_chunked(4096):
                                    f.write(chunk)
                    except ClientError as e:
                        debug.log(f"copy_images failed: {e.__class__.__name__}: {e}")
                        return get_source_url(image, image)
                if "." not in target:
                    with open(target, "rb") as f:
                        extension = is_accepted_format(f.read(12)).split("/")[-1]
                        extension = "jpg" if extension == "jpeg" else extension
                        new_target = f"{target}.{extension}"
                        os.rename(target, new_target)
                        target = new_target
            finally:
                if "." not in target and os.path.exists(target):
                    os.unlink(target)
            return f"/images/{os.path.basename(target)}{'?url=' + image if add_url and not image.startswith('data:') else ''}"

        return await asyncio.gather(*[copy_image(image, target) for image in images])
