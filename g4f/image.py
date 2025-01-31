from __future__ import annotations

import os
import re
import io
import time
import uuid
import base64
import asyncio
import hashlib
from urllib.parse import quote_plus
from io import BytesIO
from pathlib import Path
from aiohttp import ClientSession, ClientError
try:
    from PIL.Image import open as open_image, new as new_image
    from PIL.Image import FLIP_LEFT_RIGHT, ROTATE_180, ROTATE_270, ROTATE_90
    has_requirements = True
except ImportError:
    has_requirements = False

from .typing import ImageType, Union, Image, Optional, Cookies
from .errors import MissingRequirementsError
from .providers.response import ImageResponse, ImagePreview
from .requests.aiohttp import get_connector
from . import debug

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'}

EXTENSIONS_MAP: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}

# Define the directory for generated images
images_dir = "./generated_images"

def to_image(image: ImageType, is_svg: bool = False) -> Image:
    """
    Converts the input image to a PIL Image object.

    Args:
        image (Union[str, bytes, Image]): The input image.

    Returns:
        Image: The converted PIL Image object.
    """
    if not has_requirements:
        raise MissingRequirementsError('Install "pillow" package for images')

    if isinstance(image, str) and image.startswith("data:"):
        is_data_uri_an_image(image)
        image = extract_data_uri(image)

    if is_svg:
        try:
            import cairosvg
        except ImportError:
            raise MissingRequirementsError('Install "cairosvg" package for svg images')
        if not isinstance(image, bytes):
            image = image.read()
        buffer = BytesIO()
        cairosvg.svg2png(image, write_to=buffer)
        return open_image(buffer)

    if isinstance(image, bytes):
        is_accepted_format(image)
        return open_image(BytesIO(image))
    elif not isinstance(image, Image):
        image = open_image(image)
        image.load()
        return image

    return image

def is_allowed_extension(filename: str) -> bool:
    """
    Checks if the given filename has an allowed extension.

    Args:
        filename (str): The filename to check.

    Returns:
        bool: True if the extension is allowed, False otherwise.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_data_uri_an_image(data_uri: str) -> bool:
    """
    Checks if the given data URI represents an image.

    Args:
        data_uri (str): The data URI to check.

    Raises:
        ValueError: If the data URI is invalid or the image format is not allowed.
    """
    # Check if the data URI starts with 'data:image' and contains an image format (e.g., jpeg, png, gif)
    if not re.match(r'data:image/(\w+);base64,', data_uri):
        raise ValueError("Invalid data URI image.")
    # Extract the image format from the data URI
    image_format = re.match(r'data:image/(\w+);base64,', data_uri).group(1).lower()
    # Check if the image format is one of the allowed formats (jpg, jpeg, png, gif)
    if image_format not in ALLOWED_EXTENSIONS and image_format != "svg+xml":
        raise ValueError("Invalid image format (from mime file type).")

def is_accepted_format(binary_data: bytes) -> str:
    """
    Checks if the given binary data represents an image with an accepted format.

    Args:
        binary_data (bytes): The binary data to check.

    Raises:
        ValueError: If the image format is not allowed.
    """
    if binary_data.startswith(b'\xFF\xD8\xFF'):
        return "image/jpeg"
    elif binary_data.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png"
    elif binary_data.startswith(b'GIF87a') or binary_data.startswith(b'GIF89a'):
        return "image/gif"
    elif binary_data.startswith(b'\x89JFIF') or binary_data.startswith(b'JFIF\x00'):
        return "image/jpeg"
    elif binary_data.startswith(b'\xFF\xD8'):
        return "image/jpeg"
    elif binary_data.startswith(b'RIFF') and binary_data[8:12] == b'WEBP':
        return "image/webp"
    else:
        raise ValueError("Invalid image format (from magic code).")

def extract_data_uri(data_uri: str) -> bytes:
    """
    Extracts the binary data from the given data URI.

    Args:
        data_uri (str): The data URI.

    Returns:
        bytes: The extracted binary data.
    """
    data = data_uri.split(",")[-1]
    data = base64.b64decode(data)
    return data

def get_orientation(image: Image) -> int:
    """
    Gets the orientation of the given image.

    Args:
        image (Image): The image.

    Returns:
        int: The orientation value.
    """
    exif_data = image.getexif() if hasattr(image, 'getexif') else image._getexif()
    if exif_data is not None:
        orientation = exif_data.get(274) # 274 corresponds to the orientation tag in EXIF
        if orientation is not None:
            return orientation

def process_image(image: Image, new_width: int, new_height: int) -> Image:
    """
    Processes the given image by adjusting its orientation and resizing it.

    Args:
        image (Image): The image to process.
        new_width (int): The new width of the image.
        new_height (int): The new height of the image.

    Returns:
        Image: The processed image.
    """
    # Fix orientation
    orientation = get_orientation(image)
    if orientation:
        if orientation > 4:
            image = image.transpose(FLIP_LEFT_RIGHT)
        if orientation in [3, 4]:
            image = image.transpose(ROTATE_180)
        if orientation in [5, 6]:
            image = image.transpose(ROTATE_270)
        if orientation in [7, 8]:
            image = image.transpose(ROTATE_90)
    # Resize image
    image.thumbnail((new_width, new_height))
    # Remove transparency
    if image.mode == "RGBA":
        image.load()
        white = new_image('RGB', image.size, (255, 255, 255))
        white.paste(image, mask=image.split()[-1])
        return white
    # Convert to RGB for jpg format
    elif image.mode != "RGB":
        image = image.convert("RGB")
    return image


def to_bytes(image: ImageType) -> bytes:
    """
    Converts the given image to bytes.

    Args:
        image (ImageType): The image to convert.

    Returns:
        bytes: The image as bytes.
    """
    if isinstance(image, bytes):
        return image
    elif isinstance(image, str) and image.startswith("data:"):
        is_data_uri_an_image(image)
        return extract_data_uri(image)
    elif isinstance(image, Image):
        bytes_io = BytesIO()
        image.save(bytes_io, image.format)
        image.seek(0)
        return bytes_io.getvalue()
    elif isinstance(image, (str, os.PathLike)):
        return Path(image).read_bytes()
    elif isinstance(image, Path):
        return image.read_bytes()
    else:
        try:
            image.seek(0)
        except (AttributeError, io.UnsupportedOperation):
            pass
        return image.read()

def to_data_uri(image: ImageType) -> str:
    if not isinstance(image, str):
        data = to_bytes(image)
        data_base64 = base64.b64encode(data).decode()
        return f"data:{is_accepted_format(data)};base64,{data_base64}"
    return image

# Function to ensure the images directory exists
def ensure_images_dir():
    os.makedirs(images_dir, exist_ok=True)

def get_image_extension(image: str) -> str:
    if match := re.search(r"(\.(?:jpe?g|png|webp))[$?&]", image):
        return match.group(1)
    return ".jpg"

async def copy_images(
    images: list[str],
    cookies: Optional[Cookies] = None,
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
        cookies=cookies
    ) as session:
        async def copy_image(image: str, target: str = None) -> str:
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
                        async with session.get(image, ssl=ssl) as response:
                            response.raise_for_status()
                            with open(target, "wb") as f:
                                async for chunk in response.content.iter_chunked(4096):
                                    f.write(chunk)
                    except ClientError as e:
                        debug.log(f"copy_images failed: {e.__class__.__name__}: {e}")
                        return image
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

class ImageDataResponse():
    def __init__(
        self,
        images: Union[str, list],
        alt: str,
    ):
        self.images = images
        self.alt = alt

    def get_list(self) -> list[str]:
        return [self.images] if isinstance(self.images, str) else self.images

class ImageRequest:
    def __init__(
        self,
        options: dict = {}
    ):
        self.options = options

    def get(self, key: str):
        return self.options.get(key)
