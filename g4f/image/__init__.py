from __future__ import annotations

import os
import re
import io
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

try:
    from PIL import Image, ImageOps
    has_requirements = True
except ImportError:
    has_requirements = False

from ..typing import ImageType
from ..errors import MissingRequirementsError
from ..files import get_bucket_dir

EXTENSIONS_MAP: dict[str, str] = {
    # Image
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    # Audio
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "ogg": "audio/ogg",
    "m4a": "audio/m4a",
     # Video
    "mkv": "video/x-matroska",
    "webm": "video/webm",
    "mp4": "video/mp4",
}

MEDIA_TYPE_MAP: dict[str, str] = {value: key for key, value in EXTENSIONS_MAP.items()}
MEDIA_TYPE_MAP["audio/webm"] = "webm"

def to_image(image: ImageType, is_svg: bool = False) -> Image.Image:
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
        return Image.open(buffer)

    if isinstance(image, bytes):
        is_accepted_format(image)
        return Image.open(BytesIO(image))
    elif not isinstance(image, Image.Image):
        image = Image.open(image)
        image.load()
        return image

    return image

def get_extension(filename: str) -> Optional[str]:
    if '.' in filename:
        ext = os.path.splitext(filename)[1].lower().lstrip('.')
        return ext if ext in EXTENSIONS_MAP else None
    return None

def is_allowed_extension(filename: str) -> Optional[str]:
    """
    Checks if the given filename has an allowed extension.

    Args:
        filename (str): The filename to check.

    Returns:
        bool: True if the extension is allowed, False otherwise.
    """
    extension = get_extension(filename)
    if extension is None:
        return None
    return EXTENSIONS_MAP[extension]

def is_data_an_media(data, filename: str = None) -> str:
    content_type = is_data_an_audio(data, filename)
    if content_type is not None:
        return content_type
    if isinstance(data, bytes):
        return is_accepted_format(data)
    if isinstance(data, str) and data.startswith(("http://", "https://")):
        path = urlparse(data).path
        extension = get_extension(path)
        if extension is not None:
            return EXTENSIONS_MAP[extension]
        return "binary/octet-stream"
    return is_data_uri_an_image(data)

def is_valid_media(data: ImageType = None, filename: str = None) -> str:
    if is_valid_audio(data, filename):
        return True
    if filename:
        extension = get_extension(filename)
        if extension is not None:
            media_type = EXTENSIONS_MAP[extension]
            if media_type.startswith("image/"):
                return media_type
    if not data:
        return False
    if isinstance(data, bytes):
        return is_accepted_format(data)
    return is_data_uri_an_image(data)

def is_data_an_audio(data_uri: str = None, filename: str = None) -> str:
    if filename:
        extension = get_extension(filename)
        if extension is not None:
            media_type = EXTENSIONS_MAP[extension]
            if media_type.startswith("audio/"):
                return media_type
    if isinstance(data_uri, str):
        audio_format = re.match(r'^data:(audio/\w+);base64,', data_uri)
        if audio_format:
            return audio_format.group(1)

def is_valid_audio(data_uri: str = None, filename: str = None) -> bool:
    mimetype = is_data_an_audio(data_uri, filename)
    if mimetype is None:
        return False
    if MEDIA_TYPE_MAP.get(mimetype) not in ("wav", "mp3"):
        return False
    return True

def is_data_uri_an_image(data_uri: str) -> bool:
    """
    Checks if the given data URI represents an image.

    Args:
        data_uri (str): The data URI to check.

    Raises:
        ValueError: If the data URI is invalid or the image format is not allowed.
    """
    if data_uri.startswith("https:") or data_uri.startswith("http:"):
        return True
    # Check if the data URI starts with 'data:image' and contains an image format (e.g., jpeg, png, gif)
    if not re.match(r'data:image/(\w+);base64,', data_uri):
        raise ValueError(f"Invalid data URI image. {data_uri[:10]}...")
    # Extract the image format from the data URI
    image_format = re.match(r'data:image/(\w+);base64,', data_uri).group(1).lower()
    # Check if the image format is one of the allowed formats (jpg, jpeg, png, gif)
    if image_format not in EXTENSIONS_MAP and image_format != "svg+xml":
        raise ValueError("Invalid image format (from mime file type).")
    return True

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



def detect_file_type(binary_data: bytes) -> tuple[str, str] | None:
    """
    Detect file type from magic number / header signature.

    Args:
        binary_data (bytes): File binary data

    Returns:
        tuple: (extension, MIME type)

    Raises:
        ValueError: If file type is unknown
    """

    # ---- Images ----
    if binary_data.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    elif binary_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    elif binary_data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif", "image/gif"
    elif binary_data.startswith(b"RIFF") and binary_data[8:12] == b"WEBP":
        return ".webp", "image/webp"
    elif binary_data.startswith(b"BM"):
        return ".bmp", "image/bmp"
    elif binary_data.startswith(b"II*\x00") or binary_data.startswith(b"MM\x00*"):
        return ".tiff", "image/tiff"
    elif binary_data.startswith(b"\x00\x00\x01\x00"):
        return ".ico", "image/x-icon"
    elif binary_data.startswith(b"\x00\x00\x00\x0cjP  \x0d\x0a\x87\x0a"):
        return ".jp2", "image/jp2"
    elif len(binary_data) > 12 and binary_data[4:8] == b"ftyp":
        brand = binary_data[8:12]
        if brand in [b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"]:
            return ".heic", "image/heif"
        elif brand in [b"avif"]:
            return ".avif", "image/avif"
    elif binary_data.lstrip().startswith((b"<?xml", b"<svg")):
        return ".svg", "image/svg+xml"

    # ---- Documents ----
    elif binary_data.startswith(b"%PDF"):
        return ".pdf", "application/pdf"
    elif binary_data.startswith(b"PK\x03\x04"):
        return ".zip", "application/zip-based"
          # could be docx/xlsx/pptx/jar/apk/odt
    elif binary_data.startswith(b"\xd0\xcf\x11\xe0"):
        return ".doc", "application/vnd.ms-office"
    elif binary_data.startswith(b"{\\rtf"):
        return ".rtf", "application/rtf"
    elif binary_data.startswith(b"7z\xbc\xaf\x27\x1c"):
        return ".7z", "application/x-7z-compressed"
    elif binary_data.startswith(b"Rar!\x1a\x07\x00"):
        return ".rar", "application/vnd.rar"
    elif binary_data.startswith(b"\x1f\x8b"):
        return ".gz", "application/gzip"
    elif binary_data.startswith(b"BZh"):
        return ".bz2", "application/x-bzip2"
    elif binary_data.startswith(b"\xfd7zXZ\x00"):
        return ".xz", "application/x-xz"

    # ---- Executables / Libraries ----
    elif binary_data.startswith(b"MZ"):
        return ".exe", "application/x-msdownload"
    elif binary_data.startswith(b"\x7fELF"):
        return ".elf", "application/x-elf"
    elif binary_data.startswith(b"\xca\xfe\xba\xbe") or binary_data.startswith(
            b"\xca\xfe\xd0\x0d"
    ):
        return ".class", "application/java-vm"
    elif (
            binary_data.startswith(b"\x50\x4b\x03\x04")
            and b"META-INF" in binary_data[:200]
    ):
        return ".jar", "application/java-archive"

    # ---- Audio ----
    elif binary_data.startswith(b"ID3") or binary_data[0:2] == b"\xff\xfb":
        return ".mp3", "audio/mpeg"
    elif binary_data.startswith(b"OggS"):
        return ".ogg", "audio/ogg"
    elif binary_data.startswith(b"fLaC"):
        return ".flac", "audio/flac"
    elif binary_data.startswith(b"RIFF") and binary_data[8:12] == b"WAVE":
        return ".wav", "audio/wav"
    elif binary_data.startswith(b"MThd"):
        return ".mid", "audio/midi"

    # ---- Video ----
    elif binary_data.startswith(b"\x00\x00\x00") and b"ftyp" in binary_data[4:12]:
        return ".mp4", "video/mp4"
    elif binary_data.startswith(b"RIFF") and binary_data[8:12] == b"AVI ":
        return ".avi", "video/x-msvideo"
    elif binary_data.startswith(b"OggS"):
        return ".ogv", "video/ogg"
    elif binary_data.startswith(b"\x1a\x45\xdf\xa3"):
        return ".mkv", "video/webm"
    elif binary_data.startswith(b"\x00\x00\x01\xba"):
        return ".mpg", "video/mpeg"

    # ---- Text / Scripts ----
    elif binary_data.lstrip().startswith(b"#!"):
        return ".sh", "text/x-script"
    elif binary_data.lstrip().startswith((b"{", b"[")):
        return ".json", "application/json"
    elif binary_data.lstrip().startswith((b"<", b"<!DOCTYPE")):
        return ".html", "text/html"
    elif binary_data.lstrip().startswith(b"<?xml"):
        return ".xml", "application/xml"
    elif all(32 <= b <= 127 or b in (9, 10, 13) for b in binary_data[:100]):
        return ".txt", "text/plain"

    else:
        raise ValueError("Unknown or unsupported file type")


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

def process_image(image: Image.Image, new_width: int = 400, new_height: int = 400, save: str = None) -> Image.Image:
    """
    Processes the given image by adjusting its orientation and resizing it.

    Args:
        image (Image): The image to process.
        new_width (int): The new width of the image.
        new_height (int): The new height of the image.

    Returns:
        Image: The processed image.
    """
    image = ImageOps.exif_transpose(image)
    # Remove transparency
    if image.mode == "RGBA":
        # image.load()
        # white = Image.new('RGB', image.size, (255, 255, 255))
        # white.paste(image, mask=image.split()[-1])
        # image = white
        pass
    # Convert to RGB for jpg format
    elif image.mode != "RGB":
        image = image.convert("RGB")
    image_size = image.size
    image.thumbnail((new_width, new_height))
    if save is not None:
        image.save(save, exif=b"")
        return image_size
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
    elif isinstance(image, str):
        if image.startswith("data:"):
            is_data_uri_an_image(image)
            return extract_data_uri(image)
        elif image.startswith("http://") or image.startswith("https://"):
            path: str = urlparse(image).path
            if path.startswith("/files/"):
                path = get_bucket_dir(*path.split("/")[2:])
                if os.path.exists(path):
                    return Path(path).read_bytes()
                else:
                    raise FileNotFoundError(f"File not found: {path}")
            else:
                resp = requests.get(image, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
                })
                if resp.ok and is_accepted_format(resp.content):
                    return resp.content
                raise ValueError("Invalid image url. Expected bytes, str, or PIL Image.")
        else:
            raise ValueError("Invalid image format. Expected bytes, str, or PIL Image.")
    elif isinstance(image, Image.Image):
        bytes_io = BytesIO()
        image.save(bytes_io, image.format)
        image.seek(0)
        return bytes_io.getvalue()
    elif isinstance(image, os.PathLike):
        return Path(image).read_bytes()
    elif isinstance(image, Path):
        return image.read_bytes()
    else:
        try:
            image.seek(0)
        except (AttributeError, io.UnsupportedOperation):
            pass
        return image.read()

def to_data_uri(image: ImageType, filename: str = None) -> str:
    if not isinstance(image, str):
        data = to_bytes(image)
        data_base64 = base64.b64encode(data).decode()
        return f"data:{is_data_an_media(data, filename)};base64,{data_base64}"
    return image

def to_input_audio(audio: ImageType, filename: str = None) -> str:
    if not isinstance(audio, str):
        if filename is not None:
            format = get_extension(filename)
            if format is None:
                raise ValueError("Invalid input audio")
            return {
                "data": base64.b64encode(to_bytes(audio)).decode(),
                "format": format
            }
        raise ValueError("Invalid input audio")
    audio = re.match(r'^data:audio/(\w+);base64,(.+?)', audio)
    if audio:
        return {
            "data": audio.group(2),
            "format": audio.group(1).replace("mpeg", "mp3")
        }
    raise ValueError("Invalid input audio")

def use_aspect_ratio(extra_body: dict, aspect_ratio: str) -> Image:
    extra_body = {key: value for key, value in extra_body.items() if value is not None}
    if extra_body.get("width") is None or extra_body.get("height") is None:
        width, height = get_width_height(
            aspect_ratio,
            extra_body.get("width"),
            extra_body.get("height")
        )
        extra_body = {
            "width": width,
            "height": height,
            **extra_body
        }
    return {key: value for key, value in extra_body.items() if value is not None}

def get_width_height(
    aspect_ratio: str,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> tuple[int, int]:
    if aspect_ratio == "1:1":
        return width or 1024, height or 1024
    elif aspect_ratio == "16:9":
        return width or 832, height or 480
    elif aspect_ratio == "9:16":
        return width or 480, height or 832,
    return width, height

class ImageRequest:
    def __init__(
        self,
        options: dict = {}
    ):
        self.options = options

    def get(self, key: str):
        return self.options.get(key)