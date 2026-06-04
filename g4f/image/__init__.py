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
    "jpg":  "image/jpeg",
    "png":  "image/png",
    "gif":  "image/gif",
    "webp": "image/webp",
    "bmp":  "image/bmp",
    "tiff": "image/tiff",
    "tif":  "image/tiff",
    "ico":  "image/x-icon",
    "svg":  "image/svg+xml",
    "avif": "image/avif",
    "heic": "image/heif",
    # Audio
    "wav":  "audio/wav",
    "mp3":  "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "ogg":  "audio/ogg",
    "m4a":  "audio/mp4",   # was "audio/m4a" — non-standard
    # Video
    "mkv":  "video/x-matroska",
    "webm": "video/webm",
    "mp4":  "video/mp4",
}

MEDIA_TYPE_MAP: dict[str, str] = {value: key for key, value in EXTENSIONS_MAP.items()}
MEDIA_TYPE_MAP["audio/webm"] = "webm"
# Handle duplicate image/jpeg (jpg wins over jpeg as the "canonical" extension)
MEDIA_TYPE_MAP["image/jpeg"] = "jpg"

# Audio formats accepted for input (OpenAI-compatible)
ACCEPTED_AUDIO_FORMATS = {"wav", "mp3", "flac", "ogg", "opus", "m4a"}


def to_image(image: ImageType, is_svg: bool = False) -> Image.Image:
    """
    Converts the input image to a PIL Image object.
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


def is_safe_url(url: str) -> bool:
    """
    Checks if the URL is safe to download (basic http/https check).
    """
    if not isinstance(url, str):
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def is_allowed_extension(filename: str) -> Optional[str]:
    """
    Returns the MIME type for allowed extensions, or None.
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


def is_data_an_audio(data_uri: str = None, filename: str = None) -> Optional[str]:
    if filename:
        extension = get_extension(filename)
        if extension is not None:
            media_type = EXTENSIONS_MAP[extension]
            if media_type.startswith("audio/"):
                return media_type
    if isinstance(data_uri, str):
        audio_format = re.match(r'^data:(audio/[\w+-]+);base64,', data_uri)
        if audio_format:
            return audio_format.group(1)
    return None


def is_valid_audio(data_uri: str = None, filename: str = None) -> bool:
    """
    Returns True if the media is a supported audio format.
    Accepted: wav, mp3, flac, ogg, opus, m4a
    """
    mimetype = is_data_an_audio(data_uri, filename)
    if mimetype is None:
        return False
    ext = MEDIA_TYPE_MAP.get(mimetype)
    return ext in ACCEPTED_AUDIO_FORMATS


def is_data_uri_an_image(data_uri: str) -> bool:
    """
    Checks if the given data URI represents an image.

    Raises:
        ValueError: If the data URI is invalid or the image format is not allowed.
    """
    if data_uri.startswith("https:") or data_uri.startswith("http:"):
        return True
    if not re.match(r'data:image/[\w+]+;base64,', data_uri):
        raise ValueError(f"Invalid data URI image. {data_uri[:10]}...")
    image_format = re.match(r'data:image/([\w+]+);base64,', data_uri).group(1).lower()
    if image_format not in EXTENSIONS_MAP and image_format != "svg+xml":
        raise ValueError("Invalid image format (from mime file type).")
    return True


def is_accepted_format(binary_data: bytes) -> str:
    """
    Checks if the given binary data represents an image with an accepted format.
    Returns the MIME type string.

    Raises:
        ValueError: If the image format is not recognized.
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


def detect_file_type(binary_data: bytes) -> tuple[str, str]:
    """
    Detect file type from magic number / header signature.

    Returns:
        tuple: (extension, MIME type)

    Raises:
        ValueError: If file type is unknown or unsupported.
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
        # ISO Base Media File Format — branch on brand to distinguish mp4/heic/avif
        brand = binary_data[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"):
            return ".heic", "image/heif"
        elif brand == b"avif":
            return ".avif", "image/avif"
        else:
            # Default to MP4 for any other ftyp brand (isom, mp41, mp42, M4V, ...)
            return ".mp4", "video/mp4"
    elif binary_data.lstrip().startswith((b"<?xml", b"<svg")):
        return ".svg", "image/svg+xml"

    # ---- Documents ----
    elif binary_data.startswith(b"%PDF"):
        return ".pdf", "application/pdf"
    elif binary_data.startswith(b"PK\x03\x04"):
        # Could be docx/xlsx/pptx/jar/apk/odt — use generic zip
        return ".zip", "application/zip"
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
    elif binary_data.startswith(b"\xca\xfe\xba\xbe") or binary_data.startswith(b"\xca\xfe\xd0\x0d"):
        return ".class", "application/java-vm"
    elif binary_data.startswith(b"\x50\x4b\x03\x04") and b"META-INF" in binary_data[:200]:
        return ".jar", "application/java-archive"

    # ---- Audio ----
    elif binary_data.startswith(b"ID3") or binary_data[0:2] == b"\xff\xfb":
        return ".mp3", "audio/mpeg"
    elif binary_data.startswith(b"fLaC"):
        return ".flac", "audio/flac"
    elif binary_data.startswith(b"RIFF") and binary_data[8:12] == b"WAVE":
        return ".wav", "audio/wav"
    elif binary_data.startswith(b"MThd"):
        return ".mid", "audio/midi"
    elif binary_data.startswith(b"OggS"):
        # Distinguish Ogg audio from Ogg video by reading the codec header
        # Vorbis/Opus → audio; Theora → video
        if b"\x80theora" in binary_data[:64] or b"theora" in binary_data[:64]:
            return ".ogv", "video/ogg"
        elif b"\x01vorbis" in binary_data[:64] or b"OpusHead" in binary_data[:64]:
            return ".ogg", "audio/ogg"
        else:
            # Default to audio/ogg for unrecognised Ogg streams
            return ".ogg", "audio/ogg"

    # ---- Video ----
    elif binary_data.startswith(b"RIFF") and binary_data[8:12] == b"AVI ":
        return ".avi", "video/x-msvideo"
    elif binary_data.startswith(b"\x1a\x45\xdf\xa3"):
        # EBML — could be MKV or WebM; check DocType in first 64 bytes
        if b"webm" in binary_data[:64]:
            return ".webm", "video/webm"
        return ".mkv", "video/x-matroska"
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
    """Extract binary data from a data URI."""
    data = data_uri.split(",")[-1]
    return base64.b64decode(data)


def process_image(
    image: Image.Image,
    new_width: int = 400,
    new_height: int = 400,
    save: str = None
) -> Image.Image:
    """
    Adjusts orientation and resizes image. Preserves transparency for PNG output.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        pass  # keep transparency for PNG output
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
                local_path = get_bucket_dir(*path.split("/")[2:])
                if os.path.exists(local_path):
                    return Path(local_path).read_bytes()
                else:
                    raise FileNotFoundError(f"File not found: {local_path}")
            else:
                if not is_safe_url(image):
                    raise ValueError("Invalid or unsafe image url")
                resp = requests.get(image, headers={
                    # Updated to Chrome/145 (current as of Mar 2026)
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
                })
                if resp.ok and is_accepted_format(resp.content):
                    return resp.content
                raise ValueError("Invalid image url. Expected bytes, str, or PIL Image.")
        else:
            raise ValueError("Invalid image format. Expected bytes, str, or PIL Image.")
    elif isinstance(image, Image.Image):
        bytes_io = BytesIO()
        image.save(bytes_io, image.format or "PNG")
        bytes_io.seek(0)   # FIX: was image.seek(0) — PIL Image has no seek()
        return bytes_io.getvalue()
    elif isinstance(image, (os.PathLike, Path)):
        return Path(image).read_bytes()
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


def to_input_audio(audio: ImageType, filename: str = None) -> dict:
    if not isinstance(audio, str):
        if filename is not None:
            fmt = get_extension(filename)
            if fmt is None:
                raise ValueError("Invalid input audio")
            return {
                "data": base64.b64encode(to_bytes(audio)).decode(),
                "format": fmt
            }
        raise ValueError("Invalid input audio")
    match = re.match(r'^data:audio/([\w+-]+);base64,(.+)', audio)
    if match:
        return {
            "data": match.group(2),
            "format": match.group(1).replace("mpeg", "mp3")
        }
    raise ValueError("Invalid input audio")


def use_aspect_ratio(extra_body: dict, aspect_ratio: str) -> dict:
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
    """
    Returns (width, height) for common aspect ratios.
    """
    ratio_map = {
        "1:1":   (1024, 1024),
        "16:9":  (1024, 576),
        "9:16":  (576,  1024),
        "4:3":   (1024, 768),
        "3:4":   (768,  1024),
        "3:2":   (1024, 682),
        "2:3":   (682,  1024),
        "21:9":  (1024, 440),
        "9:21":  (440,  1024),
        "4:5":   (832,  1040),
        "5:4":   (1040, 832),
        "2:1":   (1024, 512),
        "1:2":   (512,  1024),
    }
    if aspect_ratio in ratio_map:
        default_w, default_h = ratio_map[aspect_ratio]
        return width or default_w, height or default_h
    return width, height


class ImageRequest:
    def __init__(self, options: dict = None):
        self.options = options or {}

    def get(self, key: str):
        return self.options.get(key)
