from __future__ import annotations

import re
from io import BytesIO
import base64
from .typing import ImageType, Union, Image

try:
    from PIL.Image import open as open_image, new as new_image
    from PIL.Image import FLIP_LEFT_RIGHT, ROTATE_180, ROTATE_270, ROTATE_90
    has_requirements = True
except ImportError:
    has_requirements = False
    
from .errors import MissingRequirementsError

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'}

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
    if isinstance(image, str):
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
        copy = image.copy()
        copy.format = image.format
        return copy
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

def is_accepted_format(binary_data: bytes) -> bool:
    """
    Checks if the given binary data represents an image with an accepted format.

    Args:
        binary_data (bytes): The binary data to check.

    Raises:
        ValueError: If the image format is not allowed.
    """
    if binary_data.startswith(b'\xFF\xD8\xFF'):
        pass # It's a JPEG image
    elif binary_data.startswith(b'\x89PNG\r\n\x1a\n'):
        pass # It's a PNG image
    elif binary_data.startswith(b'GIF87a') or binary_data.startswith(b'GIF89a'):
        pass # It's a GIF image
    elif binary_data.startswith(b'\x89JFIF') or binary_data.startswith(b'JFIF\x00'):
        pass # It's a JPEG image
    elif binary_data.startswith(b'\xFF\xD8'):
        pass # It's a JPEG image
    elif binary_data.startswith(b'RIFF') and binary_data[8:12] == b'WEBP':
        pass # It's a WebP image
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
    data = data_uri.split(",")[1]
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

def process_image(img: Image, new_width: int, new_height: int) -> Image:
    """
    Processes the given image by adjusting its orientation and resizing it.

    Args:
        img (Image): The image to process.
        new_width (int): The new width of the image.
        new_height (int): The new height of the image.

    Returns:
        Image: The processed image.
    """
    # Fix orientation
    orientation = get_orientation(img)
    if orientation:
        if orientation > 4:
            img = img.transpose(FLIP_LEFT_RIGHT)
        if orientation in [3, 4]:
            img = img.transpose(ROTATE_180)
        if orientation in [5, 6]:
            img = img.transpose(ROTATE_270)
        if orientation in [7, 8]:
            img = img.transpose(ROTATE_90)
    # Resize image
    img.thumbnail((new_width, new_height))
    # Remove transparency
    if img.mode != "RGB":
        img.load()
        white = new_image('RGB', img.size, (255, 255, 255))
        white.paste(img, mask=img.split()[3]) 
        return white
    return img

def to_base64_jpg(image: Image, compression_rate: float) -> str:
    """
    Converts the given image to a base64-encoded string.

    Args:
        image (Image.Image): The image to convert.
        compression_rate (float): The compression rate (0.0 to 1.0).

    Returns:
        str: The base64-encoded image.
    """
    output_buffer = BytesIO()
    image.save(output_buffer, format="JPEG", quality=int(compression_rate * 100))
    return base64.b64encode(output_buffer.getvalue()).decode()

def format_images_markdown(images, alt: str, preview: str = None) -> str:
    """
    Formats the given images as a markdown string.

    Args:
        images: The images to format.
        alt (str): The alt for the images.
        preview (str, optional): The preview URL format. Defaults to "{image}?w=200&h=200".

    Returns:
        str: The formatted markdown string.
    """
    if isinstance(images, str):
        images = f"[![{alt}]({preview.replace('{image}', images) if preview else images})]({images})"
    else:
        images = [
            f"[![#{idx+1} {alt}]({preview.replace('{image}', image) if preview else image})]({image})"
            for idx, image in enumerate(images)
        ]
        images = "\n".join(images)
    start_flag = "<!-- generated images start -->\n"
    end_flag = "<!-- generated images end -->\n"
    return f"\n{start_flag}{images}\n{end_flag}\n"

def to_bytes(image: Image) -> bytes:
    """
    Converts the given image to bytes.

    Args:
        image (Image.Image): The image to convert.

    Returns:
        bytes: The image as bytes.
    """
    bytes_io = BytesIO()
    image.save(bytes_io, image.format)
    image.seek(0)
    return bytes_io.getvalue()

class ImageResponse:
    def __init__(
        self,
        images: Union[str, list],
        alt: str,
        options: dict = {}
    ):
        self.images = images
        self.alt = alt
        self.options = options
        
    def __str__(self) -> str:
        return format_images_markdown(self.images, self.alt, self.get("preview"))
    
    def get(self, key: str):
        return self.options.get(key)
    
class ImageRequest:
    def __init__(
        self,
        options: dict = {}
    ):
        self.options = options
    
    def get(self, key: str):
        return self.options.get(key)