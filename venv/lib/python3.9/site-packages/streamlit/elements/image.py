# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Some casts in this file are only occasionally necessary depending on the
# user's Python version, and mypy doesn't have a good way of toggling this
# specific config option at a per-line level.
# mypy: no-warn-unused-ignores

"""Image marshalling."""

import imghdr
import io
import mimetypes
import re
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional, Sequence, Union, cast
from urllib.parse import urlparse

import numpy as np
from PIL import GifImagePlugin, Image, ImageFile
from typing_extensions import Final, Literal, TypeAlias

from streamlit import runtime
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.Image_pb2 import ImageList as ImageListProto
from streamlit.runtime import caching
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

    from streamlit.delta_generator import DeltaGenerator

LOGGER: Final = get_logger(__name__)

# This constant is related to the frontend maximum content width specified
# in App.jsx main container
# 730 is the max width of element-container in the frontend, and 2x is for high
# DPI.
MAXIMUM_CONTENT_WIDTH: Final[int] = 2 * 730

PILImage: TypeAlias = Union[
    ImageFile.ImageFile, Image.Image, GifImagePlugin.GifImageFile
]
AtomicImage: TypeAlias = Union[PILImage, "npt.NDArray[Any]", io.BytesIO, str]
ImageOrImageList: TypeAlias = Union[AtomicImage, List[AtomicImage]]
UseColumnWith: TypeAlias = Optional[Union[Literal["auto", "always", "never"], bool]]
Channels: TypeAlias = Literal["RGB", "BGR"]
ImageFormat: TypeAlias = Literal["JPEG", "PNG", "GIF"]
ImageFormatOrAuto: TypeAlias = Literal[ImageFormat, "auto"]


class WidthBehaviour(IntEnum):
    """
    Special values that are recognized by the frontend and allow us to change the
    behavior of the displayed image.
    """

    ORIGINAL = -1
    COLUMN = -2
    AUTO = -3


WidthBehaviour.ORIGINAL.__doc__ = """Display the image at its original width"""
WidthBehaviour.COLUMN.__doc__ = (
    """Display the image at the width of the column it's in."""
)
WidthBehaviour.AUTO.__doc__ = """Display the image at its original width, unless it
would exceed the width of its column in which case clamp it to
its column width"""


class ImageMixin:
    @gather_metrics("image")
    def image(
        self,
        image: ImageOrImageList,
        # TODO: Narrow type of caption, dependent on type of image,
        #  by way of overload
        caption: Optional[Union[str, List[str]]] = None,
        width: Optional[int] = None,
        use_column_width: UseColumnWith = None,
        clamp: bool = False,
        channels: Channels = "RGB",
        output_format: ImageFormatOrAuto = "auto",
    ) -> "DeltaGenerator":
        """Display an image or list of images.

        Parameters
        ----------
        image : numpy.ndarray, [numpy.ndarray], BytesIO, str, or [str]
            Monochrome image of shape (w,h) or (w,h,1)
            OR a color image of shape (w,h,3)
            OR an RGBA image of shape (w,h,4)
            OR a URL to fetch the image from
            OR a path of a local image file
            OR an SVG XML string like `<svg xmlns=...</svg>`
            OR a list of one of the above, to display multiple images.
        caption : str or list of str
            Image caption. If displaying multiple images, caption should be a
            list of captions (one for each image).
        width : int or None
            Image width. None means use the image width,
            but do not exceed the width of the column.
            Should be set for SVG images, as they have no default image width.
        use_column_width : 'auto' or 'always' or 'never' or bool
            If 'auto', set the image's width to its natural size,
            but do not exceed the width of the column.
            If 'always' or True, set the image's width to the column width.
            If 'never' or False, set the image's width to its natural size.
            Note: if set, `use_column_width` takes precedence over the `width` parameter.
        clamp : bool
            Clamp image pixel values to a valid range ([0-255] per channel).
            This is only meaningful for byte array images; the parameter is
            ignored for image URLs. If this is not set, and an image has an
            out-of-range value, an error will be thrown.
        channels : 'RGB' or 'BGR'
            If image is an nd.array, this parameter denotes the format used to
            represent color information. Defaults to 'RGB', meaning
            `image[:, :, 0]` is the red channel, `image[:, :, 1]` is green, and
            `image[:, :, 2]` is blue. For images coming from libraries like
            OpenCV you should set this to 'BGR', instead.
        output_format : 'JPEG', 'PNG', or 'auto'
            This parameter specifies the format to use when transferring the
            image data. Photos should use the JPEG format for lossy compression
            while diagrams should use the PNG format for lossless compression.
            Defaults to 'auto' which identifies the compression type based
            on the type and format of the image argument.

        Example
        -------
        >>> import streamlit as st
        >>> from PIL import Image
        >>>
        >>> image = Image.open('sunrise.jpg')
        >>>
        >>> st.image(image, caption='Sunrise by the mountains')

        .. output::
           https://doc-image.streamlitapp.com/
           height: 710px

        """

        if use_column_width == "auto" or (use_column_width is None and width is None):
            width = WidthBehaviour.AUTO
        elif use_column_width == "always" or use_column_width == True:
            width = WidthBehaviour.COLUMN
        elif width is None:
            width = WidthBehaviour.ORIGINAL
        elif width <= 0:
            raise StreamlitAPIException("Image width must be positive.")

        image_list_proto = ImageListProto()
        marshall_images(
            self.dg._get_delta_path_str(),
            image,
            caption,
            width,
            image_list_proto,
            clamp,
            channels,
            output_format,
        )
        return self.dg._enqueue("imgs", image_list_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def _image_may_have_alpha_channel(image: PILImage) -> bool:
    if image.mode in ("RGBA", "LA", "P"):
        return True
    else:
        return False


def _image_is_gif(image: PILImage) -> bool:
    return bool(image.format == "GIF")


def _validate_image_format_string(
    image_data: Union[bytes, PILImage], format: str
) -> ImageFormat:
    """Return either "JPEG", "PNG", or "GIF", based on the input `format` string.

    - If `format` is "JPEG" or "JPG" (or any capitalization thereof), return "JPEG"
    - If `format` is "PNG" (or any capitalization thereof), return "PNG"
    - For all other strings, return "PNG" if the image has an alpha channel,
    "GIF" if the image is a GIF, and "JPEG" otherwise.
    """
    format = format.upper()
    if format == "JPEG" or format == "PNG":
        return cast(ImageFormat, format)

    # We are forgiving on the spelling of JPEG
    if format == "JPG":
        return "JPEG"

    if isinstance(image_data, bytes):
        pil_image = Image.open(io.BytesIO(image_data))
    else:
        pil_image = image_data

    if _image_is_gif(pil_image):
        return "GIF"

    if _image_may_have_alpha_channel(pil_image):
        return "PNG"

    return "JPEG"


def _PIL_to_bytes(
    image: PILImage,
    format: ImageFormat = "JPEG",
    quality: int = 100,
) -> bytes:
    """Convert a PIL image to bytes."""
    tmp = io.BytesIO()

    # User must have specified JPEG, so we must convert it
    if format == "JPEG" and _image_may_have_alpha_channel(image):
        image = image.convert("RGB")

    image.save(tmp, format=format, quality=quality)

    return tmp.getvalue()


def _BytesIO_to_bytes(data: io.BytesIO) -> bytes:
    data.seek(0)
    return data.getvalue()


def _np_array_to_bytes(array: "npt.NDArray[Any]", output_format="JPEG") -> bytes:
    img = Image.fromarray(array.astype(np.uint8))
    format = _validate_image_format_string(img, output_format)

    return _PIL_to_bytes(img, format)


def _4d_to_list_3d(array: "npt.NDArray[Any]") -> List["npt.NDArray[Any]"]:
    return [array[i, :, :, :] for i in range(0, array.shape[0])]


def _verify_np_shape(array: "npt.NDArray[Any]") -> "npt.NDArray[Any]":
    if len(array.shape) not in (2, 3):
        raise StreamlitAPIException("Numpy shape has to be of length 2 or 3.")
    if len(array.shape) == 3 and array.shape[-1] not in (1, 3, 4):
        raise StreamlitAPIException(
            "Channel can only be 1, 3, or 4 got %d. Shape is %s"
            % (array.shape[-1], str(array.shape))
        )

    # If there's only one channel, convert is to x, y
    if len(array.shape) == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]

    return array


def _get_image_format_mimetype(image_format: ImageFormat) -> str:
    """Get the mimetype string for the given ImageFormat."""
    return f"image/{image_format.lower()}"


def _ensure_image_size_and_format(
    image_data: bytes, width: int, image_format: ImageFormat
) -> bytes:
    """Resize an image if it exceeds the given width, or if exceeds
    MAXIMUM_CONTENT_WIDTH. Ensure the image's format corresponds to the given
    ImageFormat. Return the (possibly resized and reformatted) image bytes.
    """
    image = Image.open(io.BytesIO(image_data))
    actual_width, actual_height = image.size

    if width < 0 and actual_width > MAXIMUM_CONTENT_WIDTH:
        width = MAXIMUM_CONTENT_WIDTH

    if width > 0 and actual_width > width:
        # We need to resize the image.
        new_height = int(1.0 * actual_height * width / actual_width)
        image = image.resize((width, new_height), resample=Image.BILINEAR)
        return _PIL_to_bytes(image, format=image_format, quality=90)

    ext = imghdr.what(None, image_data)
    if ext != image_format.lower():
        # We need to reformat the image.
        return _PIL_to_bytes(image, format=image_format, quality=90)

    # No resizing or reformatting necessary - return the original bytes.
    return image_data


def _clip_image(image: "npt.NDArray[Any]", clamp: bool) -> "npt.NDArray[Any]":
    data = image
    if issubclass(image.dtype.type, np.floating):
        if clamp:
            data = np.clip(image, 0, 1.0)
        else:
            if np.amin(image) < 0.0 or np.amax(image) > 1.0:
                raise RuntimeError("Data is outside [0.0, 1.0] and clamp is not set.")
        data = data * 255
    else:
        if clamp:
            data = np.clip(image, 0, 255)
        else:
            if np.amin(image) < 0 or np.amax(image) > 255:
                raise RuntimeError("Data is outside [0, 255] and clamp is not set.")
    return data


def image_to_url(
    image: AtomicImage,
    width: int,
    clamp: bool,
    channels: Channels,
    output_format: ImageFormatOrAuto,
    image_id: str,
) -> str:
    """Return a URL that an image can be served from.
    If `image` is already a URL, return it unmodified.
    Otherwise, add the image to the MediaFileManager and return the URL.

    (When running in "raw" mode, we won't actually load data into the
    MediaFileManager, and we'll return an empty URL.)
    """

    image_data: bytes

    # Strings
    if isinstance(image, str):
        # If it's a url, return it directly.
        try:
            p = urlparse(image)
            if p.scheme:
                return image
        except UnicodeDecodeError:
            # If the string runs into a UnicodeDecodeError, we assume it is not a valid URL.
            pass

        # Otherwise, try to open it as a file.
        try:
            with open(image, "rb") as f:
                image_data = f.read()
        except Exception:
            # When we aren't able to open the image file, we still pass the path to
            # the MediaFileManager - its storage backend may have access to files
            # that Streamlit does not.
            mimetype, _ = mimetypes.guess_type(image)
            if mimetype is None:
                mimetype = "application/octet-stream"

            url = runtime.get_instance().media_file_mgr.add(image, mimetype, image_id)
            caching.save_media_data(image, mimetype, image_id)
            return url

    # PIL Images
    elif isinstance(image, (ImageFile.ImageFile, Image.Image)):
        format = _validate_image_format_string(image, output_format)
        image_data = _PIL_to_bytes(image, format)

    # BytesIO
    # Note: This doesn't support SVG. We could convert to png (cairosvg.svg2png)
    # or just decode BytesIO to string and handle that way.
    elif isinstance(image, io.BytesIO):
        image_data = _BytesIO_to_bytes(image)

    # Numpy Arrays (ie opencv)
    elif isinstance(image, np.ndarray):
        image = _clip_image(
            _verify_np_shape(image),
            clamp,
        )

        if channels == "BGR":
            if len(image.shape) == 3:
                image = image[:, :, [2, 1, 0]]
            else:
                raise StreamlitAPIException(
                    'When using `channels="BGR"`, the input image should '
                    "have exactly 3 color channels"
                )

        # Depending on the version of numpy that the user has installed, the
        # typechecker may not be able to deduce that indexing into a
        # `npt.NDArray[Any]` returns a `npt.NDArray[Any]`, so we need to
        # ignore redundant casts below.
        image_data = _np_array_to_bytes(
            array=cast("npt.NDArray[Any]", image),  # type: ignore[redundant-cast]
            output_format=output_format,
        )

    # Raw bytes
    else:
        image_data = image

    # Determine the image's format, resize it, and get its mimetype
    image_format = _validate_image_format_string(image_data, output_format)
    image_data = _ensure_image_size_and_format(image_data, width, image_format)
    mimetype = _get_image_format_mimetype(image_format)

    if runtime.exists():
        url = runtime.get_instance().media_file_mgr.add(image_data, mimetype, image_id)
        caching.save_media_data(image_data, mimetype, image_id)
        return url
    else:
        # When running in "raw mode", we can't access the MediaFileManager.
        return ""


def marshall_images(
    coordinates: str,
    image: ImageOrImageList,
    caption: Optional[Union[str, "npt.NDArray[Any]", List[str]]],
    width: Union[int, WidthBehaviour],
    proto_imgs: ImageListProto,
    clamp: bool,
    channels: Channels = "RGB",
    output_format: ImageFormatOrAuto = "auto",
) -> None:
    """Fill an ImageListProto with a list of images and their captions.

    The images will be resized and reformatted as necessary.

    Parameters
    ----------
    coordinates
        A string indentifying the images' location in the frontend.
    image
        The image or images to include in the ImageListProto.
    caption
        Image caption. If displaying multiple images, caption should be a
        list of captions (one for each image).
    width
        The desired width of the image or images. This parameter will be
        passed to the frontend.
        Positive values set the image width explicitly.
        Negative values has some special. For details, see: `WidthBehaviour`
    proto_imgs
        The ImageListProto to fill in.
    clamp
        Clamp image pixel values to a valid range ([0-255] per channel).
        This is only meaningful for byte array images; the parameter is
        ignored for image URLs. If this is not set, and an image has an
        out-of-range value, an error will be thrown.
    channels
        If image is an nd.array, this parameter denotes the format used to
        represent color information. Defaults to 'RGB', meaning
        `image[:, :, 0]` is the red channel, `image[:, :, 1]` is green, and
        `image[:, :, 2]` is blue. For images coming from libraries like
        OpenCV you should set this to 'BGR', instead.
    output_format
        This parameter specifies the format to use when transferring the
        image data. Photos should use the JPEG format for lossy compression
        while diagrams should use the PNG format for lossless compression.
        Defaults to 'auto' which identifies the compression type based
        on the type and format of the image argument.
    """
    channels = cast(Channels, channels.upper())

    # Turn single image and caption into one element list.
    images: Sequence[AtomicImage]
    if isinstance(image, list):
        images = image
    elif isinstance(image, np.ndarray) and len(image.shape) == 4:
        images = _4d_to_list_3d(image)
    else:
        images = [image]

    if type(caption) is list:
        captions: Sequence[Optional[str]] = caption
    else:
        if isinstance(caption, str):
            captions = [caption]
        # You can pass in a 1-D Numpy array as captions.
        elif isinstance(caption, np.ndarray) and len(caption.shape) == 1:
            captions = caption.tolist()
        # If there are no captions then make the captions list the same size
        # as the images list.
        elif caption is None:
            captions = [None] * len(images)
        else:
            captions = [str(caption)]

    assert type(captions) == list, "If image is a list then caption should be as well"
    assert len(captions) == len(images), "Cannot pair %d captions with %d images." % (
        len(captions),
        len(images),
    )

    proto_imgs.width = int(width)
    # Each image in an image list needs to be kept track of at its own coordinates.
    for coord_suffix, (image, caption) in enumerate(zip(images, captions)):
        proto_img = proto_imgs.imgs.add()
        if caption is not None:
            proto_img.caption = str(caption)

        # We use the index of the image in the input image list to identify this image inside
        # MediaFileManager. For this, we just add the index to the image's "coordinates".
        image_id = "%s-%i" % (coordinates, coord_suffix)

        is_svg = False
        if isinstance(image, str):
            # Unpack local SVG image file to an SVG string
            if image.endswith(".svg") and not image.startswith(("http://", "https://")):
                with open(image) as textfile:
                    image = textfile.read()

            # Following regex allows svg image files to start either via a "<?xml...>" tag eventually followed by a "<svg...>" tag or directly starting with a "<svg>" tag
            if re.search(r"(^\s?(<\?xml[\s\S]*<svg\s)|^\s?<svg\s|^\s?<svg>\s)", image):
                if "xlink" in image or "xmlns" not in image:
                    proto_img.markup = f"data:image/svg+xml,{image}"
                else:
                    proto_img.url = f"data:image/svg+xml,{image}"
                is_svg = True

        if not is_svg:
            proto_img.url = image_to_url(
                image, width, clamp, channels, output_format, image_id
            )
