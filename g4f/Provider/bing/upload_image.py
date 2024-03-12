"""
Module to handle image uploading and processing for Bing AI integrations.
"""
from __future__ import annotations

import json
import math
from aiohttp import ClientSession, FormData

from ...typing import ImageType, Tuple
from ...image import to_image, process_image, to_base64_jpg, ImageRequest, Image
from ...requests import raise_for_status

IMAGE_CONFIG = {
    "maxImagePixels": 360000,
    "imageCompressionRate": 0.7,
    "enableFaceBlurDebug": False,
}

async def upload_image(
    session: ClientSession, 
    image_data: ImageType, 
    tone: str, 
    headers: dict
) -> ImageRequest:
    """
    Uploads an image to Bing's AI service and returns the image response.

    Args:
        session (ClientSession): The active session.
        image_data (bytes): The image data to be uploaded.
        tone (str): The tone of the conversation.
        proxy (str, optional): Proxy if any. Defaults to None.

    Raises:
        RuntimeError: If the image upload fails.

    Returns:
        ImageRequest: The response from the image upload.
    """
    image = to_image(image_data)
    new_width, new_height = calculate_new_dimensions(image)
    image = process_image(image, new_width, new_height)
    img_binary_data = to_base64_jpg(image, IMAGE_CONFIG['imageCompressionRate'])

    data = build_image_upload_payload(img_binary_data, tone)

    async with session.post("https://www.bing.com/images/kblob", data=data, headers=prepare_headers(headers)) as response:
        await raise_for_status(response, "Failed to upload image")
        return parse_image_response(await response.json())

def calculate_new_dimensions(image: Image) -> Tuple[int, int]:
    """
    Calculates the new dimensions for the image based on the maximum allowed pixels.

    Args:
        image (Image): The PIL Image object.

    Returns:
        Tuple[int, int]: The new width and height for the image.
    """
    width, height = image.size
    max_image_pixels = IMAGE_CONFIG['maxImagePixels']
    if max_image_pixels / (width * height) < 1:
        scale_factor = math.sqrt(max_image_pixels / (width * height))
        return int(width * scale_factor), int(height * scale_factor)
    return width, height

def build_image_upload_payload(image_bin: str, tone: str) -> FormData:
    """
    Builds the payload for image uploading.

    Args:
        image_bin (str): Base64 encoded image binary data.
        tone (str): The tone of the conversation.

    Returns:
        Tuple[str, str]: The data and boundary for the payload.
    """
    data = FormData()
    knowledge_request = json.dumps(build_knowledge_request(tone), ensure_ascii=False)
    data.add_field('knowledgeRequest', knowledge_request, content_type="application/json")
    data.add_field('imageBase64', image_bin)
    return data

def build_knowledge_request(tone: str) -> dict:
    """
    Builds the knowledge request payload.

    Args:
        tone (str): The tone of the conversation.

    Returns:
        dict: The knowledge request payload.
    """
    return {
        "imageInfo": {},
        "knowledgeRequest": {
            'invokedSkills': ["ImageById"],
            'subscriptionId': "Bing.Chat.Multimodal",
            'invokedSkillsRequestData': {
                'enableFaceBlur': True
            },
            'convoData': {
                'convoid': "",
                'convotone': tone
            }
        }
    }

def prepare_headers(headers: dict) -> dict:
    """
    Prepares the headers for the image upload request.

    Args:
        session (ClientSession): The active session.
        boundary (str): The boundary string for the multipart/form-data.

    Returns:
        dict: The headers for the request.
    """
    headers["Referer"] = 'https://www.bing.com/search?q=Bing+AI&showconv=1&FORM=hpcodx'
    headers["Origin"] = 'https://www.bing.com'
    return headers

def parse_image_response(response: dict) -> ImageRequest:
    """
    Parses the response from the image upload.

    Args:
        response (dict): The response dictionary.

    Raises:
        RuntimeError: If parsing the image info fails.

    Returns:
        ImageRequest: The parsed image response.
    """
    if not response.get('blobId'):
        raise RuntimeError("Failed to parse image info.")

    result = {'bcid': response.get('blobId', ""), 'blurredBcid': response.get('processedBlobId', "")}
    result["imageUrl"] = f"https://www.bing.com/images/blob?bcid={result['blurredBcid'] or result['bcid']}"

    result['originalImageUrl'] = (
        f"https://www.bing.com/images/blob?bcid={result['blurredBcid']}"
        if IMAGE_CONFIG["enableFaceBlurDebug"] else
        f"https://www.bing.com/images/blob?bcid={result['bcid']}"
    )
    return ImageRequest(result)