from __future__ import annotations

import string
import random
import json
import numpy as np
from ...typing import ImageType
from aiohttp import ClientSession
from ...image import to_image, process_image, to_base64

image_config = {
    "maxImagePixels": 360000,
    "imageCompressionRate": 0.7,
    "enableFaceBlurDebug": 0,
}

async def upload_image(
    session: ClientSession,
    image: ImageType,
    tone: str,
    proxy: str = None
) -> dict:
    image = to_image(image)
    width, height = image.size
    max_image_pixels = image_config['maxImagePixels']
    if max_image_pixels / (width * height) < 1:
        new_width = int(width * np.sqrt(max_image_pixels / (width * height)))
        new_height = int(height * np.sqrt(max_image_pixels / (width * height)))
    else:
        new_width = width
        new_height = height
    new_img = process_image(image, new_width, new_height)
    new_img_binary_data = to_base64(new_img, image_config['imageCompressionRate'])
    data, boundary = build_image_upload_api_payload(new_img_binary_data, tone)
    headers = session.headers.copy()
    headers["content-type"] = f'multipart/form-data; boundary={boundary}'
    headers["referer"] = 'https://www.bing.com/search?q=Bing+AI&showconv=1&FORM=hpcodx'
    headers["origin"] = 'https://www.bing.com'
    async with session.post("https://www.bing.com/images/kblob", data=data, headers=headers, proxy=proxy) as response:
        if response.status != 200:
            raise RuntimeError("Failed to upload image.")
        image_info = await response.json()
        if not image_info.get('blobId'):
            raise RuntimeError("Failed to parse image info.")
        result = {'bcid': image_info.get('blobId', "")}
        result['blurredBcid'] = image_info.get('processedBlobId', "")
        if result['blurredBcid'] != "":
            result["imageUrl"] = "https://www.bing.com/images/blob?bcid=" + result['blurredBcid']
        elif result['bcid'] != "":
            result["imageUrl"] = "https://www.bing.com/images/blob?bcid=" + result['bcid']
        result['originalImageUrl'] = (
            "https://www.bing.com/images/blob?bcid="
            + result['blurredBcid']
            if image_config["enableFaceBlurDebug"]
            else "https://www.bing.com/images/blob?bcid="
            + result['bcid']
        )
        return result
    

def build_image_upload_api_payload(image_bin: str, tone: str):
    payload = {
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
    knowledge_request = {
        'imageInfo': {},
        'knowledgeRequest': payload
    }
    boundary="----WebKitFormBoundary" + ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    data = (
        f'--{boundary}'
        + '\r\nContent-Disposition: form-data; name="knowledgeRequest"\r\n\r\n'
        + json.dumps(knowledge_request, ensure_ascii=False)
        + "\r\n--"
        + boundary
        + '\r\nContent-Disposition: form-data; name="imageBase64"\r\n\r\n'
        + image_bin
        + "\r\n--"
        + boundary
        + "--\r\n"
    )
    return data, boundary