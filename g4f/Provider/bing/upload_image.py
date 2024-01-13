from __future__ import annotations

import string
import random
import json
import re
import io
import base64
import numpy as np
from PIL import Image
from aiohttp import ClientSession

async def upload_image(
    session: ClientSession,
    image: str,
    tone: str,
    proxy: str = None
):
    try:
        image_config = {
            "maxImagePixels": 360000,
            "imageCompressionRate": 0.7,
            "enableFaceBlurDebug": 0,
        }
        is_data_uri_an_image(image)
        img_binary_data = extract_data_uri(image)
        is_accepted_format(img_binary_data)
        img = Image.open(io.BytesIO(img_binary_data))
        width, height = img.size
        max_image_pixels = image_config['maxImagePixels']
        if max_image_pixels / (width * height) < 1:
            new_width = int(width * np.sqrt(max_image_pixels / (width * height)))
            new_height = int(height * np.sqrt(max_image_pixels / (width * height)))
        else:
            new_width = width
            new_height = height
        try:
            orientation = get_orientation(img)
        except Exception:
            orientation = None
        new_img = process_image(orientation, img, new_width, new_height)
        new_img_binary_data = compress_image_to_base64(new_img, image_config['imageCompressionRate'])
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
    except Exception as e:
        raise RuntimeError(f"Upload image failed: {e}")
    

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

def is_data_uri_an_image(data_uri: str):
    # Check if the data URI starts with 'data:image' and contains an image format (e.g., jpeg, png, gif)
    if not re.match(r'data:image/(\w+);base64,', data_uri):
        raise ValueError("Invalid data URI image.")
        # Extract the image format from the data URI
    image_format = re.match(r'data:image/(\w+);base64,', data_uri).group(1)
    # Check if the image format is one of the allowed formats (jpg, jpeg, png, gif)
    if image_format.lower() not in ['jpeg', 'jpg', 'png', 'gif']:
        raise ValueError("Invalid image format (from mime file type).")

def is_accepted_format(binary_data: bytes) -> bool:
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
    data = data_uri.split(",")[1]
    data = base64.b64decode(data)
    return data

def get_orientation(data: bytes) -> int:
    if data[:2] != b'\xFF\xD8':
        raise Exception('NotJpeg')
    with Image.open(data) as img:
        exif_data = img._getexif()
        if exif_data is not None:
            orientation = exif_data.get(274)  # 274 corresponds to the orientation tag in EXIF
            if orientation is not None:
                return orientation

def process_image(orientation: int, img: Image.Image, new_width: int, new_height: int) -> Image.Image:
    # Initialize the canvas
    new_img = Image.new("RGB", (new_width, new_height), color="#FFFFFF")
    if orientation:
        if orientation > 4:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if orientation in [3, 4]:
            img = img.transpose(Image.ROTATE_180)
        if orientation in [5, 6]:
            img = img.transpose(Image.ROTATE_270)
        if orientation in [7, 8]:
            img = img.transpose(Image.ROTATE_90)
    new_img.paste(img, (0, 0))
    return new_img
    
def compress_image_to_base64(image: Image.Image, compression_rate: float) -> str:
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="JPEG", quality=int(compression_rate * 100))
    return base64.b64encode(output_buffer.getvalue()).decode('utf-8')