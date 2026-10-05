"""Resolve image assets from the requested turn of a ChatGPT conversation."""

import asyncio

from ...errors import ResponseError
from ...providers.response import ImagePreview, ImageResponse
from ...requests.raise_for_status import raise_for_status


async def download_image(session, auth, element, prompt, conversation_id, status, base_url='https://chatgpt.com'):
    if prompt is None and isinstance(element, dict):
        prompt = element.get('metadata', {}).get('dalle', {}).get('prompt')
    pointer = element.get('asset_pointer') if isinstance(element, dict) else element
    if not isinstance(pointer, str):
        raise ResponseError('Image asset pointer is missing')
    if pointer.startswith('sediment://'):
        if not conversation_id:
            raise ResponseError('Conversation id is required for this image')
        asset = pointer[len('sediment://'):]
        url = f'{base_url}/backend-api/conversation/{conversation_id}/attachment/{asset}/download'
    elif pointer.startswith('file-service://'):
        asset = pointer[len('file-service://'):]
        url = f'{base_url}/backend-api/files/{asset}/download'
    else:
        raise ResponseError('Unsupported ChatGPT image asset pointer')
    if not asset or any(char in asset for char in '/\\?#'):
        raise ResponseError('Invalid ChatGPT image asset id')
    async with session.get(url, headers=auth.headers) as response:
        if response.status == 404:
            return None  # An asset can become downloadable after the stream ends.
        await raise_for_status(response)
        url = (await response.json()).get('download_url')
    if not url:
        return None
    result_type = ImagePreview if pointer.startswith('sediment://') and status != 'finished_successfully' else ImageResponse
    return result_type([url], prompt or '', {'status': status, 'headers': auth.headers})


def turn_messages(record, user_message_id):
    mapping = record.get('mapping') or {}
    node_id = record.get('current_node')
    messages, seen = [], set()
    while node_id and node_id not in seen and len(seen) < len(mapping):
        seen.add(node_id)
        node = mapping.get(node_id) or {}
        message = node.get('message') or {}
        if message:
            messages.append(message)
        if node_id == user_message_id or message.get('id') == user_message_id:
            return list(reversed(messages))
        node_id = node.get('parent')
    return []  # Never use an image from another turn or branch.


async def poll_images(session, auth, conversation_id, user_message_id, prompt='', timeout=180,
                      poll_interval=2, base_url='https://chatgpt.com'):
    if not conversation_id or not user_message_id:
        raise ResponseError('ChatGPT returned no conversation or message id for image generation')
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    async def read_conversation():
        async with session.get(f'{base_url}/backend-api/conversation/{conversation_id}', headers=auth.headers) as response:
            await raise_for_status(response)
            return await response.json()

    while loop.time() < deadline:
        record = await asyncio.wait_for(read_conversation(), timeout=max(0, deadline - loop.time()))
        images, seen = [], set()
        messages = turn_messages(record, user_message_id)
        for message in messages:
            if message.get('author', {}).get('role') not in ('assistant', 'tool'):
                continue
            if message.get('status') != 'finished_successfully':
                continue
            content = message.get('content') or {}
            for part in content.get('parts') or []:
                if not isinstance(part, dict) or part.get('content_type') != 'image_asset_pointer':
                    continue
                pointer = part.get('asset_pointer')
                if pointer in seen:
                    continue
                seen.add(pointer)
                image = await asyncio.wait_for(
                    download_image(session, auth, part, prompt, conversation_id, message['status'], base_url),
                    timeout=max(0, deadline - loop.time()),
                )
                if image:
                    images.append(image)
        if images:
            for image in images:
                yield image
            return
        if messages:
            last = messages[-1]
            if last.get('status') in ('failed', 'cancelled'):
                raise ResponseError('ChatGPT image generation failed or was cancelled')
            if (last.get('author', {}).get('role') == 'assistant' and last.get('recipient', 'all') == 'all'
                    and last.get('status') == 'finished_successfully'
                    and last.get('content', {}).get('content_type') == 'text'):
                raise ResponseError('ChatGPT completed this turn without generating an image')
        await asyncio.sleep(min(poll_interval, max(0, deadline - loop.time())))
    raise TimeoutError('Timed out waiting for the ChatGPT image')
