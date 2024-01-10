from __future__ import annotations

import re
import asyncio
from ..typing import CreateResult, Messages
from ..base_provider import BaseProvider, ProviderType

system_message = """
You can generate custom images with the DALL-E 3 image generator.
To generate a image with a prompt, do this:
<img data-prompt=\"keywords for the image\">
Don't use images with data uri. It is important to use a prompt instead.
<img data-prompt=\"image caption\">
"""

class CreateImagesProvider(BaseProvider):
    def __init__(
        self,
        provider: ProviderType,
        create_images: callable,
        create_async: callable,
        system_message: str = system_message,
        include_placeholder: bool = True
    ) -> None:
        self.provider = provider
        self.create_images = create_images
        self.create_images_async = create_async
        self.system_message = system_message
        self.__name__ = provider.__name__
        self.working = provider.working
        self.supports_stream = provider.supports_stream
        self.include_placeholder = include_placeholder
        if hasattr(provider, "url"):
            self.url = provider.url

    def create_completion(
        self,
        model: str,
        messages: Messages,
        stream: bool = False,
        **kwargs
    ) -> CreateResult:
        messages.insert(0, {"role": "system", "content": self.system_message})
        buffer = ""
        for chunk in self.provider.create_completion(model, messages, stream, **kwargs):
            if buffer or "<" in chunk:
                buffer += chunk
                if ">" in buffer:
                    match = re.search(r'<img data-prompt="(.*?)">', buffer)
                    if match:
                        placeholder, prompt = match.group(0), match.group(1)
                        start, append = buffer.split(placeholder, 1)
                        if start:
                            yield start
                        if self.include_placeholder:
                            yield placeholder
                        yield from self.create_images(prompt)
                        if append:
                            yield append
                    else:
                        yield buffer
                    buffer = ""
            else:
                yield chunk

    async def create_async(
        self,
        model: str,
        messages: Messages,
        **kwargs
    ) -> str:
        messages.insert(0, {"role": "system", "content": self.system_message})
        response = await self.provider.create_async(model, messages, **kwargs)
        matches = re.findall(r'(<img data-prompt="(.*?)">)', result)
        results = []
        for _, prompt in matches:
            results.append(self.create_images_async(prompt))
        results = await asyncio.gather(*results)
        for idx, result in enumerate(results):
            placeholder = matches[idx][0]
            if self.include_placeholder:
                result = placeholder + result
            response = response.replace(placeholder, result)
        return result