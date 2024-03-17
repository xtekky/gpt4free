from __future__ import annotations

import re
import asyncio

from .. import debug
from ..typing import CreateResult, Messages
from .types import BaseProvider, ProviderType
from ..image import ImageResponse

system_message = """
You can generate images, pictures, photos or img with the DALL-E 3 image generator.
To generate an image with a prompt, do this:

<img data-prompt=\"keywords for the image\">

Never use own image links. Don't wrap it in backticks.
It is important to use a only a img tag with a prompt.

<img data-prompt=\"image caption\">
"""

class CreateImagesProvider(BaseProvider):
    """
    Provider class for creating images based on text prompts.

    This provider handles image creation requests embedded within message content, 
    using provided image creation functions.

    Attributes:
        provider (ProviderType): The underlying provider to handle non-image related tasks.
        create_images (callable): A function to create images synchronously.
        create_images_async (callable): A function to create images asynchronously.
        system_message (str): A message that explains the image creation capability.
        include_placeholder (bool): Flag to determine whether to include the image placeholder in the output.
        __name__ (str): Name of the provider.
        url (str): URL of the provider.
        working (bool): Indicates if the provider is operational.
        supports_stream (bool): Indicates if the provider supports streaming.
    """

    def __init__(
        self,
        provider: ProviderType,
        create_images: callable,
        create_async: callable,
        system_message: str = system_message,
        include_placeholder: bool = True
    ) -> None:
        """
        Initializes the CreateImagesProvider.

        Args:
            provider (ProviderType): The underlying provider.
            create_images (callable): Function to create images synchronously.
            create_async (callable): Function to create images asynchronously.
            system_message (str, optional): System message to be prefixed to messages. Defaults to a predefined message.
            include_placeholder (bool, optional): Whether to include image placeholders in the output. Defaults to True.
        """
        self.provider = provider
        self.create_images = create_images
        self.create_images_async = create_async
        self.system_message = system_message
        self.include_placeholder = include_placeholder
        self.__name__ = provider.__name__
        self.url = provider.url
        self.working = provider.working
        self.supports_stream = provider.supports_stream

    def create_completion(
        self,
        model: str,
        messages: Messages,
        stream: bool = False,
        **kwargs
    ) -> CreateResult:
        """
        Creates a completion result, processing any image creation prompts found within the messages.

        Args:
            model (str): The model to use for creation.
            messages (Messages): The messages to process, which may contain image prompts.
            stream (bool, optional): Indicates whether to stream the results. Defaults to False.
            **kwargs: Additional keywordarguments for the provider.

        Yields:
            CreateResult: Yields chunks of the processed messages, including image data if applicable.

        Note:
            This method processes messages to detect image creation prompts. When such a prompt is found, 
            it calls the synchronous image creation function and includes the resulting image in the output.
        """
        messages.insert(0, {"role": "system", "content": self.system_message})
        buffer = ""
        for chunk in self.provider.create_completion(model, messages, stream, **kwargs):
            if isinstance(chunk, ImageResponse):
                yield chunk
            elif isinstance(chunk, str) and buffer or "<" in chunk:
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
                        if debug.logging:
                            print(f"Create images with prompt: {prompt}")
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
        """
        Asynchronously creates a response, processing any image creation prompts found within the messages.

        Args:
            model (str): The model to use for creation.
            messages (Messages): The messages to process, which may contain image prompts.
            **kwargs: Additional keyword arguments for the provider.

        Returns:
            str: The processed response string, including asynchronously generated image data if applicable.

        Note:
            This method processes messages to detect image creation prompts. When such a prompt is found, 
            it calls the asynchronous image creation function and includes the resulting image in the output.
        """
        messages.insert(0, {"role": "system", "content": self.system_message})
        response = await self.provider.create_async(model, messages, **kwargs)
        matches = re.findall(r'(<img data-prompt="(.*?)">)', response)
        results = []
        placeholders = []
        for placeholder, prompt in matches:
            if placeholder not in placeholders:
                if debug.logging:
                    print(f"Create images with prompt: {prompt}")
                results.append(self.create_images_async(prompt))
                placeholders.append(placeholder)
        results = await asyncio.gather(*results)
        for idx, result in enumerate(results):
            placeholder = placeholder[idx]
            if self.include_placeholder:
                result = placeholder + result
            response = response.replace(placeholder, result)
        return response