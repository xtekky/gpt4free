from __future__ import annotations

import asyncio
import uuid
import json
import os

try:
    from py_arkose_generator.arkose import get_values_for_request
    from async_property import async_cached_property
    has_requirements = True
except ImportError:
    async_cached_property = property
    has_requirements = False
try:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    pass

from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, get_cookies
from ...webdriver import get_browser, get_driver_cookies
from ...typing import AsyncResult, Messages, Cookies, ImageType
from ...requests import StreamSession
from ...image import to_image, to_bytes, ImageResponse, ImageRequest
from ...errors import MissingRequirementsError, MissingAuthError


class OpenaiChat(AsyncGeneratorProvider, ProviderModelMixin):
    """A class for creating and managing conversations with OpenAI chat service"""
    
    url = "https://chat.openai.com"
    working = True
    needs_auth = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    default_model = None
    models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-gizmo"]
    model_aliases = {"text-davinci-002-render-sha": "gpt-3.5-turbo"}
    _cookies: dict = {}

    @classmethod
    async def create(
        cls,
        prompt: str = None,
        model: str = "",
        messages: Messages = [],
        history_disabled: bool = False,
        action: str = "next",
        conversation_id: str = None,
        parent_id: str = None,
        image: ImageType = None,
        **kwargs
    ) -> Response:
        """
        Create a new conversation or continue an existing one
        
        Args:
            prompt: The user input to start or continue the conversation
            model: The name of the model to use for generating responses
            messages: The list of previous messages in the conversation
            history_disabled: A flag indicating if the history and training should be disabled
            action: The type of action to perform, either "next", "continue", or "variant"
            conversation_id: The ID of the existing conversation, if any
            parent_id: The ID of the parent message, if any
            image: The image to include in the user input, if any
            **kwargs: Additional keyword arguments to pass to the generator
        
        Returns:
            A Response object that contains the generator, action, messages, and options
        """
        # Add the user input to the messages list
        if prompt:
            messages.append({
                "role": "user",
                "content": prompt
            })
        generator = cls.create_async_generator(
            model,
            messages,
            history_disabled=history_disabled,
            action=action,
            conversation_id=conversation_id,
            parent_id=parent_id,
            image=image,
            response_fields=True,
            **kwargs
        )
        return Response(
            generator,
            action,
            messages,
            kwargs
        )
    
    @classmethod
    async def upload_image(
        cls,
        session: StreamSession,
        headers: dict,
        image: ImageType,
        image_name: str = None
    ) -> ImageRequest:
        """
        Upload an image to the service and get the download URL
        
        Args:
            session: The StreamSession object to use for requests
            headers: The headers to include in the requests
            image: The image to upload, either a PIL Image object or a bytes object
        
        Returns:
            An ImageRequest object that contains the download URL, file name, and other data
        """
        # Convert the image to a PIL Image object and get the extension
        image = to_image(image)
        extension = image.format.lower()
        # Convert the image to a bytes object and get the size
        data_bytes = to_bytes(image)
        data = {
            "file_name": image_name if image_name else f"{image.width}x{image.height}.{extension}",
            "file_size": len(data_bytes),
            "use_case":	"multimodal"
        }
        # Post the image data to the service and get the image data
        async with session.post(f"{cls.url}/backend-api/files", json=data, headers=headers) as response:
            response.raise_for_status()
            image_data = {
                **data,
                **await response.json(),
                "mime_type": f"image/{extension}",
                "extension": extension,
                "height": image.height,
                "width": image.width
            }
        # Put the image bytes to the upload URL and check the status
        async with session.put(
            image_data["upload_url"],
            data=data_bytes,
            headers={
                "Content-Type": image_data["mime_type"],
                "x-ms-blob-type": "BlockBlob"
            }
        ) as response:
            response.raise_for_status()
        # Post the file ID to the service and get the download URL
        async with session.post(
            f"{cls.url}/backend-api/files/{image_data['file_id']}/uploaded",
            json={},
            headers=headers
        ) as response:
            response.raise_for_status()
            image_data["download_url"] = (await response.json())["download_url"]
        return ImageRequest(image_data)
    
    @classmethod
    async def get_default_model(cls, session: StreamSession, headers: dict):
        """
        Get the default model name from the service
        
        Args:
            session: The StreamSession object to use for requests
            headers: The headers to include in the requests
        
        Returns:
            The default model name as a string
        """
        if not cls.default_model:
            async with session.get(f"{cls.url}/backend-api/models", headers=headers) as response:
                data = await response.json()
                if "categories" in data:
                    cls.default_model = data["categories"][-1]["default_model"]
                else:
                    raise RuntimeError(f"Response: {data}")
        return cls.default_model
    
    @classmethod
    def create_messages(cls, prompt: str, image_request: ImageRequest = None):
        """
        Create a list of messages for the user input
        
        Args:
            prompt: The user input as a string
            image_response: The image response object, if any
        
        Returns:
            A list of messages with the user input and the image, if any
        """
        # Check if there is an image response
        if not image_request:
            # Create a content object with the text type and the prompt
            content = {"content_type": "text", "parts": [prompt]}
        else:
            # Create a content object with the multimodal text type and the image and the prompt
            content = {
                "content_type": "multimodal_text",
                "parts": [{
                    "asset_pointer": f"file-service://{image_request.get('file_id')}",
                    "height": image_request.get("height"),
                    "size_bytes": image_request.get("file_size"),
                    "width": image_request.get("width"),
                }, prompt]
            }
        # Create a message object with the user role and the content
        messages = [{
            "id": str(uuid.uuid4()),
            "author": {"role": "user"},
            "content": content,
        }]
        # Check if there is an image response
        if image_request:
            # Add the metadata object with the attachments
            messages[0]["metadata"] = {
                "attachments": [{
                    "height": image_request.get("height"),
                    "id": image_request.get("file_id"),
                    "mimeType": image_request.get("mime_type"),
                    "name": image_request.get("file_name"),
                    "size": image_request.get("file_size"),
                    "width": image_request.get("width"),
                }]
            }
        return messages
    
    @classmethod
    async def get_generated_image(cls, session: StreamSession, headers: dict, line: dict) -> ImageResponse:
        """
        Retrieves the image response based on the message content.

        This method processes the message content to extract image information and retrieves the 
        corresponding image from the backend API. It then returns an ImageResponse object containing 
        the image URL and the prompt used to generate the image.

        Args:
            session (StreamSession): The StreamSession object used for making HTTP requests.
            headers (dict): HTTP headers to be used for the request.
            line (dict): A dictionary representing the line of response that contains image information.

        Returns:
            ImageResponse: An object containing the image URL and the prompt, or None if no image is found.

        Raises:
            RuntimeError: If there'san error in downloading the image, including issues with the HTTP request or response.
        """
        if "parts" not in line["message"]["content"]:
            return
        first_part = line["message"]["content"]["parts"][0]
        if "asset_pointer" not in first_part or "metadata" not in first_part:
            return
        file_id = first_part["asset_pointer"].split("file-service://", 1)[1]
        prompt = first_part["metadata"]["dalle"]["prompt"]
        try:
            async with session.get(f"{cls.url}/backend-api/files/{file_id}/download", headers=headers) as response:
                response.raise_for_status()
                download_url = (await response.json())["download_url"]
                return ImageResponse(download_url, prompt)
        except Exception as e:
            raise RuntimeError(f"Error in downloading image: {e}")

    @classmethod
    async def delete_conversation(cls, session: StreamSession, headers: dict, conversation_id: str):
        """
        Deletes a conversation by setting its visibility to False.

        This method sends an HTTP PATCH request to update the visibility of a conversation. 
        It's used to effectively delete a conversation from being accessed or displayed in the future.

        Args:
            session (StreamSession): The StreamSession object used for making HTTP requests.
            headers (dict): HTTP headers to be used for the request.
            conversation_id (str): The unique identifier of the conversation to be deleted.

        Raises:
            HTTPError: If the HTTP request fails or returns an unsuccessful status code.
        """
        async with session.patch(
            f"{cls.url}/backend-api/conversation/{conversation_id}",
            json={"is_visible": False},
            headers=headers
        ) as response:
            response.raise_for_status()

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        access_token: str = None,
        cookies: Cookies = None,
        auto_continue: bool = False,
        history_disabled: bool = True,
        action: str = "next",
        conversation_id: str = None,
        parent_id: str = None,
        image: ImageType = None,
        response_fields: bool = False,
        **kwargs
    ) -> AsyncResult:
        """
        Create an asynchronous generator for the conversation.

        Args:
            model (str): The model name.
            messages (Messages): The list of previous messages.
            proxy (str): Proxy to use for requests.
            timeout (int): Timeout for requests.
            access_token (str): Access token for authentication.
            cookies (dict): Cookies to use for authentication.
            auto_continue (bool): Flag to automatically continue the conversation.
            history_disabled (bool): Flag to disable history and training.
            action (str): Type of action ('next', 'continue', 'variant').
            conversation_id (str): ID of the conversation.
            parent_id (str): ID of the parent message.
            image (ImageType): Image to include in the conversation.
            response_fields (bool): Flag to include response fields in the output.
            **kwargs: Additional keyword arguments.

        Yields:
            AsyncResult: Asynchronous results from the generator.

        Raises:
            RuntimeError: If an error occurs during processing.
        """
        if not has_requirements:
            raise MissingRequirementsError('Install "py-arkose-generator" and "async_property" package')
        if not parent_id:
            parent_id = str(uuid.uuid4())
        if not cookies:
            cookies = cls._cookies or get_cookies("chat.openai.com", False)
        if not access_token and "access_token" in cookies:
            access_token = cookies["access_token"]
        if not access_token:
            login_url = os.environ.get("G4F_LOGIN_URL")
            if login_url:
                yield f"Please login: [ChatGPT]({login_url})\n\n"
            try:
                access_token, cookies = cls.browse_access_token(proxy)
            except MissingRequirementsError:
                raise MissingAuthError(f'Missing "access_token"')
            cls._cookies = cookies

        auth_headers = {"Authorization": f"Bearer {access_token}"}
        async with StreamSession(
            proxies={"https": proxy},
            impersonate="chrome110",
            timeout=timeout,
            headers={"Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items())}
        ) as session:
            try:
                image_response = None
                if image:
                    image_response = await cls.upload_image(session, auth_headers, image, kwargs.get("image_name"))
            except Exception as e:
                yield e
            end_turn = EndTurn()
            model = cls.get_model(model or await cls.get_default_model(session, auth_headers))
            model = "text-davinci-002-render-sha" if model == "gpt-3.5-turbo" else model
            while not end_turn.is_end:
                arkose_token = await cls.get_arkose_token(session)
                data = {
                    "action": action,
                    "arkose_token": arkose_token,
                    "conversation_mode": {"kind": "primary_assistant"},
                    "force_paragen": False,
                    "force_rate_limit": False,
                    "conversation_id": conversation_id,
                    "parent_message_id": parent_id,
                    "model": model,
                    "history_and_training_disabled": history_disabled and not auto_continue,
                }
                if action != "continue":
                    prompt = format_prompt(messages) if not conversation_id else messages[-1]["content"]
                    data["messages"] = cls.create_messages(prompt, image_response)
                async with session.post(
                    f"{cls.url}/backend-api/conversation",
                    json=data,
                    headers={
                        "Accept": "text/event-stream",
                        "OpenAI-Sentinel-Arkose-Token": arkose_token,
                        **auth_headers
                    }
                ) as response:
                    if not response.ok:
                        raise RuntimeError(f"Response {response.status}: {await response.text()}")
                    last_message: int = 0
                    async for line in response.iter_lines():
                        if not line.startswith(b"data: "):
                            continue
                        elif line.startswith(b"data: [DONE]"):
                            break
                        try:
                            line = json.loads(line[6:])
                        except:
                            continue
                        if "message" not in line:
                            continue
                        if "error" in line and line["error"]:
                            raise RuntimeError(line["error"])
                        if "message_type" not in line["message"]["metadata"]:
                            continue
                        try:
                            image_response = await cls.get_generated_image(session, auth_headers, line)
                            if image_response:
                                yield image_response
                        except Exception as e:
                            yield e
                        if line["message"]["author"]["role"] != "assistant":
                            continue
                        if line["message"]["content"]["content_type"] != "text":
                            continue
                        if line["message"]["metadata"]["message_type"] not in ("next", "continue", "variant"):
                            continue
                        conversation_id = line["conversation_id"]
                        parent_id = line["message"]["id"]
                        if response_fields:
                            response_fields = False
                            yield ResponseFields(conversation_id, parent_id, end_turn)
                        if "parts" in line["message"]["content"]:
                            new_message = line["message"]["content"]["parts"][0]
                            if len(new_message) > last_message:
                                yield new_message[last_message:]
                            last_message = len(new_message)
                        if "finish_details" in line["message"]["metadata"]:
                            if line["message"]["metadata"]["finish_details"]["type"] == "stop":
                                end_turn.end()
                if not auto_continue:
                    break
                action = "continue"
                await asyncio.sleep(5)
            if history_disabled and auto_continue:
                await cls.delete_conversation(session, auth_headers, conversation_id)

    @classmethod
    def browse_access_token(cls, proxy: str = None, timeout: int = 1200) -> tuple[str, dict]:
        """
        Browse to obtain an access token.

        Args:
            proxy (str): Proxy to use for browsing.

        Returns:
            tuple[str, dict]: A tuple containing the access token and cookies.
        """
        driver = get_browser(proxy=proxy)
        try:
            driver.get(f"{cls.url}/")
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.ID, "prompt-textarea")))
            access_token = driver.execute_script(
                "let session = await fetch('/api/auth/session');"
                "let data = await session.json();"
                "let accessToken = data['accessToken'];"
                "let expires = new Date(); expires.setTime(expires.getTime() + 60 * 60 * 4);"
                "document.cookie = 'access_token=' + accessToken + ';expires=' + expires.toUTCString() + ';path=/';"
                "return accessToken;"
            )
            return access_token, get_driver_cookies(driver)
        finally:
            driver.close() 

    @classmethod
    async def get_arkose_token(cls, session: StreamSession) -> str:
        """
        Obtain an Arkose token for the session.

        Args:
            session (StreamSession): The session object.

        Returns:
            str: The Arkose token.

        Raises:
            RuntimeError: If unable to retrieve the token.
        """
        config = {
            "pkey": "3D86FBBA-9D22-402A-B512-3420086BA6CC",
            "surl": "https://tcr9i.chat.openai.com",
            "headers": {
                "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
            },
            "site": cls.url,
        }
        args_for_request = get_values_for_request(config)
        async with session.post(**args_for_request) as response:
            response.raise_for_status()
            decoded_json = await response.json()
            if "token" in decoded_json:
                return decoded_json["token"]
            raise RuntimeError(f"Response: {decoded_json}")

class EndTurn:
    """
    Class to represent the end of a conversation turn.
    """
    def __init__(self):
        self.is_end = False

    def end(self):
        self.is_end = True

class ResponseFields:
    """
    Class to encapsulate response fields.
    """
    def __init__(self, conversation_id: str, message_id: str, end_turn: EndTurn):
        self.conversation_id = conversation_id
        self.message_id = message_id
        self._end_turn = end_turn

class Response():
    """
    Class to encapsulate a response from the chat service.
    """
    def __init__(
        self,
        generator: AsyncResult,
        action: str,
        messages: Messages,
        options: dict
    ):
        self._generator = generator
        self.action = action
        self.is_end = False
        self._message = None
        self._messages = messages
        self._options = options
        self._fields = None

    async def generator(self):
        if self._generator:
            self._generator = None
            chunks = []
            async for chunk in self._generator:
                if isinstance(chunk, ResponseFields):
                    self._fields = chunk
                else:
                    yield chunk
                    chunks.append(str(chunk))
            self._message = "".join(chunks)
            if not self._fields:
                raise RuntimeError("Missing response fields")
            self.is_end = self._fields._end_turn.is_end

    def __aiter__(self):
        return self.generator()

    @async_cached_property
    async def message(self) -> str:
        await self.generator()
        return self._message

    async def get_fields(self):
        await self.generator()
        return {"conversation_id": self._fields.conversation_id, "parent_id": self._fields.message_id}

    async def next(self, prompt: str, **kwargs) -> Response:
        return await OpenaiChat.create(
            **self._options,
            prompt=prompt,
            messages=await self.messages,
            action="next",
            **await self.get_fields(),
            **kwargs
        )

    async def do_continue(self, **kwargs) -> Response:
        fields = await self.get_fields()
        if self.is_end:
            raise RuntimeError("Can't continue message. Message already finished.")
        return await OpenaiChat.create(
            **self._options,
            messages=await self.messages,
            action="continue",
            **fields,
            **kwargs
        )

    async def variant(self, **kwargs) -> Response:
        if self.action != "next":
            raise RuntimeError("Can't create variant from continue or variant request.")
        return await OpenaiChat.create(
            **self._options,
            messages=self._messages,
            action="variant",
            **await self.get_fields(),
            **kwargs
        )

    @async_cached_property
    async def messages(self):
        messages = self._messages
        messages.append({"role": "assistant", "content": await self.message})
        return messages