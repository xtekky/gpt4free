from __future__ import annotations

import asyncio
import uuid
import json
import base64
import time
from aiohttp import ClientWebSocketResponse
from copy import copy

try:
    import webview
    has_webview = True
except ImportError:
    has_webview = False

try:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    pass

from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...webdriver import get_browser
from ...typing import AsyncResult, Messages, Cookies, ImageType, Union, AsyncIterator
from ...requests import get_args_from_browser, raise_for_status
from ...requests.aiohttp import StreamSession
from ...image import to_image, to_bytes, ImageResponse, ImageRequest
from ...errors import MissingAuthError
from ...providers.conversation import BaseConversation
from ..openai.har_file import getArkoseAndAccessToken
from ... import debug

class OpenaiChat(AsyncGeneratorProvider, ProviderModelMixin):
    """A class for creating and managing conversations with OpenAI chat service"""

    url = "https://chat.openai.com"
    working = True
    needs_auth = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_message_history = True
    supports_system_message = True
    default_model = None
    models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-gizmo"]
    model_aliases = {"text-davinci-002-render-sha": "gpt-3.5-turbo", "": "gpt-3.5-turbo"}
    _api_key: str = None
    _headers: dict = None
    _cookies: Cookies = None
    _expires: int = None

    @classmethod
    async def create(
        cls,
        prompt: str = None,
        model: str = "",
        messages: Messages = [],
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
        if prompt is not None:
            messages.append({
                "role": "user",
                "content": prompt
            })
        generator = cls.create_async_generator(
            model,
            messages,
            return_conversation=True,
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
            cls._update_request_args()
            await raise_for_status(response)
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
            await raise_for_status(response)
        # Post the file ID to the service and get the download URL
        async with session.post(
            f"{cls.url}/backend-api/files/{image_data['file_id']}/uploaded",
            json={},
            headers=headers
        ) as response:
            cls._update_request_args(session)
            await raise_for_status(response)
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
                cls._update_request_args(session)
                await raise_for_status(response)
                data = await response.json()
                if "categories" in data:
                    cls.default_model = data["categories"][-1]["default_model"]
                    return cls.default_model 
                raise RuntimeError(f"Response: {data}")
        return cls.default_model

    @classmethod
    def create_messages(cls, messages: Messages, image_request: ImageRequest = None):
        """
        Create a list of messages for the user input
        
        Args:
            prompt: The user input as a string
            image_response: The image response object, if any
        
        Returns:
            A list of messages with the user input and the image, if any
        """
        # Create a message object with the user role and the content
        messages = [{
            "id": str(uuid.uuid4()),
            "author": {"role": message["role"]},
            "content": {"content_type": "text", "parts": [message["content"]]},
        } for message in messages]

        # Check if there is an image response
        if image_request is not None:
            # Change content in last user message
            messages[-1]["content"] = {
                "content_type": "multimodal_text",
                "parts": [{
                    "asset_pointer": f"file-service://{image_request.get('file_id')}",
                    "height": image_request.get("height"),
                    "size_bytes": image_request.get("file_size"),
                    "width": image_request.get("width"),
                }, messages[-1]["content"]["parts"][0]]
            }
            # Add the metadata object with the attachments
            messages[-1]["metadata"] = {
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
        if first_part["metadata"] is None:
            return
        prompt = first_part["metadata"]["dalle"]["prompt"]
        file_id = first_part["asset_pointer"].split("file-service://", 1)[1]
        try:
            async with session.get(f"{cls.url}/backend-api/files/{file_id}/download", headers=headers) as response:
                cls._update_request_args(session)
                await raise_for_status(response)
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
            cls._update_request_args(session)
            ...

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        api_key: str = None,
        cookies: Cookies = None,
        auto_continue: bool = False,
        history_disabled: bool = True,
        action: str = "next",
        conversation_id: str = None,
        conversation: Conversation = None,
        parent_id: str = None,
        image: ImageType = None,
        image_name: str = None,
        return_conversation: bool = False,
        **kwargs
    ) -> AsyncResult:
        """
        Create an asynchronous generator for the conversation.

        Args:
            model (str): The model name.
            messages (Messages): The list of previous messages.
            proxy (str): Proxy to use for requests.
            timeout (int): Timeout for requests.
            api_key (str): Access token for authentication.
            cookies (dict): Cookies to use for authentication.
            auto_continue (bool): Flag to automatically continue the conversation.
            history_disabled (bool): Flag to disable history and training.
            action (str): Type of action ('next', 'continue', 'variant').
            conversation_id (str): ID of the conversation.
            parent_id (str): ID of the parent message.
            image (ImageType): Image to include in the conversation.
            return_conversation (bool): Flag to include response fields in the output.
            **kwargs: Additional keyword arguments.

        Yields:
            AsyncResult: Asynchronous results from the generator.

        Raises:
            RuntimeError: If an error occurs during processing.
        """
        if parent_id is None:
            parent_id = str(uuid.uuid4())

        async with StreamSession(
            proxies={"https": proxy},
            impersonate="chrome",
            timeout=timeout
        ) as session:
            api_key = kwargs["access_token"] if "access_token" in kwargs else api_key

            if api_key is not None:
                cls._create_request_args(cookies)
                cls._set_api_key(api_key)

            if cls.default_model is None and cls._headers is not None:
                try:
                    if not model:
                        cls.default_model = cls.get_model(await cls.get_default_model(session, cls._headers))
                    else:
                        cls.default_model = cls.get_model(model)
                except Exception as e:
                    if debug.logging:
                        print("OpenaiChat: Load default_model failed")
                        print(f"{e.__class__.__name__}: {e}")

            arkose_token = None
            if cls.default_model is None:
                arkose_token, api_key, cookies = await getArkoseAndAccessToken(proxy)
                cls._create_request_args(cookies)
                cls._set_api_key(api_key)
                cls.default_model = cls.get_model(await cls.get_default_model(session, cls._headers))

            async with session.post(
                f"{cls.url}/backend-api/sentinel/chat-requirements",
                json={"conversation_mode_kind": "primary_assistant"},
                headers=cls._headers
            ) as response:
                cls._update_request_args(session)
                await raise_for_status(response)
                data = await response.json()
                blob = data["arkose"]["dx"]
                need_arkose = data["arkose"]["required"]
                chat_token = data["token"]

            if need_arkose and arkose_token is None:
                arkose_token, api_key, cookies = await getArkoseAndAccessToken(proxy)
                cls._create_request_args(cookies)
                cls._set_api_key(api_key)
                if arkose_token is None:
                    raise MissingAuthError("No arkose token found in .har file")

            try:
                image_request = await cls.upload_image(session, cls._headers, image, image_name) if image else None
            except Exception as e:
                if debug.logging:
                    print("OpenaiChat: Upload image failed")
                    print(f"{e.__class__.__name__}: {e}")

            model = cls.get_model(model).replace("gpt-3.5-turbo", "text-davinci-002-render-sha")
            fields = Conversation(conversation_id, parent_id) if conversation is None else copy(conversation)
            fields.finish_reason = None
            while fields.finish_reason is None:
                websocket_request_id = str(uuid.uuid4())
                data = {
                    "action": action,
                    "conversation_mode": {"kind": "primary_assistant"},
                    "force_paragen": False,
                    "force_rate_limit": False,
                    "conversation_id": fields.conversation_id,
                    "parent_message_id": fields.message_id,
                    "model": model,
                    "history_and_training_disabled": history_disabled and not auto_continue and not return_conversation,
                    "websocket_request_id": websocket_request_id
                }
                if action != "continue":
                    messages = messages if conversation_id is None else [messages[-1]]
                    data["messages"] = cls.create_messages(messages, image_request)
                headers = {
                    "Accept": "text/event-stream",
                    "OpenAI-Sentinel-Chat-Requirements-Token": chat_token,
                    **cls._headers
                }
                if need_arkose:
                    headers["OpenAI-Sentinel-Arkose-Token"] = arkose_token
                async with session.post(
                    f"{cls.url}/backend-api/conversation",
                    json=data,
                    headers=headers
                ) as response:
                    cls._update_request_args(session)
                    await raise_for_status(response)
                    async for chunk in cls.iter_messages_chunk(response.iter_lines(), session, fields):
                        if return_conversation:
                            history_disabled = False
                            return_conversation = False
                            yield fields
                        yield chunk
                if not auto_continue:
                    break
                action = "continue"
                await asyncio.sleep(5)
            if history_disabled and auto_continue:
                await cls.delete_conversation(session, cls._headers, fields.conversation_id)

    @staticmethod
    async def iter_messages_ws(ws: ClientWebSocketResponse, conversation_id: str, is_curl: bool) -> AsyncIterator:
        while True:
            if is_curl:
                message = json.loads(ws.recv()[0])
            else:
                message = await ws.receive_json()
            if message["conversation_id"] == conversation_id:
                yield base64.b64decode(message["body"])

    @classmethod
    async def iter_messages_chunk(
        cls,
        messages: AsyncIterator,
        session: StreamSession,
        fields: Conversation
    ) -> AsyncIterator:
        last_message: int = 0
        async for message in messages:
            if message.startswith(b'{"wss_url":'):
                message = json.loads(message)
                ws = await session.ws_connect(message["wss_url"])
                try:
                    async for chunk in cls.iter_messages_chunk(
                        cls.iter_messages_ws(ws, message["conversation_id"], hasattr(ws, "recv")),
                        session, fields
                    ):
                        yield chunk
                finally:
                    await ws.aclose() if hasattr(ws, "aclose") else await ws.close()
                break
            async for chunk in cls.iter_messages_line(session, message, fields):
                if fields.finish_reason is not None:
                    break
                elif isinstance(chunk, str):
                    if len(chunk) > last_message:
                        yield chunk[last_message:]
                    last_message = len(chunk)
                else:
                    yield chunk
            if fields.finish_reason is not None:
                break

    @classmethod
    async def iter_messages_line(cls, session: StreamSession, line: bytes, fields: Conversation) -> AsyncIterator:
        if not line.startswith(b"data: "):
            return
        elif line.startswith(b"data: [DONE]"):
            if fields.finish_reason is None:
                fields.finish_reason = "error"
            return
        try:
            line = json.loads(line[6:])
        except:
            return
        if "message" not in line:
            return
        if "error" in line and line["error"]:
            raise RuntimeError(line["error"])
        if "message_type" not in line["message"]["metadata"]:
            return
        try:
            image_response = await cls.get_generated_image(session, cls._headers, line)
            if image_response is not None:
                yield image_response
        except Exception as e:
            yield e
        if line["message"]["author"]["role"] != "assistant":
            return
        if line["message"]["content"]["content_type"] != "text":
            return
        if line["message"]["metadata"]["message_type"] not in ("next", "continue", "variant"):
            return
        if fields.conversation_id is None:
            fields.conversation_id = line["conversation_id"]
            fields.message_id = line["message"]["id"]
        if "parts" in line["message"]["content"]:
            yield line["message"]["content"]["parts"][0]
        if "finish_details" in line["message"]["metadata"]:
            fields.finish_reason = line["message"]["metadata"]["finish_details"]["type"]

    @classmethod
    async def webview_access_token(cls) -> str:
        window = webview.create_window("OpenAI Chat", cls.url)
        await asyncio.sleep(3)
        prompt_input = None
        while not prompt_input:
            try:
                await asyncio.sleep(1)
                prompt_input = window.dom.get_element("#prompt-textarea")
            except:
                ...
        window.evaluate_js("""
this._fetch = this.fetch;
this.fetch = async (url, options) => {
    const response = await this._fetch(url, options);
    if (url == "https://chat.openai.com/backend-api/conversation") {
        this._headers = options.headers;
        return response;
    }
    return response;
};
""")
        window.evaluate_js("""
            document.querySelector('.from-token-main-surface-secondary').click();
        """)
        headers = None
        while headers is None:
            headers = window.evaluate_js("this._headers")
            await asyncio.sleep(1)
        headers["User-Agent"] = window.evaluate_js("this.navigator.userAgent")
        cookies = [list(*cookie.items()) for cookie in window.get_cookies()]
        window.destroy()
        cls._cookies = dict([(name, cookie.value) for name, cookie in cookies])
        cls._headers = headers
        cls._expires = int(time.time()) + 60 * 60 * 4
        cls._update_cookie_header()

    @classmethod
    def browse_access_token(cls, proxy: str = None, timeout: int = 1200) -> None:
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
                "let expires = new Date(); expires.setTime(expires.getTime() + 60 * 60 * 4 * 1000);"
                "document.cookie = 'access_token=' + accessToken + ';expires=' + expires.toUTCString() + ';path=/';"
                "return accessToken;"
            )
            args = get_args_from_browser(f"{cls.url}/", driver, do_bypass_cloudflare=False)
            cls._headers = args["headers"]
            cls._cookies = args["cookies"]
            cls._update_cookie_header()
            cls._set_api_key(access_token)
        finally:
            driver.close()

    @classmethod
    async def fetch_access_token(cls, session: StreamSession, headers: dict):
        async with session.get(
            f"{cls.url}/api/auth/session",
            headers=headers
        ) as response:
            if response.ok:
                data = await response.json()
                if "accessToken" in data:
                    return data["accessToken"]

    @staticmethod
    def _format_cookies(cookies: Cookies):
        return "; ".join(f"{k}={v}" for k, v in cookies.items() if k != "access_token")

    @classmethod
    def _create_request_args(cls, cookies: Union[Cookies, None]):
        cls._headers = {
            "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
        }
        cls._cookies = {} if cookies is None else cookies
        cls._update_cookie_header()

    @classmethod
    def _update_request_args(cls, session: StreamSession):
        for c in session.cookie_jar if hasattr(session, "cookie_jar") else session.cookies.jar:
            cls._cookies[c.key if hasattr(c, "key") else c.name] = c.value
        cls._update_cookie_header()

    @classmethod
    def _set_api_key(cls, api_key: str):
        cls._api_key = api_key
        cls._expires = int(time.time()) + 60 * 60 * 4
        cls._headers["Authorization"] = f"Bearer {api_key}"

    @classmethod
    def _update_cookie_header(cls):
        cls._headers["Cookie"] = cls._format_cookies(cls._cookies)

class Conversation(BaseConversation):
    """
    Class to encapsulate response fields.
    """
    def __init__(self, conversation_id: str = None, message_id: str = None, finish_reason: str = None):
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.finish_reason = finish_reason

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

    async def generator(self) -> AsyncIterator:
        if self._generator is not None:
            self._generator = None
            chunks = []
            async for chunk in self._generator:
                if isinstance(chunk, Conversation):
                    self._fields = chunk
                else:
                    yield chunk
                    chunks.append(str(chunk))
            self._message = "".join(chunks)
            if self._fields is None:
                raise RuntimeError("Missing response fields")
            self.is_end = self._fields.finish_reason == "stop"

    def __aiter__(self):
        return self.generator()

    async def get_message(self) -> str:
        await self.generator()
        return self._message

    async def get_fields(self) -> dict:
        await self.generator()
        return {
            "conversation_id": self._fields.conversation_id,
            "parent_id": self._fields.message_id
        }

    async def create_next(self, prompt: str, **kwargs) -> Response:
        return await OpenaiChat.create(
            **self._options,
            prompt=prompt,
            messages=await self.get_messages(),
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
            messages=await self.get_messages(),
            action="continue",
            **fields,
            **kwargs
        )

    async def create_variant(self, **kwargs) -> Response:
        if self.action != "next":
            raise RuntimeError("Can't create variant from continue or variant request.")
        return await OpenaiChat.create(
            **self._options,
            messages=self._messages,
            action="variant",
            **await self.get_fields(),
            **kwargs
        )

    async def get_messages(self) -> list:
        messages = self._messages
        messages.append({"role": "assistant", "content": await self.message()})
        return messages