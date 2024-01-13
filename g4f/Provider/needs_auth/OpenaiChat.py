from __future__ import annotations

import uuid, json, asyncio, os
from py_arkose_generator.arkose import get_values_for_request
from async_property import async_cached_property
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from ..base_provider import AsyncGeneratorProvider
from ..helper import format_prompt, get_cookies
from ...webdriver import get_browser, get_driver_cookies
from ...typing import AsyncResult, Messages
from ...requests import StreamSession
from ...image import to_image, to_bytes, ImageType, ImageResponse
from ... import debug

models = {
    "gpt-3.5":       "text-davinci-002-render-sha",
    "gpt-3.5-turbo": "text-davinci-002-render-sha",
    "gpt-4":         "gpt-4",
    "gpt-4-gizmo":   "gpt-4-gizmo"
}

class OpenaiChat(AsyncGeneratorProvider):
    url                   = "https://chat.openai.com"
    working               = True
    needs_auth            = True
    supports_gpt_35_turbo = True
    supports_gpt_4        = True
    _cookies: dict        = {}
    _default_model: str   = None

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
        image: ImageType
    ) -> ImageResponse:
        image = to_image(image)
        extension = image.format.lower()
        data_bytes = to_bytes(image)
        data = {
            "file_name": f"{image.width}x{image.height}.{extension}",
            "file_size": len(data_bytes),
            "use_case":	"multimodal"
        }
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
        async with session.put(
            image_data["upload_url"],
            data=data_bytes,
            headers={
                "Content-Type": image_data["mime_type"],
                "x-ms-blob-type": "BlockBlob"
            }
        ) as response:
            response.raise_for_status()
        async with session.post(
            f"{cls.url}/backend-api/files/{image_data['file_id']}/uploaded",
            json={},
            headers=headers
        ) as response:
            response.raise_for_status()
            download_url = (await response.json())["download_url"]
        return ImageResponse(download_url, image_data["file_name"], image_data)
    
    @classmethod
    async def get_default_model(cls, session: StreamSession, headers: dict):
        if cls._default_model:
            model =  cls._default_model
        else:
            async with session.get(f"{cls.url}/backend-api/models", headers=headers) as response:
                data = await response.json()
                if "categories" in data:
                    model = data["categories"][-1]["default_model"]
                else:
                    RuntimeError(f"Response: {data}")
            cls._default_model = model
        return model
    
    @classmethod
    def create_messages(cls, prompt: str, image_response: ImageResponse = None):
        if not image_response:
            content = {"content_type": "text", "parts": [prompt]}
        else:
            content = {
                "content_type": "multimodal_text",
                "parts": [{
                    "asset_pointer": f"file-service://{image_response.get('file_id')}",
                    "height": image_response.get("height"),
                    "size_bytes": image_response.get("file_size"),
                    "width": image_response.get("width"),
                }, prompt]
            }
        messages = [{
            "id": str(uuid.uuid4()),
            "author": {"role": "user"},
            "content": content,
        }]
        if image_response:
            messages[0]["metadata"] = {
                "attachments": [{
                    "height": image_response.get("height"),
                    "id": image_response.get("file_id"),
                    "mimeType": image_response.get("mime_type"),
                    "name": image_response.get("file_name"),
                    "size": image_response.get("file_size"),
                    "width": image_response.get("width"),
                }]
            }
        return messages
    
    @classmethod
    async def get_image_response(cls, session: StreamSession, headers: dict, line: dict):
        if "parts" in line["message"]["content"]:
            part = line["message"]["content"]["parts"][0]
            if "asset_pointer" in part and part["metadata"]:
                file_id = part["asset_pointer"].split("file-service://", 1)[1]
                prompt = part["metadata"]["dalle"]["prompt"]
                async with session.get(
                    f"{cls.url}/backend-api/files/{file_id}/download",
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    download_url = (await response.json())["download_url"]
                    return ImageResponse(download_url, prompt)

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        access_token: str = None,
        cookies: dict = None,
        auto_continue: bool = False,
        history_disabled: bool = True,
        action: str = "next",
        conversation_id: str = None,
        parent_id: str = None,
        image: ImageType = None,
        response_fields: bool = False,
        **kwargs
    ) -> AsyncResult:
        if model in models:
            model = models[model]
        if not parent_id:
            parent_id = str(uuid.uuid4())
        if not cookies:
            cookies = cls._cookies
        if not access_token:
            if not cookies:
                cls._cookies = cookies = get_cookies("chat.openai.com")
            if "access_token" in cookies:
                access_token = cookies["access_token"]
        if not access_token:
            login_url = os.environ.get("G4F_LOGIN_URL")
            if login_url:
                yield f"Please login: [ChatGPT]({login_url})\n\n"
            access_token, cookies = cls.browse_access_token(proxy)
            cls._cookies = cookies
        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        async with StreamSession(
            proxies={"https": proxy},
            impersonate="chrome110",
            timeout=timeout,
            cookies=dict([(name, value) for name, value in cookies.items() if name == "_puid"])
        ) as session:
            if not model:
                model =  await cls.get_default_model(session, headers)
            try:
                image_response = None
                if image:
                    image_response = await cls.upload_image(session, headers, image)
                    yield image_response
            except Exception as e:
                yield e
            end_turn = EndTurn()
            while not end_turn.is_end:
                data = {
                    "action": action,
                    "arkose_token": await cls.get_arkose_token(session),
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
                    headers={"Accept": "text/event-stream", **headers}
                ) as response:
                    try:
                        response.raise_for_status()
                    except:
                        raise RuntimeError(f"Response {response.status_code}: {await response.text()}")
                    try:
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
                                image_response = await cls.get_image_response(session, headers, line)
                                if image_response:
                                    yield image_response
                            except Exception as e:
                                yield e
                            if line["message"]["author"]["role"] != "assistant":
                                continue
                            if line["message"]["metadata"]["message_type"] in ("next", "continue", "variant"):
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
                                    break
                    except Exception as e:
                        yield e
                if not auto_continue:
                    break
                action = "continue"
                await asyncio.sleep(5)
            if history_disabled:
                async with session.patch(
                    f"{cls.url}/backend-api/conversation/{conversation_id}",
                    json={"is_visible": False},
                    headers=headers
                ) as response:
                    response.raise_for_status()

    @classmethod
    def browse_access_token(cls, proxy: str = None) -> tuple[str, dict]:
        driver = get_browser(proxy=proxy)
        try:
            driver.get(f"{cls.url}/")
            WebDriverWait(driver, 1200).until(
                EC.presence_of_element_located((By.ID, "prompt-textarea"))
            )
            javascript = """
access_token = (await (await fetch('/api/auth/session')).json())['accessToken'];
expires = new Date(); expires.setTime(expires.getTime() + 60 * 60 * 24 * 7); // One week
document.cookie = 'access_token=' + access_token + ';expires=' + expires.toUTCString() + ';path=/';
return access_token;
"""
            return driver.execute_script(javascript), get_driver_cookies(driver)
        finally:
            driver.quit()

    @classmethod     
    async def get_arkose_token(cls, session: StreamSession) -> str:
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
        
class EndTurn():
    def __init__(self):
        self.is_end = False

    def end(self):
        self.is_end = True

class ResponseFields():
    def __init__(
        self,
        conversation_id: str,
        message_id: str,
        end_turn: EndTurn
    ):
        self.conversation_id = conversation_id
        self.message_id = message_id
        self._end_turn = end_turn
        
class Response():
    def __init__(
        self,
        generator: AsyncResult,
        action: str,
        messages: Messages,
        options: dict
    ):
        self._generator = generator
        self.action: str = action
        self.is_end: bool = False
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
        [_ async for _ in self.generator()]
        return self._message
    
    async def get_fields(self):
        [_ async for _ in self.generator()]
        return {
            "conversation_id": self._fields.conversation_id,
            "parent_id": self._fields.message_id,
        }
    
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
        messages.append({
            "role": "assistant", "content": await self.message
        })
        return messages