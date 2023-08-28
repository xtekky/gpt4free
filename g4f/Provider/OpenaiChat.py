has_module = True
try:
    from revChatGPT.V1 import AsyncChatbot
except ImportError:
    has_module = False

from .base_provider import AsyncGeneratorProvider, get_cookies, format_prompt
from ..typing import AsyncGenerator
from httpx import AsyncClient
import json


class OpenaiChat(AsyncGeneratorProvider):
    url                   = "https://chat.openai.com"
    needs_auth            = True
    working               = has_module
    supports_gpt_35_turbo = True
    supports_gpt_4        = True
    supports_stream       = True
    _access_token         = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        access_token: str = _access_token,
        cookies: dict = None,
        **kwargs: dict
    ) -> AsyncGenerator:
        
        config = {"access_token": access_token, "model": model}
        if proxy:
            if "://" not in proxy:
                proxy = f"http://{proxy}"
            config["proxy"] = proxy

        bot = AsyncChatbot(
            config=config
        )

        if not access_token:
            cookies = cookies if cookies else get_cookies("chat.openai.com")
            cls._access_token = await get_access_token(bot.session, cookies)
            bot.set_access_token(cls._access_token)

        returned = None
        async for message in bot.ask(format_prompt(messages)):
            message = message["message"]
            if returned:
                if message.startswith(returned):
                    new = message[len(returned):]
                    if new:
                        yield new
            else:
                yield message
            returned = message
        
        await bot.delete_conversation(bot.conversation_id)


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
    

async def get_access_token(session: AsyncClient, cookies: dict):
    response = await session.get("https://chat.openai.com/api/auth/session", cookies=cookies)
    response.raise_for_status()
    try:
        return response.json()["accessToken"]
    except json.decoder.JSONDecodeError:
        raise RuntimeError(f"Response: {response.text}")