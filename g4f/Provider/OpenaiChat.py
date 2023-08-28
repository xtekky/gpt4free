has_module = True
try:
    from revChatGPT.V1 import AsyncChatbot
except ImportError:
    has_module = False

from .base_provider import AsyncGeneratorProvider, get_cookies
from ..typing       import AsyncGenerator

class OpenaiChat(AsyncGeneratorProvider):
    url                   = "https://chat.openai.com"
    needs_auth            = True
    working               = has_module
    supports_gpt_35_turbo = True
    supports_gpt_4        = True
    supports_stream       = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        access_token: str = None,
        cookies: dict = None,
        **kwargs
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
            cookies      = cookies if cookies else get_cookies("chat.openai.com")
            response     = await bot.session.get("https://chat.openai.com/api/auth/session", cookies=cookies)
            access_token = response.json()["accessToken"]
            bot.set_access_token(access_token)

        if len(messages) > 1:
            formatted = "\n".join(
                ["%s: %s" % ((message["role"]).capitalize(), message["content"]) for message in messages]
            )
            prompt = f"{formatted}\nAssistant:"
        else:
            prompt = messages.pop()["content"]

        returned = None
        async for message in bot.ask(prompt):
            message = message["message"]
            if returned:
                if message.startswith(returned):
                    new = message[len(returned):]
                    if new:
                        yield new
            else:
                yield message
            returned = message

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
