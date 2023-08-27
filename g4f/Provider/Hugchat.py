has_module = False
try:
    from hugchat.hugchat import ChatBot
except ImportError:
    has_module = False

from .base_provider import BaseProvider, get_cookies
from g4f.typing     import CreateResult

class Hugchat(BaseProvider):
    url        = "https://huggingface.co/chat/"
    needs_auth = True
    working    = has_module
    llms       = ['OpenAssistant/oasst-sft-6-llama-30b-xor', 'meta-llama/Llama-2-70b-chat-hf']

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        proxy: str = None,
        cookies: str = get_cookies(".huggingface.co"), **kwargs) -> CreateResult:
        
        bot = ChatBot(
            cookies=cookies)
        
        if proxy and "://" not in proxy:
            proxy = f"http://{proxy}"
            bot.session.proxies = {"http": proxy, "https": proxy}

        if model:
            try:
                if not isinstance(model, int):
                    model = cls.llms.index(model)
                bot.switch_llm(model)
            except:
                raise RuntimeError(f"Model are not supported: {model}")

        if len(messages) > 1:
            formatted = "\n".join(
                ["%s: %s" % (message["role"], message["content"]) for message in messages]
            )
            prompt = f"{formatted}\nAssistant:"
        else:
            prompt = messages.pop()["content"]

        try:
            yield bot.chat(prompt, **kwargs)
        finally:
            bot.delete_conversation(bot.current_conversation)
            bot.current_conversation = ""
            pass

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
