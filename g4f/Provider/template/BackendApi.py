from __future__ import annotations

import re
import json
import time
from urllib.parse import quote_plus

from ...typing import Messages, AsyncResult
from ...requests import StreamSession
from ...providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...providers.response import *
from ...image import get_image_extension
from ...errors import ModelNotSupportedError
from ..needs_auth.OpenaiAccount import OpenaiAccount
from ..hf.HuggingChat import HuggingChat
from ... import debug

class BackendApi(AsyncGeneratorProvider, ProviderModelMixin):
    ssl = False

    models = [
        *OpenaiAccount.get_models(),
        *HuggingChat.get_models(),
        "flux",
        "flux-pro",
        "MiniMax-01",
        "Microsoft Copilot",
    ]

    @classmethod
    def get_model(cls, model: str):
        if "MiniMax" in model:
            model = "MiniMax"
        elif "Copilot" in model:
            model = "Copilot"
        elif "FLUX" in model:
            model = f"flux-{model.split('-')[-1]}"
        elif "flux" in model:
            model = model.split(' ')[-1]
        elif model in OpenaiAccount.get_models():
            pass
        elif model in HuggingChat.get_models():
            pass
        else:
            raise ModelNotSupportedError(f"Model: {model}")
        return model

    @classmethod
    def get_provider(cls, model: str):
        if model.startswith("MiniMax"):
            return "HailuoAI"
        elif model == "Copilot":
            return "CopilotAccount"
        elif model in OpenaiAccount.get_models():
            return "OpenaiAccount"
        elif model in HuggingChat.get_models():
            return "HuggingChat"
        return None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        proxy: str = None,
        timeout: int = 0,
        **kwargs
    ) -> AsyncResult:
        debug.log(f"{__name__}: {api_key}")

        async with StreamSession(
            proxy=proxy,
            headers={"Accept": "text/event-stream"},
            timeout=timeout
        ) as session:
            model = cls.get_model(model)
            provider = cls.get_provider(model)
            async with session.post(f"{cls.url}/backend-api/v2/conversation", json={
                "model": model,
                "messages": messages,
                "provider": provider,
                **kwargs
            }, ssl=cls.ssl) as response:
                async for line in response.iter_lines():
                    data = json.loads(line)
                    data_type = data.pop("type")
                    if data_type == "provider":
                        yield ProviderInfo(**data[data_type])
                        provider = data[data_type]["name"]
                    elif data_type == "conversation":
                        yield JsonConversation(**data[data_type][provider] if provider in data[data_type] else data[data_type][""])
                    elif data_type == "conversation_id":
                        pass
                    elif data_type == "message":
                        yield Exception(data)
                    elif data_type == "preview":
                        yield PreviewResponse(data[data_type])
                    elif data_type == "content":
                        def on_image(match):
                            extension = get_image_extension(match.group(3))
                            filename = f"{int(time.time())}_{quote_plus(match.group(1)[:100], '')}{extension}"
                            download_url = f"/download/{filename}?url={cls.url}{match.group(3)}"
                            return f"[![{match.group(1)}]({download_url})](/images/{filename})"
                        yield re.sub(r'\[\!\[(.+?)\]\(([^)]+?)\)\]\(([^)]+?)\)', on_image, data["content"])
                    elif data_type =="synthesize":
                        yield SynthesizeData(**data[data_type])
                    elif data_type == "parameters":
                        yield Parameters(**data[data_type])
                    elif data_type == "usage":
                        yield Usage(**data[data_type])
                    elif data_type == "reasoning":
                        yield Reasoning(**data)
                    elif data_type == "login":
                        pass
                    elif data_type == "title":
                        yield TitleGeneration(data[data_type])
                    elif data_type == "finish":
                        yield FinishReason(data[data_type]["reason"])
                    elif data_type == "log":
                        yield DebugResponse.from_dict(data[data_type])
                    else:
                        yield DebugResponse.from_dict(data)
